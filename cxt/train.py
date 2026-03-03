"""Unified training script for cxt models.

Replaces train.py, train2.py, train2_n10.py with a single entry point.
Select model variant and training mode via CLI flags:

    python -m cxt.train --model broad --gpus 0 1 2 --epochs 2
    python -m cxt.train --model broad_w200 --checkpoint broad --lr 3e-5
    python -m cxt.train --model broad --adapter --adapter-samples 10 --checkpoint broad
    python -m cxt.train --model broad
    python -m cxt.train --model w200_wmissing --checkpoint broad_w200 --lr 3e-5
"""

from __future__ import annotations

import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader

from cxt.config import PRESETS, TrainingConfig
from cxt.model import TokenFreeDecoder
from cxt.translate import generate_causal_mask


# ---------------------------------------------------------------------------
# Adapter modules (kept from train2_n10.py, cleaned up)
# ---------------------------------------------------------------------------

class IEAdapter(nn.Module):
    """Sample-size adapter with SwiGLU bottleneck and optional squeeze-excite."""

    def __init__(self, ie_in: int, ie_out: int = 50, bottleneck: int = 64,
                 dropout: float = 0.1, use_se: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(ie_in)
        self.use_se = use_se
        if use_se:
            hidden = max(ie_in // 4, 1)
            self.se = nn.Sequential(
                nn.Linear(ie_in, hidden, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, ie_in, bias=True),
                nn.Sigmoid(),
            )
        self.fc_in = nn.Linear(ie_in, 2 * bottleneck, bias=False)
        self.fc_out = nn.Linear(bottleneck, ie_out, bias=True)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Identity() if ie_in == ie_out else nn.Linear(ie_in, ie_out, bias=True)
        self.alpha = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x):
        if self.use_se:
            se_scale = self.se(x.mean(dim=(1, 2, 3)))
            x = x * se_scale[:, None, None, None, :]
        x_res = self.proj(x)
        z = self.ln(x)
        z = self.fc_in(z)
        v, g = torch.chunk(z, 2, dim=-1)
        z = F.silu(g) * v
        z = self.drop(z)
        z = self.fc_out(z)
        return x_res + self.alpha * z


def _set_trainable(module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)


def _select_backbone_params(backbone, strategy: str = "none", last_n: int = 2):
    if strategy == "none":
        return []
    wants_ln = "ln" in strategy
    wants_last = "lastN" in strategy
    _set_trainable(backbone, False)
    params = []
    named = list(backbone.named_parameters())
    if wants_ln:
        for name, p in named:
            if any(k in name.lower() for k in ("ln", "norm")):
                p.requires_grad_(True)
                params.append(p)
    if wants_last and hasattr(backbone, "transformer") and hasattr(backbone.transformer, "h"):
        for blk in backbone.transformer.h[-last_n:]:
            _set_trainable(blk, True)
        for _, p in named:
            if p.requires_grad:
                params.append(p)
    return list({id(p): p for p in params}.values())


class FrozenDecoderWithAdapter(nn.Module):
    """Backbone + adapter wrapper for low-sample-size fine-tuning."""

    def __init__(self, backbone: nn.Module, ie_in: int, ie_out: int = 50,
                 adapter_bottleneck: int = 196, adapter_dropout: float = 0.1,
                 new_mask_index: int | None = 0,
                 unfreeze: str = "ln_lastN", last_n: int = 2):
        super().__init__()
        _set_trainable(backbone, False)
        backbone.eval()
        self.backbone = backbone
        self.adapter = IEAdapter(ie_in, ie_out, adapter_bottleneck, adapter_dropout, use_se=True)
        self.new_mask_index = new_mask_index
        if new_mask_index is not None:
            self.mask_token = nn.Parameter(torch.zeros(()))
        self.trainable_backbone_params = _select_backbone_params(backbone, strategy=unfreeze, last_n=last_n)

    def forward(self, x, y, attn_mask):
        if self.new_mask_index is not None:
            x[..., self.new_mask_index] = self.mask_token
        x = self.adapter(x)
        return self.backbone(x, y, attn_mask)


# ---------------------------------------------------------------------------
# Lightning modules
# ---------------------------------------------------------------------------

class LitDecoder(L.LightningModule):
    """Standard (non-adapter) training."""

    def __init__(self, gpt_config, training_config: dict | None = None):
        super().__init__()
        self.model = TokenFreeDecoder(gpt_config)
        self.training_config = training_config or TrainingConfig().__dict__
        if isinstance(self.training_config, TrainingConfig):
            self.training_config = self.training_config.__dict__
        self.save_hyperparameters(ignore=["model"])

    def training_step(self, batch, batch_idx):
        x, y = batch
        attn_mask = generate_causal_mask(1001, full_attention_n=501, device=x.device)
        attn_mask = attn_mask.repeat(x.size(0), 1, 1, 1)
        _, loss = self.model(x, y, attn_mask)
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        attn_mask = generate_causal_mask(1001, full_attention_n=501, device=x.device)
        attn_mask = attn_mask.repeat(x.size(0), 1, 1, 1)
        _, loss = self.model(x, y, attn_mask)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        tc = self.training_config
        opt = self.model.configure_optimizers(
            weight_decay=tc["weight_decay"],
            learning_rate=tc["max_lr"],
            betas=tc["betas"],
            device_type=self.device.type,
        )
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=self._lr_lambda)
        return [opt], [{"scheduler": sch, "interval": "step", "frequency": 1}]

    def _lr_lambda(self, step):
        tc = self.training_config
        if step < tc["warmup_iters"]:
            return float(step) / max(1, tc["warmup_iters"])
        if step > tc["lr_decay_iters"]:
            return tc["min_lr"] / tc["max_lr"]
        ratio = (step - tc["warmup_iters"]) / (tc["lr_decay_iters"] - tc["warmup_iters"])
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return tc["min_lr"] / tc["max_lr"] + coeff * (1.0 - tc["min_lr"] / tc["max_lr"])


class LitAdapterDecoder(L.LightningModule):
    """Adapter-based fine-tuning with frozen backbone."""

    def __init__(self, gpt_config, pretrained_ckpt: str | None = None,
                 ie_new: int | None = None, adapter_bottleneck: int = 32,
                 adapter_dropout: float = 0.0, new_mask_index: int | None = 0,
                 training_config: dict | None = None):
        super().__init__()
        self.training_config = training_config if isinstance(training_config, dict) else TrainingConfig().__dict__

        backbone = TokenFreeDecoder(gpt_config)
        if pretrained_ckpt and pretrained_ckpt != "1":
            ckpt = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            cleaned = {k.removeprefix("model."): v for k, v in state.items()}
            backbone.load_state_dict(cleaned, strict=False)

        if ie_new is not None and ie_new != gpt_config.num_samples:
            self.model = FrozenDecoderWithAdapter(
                backbone=backbone, ie_in=ie_new, ie_out=gpt_config.num_samples,
                adapter_bottleneck=adapter_bottleneck, adapter_dropout=adapter_dropout,
                new_mask_index=new_mask_index,
            )
        else:
            self.model = backbone

        self.save_hyperparameters(ignore=["model"])

    def training_step(self, batch, batch_idx):
        x, y = batch
        attn_mask = generate_causal_mask(1001, full_attention_n=501, device=x.device)
        attn_mask = attn_mask.repeat(x.size(0), 1, 1, 1)
        _, loss = self.model(x, y, attn_mask)
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        attn_mask = generate_causal_mask(1001, full_attention_n=501, device=x.device)
        attn_mask = attn_mask.repeat(x.size(0), 1, 1, 1)
        _, loss = self.model(x, y, attn_mask)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        tc = self.training_config
        ad_lr = tc["max_lr"]
        bb_lr = ad_lr * 0.05
        wd = tc["weight_decay"]

        groups = [{"params": list(self.model.adapter.parameters()), "lr": ad_lr, "weight_decay": wd}]
        bb_params = getattr(self.model, "trainable_backbone_params", [])
        if bb_params:
            decay = [p for p in bb_params if p.ndim >= 2]
            nodecay = [p for p in bb_params if p.ndim < 2]
            if decay:
                groups.append({"params": decay, "lr": bb_lr, "weight_decay": wd})
            if nodecay:
                groups.append({"params": nodecay, "lr": bb_lr, "weight_decay": 0.0})

        opt = torch.optim.AdamW(groups, betas=tc["betas"])
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=self._lr_lambda)
        return [opt], [{"scheduler": sch, "interval": "step"}]

    def _lr_lambda(self, step):
        tc = self.training_config
        if step < tc["warmup_iters"]:
            return float(step) / max(1, tc["warmup_iters"])
        if step > tc["lr_decay_iters"]:
            return tc["min_lr"] / tc["max_lr"]
        ratio = (step - tc["warmup_iters"]) / (tc["lr_decay_iters"] - tc["warmup_iters"])
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return tc["min_lr"] / tc["max_lr"] + coeff * (1.0 - tc["min_lr"] / tc["max_lr"])


# ---------------------------------------------------------------------------
# Backward compatibility aliases for checkpoint loading
# ---------------------------------------------------------------------------
LitTokenFreeDecoder = LitDecoder


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train cxt models")
    parser.add_argument("--model", type=str, default="broad", choices=list(PRESETS),
                        help="Model preset name")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to PairDataset root")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=6)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint path or model type name")
    parser.add_argument("--adapter", action="store_true",
                        help="Train adapter on frozen backbone")
    parser.add_argument("--adapter-samples", type=int, default=10,
                        help="Adapter input sample dimension (ie_new)")
    parser.add_argument("--adapter-bottleneck", type=int, default=32)
    parser.add_argument("--adapter-dropout", type=float, default=0.0)
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Root directory for lightning_logs (default: cwd)")

    args = parser.parse_args()

    # Config
    model_cfg = PRESETS[args.model].for_training(batch_size=args.batch_size, device="cuda")
    train_cfg = TrainingConfig(
        max_lr=args.lr,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        num_workers=args.workers,
    )

    # Dataset
    from cxt.dataset import PairDataset
    train_ds = PairDataset(args.dataset_path, split="train", mmap=True)
    test_ds = PairDataset(args.dataset_path, split="test", mmap=True)
    print(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers, pin_memory=True,
        shuffle=True, persistent_workers=True,
        prefetch_factor=train_cfg.prefetch_factor, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers, pin_memory=True,
        shuffle=False, persistent_workers=True,
        prefetch_factor=train_cfg.prefetch_factor, drop_last=True,
    )

    # Model
    if args.adapter:
        ckpt_path = args.checkpoint
        lit_model = LitAdapterDecoder(
            gpt_config=model_cfg,
            pretrained_ckpt=ckpt_path,
            ie_new=args.adapter_samples,
            adapter_bottleneck=args.adapter_bottleneck,
            adapter_dropout=args.adapter_dropout,
            new_mask_index=0,
            training_config=train_cfg.__dict__,
        )
    else:
        if args.checkpoint:
            lit_model = LitDecoder.load_from_checkpoint(
                args.checkpoint, gpt_config=model_cfg,
                training_config=train_cfg.__dict__,
            )
        else:
            lit_model = LitDecoder(model_cfg, training_config=train_cfg.__dict__)

    torch.set_float32_matmul_precision("medium")
    trainer_kwargs = dict(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=args.gpus,
        precision="bf16-mixed",
        strategy="ddp",
        accumulate_grad_batches=train_cfg.grad_accum_steps,
    )
    if args.log_dir:
        trainer_kwargs["default_root_dir"] = args.log_dir
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    main()
