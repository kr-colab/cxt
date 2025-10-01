# TokenFreeDecoder
import os
import torch
import torch.nn as nn
import numpy as np
import lightning as L
torch.random.manual_seed(0)
from dataclasses import dataclass
from cxt.model import TokenFreeDecoder
from torch.utils.data import DataLoader
from cxt.dataset import LazyDataset, MultiDirLazyDataset
import math
import argparse


def generate_causal_mask(seq_len, full_attention_n=None, device="cpu"):
    full_attention_n = full_attention_n if full_attention_n is not None else 0
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    mask[:full_attention_n, :full_attention_n] = 1  # Full attention for first n tokens
    return mask.bool().unsqueeze(0).unsqueeze(0)

num_gpus = 2
config = {
    'training': {
        'max_steps': 50_000 * 2,
        'max_lr': 3e-4,
        'min_lr': 3e-4 * 0.1,
        'warmup_iters': 10,
        'lr_decay_iters': 50_000 * 3,
        'batch_size': 128,#196, #32
        'grad_accum_steps': 6,#1, 
        'weight_decay': 0.1,
        'betas': (0.9, 0.95),
        'num_workers': 16,#8,
        'prefetch_factor': 2,
    }
}


import torch
import torch.nn as nn
import torch.nn.functional as F

class IEAdapter(nn.Module):
    def __init__(self, ie_in: int, ie_out: int = 50, bottleneck: int = 64, dropout: float = 0.1, use_se: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(ie_in)
        self.use_se = use_se
        if use_se:
            hidden = max(ie_in // 4, 1)
            self.se = nn.Sequential(
                nn.Linear(ie_in, hidden, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, ie_in, bias=True),
                nn.Sigmoid()
            )

        # SwiGLU-style gated bottleneck
        self.fc_in  = nn.Linear(ie_in, 2 * bottleneck, bias=False)
        self.fc_out = nn.Linear(bottleneck, ie_out, bias=True)
        self.drop   = nn.Dropout(dropout)

        # Residual projection (identity if shapes match)
        self.proj = nn.Identity() if ie_in == ie_out else nn.Linear(ie_in, ie_out, bias=True)
        # Small learnable residual scale to stabilize early training
        self.alpha = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x):
        # x: [B, XX, WS, NW, IE_in]
        if self.use_se:
            # channel-wise squeeze-excite across spatial dims
            se_scale = self.se(x.mean(dim=(1,2,3)))  # [B, IE_in]
            x = x * se_scale[:, None, None, None, :]

        x_res = self.proj(x)

        z = self.ln(x)
        z = self.fc_in(z)
        v, g = torch.chunk(z, 2, dim=-1)
        z = F.silu(g) * v
        z = self.drop(z)
        z = self.fc_out(z)

        return x_res + self.alpha * z


def set_trainable(module, flag: bool):
    for p in module.parameters(): p.requires_grad_(flag)

def select_backbone_params(backbone, strategy: str = "none", last_n: int = 2):
    """
    strategy ∈ {"none", "ln", "lastN", "ln_lastN"}.
    - "ln": all LayerNorm/Norm scale+bias
    - "lastN": parameters in the last N transformer blocks
    - "ln_lastN": union of the above
    """
    if strategy == "none": return []
    wants_ln = "ln" in strategy
    wants_last = "lastN" in strategy

    params = []
    named = list(backbone.named_parameters())
    # mark all frozen first
    set_trainable(backbone, False)

    # (A) LayerNorms
    if wants_ln:
        for name, p in named:
            if any(k in name.lower() for k in ("ln", "norm")):
                p.requires_grad_(True); params.append(p)

    # (B) last N blocks (assumes backbone.transformer.h is a list of blocks)
    if wants_last and hasattr(backbone, "transformer") and hasattr(backbone.transformer, "h"):
        blocks = backbone.transformer.h[-last_n:]
        for blk in blocks:
            set_trainable(blk, True)
        for name, p in named:
            if p.requires_grad: params.append(p)

    # de-dup
    return list({id(p): p for p in params}.values())
# ---------------------------------------------------------------------------

class FrozenDecoderWithAdapter(nn.Module):
    def __init__(self, backbone: nn.Module, ie_in: int, ie_out: int = 50,
                 adapter_bottleneck: int = 196, adapter_dropout: float = 0.1,
                 new_mask_index: int | None = 0,
                 unfreeze: str = "ln_lastN", last_n: int = 2):
        super().__init__()
        # freeze all, then selectively unfreeze
        set_trainable(backbone, False)
        backbone.eval()

        self.backbone = backbone
        self.adapter  = IEAdapter(ie_in, ie_out, adapter_bottleneck, adapter_dropout, use_se=True)

        self.new_mask_index = new_mask_index
        if new_mask_index is not None:
            self.mask_token = nn.Parameter(torch.zeros(()))

        # record which backbone params to train
        self.trainable_backbone_params = select_backbone_params(backbone, strategy=unfreeze, last_n=last_n)

    def forward(self, x, y, attn_mask):
        if self.new_mask_index is not None:
            x[..., self.new_mask_index] = self.mask_token
        x = self.adapter(x)
        return self.backbone(x, y, attn_mask)




import lightning as L
import math

class LitTokenFreeDecoder(L.LightningModule):
    def __init__(self, gpt_config, pretrained_ckpt: str | None = None,
                 ie_new: int | None = None, adapter_bottleneck: int = 32, adapter_dropout: float = 0.0,
                 new_mask_index: int | None = 0, training_config: dict | None = None):
        super().__init__()
        from cxt.model import TokenFreeDecoder  # your original model

        self.training_config = training_config or {
            'max_lr': 3e-4,
            'min_lr': 3e-5,
            'warmup_iters': 10,
            'lr_decay_iters': 150_000,
            'weight_decay': 0.1,
            'betas': (0.9, 0.95),
        }

        # 1) build backbone
        backbone = TokenFreeDecoder(gpt_config)

        # 2) (optional) load pretrained checkpoint weights into backbone
        if pretrained_ckpt:
            ckpt = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            # strip 'model.' prefix if it exists (Lightning checkpoint)
            new_state = {}
            for k, v in state.items():
                kk = k
                if kk.startswith("model."):
                    kk = kk[len("model."):]
                new_state[kk] = v
            missing, unexpected = backbone.load_state_dict(new_state, strict=False)
            if len(missing) or len(unexpected):
                print("[load_state] missing:", missing)
                print("[load_state] unexpected:", unexpected)

        # 3) wrap with adapter if ie_new is provided and != 50
        if ie_new is not None and ie_new != gpt_config.num_samples:
            self.model = FrozenDecoderWithAdapter(
                backbone=backbone,
                ie_in=ie_new,
                ie_out=gpt_config.num_samples,            # 50
                adapter_bottleneck=adapter_bottleneck,
                adapter_dropout=adapter_dropout,
                new_mask_index=new_mask_index,
            )
            self._adapter_only = True
        else:
            # plain training / finetuning (not your use case now)
            self.model = backbone
            self._adapter_only = False

        self.save_hyperparameters(ignore=['model'])

    def training_step(self, batch, batch_idx):
        x, y = batch
        attn_mask = generate_causal_mask(1001, full_attention_n=501, device=x.device).repeat(x.size(0), 1, 1, 1)
        logits, loss = self.model(x, y, attn_mask)
        self.log("train_loss", loss, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        attn_mask = generate_causal_mask(1001, full_attention_n=501, device=x.device).repeat(x.size(0), 1, 1, 1)
        logits, loss = self.model(x, y, attn_mask)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        ad_lr  = self.training_config['max_lr']         # e.g. 3e-4 to 1e-3
        bb_lr  = ad_lr * 0.05                           # tiny (e.g. 1/20)
        wd     = self.training_config['weight_decay']

        groups = [
            {"params": list(self.model.adapter.parameters()), "lr": ad_lr, "weight_decay": wd},
        ]
        bb_params = getattr(self.model, "trainable_backbone_params", [])
        if bb_params:
            # no weight decay on norms/biases
            decay, nodecay = [], []
            for p in bb_params:
                (nodecay if p.ndim == 1 else decay).append(p)
            if decay:   groups.append({"params": decay,   "lr": bb_lr, "weight_decay": wd})
            if nodecay: groups.append({"params": nodecay, "lr": bb_lr, "weight_decay": 0.0})

        opt = torch.optim.AdamW(groups, betas=self.training_config['betas'])
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=self._get_lr_schedule_function)
        return [opt], [{"scheduler": sch, "interval": "step"}]



    def _get_lr_schedule_function(self, current_step):
        config = self.training_config
        # Linear warmup
        if current_step < config['warmup_iters']:
            return float(current_step) / float(max(1, config['warmup_iters']))
        # Cosine decay
        if current_step > config['lr_decay_iters']:
            return config['min_lr'] / config['max_lr']
        decay_ratio = (current_step - config['warmup_iters']) / (
            config['lr_decay_iters'] - config['warmup_iters']
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config['min_lr'] / config['max_lr'] + coeff * (1.0 - config['min_lr'] / config['max_lr'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TokenFreeDecoder model")
    parser.add_argument('--dataset_path', type=str, default='/sietch_colab/kkor/cxt/ts/processed_n10', help='Path to the dataset')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2], help='List of GPUs to use')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the checkpoint')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for the optimizer')
    parser.add_argument('--test_batches', type=int, default=100, help='Tiny batch size of 1000')

    args = parser.parse_args()
    dataset_path = args.dataset_path
    gpus = args.gpus
    checkpoint_path = args.checkpoint_path
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    test_batches = args.test_batches
    config['training']['max_lr'] = learning_rate

    # narrow model
    @dataclass
    class TokenFreeDecoderConfig:
        num_samples: int = 50
        sample_scale_embd: int = 2
        output_dim: int = 256+2
        n_embd: int = 400 #768
        combined_dim: int = 1001
        n_layer: int = 6 #12
        bias: bool = False
        dropout: float = 0.1
        n_head: int = 4 #8
        device: str = "cuda"
        batch_size: int = config['training']['batch_size']

    
    # broad model
    @dataclass
    class TokenFreeDecoderConfig:
        num_samples: int = 50
        sample_scale_embd: int = 2
        output_dim: int = 324+2
        n_embd: int = 400 
        combined_dim: int = 1001
        n_layer: int = 10
        bias: bool = False
        dropout: float = 0.1
        n_head: int = 4
        device: str = "cuda"
        batch_size: int = config['training']['batch_size']
    

    

    # Check if dataset_path contains multiple subdirectories
    """
    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    if len(subdirs) > 1:
        print("Using multi directory dataset!")
        train_dataset = MultiDirLazyDataset(root_dir=dataset_path, split='train', test_ratio=0.1)
        test_dataset = MultiDirLazyDataset(root_dir=dataset_path, split='test', test_ratio=0.1)
        print(f"Training samples: {len(train_dataset)}")
        print(f"Testing samples: {len(test_dataset)}")
    else:
        print("Using single directory dataset!")
        train_dataset = LazyDataset(dataset_path, split='train', test_batches=test_batches)
        test_dataset = LazyDataset(dataset_path, split='test', test_batches=test_batches)
        print(f"training dataset {len(train_dataset)} samples")
        print(f"test dataset {len(test_dataset)} samples")
    """

    
    from cxt.dataset2 import PairDataset, ShuffleBufferDataset
    from torch.utils.data import DistributedSampler


    train_dataset = PairDataset(root=dataset_path, split="train", mmap=True)
    #sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)

    #train_dataset.shuffle_files(seed=1234)  # O(#files), no disk I/O
    #train_dataset = ShuffleBufferDataset(train_dataset, buffer_size=4096*8, seed=1234)

    #loader = DataLoader(ds, batch_size=196, shuffle=False, num_workers=32, drop_last=True, prefetch_factor=4)
    test_dataset = PairDataset(root=dataset_path, split="test", mmap=True)
    #test_dataset = ShuffleBufferDataset(test_dataset, buffer_size=4096*8, seed=5678)


    

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        shuffle=True, # taken care by shuffle buffer
        persistent_workers=True,
        prefetch_factor=config['training']['prefetch_factor'],
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        shuffle=False,
        persistent_workers=True,
        prefetch_factor=config['training']['prefetch_factor'],
        drop_last=True
    )
    gpt_config = TokenFreeDecoderConfig()
    

    lit_model = LitTokenFreeDecoder(
        gpt_config=gpt_config,
        pretrained_ckpt='/home/kkor/cxt/cxt/lightning_logs/version_20/checkpoints/epoch=1-step=5280.ckpt',    
        ie_new=10,          
        adapter_bottleneck=32,
        adapter_dropout=0.0,
        new_mask_index=0,
        training_config=config['training'],
    )
    

    torch.set_float32_matmul_precision('medium')
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices=gpus,
        precision="bf16-mixed",
        strategy="ddp", # not in interactive mode
        #detect_anomaly=True,
        #gradient_clip_val=1.0, # because of fused optim 
        accumulate_grad_batches=config['training']['grad_accum_steps']
    )
    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=test_loader
    )