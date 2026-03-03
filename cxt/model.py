"""TokenFreeDecoder -- the core cxt transformer model.

State-dict keys are preserved from the original implementation so that
existing checkpoints load without remapping:

    transformer.bt2ls.*          MutationsToLatentSpace
    transformer.ote.*            output token embedding
    transformer.drop.*           dropout
    transformer.h.{i}.*         Block layers
    transformer.ln_f.*           final LayerNorm
    lm_head.*                    prediction head
"""

import math
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from cxt.modules import MutationsToLatentSpace, Block, LayerNorm


class TokenFreeDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            bt2ls=MutationsToLatentSpace(config=config),
            ote=nn.Embedding(config.output_dim, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, use_bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.output_dim, bias=False)
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    # KV-cache helpers
    # ------------------------------------------------------------------
    def clear_cache(self):
        for block in self.transformer.h:
            if block.attn.cache_k is not None:
                block.attn.cache_k.zero_()
                block.attn.cache_v.zero_()

    def cache_to_device(self, device):
        for block in self.transformer.h:
            if block.attn.cache_k is not None:
                block.attn.cache_k = block.attn.cache_k.to(device)
                block.attn.cache_v = block.attn.cache_v.to(device)

    def enable_kv_cache(self, batch_size: int = 1):
        """Allocate KV caches after construction (e.g. for inference)."""
        device = next(self.parameters()).device
        for block in self.transformer.h:
            attn = block.attn
            shape = (batch_size, attn.n_head, 1001, attn.head_size)
            attn.cache_k = torch.zeros(shape, device=device)
            attn.cache_v = torch.zeros(shape, device=device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x, y, attn_mask, position=None, use_cache=False, calculate_loss=True):
        use_cache = use_cache or self.config.use_kv_cache

        if use_cache:
            if position is None or position == 0:
                src = self.transformer.bt2ls(x)
            else:
                src = self.transformer.ote(y).contiguous()

            for block in self.transformer.h:
                src = block(src, attn_mask, use_cache=True, position=position)

            src = self.transformer.ln_f(src)
            return self.lm_head(src).contiguous()

        # Training / prefill path
        B, Ty = y.size()
        NW = x.shape[3]

        x_enc = self.transformer.bt2ls(x)
        y_emb = self.transformer.ote(y)
        src = self.transformer.drop(torch.cat([x_enc, y_emb], dim=1))

        for block in self.transformer.h:
            src = block(src, attn_mask, use_cache=False)

        src = self.transformer.ln_f(src)
        logits = self.lm_head(src)[:, NW:, :].contiguous()

        if not calculate_loss:
            return logits

        pred = logits[:, :-1, :].reshape(-1, logits.size(-1))
        tgt = y[:, 1:].reshape(-1).long()
        loss = F.cross_entropy(pred, tgt)
        return logits, loss

    # ------------------------------------------------------------------
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
