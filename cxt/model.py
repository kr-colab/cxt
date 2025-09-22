import math
import torch
import inspect
import torch.nn as nn
import torch.nn.functional as F

from cxt.modules import MutationsToLatentSpace
from cxt.modules import Block
from cxt.modules import LayerNorm

class TokenFreeDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            bt2ls = MutationsToLatentSpace(config=config),
            ote = nn.Embedding(config.output_dim, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, use_bias=config.bias),))
        self.lm_head = nn.Linear(config.n_embd, config.output_dim, bias=False)
        self.cached_src = None
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def clear_cache(self):
        for i in range(self.config.n_layer):
            self.transformer.h[i].attn.cache_k *= 0. 
            self.transformer.h[i].attn.cache_v *= 0. 
    
    def cache_to_device(self, device):
        for i in range(self.config.n_layer):
            self.transformer.h[i].attn.cache_k = self.transformer.h[i].attn.cache_k.to(device)
            self.transformer.h[i].attn.cache_v = self.transformer.h[i].attn.cache_v.to(device)

    def forward(self, x, y, attn_mask, position=None, use_cache=False, calculate_loss=True):
        # ---------------------------
        # Decode / KV-cache path
        # ---------------------------
        if use_cache:
            if position is None or position == 0:
                src = self.transformer.bt2ls(x)              # [B, NW, D]
            else:
                src = self.transformer.ote(y).contiguous()   # [B, Ty_step, D] (usually 1)

            for block in self.transformer.h:
                src = block(src, attn_mask, use_cache=True, position=position)

            src = self.transformer.ln_f(src)
            logits = self.lm_head(src).contiguous()          # [B, T_in, V]; same as original
            return logits

        # ---------------------------
        # Prefill / training path
        # ---------------------------
        B, Ty = y.size()          # Ty = 501  (BOS + 500)
        NW     = x.shape[3]       # 500

        x_enc = self.transformer.bt2ls(x)                    # [B, NW, D]
        y_emb = self.transformer.ote(y)                      # [B, Ty, D]
        src   = self.transformer.drop(torch.cat([x_enc, y_emb], dim=1))  # [B, NW+Ty, D]

        for block in self.transformer.h:
            src = block(src, attn_mask, use_cache=False)

        src    = self.transformer.ln_f(src)
        logits = self.lm_head(src)[:, NW:, :].contiguous()   # keep only target segment -> [B, Ty, V]

        if not calculate_loss:
            return logits                                    # [B, 501, V], identical to your original

        # Train on 500 real targets: predict y[:,1:] using logits[:, :-1, :]
        pred = logits[:, :-1, :].reshape(-1, logits.size(-1))  # [B*500, V]
        tgt  = y[:, 1:].reshape(-1).long()                      # [B*500]
        loss = F.cross_entropy(pred, tgt)
        return logits, loss



    """
    def forward(self, x, y, attn_mask, position=None, use_cache=False, calculate_loss=True):

        if use_cache:
            if position == 0: src = self.transformer.bt2ls(x)   
            else: src = self.transformer.ote(y).contiguous()    
            for block in self.transformer.h:
                src = block(src, attn_mask, use_cache=use_cache, position=position)
            src = self.transformer.ln_f(src)
            logits = self.lm_head(src)
            if position == 0: logits = logits[:, :, :].contiguous()
            else: logits = logits[:, :, :].contiguous()
            return logits
        else:
            B, _ = y.size()
            B, _, _, NW, _ = x.size()

            x = self.transformer.bt2ls(x)   
            y2 = self.transformer.ote(y)    
            src = torch.cat([x, y2], dim=1)  
            src = self.transformer.drop(src)

            if calculate_loss:
                zero_col = torch.zeros(B, 1, device=y.device)
                # removing start token
                tgt = torch.cat((y[:, 1:], zero_col), dim=1)

            for block in self.transformer.h:
                src = block(src, attn_mask, use_cache=use_cache)
            src = self.transformer.ln_f(src)
            logits = self.lm_head(src)
            logits = logits[:, NW:, :].contiguous()
            
            if calculate_loss:
                tgt = tgt.long()
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                        tgt.reshape(-1))
                return logits, loss
            else: return logits
    """

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer
