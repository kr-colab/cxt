import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from einops import rearrange


class MutationsToLatentSpace(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj01   = nn.Linear(
            config.num_samples,
            config.sample_scale_embd * config.num_samples,
            bias=True)
        self.gelu    = nn.GELU()
        self.proj02  = nn.Linear(
            config.sample_scale_embd * config.num_samples,
            config.sample_scale_embd * config.num_samples,
            bias=True)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('weights', torch.tensor([0.7, 0.3]).view(1, 2, 1, 1, 1))
    def forward(self, x):
        B, XX, WS, NW, IE = x.size()
        x[..., 0] = 0. # masks singeltons                          
        x = self.proj01(x)
        x = self.gelu(x)
        x = self.proj02(x)
        x = self.dropout(x)
        x = (x * self.weights).sum(dim=1)
        x = x.transpose(1, 2).reshape(B, NW, -1)
        return x


class MLP(nn.Module):    
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    """
    Layer normalization module.
    Args:
        ndim (int): The number of dimensions in the input tensor.
        use_bias (bool): If True, adds a learnable bias to the normalized tensor.
    Attributes:
        weight (torch.nn.Parameter): Learnable scaling parameter of shape (ndim,).
        bias (torch.nn.Parameter or None): Learnable bias parameter of shape (ndim,) if bias is True, otherwise None.
    """
    def __init__(self, ndim: int, use_bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if use_bias else None
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


"""
class CausalSelfAttention(nn.Module):
    def __init__(self, config, i):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by the number of heads"
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.rotary_emb = RotaryPositionalEmbeddings(dim=self.head_size, max_seq_len=1001)
        self.layer_index = torch.tensor(i)
        # KV Cache
        #self.cache_k = torch.zeros((config.batch_size, self.n_head, 1001, self.head_size), dtype=torch.bfloat16).to(config.device)
        #self.cache_v = torch.zeros((config.batch_size, self.n_head, 1001, self.head_size), dtype=torch.bfloat16).to(config.device)

        self.cache_k = torch.zeros((config.batch_size, self.n_head, 1001, self.head_size), requires_grad=False).to(config.device)
        self.cache_v = torch.zeros((config.batch_size, self.n_head, 1001, self.head_size), requires_grad=False).to(config.device)


    def forward(self, x, attn_mask, position=None, use_cache=False):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
     
        if use_cache:
            if position == 0:
                q = self.rotary_emb(q.transpose(1, 2), input_pos=torch.arange(T, device=q.device)).transpose(1, 2)
                k = self.rotary_emb(k.transpose(1, 2), input_pos=torch.arange(T, device=k.device)).transpose(1, 2)
                self.cache_k[:B, :, :T, :] = k
                self.cache_v[:B, :, :T, :] = v
            else:
                q = self.rotary_emb(q.transpose(1, 2), input_pos=position).transpose(1, 2)
                k = self.rotary_emb(k.transpose(1, 2), input_pos=position).transpose(1, 2)
                self.cache_k[:B, :, position:position + T, :] = k
                self.cache_v[:B, :, position:position + T, :] = v
                k = self.cache_k[:B, :, :position + T, :]
                v = self.cache_v[:B, :, :position + T, :]
        else:
            q = self.rotary_emb(q.transpose(1, 2), input_pos=torch.arange(T, device=q.device)).transpose(1, 2)
            k = self.rotary_emb(k.transpose(1, 2), input_pos=torch.arange(T, device=k.device)).transpose(1, 2)

        if use_cache:
            if position == 0: attn_mask = attn_mask[:, :, :q.size(-2), :k.size(-2)]
            else: attn_mask = attn_mask[:, :, position, :k.size(-2)].unsqueeze(2)

        #if use_cache:
        #    original_dtype = q.dtype
        #    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
        #    attn_mask = attn_mask.to(torch.bfloat16)
        #    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        #    y = y.to(original_dtype)
        #else:
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).contiguous().view(B, -1, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
"""


class CausalSelfAttention(nn.Module):
    def __init__(self, config, i):
        super().__init__(); assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout); self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head, self.n_embd = config.n_head, config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.rotary_emb = RotaryPositionalEmbeddings(self.head_size, max_seq_len=1001)
        self.layer_index = torch.tensor(i)
        self.cache_k = torch.zeros((config.batch_size, self.n_head, 1001, self.head_size), requires_grad=False, device=config.device)
        self.cache_v = torch.zeros_like(self.cache_k)

    def forward(self, x, attn_mask, position=None, use_cache=False):
        B, T, _ = x.size()

        r = lambda t: rearrange(t, 'b t (h d)-> b h t d', h=self.n_head)
        rope = lambda t: rearrange(self.rotary_emb(rearrange(t, 'b h t d->b t h d'), input_pos=pos), 'b t h d->b h t d')

        q, k, v = [r(t) for t in self.c_attn(x).split(self.n_embd, dim=-1)]
        pos0 = (position is None) or (position == 0)
        pos = torch.arange(T, device=x.device) if pos0 else position
        q, k = rope(q), rope(k)

        if use_cache:
            start = 0 if pos0 else int(position); end = T if pos0 else start + T
            self.cache_k[:B, :, start:end], self.cache_v[:B, :, start:end] = k, v
            k, v = self.cache_k[:B, :, :end], self.cache_v[:B, :, :end]
            attn_mask = attn_mask[:, :, :q.size(-2), :k.size(-2)] if pos0 else attn_mask[:, :, start, :k.size(-2)].unsqueeze(2)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.resid_dropout(self.c_proj(rearrange(y, 'b h t d->b t (h d)')))


class Block(nn.Module):
    def __init__(self, config, i):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, use_bias=config.bias)
        self.attn = CausalSelfAttention(config, i)
        self.ln_2 = LayerNorm(config.n_embd, use_bias=config.bias)
        self.mlp = MLP(config)
        
    def forward(self, x, attn_mask, position=None, use_cache=False):
        x = x + self.attn(self.ln_1(x),
        attn_mask=attn_mask, position=position, use_cache=use_cache)
        x = x + self.mlp(self.ln_2(x))
        return x







