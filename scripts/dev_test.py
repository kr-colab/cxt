import torch
from cxt.train import LitTokenFreeDecoder
import torch.nn.functional as F

class TokenFreeDecoderConfig:
    num_samples: int = 50
    sample_scale_embd: int = 2 # mini vs mini2
    output_dim: int = 256+2
    n_embd: int = 400#800
    combined_dim: int = 1001
    n_layer: int = 6
    bias: bool = False
    dropout: float = 0.1
    n_head: int = 4
    device: str = 'cuda'
model = LitTokenFreeDecoder(TokenFreeDecoderConfig())
model_path = '../cxt/lightning_logs/version_8/checkpoints/epoch=4-step=16160.ckpt'
model.load_state_dict(torch.load(model_path, weights_only=False)['state_dict'], strict=False)

model = model.model
model.to(device='cuda')
model.eval()
print()

model = torch.compile(model)

import msprime
import numpy as np
from cxt.utils import simulate_parameterized_tree_sequence
from cxt.utils import interpolate_tmrcas
from tqdm import tqdm
from cxt.utils import ts2X_vectorized, xor, xnor
from cxt.inference import generate_kv
from cxt.dataset import discretize
from tqdm import tqdm

torch.set_float32_matmul_precision('high')
TIMES = np.linspace(3, 14, 256)

def ts2input(ts, pivot_A, pivot_B):
    Xxor = ts2X_vectorized(ts, window_size=2000, xor_ops=xor, pivot_A=pivot_A, pivot_B=pivot_B).astype(np.float16)
    Xxnor = ts2X_vectorized(ts, window_size=2000, xor_ops=xnor, pivot_A=pivot_A, pivot_B=pivot_B).astype(np.float16)
    src = np.stack([Xxor, Xxnor], axis=0).astype(np.float16)
    tgt = np.log(interpolate_tmrcas(ts.simplify(samples=[pivot_A, pivot_B]), window_size=2000)).astype(np.float16)
    src = torch.from_numpy(src).float()
    src = torch.log1p(src)
    tgt = np.array(discretize(tgt, np.linspace(3, 14, 256)))
    tgt = torch.from_numpy(tgt).long() + 2
    tgt = torch.cat([torch.tensor([1]), tgt])
    return src, tgt

def generate_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

SEED = 103370001
ts = simulate_parameterized_tree_sequence(SEED, sequence_length=1e6)

device = 'cuda'
attn_mask = generate_causal_mask(1001, 'cuda')
attn_mask = attn_mask.repeat(20, 1, 1, 1)
idx = torch.ones(20, 1).long().to("cuda")
src, tgt = ts2input(ts, 0, 1)
src = src.unsqueeze(0).repeat(20, 1, 1, 1, 1).cuda()


"""

top_k = 50
with torch.no_grad():
    logits = model(src, idx[:, -1:], attn_mask, calculate_loss=False, use_cache=True, position=0)
    logits = logits[:, -1, :]
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    idx = torch.cat([idx, next_token], dim=1)

with torch.no_grad():
    for i in range(501, 1000):
            print(i)
        #with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(src, idx[:, -1:], attn_mask, calculate_loss=False, use_cache=True, position=i)
            logits = logits[:, -1, :]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
"""


B = 20
L = 6
n_head = 4
T = 1001
head_dim = 100
past_key_values = torch.zeros(2, L, B, n_head, T, head_dim).to('cuda')

model.inference_initalization(src, None, attn_mask, calculate_loss=False, use_cache=True, past_key_values=past_key_values)

top_k = 50
with torch.no_grad():
    for i in range(500, 999):
            print(i)
            logits = model(src, idx[:, -1:], attn_mask, calculate_loss=False, use_cache=True, past_key_values=past_key_values)
            logits = logits[:, -1, :]
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

