from __future__ import annotations
from dataclasses import dataclass, field, replace
from copy import deepcopy


@dataclass
class ModelConfig:
    """Unified configuration for all cxt model variants.

    Instead of separate NarrowModelConfig / BroadModelConfig classes with
    commented-out blocks, select a preset via ``PRESETS["broad"]`` etc.
    """
    n_layer: int = 10
    n_embd: int = 400
    n_head: int = 4
    output_dim: int = 326          # 324 discretization bins + 2 special tokens
    num_samples: int = 50
    sample_scale_embd: int = 2
    combined_dim: int = 1001       # source (500) + target (501)
    window_size: int = 2000        # base SFS window in bp

    mask_singletons: bool = True
    use_kv_cache: bool = False     # enable only at inference time
    bias: bool = False
    dropout: float = 0.1

    device: str = "cpu"
    batch_size: int = 1            # only matters when use_kv_cache=True

    def for_inference(self, batch_size: int = 1, device: str = "cpu") -> ModelConfig:
        """Return a copy configured for autoregressive decoding."""
        return replace(self, use_kv_cache=True, batch_size=batch_size, device=device)

    def for_training(self, batch_size: int = 128, device: str = "cuda") -> ModelConfig:
        """Return a copy configured for training (no KV cache)."""
        return replace(self, use_kv_cache=False, batch_size=batch_size, device=device)


PRESETS: dict[str, ModelConfig] = {
    "narrow":        ModelConfig(n_layer=6),
    "broad":         ModelConfig(n_layer=10),
    "broad_w200":    ModelConfig(n_layer=10, window_size=200),
    "residual":      ModelConfig(n_layer=10),
    "w200_wmissing": ModelConfig(n_layer=10, window_size=200, mask_singletons=False),
}


@dataclass
class AdapterConfig:
    """Configuration for the sample-size adapter (IEAdapter)."""
    ie_in: int = 10
    ie_out: int = 50
    bottleneck: int = 32
    dropout: float = 0.0
    use_se: bool = True
    new_mask_index: int | None = 0
    unfreeze_strategy: str = "ln_lastN"
    last_n: int = 2


@dataclass
class TrainingConfig:
    """Hyperparameters for Lightning training."""
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 10
    lr_decay_iters: int = 150_000
    batch_size: int = 128
    grad_accum_steps: int = 6
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    num_workers: int = 16
    prefetch_factor: int = 2


# ---------------------------------------------------------------------------
# Backward compatibility aliases so old checkpoints can be unpickled.
# Lightning saves the config class name; these let ``torch.load`` find them.
# ---------------------------------------------------------------------------
NarrowModelConfig = ModelConfig
BroadModelConfig = ModelConfig
TokenFreeDecoderConfig = ModelConfig
