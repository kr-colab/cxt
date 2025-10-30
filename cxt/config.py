from dataclasses import dataclass


@dataclass
class NarrowModelConfig:
    num_samples: int = 50
    sample_scale_embd: int = 2 
    output_dim: int = 324+2#256+2
    n_embd: int = 400
    combined_dim: int = 1001
    n_layer: int = 6
    bias: bool = False
    dropout: float = 0.1
    n_head: int = 4
    device: str = 'cuda'
    batch_size: int = 1225
    mask_singletons: bool = True


@dataclass
class BroadModelConfig:
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
    batch_size: int = 1225
    mask_singletons: bool = True