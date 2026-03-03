"""Checkpoint download, caching, and model loading.

Consolidates the two ``setup_cxt_model()`` implementations from ``utils.py``
into a single ``load_model()`` factory that:

* resolves the correct ``ModelConfig`` preset (no post-load hacking),
* downloads checkpoints from GitHub LFS on first use, and
* handles adapter wrapping internally.
"""

from __future__ import annotations

import os
import sys
import importlib
from pathlib import Path
from typing import Optional

import torch

from cxt.config import ModelConfig, PRESETS, AdapterConfig


# ---------------------------------------------------------------------------
# Checkpoint registry
# ---------------------------------------------------------------------------

GITHUB_BASE = "https://github.com/kevinkorfmann/cxt/raw/main/checkpoints"

CHECKPOINT_REGISTRY: dict[str, dict] = {
    "broad": {
        "filename": "broad_epoch=1-step=5280.ckpt",
    },
    "broad+adapter": {
        "filename": "broad_adapter_epoch=2-step=792.ckpt",
        "base_preset": "broad",
        "adapter": AdapterConfig(ie_in=10),
    },
    "narrow": {
        "filename": "narrow_epoch=5-step=4692.ckpt",
    },
    "broad_w200": {
        "filename": "broad_w200_epoch=1-step=944.ckpt",
    },
    "residual": {
        "filename": "residual_epoch=1-step=5280.ckpt",
    },
    "w200_wmissing": {
        "filename": "w200_wmissing_epoch=1-step=944.ckpt",
    },
    "w200_wmissing_adapter": {
        "filename": "w200_wmissing_adapter_epoch=9-step=480.ckpt",
        "base_preset": "w200_wmissing",
        "adapter": AdapterConfig(ie_in=10),
    },
}

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_dir() -> Path:
    override = os.environ.get("CXT_CHECKPOINT_CACHE")
    if override:
        d = Path(override)
    else:
        d = Path.home() / ".cache" / "cxt" / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dest: Path) -> None:
    import requests

    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                print(f"\r  {downloaded / total * 100:.0f}%", end="", flush=True)
    print(f"\n  Saved to {dest}")


def get_checkpoint_path(model_type: str, force_local: str | None = None) -> Path:
    """Return local path to checkpoint, downloading if necessary."""
    if force_local:
        return Path(force_local)

    info = CHECKPOINT_REGISTRY.get(model_type)
    if info is None:
        raise ValueError(
            f"Unknown model_type {model_type!r}. "
            f"Available: {sorted(CHECKPOINT_REGISTRY)}"
        )

    dest = _cache_dir() / model_type / info["filename"]
    if dest.exists():
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{GITHUB_BASE}/{model_type}/{info['filename']}"
    _download(url, dest)
    return dest


# ---------------------------------------------------------------------------
# Unpickle compatibility shim
# ---------------------------------------------------------------------------

def _inject_compat_aliases():
    """Inject ModelConfig under the names old checkpoints expect."""
    from cxt.config import ModelConfig

    sys.modules.setdefault("__main__", type(sys)("__main__"))
    main = sys.modules["__main__"]
    for alias in ("TokenFreeDecoderConfig", "NarrowModelConfig", "BroadModelConfig"):
        if not hasattr(main, alias):
            setattr(main, alias, ModelConfig)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_model(
    model_type: str = "broad",
    device: str = "cpu",
    force_local: str | None = None,
):
    """Load a pretrained cxt model ready for inference.

    Parameters
    ----------
    model_type : str
        One of: broad, broad+adapter, narrow, broad_w200, residual,
        w200_wmissing, w200_wmissing_adapter.
    device : str
        Target device (model is loaded on CPU first, then moved).
    force_local : str, optional
        Use this checkpoint path instead of downloading.

    Returns
    -------
    model : TokenFreeDecoder or FrozenDecoderWithAdapter
        Model in eval mode with KV caches allocated.
    """
    from cxt.model import TokenFreeDecoder

    _inject_compat_aliases()

    info = CHECKPOINT_REGISTRY.get(model_type, {})
    adapter_cfg = info.get("adapter")
    base_preset_name = info.get("base_preset", model_type)

    if base_preset_name not in PRESETS:
        raise ValueError(f"No preset for {base_preset_name!r}")

    config = PRESETS[base_preset_name].for_inference(batch_size=1, device="cpu")

    ckpt_path = get_checkpoint_path(model_type, force_local=force_local)
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    if adapter_cfg is not None:
        return _load_adapter_model(config, adapter_cfg, ckpt_path, device)

    state = ckpt.get("state_dict", ckpt)
    cleaned = {}
    for k, v in state.items():
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned[k] = v

    model = TokenFreeDecoder(config)
    model.load_state_dict(cleaned, strict=False)
    model.enable_kv_cache(batch_size=1)
    return model.to(device).eval()


def _load_adapter_model(config, adapter_cfg, ckpt_path, device):
    """Load an adapter-wrapped model from a Lightning checkpoint."""
    from cxt.train import LitAdapterDecoder

    _inject_compat_aliases()

    lit = LitAdapterDecoder.load_from_checkpoint(
        str(ckpt_path),
        gpt_config=config,
        ie_new=adapter_cfg.ie_in,
        adapter_bottleneck=adapter_cfg.bottleneck,
        adapter_dropout=adapter_cfg.dropout,
        new_mask_index=adapter_cfg.new_mask_index,
        training_config=1,  # dummy, not used at inference
    )
    wrapped = lit.model
    wrapped.backbone.enable_kv_cache(batch_size=1)
    return wrapped.to(device).eval()
