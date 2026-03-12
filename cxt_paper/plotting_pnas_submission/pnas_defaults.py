"""
PNAS figure style defaults.

Figure size specs from PNAS Author Center:
  - Single column:  8.7 cm  (~3.42 in)
  - 1.5 column:    11.4 cm  (~4.49 in)
  - Double column:  17.8 cm  (~7.01 in)

Font requirements: 6–12 pt after reduction, consistent within each graphic.
Color mode: RGB.
Preferred output: PDF (vector) for line art, TIFF for raster.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── Dimensions (inches) ─────────────────────────────────────────────────
CM = 1 / 2.54
SINGLE_COL = 8.7 * CM       # ~3.42 in
ONE_HALF_COL = 11.4 * CM    # ~4.49 in
DOUBLE_COL = 17.8 * CM      # ~7.01 in

# ── Cache search paths (tried in order) ─────────────────────────────────
_CACHE_ROOTS = [
    os.environ.get("PNAS_CACHE", ""),
    "/sietch_colab/data_share/cxt_scratch/figures/output",
    os.path.join(os.path.dirname(__file__), "../../figures/output"),
]

_REVISION_DIRS = [
    os.path.join(os.path.dirname(__file__), "../cxtkit_manuscript/figures_revision"),
    os.path.join(os.path.dirname(__file__), "../cxtkit_manuscript/figures_v2"),
    os.path.join(os.path.dirname(__file__), "../cxtkit_manuscript/figures"),
]

DEFAULT_OUTPUT = os.environ.get(
    "PNAS_OUTPUT",
    os.path.join(os.path.dirname(__file__), "output"),
)


def resolve_cache(*subpath):
    """Find the first existing cache path across all cache roots."""
    rel = os.path.join(*subpath)
    for root in _CACHE_ROOTS:
        if not root:
            continue
        full = os.path.join(root, rel)
        if os.path.exists(full):
            return full
    return os.path.join(_CACHE_ROOTS[1], rel)


def resolve_cache_dir(*subpath):
    """Find the first existing cache directory across all cache roots."""
    rel = os.path.join(*subpath)
    for root in _CACHE_ROOTS:
        if not root:
            continue
        full = os.path.join(root, rel)
        if os.path.isdir(full):
            return full
    return os.path.join(_CACHE_ROOTS[1], rel)


def resolve_revision_fig(filename):
    """Find a figure file in the cxtkit_manuscript revision directories."""
    for d in _REVISION_DIRS:
        full = os.path.join(d, filename)
        if os.path.exists(full):
            return full
    return None


# ── rcParams for consistent PNAS style ──────────────────────────────────
PNAS_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "lines.linewidth": 0.8,
    "patch.linewidth": 0.5,
    "grid.linewidth": 0.3,
    "axes.grid": False,
}

# ── TIMES grid (log-space, matches cxt.utils.TIMES) ─────────────────────
GRID_SIZE = 324
TIMES = np.linspace(3, 17, GRID_SIZE)

# ── Color palette ────────────────────────────────────────────────────────
CXT_BLUE = "dodgerblue"
CXT_FILL = "dodgerblue"
SINGER_NAVY = "darkblue"
SMCPP_CYAN = "cyan"
TRUE_BLACK = "black"
REGION_RED = "crimson"


def apply_pnas_style():
    """Apply PNAS rcParams globally."""
    mpl.rcParams.update(PNAS_RC)


def savefig(fig, name, output_dir=None, formats=("pdf", "png"), **kwargs):
    """Save figure in one or more formats."""
    output_dir = output_dir or DEFAULT_OUTPUT
    os.makedirs(output_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, format=fmt, **kwargs)
        print(f"  -> {path}")
    plt.close(fig)


def panel_label(ax, letter, x=-0.12, y=1.08, **kwargs):
    """Add a bold panel label (A, B, C, ...) to an axes."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left", **kwargs)
