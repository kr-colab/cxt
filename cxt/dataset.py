"""Training datasets for cxt.

``PairDataset`` is the canonical dataset: each item is one pair from one
tree-sequence simulation, stored as ``(X.npy, y.npy)`` per directory.
Memory-mapped loading with O(files) shuffle.

The legacy ``LazyDataset`` / ``MultiDirLazyDataset`` classes have been removed.
"""

from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset


# ---------------------------------------------------------------------------
# Discretization grids (shared constants)
# ---------------------------------------------------------------------------

GRID_SIZE = 324
TIMES = np.linspace(3, 17, GRID_SIZE)


def discretize(sequence, grid):
    idx = np.searchsorted(grid, sequence, side="right") - 1
    np.clip(idx, 0, len(grid) - 1, out=idx)
    return idx


# ---------------------------------------------------------------------------
# PairDataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """One pair per item; file-order shuffle without disk I/O.

    Parameters
    ----------
    root : str
        Root directory containing ``train/`` and ``test/`` subdirectories.
    split : str
        ``"train"`` or ``"test"``.
    mmap : bool
        Use memory-mapped loading (recommended).
    """

    def __init__(self, root: str, split: str = "train", mmap: bool = True):
        self.root = root
        self.split = split
        self.mmap = mmap

        self.items: list[tuple[str, str, int]] = []
        self._file_spans: list[tuple[int, int]] = []
        self._file_perm = None
        self._cum_lengths_perm = None

        split_dir = os.path.join(root, split)
        for dirpath, _dirnames, filenames in os.walk(split_dir):
            if "X.npy" in filenames and "y.npy" in filenames:
                X_path = os.path.join(dirpath, "X.npy")
                y_path = os.path.join(dirpath, "y.npy")
                Y = np.load(y_path, mmap_mode="r") if mmap else np.load(y_path)
                P = int(Y.shape[0])
                if mmap:
                    del Y
                start = len(self.items)
                for p_idx in range(P):
                    self.items.append((X_path, y_path, p_idx))
                self._file_spans.append((start, P))

    def __len__(self):
        return len(self.items)

    def shuffle_files(self, seed=None):
        """Shuffle file-block order. O(files), no disk I/O."""
        F = len(self._file_spans)
        if F <= 1:
            self._file_perm = self._cum_lengths_perm = None
            return
        rng = np.random.default_rng(int(seed) if seed is not None else None)
        perm = rng.permutation(F)
        lengths = np.fromiter((L for _, L in self._file_spans), dtype=np.int64, count=F)
        self._file_perm = perm.astype(np.int32)
        self._cum_lengths_perm = np.cumsum(lengths[perm], dtype=np.int64)

    def set_epoch(self, epoch: int):
        self.shuffle_files(seed=epoch)

    def _map_idx(self, i: int) -> int:
        if self._file_perm is None:
            return i
        k = int(np.searchsorted(self._cum_lengths_perm, i, side="right"))
        prev = 0 if k == 0 else int(self._cum_lengths_perm[k - 1])
        file_idx = int(self._file_perm[k])
        start, L = self._file_spans[file_idx]
        return start + min(i - prev, L - 1)

    def __getitem__(self, i):
        j = self._map_idx(int(i))
        X_path, y_path, p_idx = self.items[j]

        X = np.load(X_path, mmap_mode="r") if self.mmap else np.load(X_path)
        y = np.load(y_path, mmap_mode="r") if self.mmap else np.load(y_path)

        Xi = torch.tensor(X[p_idx]).float()
        Xi = torch.log1p(Xi)

        yi = y[p_idx]
        yi = torch.tensor(discretize(yi, TIMES)).long() + 2
        yi = torch.cat([torch.tensor([1]), yi])

        return Xi, yi


# ---------------------------------------------------------------------------
# Optional shuffle-buffer wrapper for distributed training
# ---------------------------------------------------------------------------

class ShuffleBufferDataset(IterableDataset):
    """Wrap any map-style dataset with a streaming shuffle buffer."""

    def __init__(self, ds: Dataset, buffer_size: int = 8192, seed: int = 1234):
        self.ds = ds
        self.buffer_size = int(buffer_size)
        self.seed = int(seed)

    def __len__(self):
        return len(self.ds)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world = torch.distributed.get_world_size()
        else:
            rank, world = 0, 1

        N = len(self.ds)
        src = iter(range(rank, N, world))

        buf = []
        for _ in range(min(self.buffer_size, (N + world - 1) // world)):
            i = next(src, None)
            if i is None:
                break
            buf.append(i)

        while buf:
            j = int(rng.integers(0, len(buf)))
            idx = buf.pop(j)
            yield self.ds[idx]
            nxt = next(src, None)
            if nxt is not None:
                buf.append(nxt)
