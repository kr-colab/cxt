import os, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from cxt.utils import TIMES
from cxt.utils import LOG_RESIDUAL_GRID

def discretize(sequence, population_time):
    idx = np.searchsorted(population_time, sequence, side="right") - 1
    np.clip(idx, 0, len(population_time) - 1, out=idx)
    return idx  # keep as numpy array

def discretize_residuals(log_predicted_times, log_residual_grid):
    r"""
    Discretize log TMRCA residuals (deviations from the mean). If $T_i$ is the TMRCA (not logged)
    $in window $i, then we compute the log residual as the relative different relative to the mean
    TMRCA. That is, $log(R_i)$ where $R_i = T_i N / (\sum_j^N T_j)$. Note that "TMRCA" is equivalent
    with expected nucleotide diversity as far as the residual is concerned, because the factor of
    $2 \mu$ cancels. The log residual is then discretized on a grid centered at zero.
    """
    #assert log_predicted_times.squeeze().ndim == 1
    #log_expected_diversity = np.log(np.mean(np.exp(log_predicted_times)))
    #log_expected_diversity = np.log(np.mean(np.exp(np.asarray(log_predicted_times))))
    #assert np.isfinite(log_expected_diversity)
    #log_residuals = log_predicted_times - log_expected_diversity
    #grid_index = np.digitize(log_residuals, log_residual_grid, right=True) - 1
    #np.clip(grid_index, 0, len(log_residual_grid) - 1, out=grid_index)
    #return grid_index  # keep as numpy array

    # ensure numpy array in float32 (handles torch.Tensor / bfloat16)
    if hasattr(log_predicted_times, "detach"):
        arr = log_predicted_times.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(log_predicted_times, dtype=np.float32)

    # stable log(mean(exp(x))) to prevent overflow
    m = arr.max()
    log_expected_diversity = float(m + np.log(np.exp(arr - m).mean()))
    assert np.isfinite(log_expected_diversity)

    log_residuals = arr - log_expected_diversity
    grid_index = np.digitize(log_residuals, log_residual_grid, right=True) - 1
    np.clip(grid_index, 0, len(log_residual_grid) - 1, out=grid_index)
    return grid_index


class PairDataset(Dataset):
    """
    Each item is ONE pair from ONE TS.
    We prebuild a global index of (X_path, y_path, p_idx) ONCE in file-major order.
    Call shuffle_files(seed) to change the order of FILE BLOCKS without touching disk.
    If `return_residuals` is True, then outputs are centered (in log space) around the
    log expected TMRCA.
    """
    def __init__(self, root, split="train", mmap=True, return_residuals=False):
        self.root = root
        self.split = split
        self.mmap = mmap
        self.return_residuals = return_residuals

        self.items = []          # list of (X_path, y_path, p_idx)  [canonical, file-major]
        self._file_spans = []    # list of (start_idx, length) per file, into `items`
        self._file_paths = []    # list of (X_path, y_path) per file  (same order as _file_spans)

        split_dir = os.path.join(root, split)
        # --- build items ONCE in file-major order & record spans ---
        for dirpath, dirnames, filenames in os.walk(split_dir):
            if "X.npy" in filenames and "y.npy" in filenames:
                X_path = os.path.join(dirpath, "X.npy")
                y_path = os.path.join(dirpath, "y.npy")
                # read P cheaply from header once
                Y = np.load(y_path, mmap_mode="r") if mmap else np.load(y_path)
                P = int(Y.shape[0])
                if mmap:
                    del Y
                start = len(self.items)
                for p_idx in range(P):
                    self.items.append((X_path, y_path, p_idx))
                self._file_spans.append((start, P))
                self._file_paths.append((X_path, y_path))

        # mapping state for shuffling FILE order (None means identity)
        self._file_perm = None                  # np.ndarray of file indices
        self._cum_lengths_perm = None           # cumulative lengths in permuted file order

    def __len__(self):
        return len(self.items)

    # ---- cheap file-order shuffle (no disk I/O, no rebuilding items) ----
    def shuffle_files(self, seed=None):
        """
        Shuffle FILE order only. Keeps pairs within each file contiguous.
        O(#files), no np.load calls, no touching `self.items`.
        """
        F = len(self._file_spans)
        if F <= 1:
            self._file_perm = None
            self._cum_lengths_perm = None
            return

        if seed is None:
            perm = np.random.permutation(F)
        else:
            perm = np.random.default_rng(int(seed)).permutation(F)

        # store the permutation and its cumulative lengths
        lengths = np.fromiter((L for (_, L) in self._file_spans), dtype=np.int64, count=F)
        lengths_perm = lengths[perm]
        cum = np.cumsum(lengths_perm, dtype=np.int64)

        self._file_perm = perm.astype(np.int32, copy=False)   # compact
        self._cum_lengths_perm = cum

    # optional: revert to canonical order
    def clear_shuffle(self):
        self._file_perm = None
        self._cum_lengths_perm = None

    # optional epoch API
    def set_epoch(self, epoch: int):
        self.shuffle_files(seed=epoch)

    def _map_logical_to_physical(self, i: int) -> int:
        """
        Map logical index i (0..N-1) in the current FILE order
        to the physical index j into self.items (canonical file-major build).
        """
        if self._file_perm is None:
            return i  # identity (canonical order)

        # find which permuted file block contains i
        k = int(np.searchsorted(self._cum_lengths_perm, i, side="right"))
        prev_cum = 0 if k == 0 else int(self._cum_lengths_perm[k-1])
        offset_in_file = i - prev_cum

        file_idx = int(self._file_perm[k])     # original file index
        start, L = self._file_spans[file_idx]
        # safety clip (should already hold)
        if offset_in_file >= L:
            offset_in_file = L - 1
        return start + offset_in_file

    def __getitem__(self, i):
        # remap i through current file permutation (cheap)
        j = self._map_logical_to_physical(int(i))
        X_path, y_path, p_idx = self.items[j]

        X = np.load(X_path, mmap_mode="r") if self.mmap else np.load(X_path)
        y = np.load(y_path, mmap_mode="r") if self.mmap else np.load(y_path)

        # features
        Xi = torch.tensor(X[p_idx])            # (2, ...)
        Xi = torch.log1p(Xi)

        # labels
        yi = torch.tensor(y[p_idx])

        if self.return_residuals:
            yi = torch.tensor(discretize_residuals(yi, LOG_RESIDUAL_GRID)).long() + 2
        else:
            yi = torch.tensor(discretize(yi, TIMES)).long() + 2
        yi = torch.cat([torch.tensor([1]), yi])

        return Xi, yi


import numpy as np, torch
from torch.utils.data import IterableDataset, DataLoader

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

class ShuffleBufferDataset(IterableDataset):
    def __init__(self, ds, buffer_size: int = 8192, seed: int = 1234):
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

        N = len(self.ds)  # your wrapped ds must support __len__
        src = iter(range(rank, N, world))  # <-- shard here

        buf = []
        for _ in range(min(self.buffer_size, (N + world - 1)//world)):
            i = next(src, None)
            if i is None: break
            buf.append(i)

        while buf:
            j = int(rng.integers(0, len(buf)))
            idx = buf.pop(j)
            yield self.ds[idx]
            nxt = next(src, None)
            if nxt is not None:
                buf.append(nxt)
