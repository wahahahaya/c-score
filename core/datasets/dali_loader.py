"""
DALI-based DataLoader for in-memory CIFAR datasets.

Replaces the CPU DataLoader + the PCIe transfer stage:
  - DALI external source feeds pre-loaded numpy uint8 arrays
  - GPU pipeline converts uint8 HWC [0,255] → float32 CHW [0,1]
  - GPUAugmentor still handles all augmentation (flip/crop/normalize/RandAugment)

Usage: enabled when `use_dali: true` in config.
"""

import numpy as np
import torch
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy


class _InfiniteSource:
    """
    Infinite cyclic source: never raises StopIteration.

    Cycles through all samples with a new shuffle each time the dataset is
    exhausted, so epoch-level shuffle is preserved without DALI pipeline resets.
    The DALILoaderWrapper controls how many batches constitute one epoch.
    """

    def __init__(self, data, labels, batch_size, shuffle=True):
        self.data       = data                     # (N, H, W, C) uint8
        self.labels     = labels.astype(np.int32)  # (N,)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self._reset_perm()

    def _reset_perm(self):
        idx = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(idx)
        self._idx = idx
        self._pos = 0

    def __iter__(self):
        self._reset_perm()
        return self

    def __next__(self):
        result_idx = []
        while len(result_idx) < self.batch_size:
            remaining = self._idx[self._pos:]
            need      = self.batch_size - len(result_idx)
            take      = remaining[:need]
            result_idx.extend(take.tolist())
            self._pos += len(take)
            if len(take) < need:          # exhausted current permutation
                self._reset_perm()        # re-shuffle for next cycle
        idx = np.array(result_idx)
        return self.data[idx], self.labels[idx]


def build_dali_loader(data, labels, batch_size, total_batches,
                      shuffle=True, device_id=0, num_threads=2,
                      prefetch_queue_depth=2):
    """
    Build a DALI-based loader and return it as a DALILoaderWrapper.

    Args:
        data              : numpy uint8 (N, H, W, C)
        labels            : numpy int64/int32 (N,)
        batch_size        : samples per batch
        total_batches     : batches to produce per epoch (before StopIteration)
        shuffle           : shuffle indices at each reset
        device_id         : CUDA device index
        num_threads       : DALI CPU decode threads
        prefetch_queue_depth : async prefetch depth

    Returns:
        _DALILoaderWrapper — iterable yielding
            (images: float32 CHW GPU tensor, labels: int64 GPU tensor)
    """
    source = _InfiniteSource(data, labels, batch_size, shuffle)

    @pipeline_def(batch_size=batch_size, num_threads=num_threads,
                  device_id=device_id, prefetch_queue_depth=prefetch_queue_depth)
    def _cifar_pipeline():
        images, labels_out = fn.external_source(
            source=source,
            num_outputs=2,
            dtype=[types.UINT8, types.INT32],
            layout=["HWC", ""],   # per-sample layout: HWC for images, scalar for labels
            batch=True,
        )
        # uint8 HWC [0, 255]  →  float32 CHW [0, 1]
        # crop_mirror_normalize with no crop params = pure normalize + layout transpose
        images = fn.crop_mirror_normalize(
            images.gpu(),
            dtype=types.FLOAT,
            mean=[0.0, 0.0, 0.0],
            std=[255.0, 255.0, 255.0],
            output_layout="CHW",
        )
        return images, labels_out

    pipe = _cifar_pipeline()
    pipe.build()
    return _DALILoaderWrapper(pipe, total_batches)


class _DALILoaderWrapper:
    """
    Makes a built DALI pipeline look like a PyTorch DataLoader.

    __iter__ creates a fresh DALIGenericIterator each epoch, which causes
    DALI to call __iter__ on the _BatchedSource → rebuilds shuffled indices.
    __next__ returns (images: float32 CHW GPU, labels: int64 GPU).
    Raises StopIteration after `size` batches.
    """

    def __init__(self, pipe, size):
        self._pipe  = pipe
        self._size  = size
        self._count = 0
        # Build ONE iterator; the infinite source means it never needs to be rebuilt
        self._dali_iter = DALIGenericIterator(
            [pipe],
            output_map=["images", "labels"],
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=False,
        )

    def __iter__(self):
        # Infinite source keeps running; just reset the epoch counter
        self._count = 0
        return self

    def __next__(self):
        if self._count >= self._size:
            raise StopIteration
        data = next(self._dali_iter)
        self._count += 1
        images = data[0]["images"]                     # float32 CHW on GPU
        labels = data[0]["labels"].long().squeeze(-1)  # int64 on GPU
        return images, labels

    def __len__(self):
        return self._size
