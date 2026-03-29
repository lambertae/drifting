"""Memory bank for storing and sampling feature vectors per class."""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch


class ArrayMemoryBank:
    """Fixed-size circular buffer of feature vectors, one buffer per class.

    Stores features as CPU numpy arrays for efficiency; returns torch.Tensor
    from `sample()`.
    """

    def __init__(self, num_classes: int = 1000, max_size: int = 64, dtype=np.float32):
        self.num_classes = int(num_classes)
        self.max_size = int(max_size)
        self.dtype = dtype

        self.bank: Optional[np.ndarray] = None
        self.feature_shape: Optional[tuple] = None
        self.ptr = np.zeros(self.num_classes, dtype=np.int32)
        self.count = np.zeros(self.num_classes, dtype=np.int32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_bank(self, sample_shape: tuple) -> None:
        self.feature_shape = tuple(sample_shape)
        self.bank = np.zeros(
            (self.num_classes, self.max_size, *self.feature_shape), dtype=self.dtype
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, samples, labels) -> None:
        """Insert a batch of feature vectors into the bank.

        Args:
            samples: array-like [B, *feature_shape]
            labels:  integer class indices [B]
        """
        samples = np.asarray(samples, dtype=self.dtype)
        labels = np.asarray(labels, dtype=np.int64).ravel()

        if self.bank is None:
            self._init_bank(samples.shape[1:])

        for i in range(labels.shape[0]):
            lbl = int(labels[i])
            idx = int(self.ptr[lbl])
            self.bank[lbl, idx] = samples[i]
            self.ptr[lbl] = (idx + 1) % self.max_size
            if self.count[lbl] < self.max_size:
                self.count[lbl] += 1

    def sample(self, labels, n_samples: int) -> torch.Tensor:
        """Sample `n_samples` feature vectors for each label in `labels`.

        Args:
            labels:    integer class indices [B]
            n_samples: number of samples per class

        Returns:
            torch.Tensor of shape [B, n_samples, *feature_shape]
        """
        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("MemoryBank is empty. Call add() before sample().")

        labels = np.asarray(labels, dtype=np.int64).ravel()
        bsz = labels.shape[0]
        sample_indices = np.empty((bsz, n_samples), dtype=np.int32)

        for i in range(bsz):
            lbl = int(labels[i])
            valid = int(self.count[lbl])
            if valid <= 0:
                sample_indices[i] = np.zeros(n_samples, dtype=np.int32)
            else:
                sample_indices[i] = np.random.choice(
                    valid, n_samples, replace=(valid < n_samples)
                )

        # Fancy-index: bank[labels, sample_indices] -> [B, n_samples, *feature_shape]
        out = self.bank[labels[:, None], sample_indices]
        return torch.tensor(out)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        filled = int((self.count > 0).sum())
        return (
            f"ArrayMemoryBank(num_classes={self.num_classes}, "
            f"max_size={self.max_size}, "
            f"classes_with_data={filled}/{self.num_classes})"
        )
