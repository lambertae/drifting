from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np


class ArrayMemoryBank:
    """Class-wise ring buffer for feature/image samples used by generator training."""

    def __init__(self, num_classes: int = 1000, max_size: int = 64, dtype=np.float32):
        self.num_classes = int(num_classes)
        self.max_size = int(max_size)
        self.dtype = dtype
        self.bank: Optional[np.ndarray] = None
        self.feature_shape: Optional[Tuple[int, ...]] = None
        self.ptr = np.zeros(self.num_classes, dtype=np.int32)
        self.count = np.zeros(self.num_classes, dtype=np.int32)

    def _init_bank(self, sample_shape: Tuple[int, ...]) -> None:
        self.feature_shape = tuple(sample_shape)
        self.bank = np.zeros((self.num_classes, self.max_size, *self.feature_shape), dtype=self.dtype)

    def add(self, samples, labels) -> None:
        """Insert samples into per-class ring buffers.

        Args:
            samples: array of shape ``(N, *feature_shape)`` to store.
            labels: integer class labels of shape ``(N,)``.
        """
        samples = np.asarray(samples)
        labels = np.asarray(labels)
        if self.bank is None:
            self._init_bank(samples.shape[1:])

        for i in range(labels.shape[0]):
            lbl = int(labels[i])
            idx = self.ptr[lbl]
            self.bank[lbl, idx] = samples[i]
            self.ptr[lbl] = (idx + 1) % self.max_size
            if self.count[lbl] < self.max_size:
                self.count[lbl] += 1

    def sample(self, labels, n_samples: int):
        """Sample stored entries for each label.

        Args:
            labels: integer class labels of shape ``(B,)``.
            n_samples: number of samples to draw per label.

        Returns:
            jnp.ndarray of shape ``(B, n_samples, *feature_shape)``.
        """
        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("MemoryBank is empty. Call add() before sample().")

        labels = np.asarray(labels)
        bsz = labels.shape[0]
        sample_indices = np.empty((bsz, n_samples), dtype=np.int32)
        for i in range(bsz):
            lbl = int(labels[i])
            valid = int(self.count[lbl])
            if valid <= 0:
                sample_indices[i] = np.zeros((n_samples,), dtype=np.int32)
            else:
                sample_indices[i] = np.random.choice(valid, n_samples, replace=(valid < n_samples))

        out = self.bank[labels[:, None], sample_indices]
        return jnp.asarray(out)
