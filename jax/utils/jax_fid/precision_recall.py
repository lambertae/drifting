from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np


def _numpy_partition(arr, kth, **kwargs):
    num_workers = min(cpu_count(), len(arr))
    chunk_size = len(arr) // num_workers
    extra = len(arr) % num_workers

    start_idx = 0
    batches = []
    for i in range(num_workers):
        size = chunk_size + (1 if i < extra else 0)
        batches.append(arr[start_idx : start_idx + size])
        start_idx += size

    with ThreadPool(num_workers) as pool:
        return list(pool.map(partial(np.partition, kth=kth, **kwargs), batches))


class ManifoldEstimator:
    """
    A helper for comparing manifolds of feature vectors using JAX.

    Adapted from https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L57
    """

    def __init__(
        self,
        row_batch_size=10000,
        col_batch_size=10000,
        nhood_sizes=(3,),
        clamp_to_percentile=None,
        eps=1e-5,
    ):
        """
        Estimate the manifold of given feature vectors.

        :param row_batch_size: row batch size to compute pairwise distances
                               (parameter to trade-off between memory usage and performance).
        :param col_batch_size: column batch size to compute pairwise distances.
        :param nhood_sizes: number of neighbors used to estimate the manifold.
        :param clamp_to_percentile: prune hyperspheres that have radius larger than
                                    the given percentile.
        :param eps: small number for numerical stability.
        """
        self.distance_block = DistanceBlock()
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.clamp_to_percentile = clamp_to_percentile
        self.eps = eps

    def warmup(self):
        feats, radii = (
                    np.zeros([1, 2048], dtype=np.float64),
        np.zeros([1, 1], dtype=np.float64),
        )
        self.evaluate_pr(feats, radii, feats, radii)

    def manifold_radii(self, features: np.ndarray) -> np.ndarray:
        num_images = len(features)

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        radii = np.zeros([num_images, self.num_nhoods], dtype=np.float64)
        distance_batch = np.zeros([self.row_batch_size, num_images], dtype=np.float64)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_images)
                col_batch = features[begin2:end2]

                # Compute distances between batches.
                distance_batch[
                    0 : end1 - begin1, begin2:end2
                ] = self.distance_block.pairwise_distances(row_batch, col_batch)

            # Find the k-nearest neighbor from the current batch.
            radii[begin1:end1, :] = np.concatenate(
                [
                    x[:, self.nhood_sizes]
                    for x in _numpy_partition(distance_batch[0 : end1 - begin1, :], seq, axis=1)
                ],
                axis=0,
            )

        if self.clamp_to_percentile is not None:
            max_distances = np.percentile(radii, self.clamp_to_percentile, axis=0)
            radii[radii > max_distances] = 0
        return radii

    def evaluate(self, features: np.ndarray, radii: np.ndarray, eval_features: np.ndarray):
        """
        Evaluate if new feature vectors are at the manifold.
        """
        num_eval_images = eval_features.shape[0]
        num_ref_images = radii.shape[0]
        distance_batch = np.zeros([self.row_batch_size, num_ref_images], dtype=np.float64)
        batch_predictions = np.zeros([num_eval_images, self.num_nhoods], dtype=np.int32)
        max_realism_score = np.zeros([num_eval_images], dtype=np.float64)
        nearest_indices = np.zeros([num_eval_images], dtype=np.int32)

        for begin1 in range(0, num_eval_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_eval_images)
            feature_batch = eval_features[begin1:end1]

            for begin2 in range(0, num_ref_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_ref_images)
                ref_batch = features[begin2:end2]

                distance_batch[
                    0 : end1 - begin1, begin2:end2
                ] = self.distance_block.pairwise_distances(feature_batch, ref_batch)

            # From the minibatch of new feature vectors, determine if they are in the estimated manifold.
            # If a feature vector is inside a hypersphere of some reference sample, then
            # the new sample lies at the estimated manifold.
            # The radii of the hyperspheres are determined from distances of neighborhood size k.
            samples_in_manifold = distance_batch[0 : end1 - begin1, :, None] <= radii
            batch_predictions[begin1:end1] = np.any(samples_in_manifold, axis=1).astype(np.int32)

            max_realism_score[begin1:end1] = np.max(
                radii[:, 0] / (distance_batch[0 : end1 - begin1, :] + self.eps), axis=1
            )
            nearest_indices[begin1:end1] = np.argmin(distance_batch[0 : end1 - begin1, :], axis=1)

        return {
            "fraction": float(np.mean(batch_predictions)),
            "batch_predictions": batch_predictions,
            "max_realisim_score": max_realism_score,
            "nearest_indices": nearest_indices,
        }

    def evaluate_pr(
        self,
        features_1: np.ndarray,
        radii_1: np.ndarray,
        features_2: np.ndarray,
        radii_2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate precision and recall efficiently.

        :param features_1: [N1 x D] feature vectors for reference batch.
        :param radii_1: [N1 x K1] radii for reference vectors.
        :param features_2: [N2 x D] feature vectors for the other batch.
        :param radii_2: [N x K2] radii for other vectors.
        :return: a tuple of arrays for (precision, recall):
                 - precision: an np.ndarray of length K1
                 - recall: an np.ndarray of length K2
        """
        features_1_status = np.zeros([len(features_1), radii_2.shape[1]], dtype=bool)
        features_2_status = np.zeros([len(features_2), radii_1.shape[1]], dtype=bool)
        for begin_1 in range(0, len(features_1), self.row_batch_size):
            end_1 = min(begin_1 + self.row_batch_size, len(features_1))
            batch_1 = features_1[begin_1:end_1]
            for begin_2 in range(0, len(features_2), self.col_batch_size):
                end_2 = min(begin_2 + self.col_batch_size, len(features_2))
                batch_2 = features_2[begin_2:end_2]
                batch_1_in, batch_2_in = self.distance_block.less_thans(
                    batch_1, radii_1[begin_1:end_1], batch_2, radii_2[begin_2:end_2]
                )
                features_1_status[begin_1:end_1] |= batch_1_in
                features_2_status[begin_2:end_2] |= batch_2_in
        return (
            np.mean(features_2_status.astype(np.float64), axis=0),
            np.mean(features_1_status.astype(np.float64), axis=0),
        )


class DistanceBlock:
    """
    Calculate pairwise distances between vectors using JAX.

    Adapted from https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L34
    """

    def __init__(self):
        # JAX functions are compiled on first use, no need for explicit graph building
        pass

    def pairwise_distances(self, U, V):
        """
        Evaluate pairwise distances between two batches of feature vectors.
        """
        # Convert to JAX arrays if needed with float64 precision
        U_jax = jnp.asarray(U, dtype=jnp.float64)
        V_jax = jnp.asarray(V, dtype=jnp.float64)
        
        # Compute distances using JAX
        distances = _batch_pairwise_distances(U_jax, V_jax)
        
        # Convert back to numpy for compatibility
        return np.array(distances)

    def less_thans(self, batch_1, radii_1, batch_2, radii_2):
        """
        Check if distances are less than radii for both batches.
        """
        # Convert to JAX arrays with float64 precision
        batch_1_jax = jnp.asarray(batch_1, dtype=jnp.float64)
        batch_2_jax = jnp.asarray(batch_2, dtype=jnp.float64)
        radii_1_jax = jnp.asarray(radii_1, dtype=jnp.float64)
        radii_2_jax = jnp.asarray(radii_2, dtype=jnp.float64)
        
        # Compute pairwise distances
        distances = _batch_pairwise_distances(batch_1_jax, batch_2_jax)
        
        # Check if distances are within radii
        # batch_1_in: for each point in batch_1, check if it's within any radius in radii_2
        batch_1_in = jnp.any(distances[..., None] <= radii_2_jax, axis=1)
        
        # batch_2_in: for each point in batch_2, check if it's within any radius in radii_1
        batch_2_in = jnp.any(distances[..., None] <= radii_1_jax[:, None], axis=0)
        
        # Convert back to numpy
        return np.array(batch_1_in), np.array(batch_2_in)


@partial(jax.jit, static_argnums=())
def _batch_pairwise_distances(U, V):
    """
    Compute pairwise distances between two batches of feature vectors using JAX.
    """
    # Squared norms of each row in U and V
    norm_u = jnp.sum(jnp.square(U), axis=1)
    norm_v = jnp.sum(jnp.square(V), axis=1)

    # norm_u as a column and norm_v as a row vectors
    norm_u = jnp.reshape(norm_u, [-1, 1])
    norm_v = jnp.reshape(norm_v, [1, -1])

    # Pairwise squared Euclidean distances
    D = jnp.maximum(norm_u - 2 * jnp.matmul(U, V.T) + norm_v, 0.0)

    return D


def compute_precision_recall(features_real, features_fake, k=3):
    """
    Compute precision and recall using ADM's manifold estimation approach.
    
    Args:
        features_real: Real image features, shape [N_real, feature_dim]
        features_fake: Generated image features, shape [N_fake, feature_dim]
        k: Number of nearest neighbors (default: 3)
    
    Returns:
        precision: Fraction of fake samples that fall within the real manifold
        recall: Fraction of real samples that fall within the fake manifold
    """
    # Ensure arrays are numpy arrays with float64 precision
    features_real = np.asarray(features_real, dtype=np.float64)
    features_fake = np.asarray(features_fake, dtype=np.float64)
    
    # Create manifold estimator
    if isinstance(k, int):
        nhood_sizes = (k,)
    else:
        assert isinstance(k, tuple) or isinstance(k, list), "k must be an integer or a tuple/list of integers"
        nhood_sizes = k
    estimator = ManifoldEstimator(nhood_sizes=nhood_sizes)
    
    # Compute manifold radii for both real and fake features
    radii_real = estimator.manifold_radii(features_real)
    radii_fake = estimator.manifold_radii(features_fake)
    
    # Evaluate precision and recall
    precision, recall = estimator.evaluate_pr(
        features_real, radii_real, features_fake, radii_fake
    )
    
    return precision[0], recall[0]  # Return scalar values for k=3