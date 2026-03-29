import numpy as np
def compute_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    sigma1 = np.atleast_1d(sigma1).astype(np.float64)
    sigma2 = np.atleast_1d(sigma2).astype(np.float64)

    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2
    tr_covmean = np.sum(np.sqrt(np.linalg.eigvals(sigma1.dot(sigma2)).astype("complex128")).real)
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return fid
