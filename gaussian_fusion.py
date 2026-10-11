"""Independent identity-observation Gaussian likelihoods; explicit optional prior.

This is a scoped Gaussian teaching example, separate from full-density Lorenz DA.
"""
import numpy as np


def _covariance(value, dimension):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (dimension, dimension) or not np.isfinite(matrix).all():
        raise ValueError("Covariance must be a finite d-by-d matrix")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
        raise ValueError("Covariance must be symmetric")
    matrix = (matrix + matrix.T) / 2
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as error:
        raise ValueError("Covariance must be positive definite") from error
    return matrix


def fuse_gaussian_measurements(means, covariances, *, prior_mean=None, prior_covariance=None):
    """Return mean/covariance for y_i | z ~ N(z, R_i), independent given z.

    Each supplied measurement contributes exactly once. Without a proper prior,
    this explicitly uses a flat improper prior on R^d; positive-definite likelihood
    covariances ensure a proper Gaussian posterior. Correlated sensors require a
    joint observation model and are outside this function's contract.
    """
    means = np.asarray(means, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    if means.ndim != 2 or not all(means.shape) or not np.isfinite(means).all():
        raise ValueError("Means must be a nonempty finite n-by-d array")
    n, d = means.shape
    if covariances.shape != (n, d, d):
        raise ValueError("Supply exactly one d-by-d covariance per measurement")
    if (prior_mean is None) != (prior_covariance is None):
        raise ValueError("Supply both prior mean and covariance, or neither")
    precision = np.zeros((d, d))
    information = np.zeros(d)
    identity = np.eye(d)
    for mean, covariance in zip(means, covariances):
        weight = np.linalg.solve(_covariance(covariance, d), identity)
        precision += weight
        information += weight @ mean
    if prior_mean is not None:
        prior_mean = np.asarray(prior_mean, dtype=float)
        if prior_mean.shape != (d,) or not np.isfinite(prior_mean).all():
            raise ValueError("Prior mean must be a finite vector of dimension d")
        weight = np.linalg.solve(_covariance(prior_covariance, d), identity)
        precision += weight
        information += weight @ prior_mean
    mean = np.linalg.solve(precision, information)
    covariance = np.linalg.solve(precision, identity)
    return mean, (covariance + covariance.T) / 2
