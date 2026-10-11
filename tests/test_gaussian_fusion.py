import unittest
import numpy as np
from gaussian_fusion import fuse_gaussian_measurements as fuse


class FusionTests(unittest.TestCase):
    def test_equal_sensors_each_count_once(self):
        mean, covariance = fuse([[0], [2]], [[[1]], [[1]]])
        np.testing.assert_allclose(mean, [1])
        np.testing.assert_allclose(covariance, [[.5]])

    def test_unequal_scalar_variances(self):
        mean, covariance = fuse([[0], [10]], [[[1]], [[9]]])
        np.testing.assert_allclose(mean, [1])
        np.testing.assert_allclose(covariance, [[.9]])

    def test_proper_prior(self):
        mean, covariance = fuse([[4]], [[[2]]], prior_mean=[0], prior_covariance=[[2]])
        np.testing.assert_allclose(mean, [2])
        np.testing.assert_allclose(covariance, [[1]])

    def test_single_sensor_flat_prior(self):
        mean, covariance = fuse([[1, 2]], [[[2, .4], [.4, 1]]])
        np.testing.assert_allclose(mean, [1, 2])
        np.testing.assert_allclose(covariance, [[2, .4], [.4, 1]])

    def test_permutation_invariance(self):
        means = np.array([[1, 2], [3, 4]])
        covs = np.array([[[2, .2], [.2, 1]], [[1, -.1], [-.1, 3]]])
        a = fuse(means, covs)
        b = fuse(means[::-1], covs[::-1])
        for x, y in zip(a, b): np.testing.assert_allclose(x, y)

    def test_product_log_density_ratios(self):
        means = np.array([[1., 2.], [3., 1.]])
        covs = np.array([[[2., .3], [.3, 1.]], [[1., -.2], [-.2, 3.]]])
        mu, cov = fuse(means, covs)
        def log_kernel(x, mean, covariance):
            residual = x-mean
            return -.5 * residual @ np.linalg.solve(covariance, residual)
        base = np.array([.5, .6])
        for x in (np.array([0., 0.]), np.array([2., 4.]), np.array([-1., 5.])):
            original = sum(log_kernel(x,m,c)-log_kernel(base,m,c) for m,c in zip(means,covs))
            posterior = log_kernel(x,mu,cov)-log_kernel(base,mu,cov)
            self.assertAlmostEqual(original, posterior, places=12)

    def test_invalid_covariances(self):
        for cov in ([[1, 2], [2, 1]], [[1, 0], [1, 1]], [[1, 0], [0, 0]],
                    [[float('nan'), 0], [0, 1]]):
            with self.assertRaises(ValueError): fuse([[0, 0]], [cov])

    def test_invalid_shapes_and_partial_prior(self):
        for means, covs in (([], []), ([[0]], []), ([[0, float('inf')]], [[[1,0],[0,1]]])):
            with self.assertRaises(ValueError): fuse(means, covs)
        with self.assertRaises(ValueError): fuse([[0]], [[[1]]], prior_mean=[0])


if __name__ == '__main__': unittest.main()
