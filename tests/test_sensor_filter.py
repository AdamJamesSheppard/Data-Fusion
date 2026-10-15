import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec = importlib.util.spec_from_file_location('sensor_filter', Path(__file__).resolve().parents[1]/'kalman filter with two sensors.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class SensorTests(unittest.TestCase):
    def test_fused_covariance(self):
        value, variance = m.fuse_measurements([[0.], [2.]], [1., 1.])
        np.testing.assert_allclose(value, [1.])
        self.assertEqual(variance, .5)
    def test_joseph_update(self):
        x, p = m.update(np.zeros(2), np.eye(2), np.ones(2), .5)
        np.testing.assert_allclose(x, np.full(2, 2/3))
        np.testing.assert_allclose(p, np.eye(2)/3)
    def test_actual_filter_metric_and_repeatability(self):
        a, truth, fused, estimates = m.run_experiment(n_steps=12)
        b, *_ = m.run_experiment(n_steps=12)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a['ekf_estimate_mae'], np.mean(abs(estimates-truth)))
        self.assertNotAlmostEqual(a['ekf_estimate_mae'], a['fused_measurement_mae'])
    def test_nonlinear_prediction(self):
        x = np.arange(8, dtype=float)
        self.assertFalse(np.allclose(m.transition_jacobian(x, .01), np.eye(8)))
    def test_invalid_noise(self):
        with self.assertRaises(ValueError): m.fuse_measurements([[1.]], [0.])
