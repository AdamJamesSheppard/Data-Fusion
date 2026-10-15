"""Seeded Lorenz-96 sensor-fusion teaching experiment, with an extended Kalman filter.

Independent isotropic sensor noise is assumed. Q is a chosen discrete model-error
covariance, not estimated from data. This experiment supplies no general accuracy
or stability guarantee. Run directly to print separate measurement/filter MAEs.
"""
import json
import numpy as np


def lorenz96(x, forcing=8.0):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + forcing


def transition(x, dt):
    """One classical RK4 step of the nonlinear Lorenz-96 transition."""
    a = lorenz96(x)
    b = lorenz96(x + dt*a/2)
    c = lorenz96(x + dt*b/2)
    d = lorenz96(x + dt*c)
    return x + dt*(a+2*b+2*c+d)/6


def transition_jacobian(x, dt):
    """Central-difference Jacobian of the actual discrete transition."""
    columns = []
    for j in range(len(x)):
        delta = np.zeros_like(x)
        delta[j] = 1e-5 * max(1.0, abs(x[j]))
        columns.append((transition(x+delta, dt)-transition(x-delta, dt))/(2*delta[j]))
    return np.column_stack(columns)


def fuse_measurements(measurements, noise_stds):
    """Return precision-weighted measurement and its isotropic noise variance."""
    values = np.asarray(measurements, dtype=float)
    stds = np.asarray(noise_stds, dtype=float)
    if stds.ndim != 1 or len(stds) != len(values) or not len(stds):
        raise ValueError("Supply one standard deviation per sensor")
    if not np.isfinite(stds).all() or np.any(stds <= 0) or not np.isfinite(values).all():
        raise ValueError("Require finite measurements and positive finite standard deviations")
    precision = 1/stds**2
    return np.tensordot(precision/precision.sum(), values, axes=(0, 0)), float(1/precision.sum())


def update(x, covariance, observation, variance):
    """Identity-observation EKF update with Joseph covariance form."""
    identity = np.eye(len(x))
    r = variance * identity
    gain = np.linalg.solve((covariance+r).T, covariance.T).T
    estimate = x + gain @ (observation-x)
    residual = identity-gain
    posterior = residual @ covariance @ residual.T + gain @ r @ gain.T
    return estimate, (posterior+posterior.T)/2


def run_experiment(seed=42, n_steps=100, n_vars=8, dt=0.01):
    if n_steps < 1 or n_vars < 4 or dt <= 0 or not np.isfinite(dt):
        raise ValueError("Require positive steps/dt and at least four state variables")
    rng = np.random.default_rng(seed)
    truth = np.empty((n_steps, n_vars))
    truth[0] = rng.random(n_vars)
    for t in range(1, n_steps):
        truth[t] = transition(truth[t-1], dt)
    stds = np.array([1.0, 3.0])
    sensors = truth[None, :, :] + rng.normal(size=(2, n_steps, n_vars))*stds[:, None, None]
    fused, variance = fuse_measurements(sensors, stds)
    estimates = np.empty_like(truth)
    x, covariance = np.zeros(n_vars), np.eye(n_vars)
    for t, observation in enumerate(fused):
        # First observation corresponds to initial time; no preceding prediction.
        if t:
            jacobian = transition_jacobian(x, dt)
            x = transition(x, dt)
            covariance = jacobian @ covariance @ jacobian.T + 0.01*np.eye(n_vars)
        x, covariance = update(x, covariance, observation, variance)
        estimates[t] = x
    report = {"seed": seed, "steps": n_steps, "dt": dt,
              "fused_measurement_variance": variance,
              "sensor_1_mae": float(np.mean(abs(sensors[0]-truth))),
              "sensor_2_mae": float(np.mean(abs(sensors[1]-truth))),
              "fused_measurement_mae": float(np.mean(abs(fused-truth))),
              "ekf_estimate_mae": float(np.mean(abs(estimates-truth))),
              "scope": "Configured synthetic software demonstration; no empirical validation"}
    return report, truth, fused, estimates


if __name__ == "__main__":
    print(json.dumps(run_experiment()[0], indent=2))
