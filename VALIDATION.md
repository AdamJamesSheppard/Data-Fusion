# Entry-point repair validation — 2026-10-15

**13/13 tests passed** with `python -m unittest discover -s tests -v`.

Notebook execution additionally returned the analytic mean 1 and covariance 0.5. Filter tests check fused noise covariance, Joseph update, nonlinear transition and actual estimate-error accounting. No real-sensor validation or general EKF performance claim.

Environment: Python 3.12.3, NumPy 1.26.4, pandas 3.0.6, SciPy 1.11.4, scikit-learn 1.9.1. Dependencies were supplied in an isolated task environment.
Self-review by implementing assistant; no independent assessment or CI success is inferred.

## Earlier validation record

# Repair validation — 2026-10-11

Historical base: `e34ce846349bed1d338933ffea1f9fbb4d6a3f4c`.
Original notebook/Kalman script unchanged. Additive utility, eight tests:
**8/8 pass** via `python -m unittest discover -s tests -v`.
Local environment: Python3.12.13, NumPy2.5.2. Elapsed suite approximately0.014s.
CI uses a separately declared NumPy version; its result is not inferred here.

Self-review by implementing assistant; no independent assessor. Analytic test
oracles and product-density-ratio checks support the scoped independent Gaussian
measurement formula. They do not validate the old plots, correlated sensors,
Lorenz filtering, non-Gaussian state approximation or full-density assimilation.
