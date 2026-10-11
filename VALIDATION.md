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
