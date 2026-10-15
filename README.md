# Bayesian data-fusion notes — historical teaching experiments

This repository studies explicitly Gaussian sensor models. It is separate from
the current full-density Lorenz Fokker–Planck/operator-learning programme, whose
eventual assimilation contract excludes Gaussian state closure as a substitution.
Neither this repository nor its Kalman example qualifies that current application.

## Audited Gaussian fusion utility

`gaussian_fusion.py` implements independent measurements
`y_i | z ~ N(z, R_i)` with identity observation maps. Each sensor contributes
once. A Gaussian prior is optional; omitting it explicitly means a flat improper
prior on the state space. Positive-definite sensor covariances yield a proper
posterior in this identity-observation setting.

Completing the square gives the precision and information vector:

```
J = sum_i solve(R_i, I) [+ solve(P0, I)]
h = sum_i solve(R_i, y_i) [+ solve(P0, m0)]
posterior_mean = solve(J, h)
posterior_covariance = solve(J, I)
```

Bracketed terms apply only to a supplied proper prior. This is a mathematical
derivation under the stated Gaussian/conditional-independence assumptions,
not an assumption-free Bayesian method. Correlated sensors, nonlinear observation
maps, uncertain noise covariances and non-Gaussian states are outside its scope.
Reference: Kevin P. Murphy, *Probabilistic Machine Learning: An Introduction*,
MIT Press, 2022, §§3.3.1–3.3.5,
[book and sensor-fusion context](https://probml.github.io/book1).
The numerical implementation uses linear solves rather than explicit inversion.

```python
from gaussian_fusion import fuse_gaussian_measurements
mean, covariance = fuse_gaussian_measurements([[0.0], [2.0]], [[[1.0]], [[1.0]]])
# mean=[1], covariance=[[0.5]]
```

```sh
python3 -m unittest discover -s tests -v
```

Requires NumPy. Tests cover analytic scalar cases, an explicit proper prior,
correlated coordinates within a sensor, permutation invariance, product-density
ratios, invalid covariances and shape errors. These software/numerical checks
do not establish sequential filtering accuracy or real-sensor calibration.

## Preserved limitations (audit: 2026-10-11)

The following describes the pre-repair versions preserved in Git history.
The current entry points were repaired on 15 October 2026 (see below).
`DataFusion.ipynb` cell9 doubles sensor1 precision and information without
declaring a second independent observation; its two-sensor interpretation is
therefore unsupported. The posterior covariance formula in cell8 also omits the
inverse of the summed precision. The new utility corrects these specific defects;
it does not retrospectively validate the notebook.

`kalman filter with two sensors.py` uses identity transition dynamics for filtering
Lorenz-96 data, chooses measurement covariance independently of fused noise, and
reports fused-measurement errors rather than Kalman estimate errors. Its plots
cannot establish accurate Lorenz filtering. The current script explicitly uses an extended Kalman filter with RK4 dynamics;
its seeded synthetic experiment remains a teaching demonstration, not scientific qualification.

## Entry-point repairs — 15 October 2026
The notebook now delegates to the tested independent-Gaussian utility and corrects the precision derivation. The sensor-filter example now uses the nonlinear RK4 transition and reports filter errors separately. Historical versions remain available in Git history.
