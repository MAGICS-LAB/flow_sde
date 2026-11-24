# Synthetic Gaussian experiments

Code for Sections 5.1-5.2, Figure 2, and Appendices F.1-F.2 of the paper.

## Run

Use Python 3.10 or newer and install dependencies in your active environment:

```bash
python -m pip install -r NumericalExp/requirements.txt
cd NumericalExp
python TV_eps_scaling.py
python Violation_rate.py
python Trajectory.py
```

These are full experiments with training and Monte Carlo evaluation. They can take substantial time. For a small CPU check, run `python -m pytest -q tests` after installing the root development requirements.

| Script | Paper panel | Generated files |
| --- | --- | --- |
| `TV_eps_scaling.py` | Figure 2(a) | `theorem4_1_loglog.pdf`, `theorem4_1_results.json` |
| `Violation_rate.py` | Figure 2(b) | `binned_violation_rate.pdf`, `violation_rate_results.json` |
| `Trajectory.py` | Figure 2(c) | `trajectory_data.pdf`, `trajectory_results.json` |

Parameters are defined in each script's experiment function. Outputs are written to the current working directory. CUDA is selected when available, with CPU fallback.

## Measurement conventions

The reference path is `X_t = (1-t) X_0 + t X_1` with independent Gaussian endpoints. The analytic marginal velocity supplies the regression target. ODE inference uses Euler integration; SDE inference uses Euler-Maruyama with constant noise level 1 and terminal time `T=0.99`.

Let `I = integral_0^T E[||u_theta-u_star||^2] dt`. The manuscript defines `epsilon=sqrt(I)`. `TV_eps_scaling.py` uses this convention. The transfer scripts retain the historical field name `eps_sqrt` for `sqrt(I)`, so this field is the paper's **epsilon**. The legacy Figure 2(b) label `sqrt(epsilon)` uses a different convention; see the [reproduction notes](../docs/REPRODUCIBILITY.md). Raw values are retained to make that distinction explicit.

The TV figure uses a different target Gaussian from the two transfer experiments, as specified in Sections 5.1 and 5.2. Original synthetic measurement files were not supplied; running these scripts produces new measurements, rather than replaying the published ones.
