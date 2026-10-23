# Heston arithmetic Asian options by quintic B-spline collocation

This repository contains the code, the raw results and the LaTeX source of the paper

> M. Sharif, *Quintic B-spline collocation for continuously monitored arithmetic Asian options under
> the Heston model*, preprint, 2026.

The paper prices a fixed-strike arithmetic Asian call with continuous monitoring when the underlying
follows the Heston model. After the similarity reduction of Rogers, Shi and Večeř the price solves a
two-dimensional PDE with time-independent coefficients. We discretise it with tensor-product quintic
B-splines, collocate at Greville points, and march in time with Crank–Nicolson after four Rannacher
steps. Because the operator does not change in time, a whole solve needs two sparse LU factorisations,
and because the strike only enters where the solution is read, one solve prices every strike.

Most of the work, and most of this repository, is about checking the solver. There are three references:
an exact price for the geometric-average version of the contract (Riccati system plus Carr–Madan), the
lower bound of Fusai and Kyriakou (2016) with the transform derived for a continuous average under Heston,
and a Monte Carlo estimator that uses this bound as a control variate and reaches a standard error of
about 1.6e-5. Everything needed to rebuild the tables and figures of the paper from these references is
here, together with the output of every run we used.

## Layout

```
src/            the numerical code (plain modules, no package install needed)
experiments/    one script per experiment; each writes its output into results/
results/        raw output of all runs used in the paper (JSON, plus console logs)
figures/        make_figs.py and the four figures of the paper (PDF and PNG)
tests/          quick checks with pytest (a few seconds)
paper/          LaTeX source of the paper
```

The modules in `src/`:

| module | what it does |
|---|---|
| `arith_order.py` | the solver used for all arithmetic results: `solve_multi` (one solve, many strikes) and `price_extrapolated` (with Richardson extrapolation in time) |
| `arith_colloc.py` | B-spline machinery (knot vectors, collocation matrices, grids) and the first single-strike version of the solver |
| `arith_adi.py` | modified Craig–Sneyd ADI scheme on the same reduced equation, used as the comparator |
| `geo_chf.py` | exact price of the continuously monitored geometric Asian call under Heston |
| `geo_colloc.py`, `geo_adi.py` | collocation and ADI solvers for the geometric twin (time-dependent operator) |
| `fk_bound.py` | the Fusai–Kyriakou lower bound for Heston with a continuous average (`FKBound`) |
| `mc_fk.py` | Monte Carlo for the difference between price and bound, which gives the reference price |
| `mc_check.py` | Andersen's QE scheme with the exact geometric price as control variate |
| `mc_euler.py` | full-truncation Euler scheme, used as an independent check in Section 5.6 |

A note on parameter names. The solvers use `kappa, theta, xi_, rho, v0`. `FKBound` follows the notation
of Fusai and Kyriakou, where `alpha` is the mean-reversion speed, `beta` the long-run variance, `gamma`
the vol-of-vol and `V0` the initial variance. The results files keep a few names from the lab notes:
`E8` is the reference price MLB + E[D], `gap` is E[D], `R2` is a price after two levels of Richardson
extrapolation in time, and `k` is the grid level, with `N_x = 17k + 1` points in the similarity variable.

## Installation

Python 3.10 or later with NumPy, SciPy and Matplotlib. The results in the paper were produced with
Python 3.11, NumPy 2.4, SciPy 1.17 and Matplotlib 3.10.

```
pip install -r requirements.txt
pytest -q tests
```

The tests check the one-dimensional collocation order (4 for quintics, 2 for cubics on a smooth
problem), the exact geometric price in the Black–Scholes limit and its stability, and the lower bound
against its closed form in the Gaussian limit.

## Pricing an option

```python
import sys; sys.path.insert(0, "src")
from arith_order import price_extrapolated

# base parameters of the paper: kappa=1, theta=0.09, xi=1, rho=-0.3, v0=0.09, S0=100, r=0.04, T=1
prices = price_extrapolated(k=16, Nv=61, Nt=1600, Ks=(90.0, 100.0, 110.0))
print(prices)
```

This takes about a minute and a half on the two-core machine used for the paper, and the discretisation error
of the result is below 1e-6 at these strikes. The default domain is x in [-1, 5] and v in [0, 1.5]. That is enough for the base
parameters but not everywhere: Section 5.4 of the paper shows that a vol-of-vol of 1.5 needs
`v_hi=3` and a five-year maturity needs a wider x-domain, and that grid refinement does not reveal
either problem. If you move far from the base parameters, move the boundaries and check that the price
does not change.

## Reproducing the paper

The `results/` directory already holds the output of every run, so the figures and the tables can be
rebuilt without running anything expensive. The scripts in `experiments/` recreate those files. Each
script puts `src/` on the path and runs inside `results/`, so they can be started from anywhere, for
example `python experiments/03b_fk_table10.py`. Times below are wall-clock times on the two-core
virtual machine used for the paper; the Monte Carlo runs dominate.

| paper | script(s) | output in `results/` | time |
|---|---|---|---|
| Sec. 2.3, condition at v_max | `06c_vmax_condition.py` | `bcv.json` | 6 min |
| Sec. 4.1, exact geometric price | `01a_geo_chf_limits.py`, `01b_geo_chf_plateau.py` | printed (`*.log`) | < 1 min |
| Sec. 4.1, check at T = 0.25, 3, 5 | `02a_mc_maturity_axis.py`, `02b_solver_long_maturity.py` | `A4b.json`, `A5.json` | 10 min |
| Sec. 4.2, transform checks | `03a_transform_identities.py` | printed (`*.log`) | < 1 min |
| Table 1 | `03b_fk_table10.py` | `B2.json` | < 1 min |
| Table 2 | `05a_reference_runs.py` (three calls, see below), `05b_reference_pool.py`, `05c_geometric_cv_reference.py` | `E8*.json`, `E8_pooled.json`, `geo_cv_reference.json` | 30 min |
| Table 3, Sec. 5.1 and 5.5 (geometric twin) | `10_geometric_twin.py` | `geo_runs.json` | 25 min |
| Tables 4 and 5 | `06a_convergence_runs.py C1` (and `C2`, `C4`), then `06d_time_and_price_tables.py` | `C1.json` | 11 min |
| Table 6, Figure 2 | `06a_convergence_runs.py C2`, `06a_convergence_runs.py C4`, `06b_convergence_analysis.py` | `C2.json`, `C4.json`, `C24.json` | 2 h |
| Table 7 | `08d`, `08e` (T = 3, 5), `08g`, `08h` (xi = 1.5) | `diag_*.json` | 70 min |
| Table 8 | `03c_fk_table9.py`, `03d_fk_table9_conventions.py`, `03e_fk_table9_collocation.py base`, `04a_table9_mc_qe.py`, `04b_table9_mc_euler.py`, `04c_table9_mc_euler_pool.py` | `B3.json`, `diagB3.json`, `fk_table9_colloc_*.json`, `mcB3_*.json`, `mc_euler*.json` | 30 min |
| Figure 1, Table 9 | `07a_solver_sets.py strikes`, `07b_reference_sets.py strikes 16000000 256` | `sol_strikes.json`, `e8_strikes.json` | 10 min |
| Figure 3, Table 10 | `07a_solver_sets.py regimes`, `07b_reference_sets.py regimes 4000000 256`, then `08a` to `08h`, merged by `08i_merge_regimes.py` | `regimes_final.json` | 3 h |
| Figure 4, Table 11 | `09_adi_comparison.py` | `comp_arith.json` | 10 min |

The reference of Table 2 was pooled from seven batches:

```
python experiments/05a_reference_runs.py 4000000 128,256,512,1024 E8.json
python experiments/05a_reference_runs.py 8000000 32,64 E8_coarse.json
python experiments/05a_reference_runs.py 4000000 2048 E8_fine.json
python experiments/05b_reference_pool.py
```

The comparison of the two control variates on the same paths (Section 4.3) is
`python ../src/mc_fk.py corsaro 2000000 256 20000`, run inside `results/`. The Table 8 simulations were
run as `04a_table9_mc_qe.py 2000000 N` for N = 128, 256, 512 and as
`04b_table9_mc_euler.py FK-1,FK-2,FK-3 4000000 100,200,400,800`. The regime diagnostics
`08b`–`08h` are small scripts, each changing one thing at a time; their docstrings say what they test.
The figures are rebuilt from `results/` with

```
python figures/make_figs.py
cp figures/*.pdf paper/figs/
```

The longer solver scripts (`03e`, `06a`, `07a`, `07b`, `08a`, `08d`, `08e`, `08h`, `09`, `10`) skip jobs
already present in their output file, so an interrupted run can be restarted; delete the file to recompute
from scratch. The other scripts overwrite their output. The Monte Carlo scripts use fixed seeds, so
a rerun gives the same numbers on the same NumPy version.

The `.log` files in `results/` are the console output of the original runs, kept as they were. Some of
them contain Russian, the language of the lab notes; the JSON files hold the same numbers.

## Building the paper

```
cd paper
latexmk -pdf main.tex
```

The source uses standard packages only (amsmath, natbib, booktabs, siunitx, hyperref).

## Citation and licence

If you use the code, please cite the paper; `CITATION.cff` has the details. The code is released under
the MIT licence (see `LICENSE`).
