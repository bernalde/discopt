# Changelog

All notable changes to discopt are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The release procedure that produces these entries is documented in
[`RELEASE.md`](RELEASE.md).

## [Unreleased]

### Added

### Changed

### Fixed

## [0.2.6] - 2026-04-12

### Added

- `RELEASE.md` -- authoritative release checklist documenting the procedure for cutting a discopt release.
- `CHANGELOG.md` -- this file, in Keep a Changelog format.
- Local `cargo-fmt` pre-commit hook so Rust formatting is enforced alongside `ruff` and `mypy`.

### Changed

- Bumped `ripopt` workspace dependency from `0.6.1` to `0.6.2` (`Cargo.toml`, `Cargo.lock`). Transitively updates `rmumps` `0.1.0` -> `0.1.1`.
- `_solve_continuous` (pure-continuous NLP fast path) now promotes the default `nlp_solver="ipm"` to `"ipopt"` for single-problem solves. The pure-JAX IPM's acceptable-tolerance check only covers variable-bound complementarity, so on problems with unbounded variables plus inequality constraints it could terminate at a non-KKT point and report OPTIMAL. Ipopt is more reliable for single solves; the JAX IPM remains the default for B&B subproblems.
- `differentiable_solve` and `differentiable_solve_l3` default backend changed from `"ipm"` to `"ipopt"` for the same reason.

### Deprecated

### Removed

### Fixed

- Jupyter Book documentation now builds with zero Sphinx warnings. Cleaned up RST-formatting issues in module docstrings for `benchmarks/problems/gas_network_minlp.py`, `modeling/core.py`, `ro/formulations/box.py`, `solvers/qp_highs.py`, and `solvers/sipopt.py`, and suppressed autoapi import-resolution warnings for the compiled `discopt._rust` extension.
- **HiGHS LP/QP false optimality on wide bounds**: `solvers/qp_highs.py` and `solvers/lp_highs.py` now clip any bound with magnitude >= `1e15` to `highspy.kHighsInf` before passing to HiGHS. Bounds like discopt's default `+/-9.999e19` fall just below HiGHS's internal infinity threshold (`1e20`) and caused HiGHS to return false-optimal solutions on convex QPs with unbounded variables.
- **Single-solve starting point**: `_solve_continuous` now clips the default starting point to `+/-10` (respecting actual bounds) instead of the previous `+/-100`, preventing ipopt from exploding on exp/log NLPs with one-sided large bounds.
- **Stationary-point starting point**: Fully unbounded variables (`|lb| > 1e15` and `|ub| > 1e15`) now start at `0.5` instead of the midpoint of `0`. Zero is a stationary point of periodic functions (sin, cos) and even functions generally; starting at 0.5 lets first-order NLP methods pick a descent direction and escape local maxima of the objective. Same fix applied in `_jax/differentiable.py::_safe_x0`.
- `_solve_qp_highs` and `_solve_qp_jax` now set `SolveResult.convex_fast_path = True` when solving a detected convex QP directly, matching the semantics of the convex NLP fast path.
- **Cutting planes with bilinear terms (#35)**: `_jax/cutting_planes.py::generate_rlt_cuts` no longer emits unsound inequalities when a bilinear term has no auxiliary `w_index`. The old no-auxiliary branch produced cuts purely in the original variable space that were not valid relaxations of the product (e.g. with `x,y` in `[0.1,5]` it emitted `0.1*x + 0.1*y <= 0.01`, excluding every feasible point). Since `detect_bilinear_terms` always returns `w_index=None`, every RLT cut fed into `_AugmentedEvaluator` made the NLP infeasible at every B&B node, so no incumbent could be accepted on mixed convex/nonconvex MINLPs when `cutting_planes=True`. The function now returns `[]` in that case. Fixed in `383985e`.

### Security

## [0.2.5] and earlier

Historical releases (`v0.2.0` through `v0.2.5`) are not backfilled in this file.
For commit-level history of those releases, see:

```bash
git log v0.2.4..v0.2.5
git log v0.2.3..v0.2.4
git log v0.2.2..v0.2.3
git log v0.2.1..v0.2.2
git log v0.2.0..v0.2.1
```

Going forward, every release will have a section above with curated entries.

[Unreleased]: https://github.com/jkitchin/discopt/compare/v0.2.6...HEAD
[0.2.6]: https://github.com/jkitchin/discopt/compare/v0.2.5...v0.2.6
