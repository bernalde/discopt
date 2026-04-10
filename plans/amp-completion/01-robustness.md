# AMP Phase 1: Bug Fixes and Robustness

Plan phases:
- 1A Big-M scaling
- 1B Original variable integrality in the MILP
- 1C OA-cut infeasibility recovery and safe constraint checking
- 1D Integer rounding fallback search
- 1E Maximize objective support

Primary files:
- `python/discopt/_jax/milp_relaxation.py`
- `python/discopt/solvers/amp.py`
- `python/discopt/solver.py`
- `python/tests/test_amp.py`

Exit criteria:
- AMP handles both minimize and maximize correctly.
- Original integer and binary variables stay integral in the MILP relaxation.
- OA-cut accumulation cannot permanently kill the MILP solve path.
- Integer fixing is robust to nearest-round infeasibility.
- New regression tests cover every bug fixed here.
