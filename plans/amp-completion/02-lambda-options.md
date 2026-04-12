# AMP Phase 2: Lambda Formulation and Solver Options

Plan phases:
- 4 Convex hull (lambda) formulation
- 6 Alpine-compatible solver options

Primary files:
- `python/discopt/_jax/milp_relaxation.py`
- `python/discopt/solvers/amp.py`
- `python/discopt/solver.py`
- `python/tests/test_amp.py`

Exit criteria:
- `convhull_formulation` can select between current McCormick and lambda relaxations.
- AMP routes Alpine-style tuning parameters end-to-end.
- `nlp3` shows a clear improvement with the lambda formulation.
- Tests cover formulation validity, tighter bounds, and option pass-through.
