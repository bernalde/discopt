# AMP Phase 3: OBBT Presolve

Plan phases:
- 2 OBBT presolve integration

Primary files:
- `python/discopt/solvers/amp.py`
- `python/discopt/_jax/obbt.py`
- `python/discopt/solver.py`
- `python/tests/test_amp.py`

Exit criteria:
- AMP can tighten bounds with OBBT before the main iteration loop.
- OBBT can be enabled or disabled from the public solver interface.
- Tests show bound tightening and at least one measurable iteration reduction.
