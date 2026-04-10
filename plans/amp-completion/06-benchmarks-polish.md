# AMP Phase 6: Validation, Performance, and Documentation

Plan phases:
- 9 MINLPTests integration
- 10 Performance and polish

Primary files:
- `python/tests/test_minlptests.py`
- `python/tests/data/known_failures.toml`
- `discopt_benchmarks/benchmarks/problems/minlptests_problems.py`
- `python/discopt/solvers/amp.py`
- `docs/notebooks/amp_global_optimization.ipynb`

Exit criteria:
- AMP is registered in the MINLPTests infrastructure with tracked known failures.
- Alpine cross-validation exists for the targeted nonconvex problems.
- Warm starts, sparse matrices, and logging improvements are validated or explicitly deferred.
- The notebook documents the algorithm and the benchmark results clearly.
