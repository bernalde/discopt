# AMP Phase 6: Validation, Performance, and Documentation

Plan phases:
- 9 MINLPTests integration
- 10 Performance and polish

Repository touchpoints for this phase:
- `python/tests/test_minlptests.py` (existing)
- `python/tests/data/known_failures.toml` (existing)
- `discopt_benchmarks/benchmarks/problems/minlptests_problems.py` (existing)
- `python/discopt/solvers/amp.py` (existing)
- `docs/notebooks/amp_global_optimization.ipynb` (to be added)

This planning PR only adds the scoped checklist. The files above are the
expected implementation touchpoints on top of the current AMP line.

Exit criteria:
- AMP is registered in the MINLPTests infrastructure with tracked known failures.
- Alpine cross-validation exists for the targeted nonconvex problems.
- Warm starts, sparse matrices, and logging improvements are validated or explicitly deferred.
- The notebook documents the algorithm and the benchmark results clearly.
