# AMP Phase 5: Advanced Formulations and Adaptive Partitioning

Plan phases:
- 7 Facet formulation and embedded binary encoding
- 8 Adaptive variable selection

Primary files:
- `python/discopt/_jax/milp_relaxation.py`
- `python/discopt/_jax/embedding.py`
- `python/discopt/_jax/partition_selection.py`
- `python/discopt/solvers/amp.py`
- `python/tests/test_amp.py`

Exit criteria:
- Facet and embedded-binary variants are available behind explicit options.
- Binary growth scales like `O(log K)` where expected.
- Variable selection can change across AMP iterations based on bound contribution.
- Tests cover correctness and expected model-size reductions.
