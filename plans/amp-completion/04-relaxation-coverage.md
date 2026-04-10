# AMP Phase 4: Trilinear and Higher-Order Monomial Coverage

Plan phases:
- 3 Trilinear decomposition
- 5 Monomial `n > 2` piecewise relaxation

Primary files:
- `python/discopt/_jax/milp_relaxation.py`
- `python/discopt/_jax/term_classifier.py`
- `python/tests/test_amp.py`

Exit criteria:
- Trilinear terms are decomposed into valid nested bilinear relaxations.
- Higher-order monomials get piecewise relaxations rather than single-point tangents.
- Tests prove valid lower bounds and end-to-end convergence on targeted models.
