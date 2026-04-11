# AMP Global Solver: Completion Plan

## Context

The AMP base from PR https://github.com/bernalde/discopt/pull/1 is merged, and the remaining work now lives in the stacked draft PRs on top of `upstream-main-sync`. The current implementation solves simple bilinear problems (nlp1: obj=58.38, certified; circle: obj=1.414, certified) but still has significant gaps vs the reference Alpine.jl implementation at `/home/bernalde/repos/Alpine.jl`. This plan covers the remaining feature work and validation against the upstream MINLPTests integration.

**What already works**:
- `term_classifier.py` -- bilinear, monomial(n=2), trilinear detection
- `partition_selection.py` -- max_cover, min_vertex_cover (HiGHS MILP)
- `discretization.py` -- DiscretizationState, init, adaptive refinement
- `milp_relaxation.py` -- piecewise McCormick (big-M delta) for bilinear, monomial(n=2)
- `solvers/amp.py` -- main loop: MILP LB -> NLP UB -> refine -> repeat
- `solver.py` routing -- `solver="amp"` dispatch

**Soundness invariant**: at every iteration, `MILP_LB <= global_opt <= NLP_UB`.

---

## Phase 1: Bug Fixes and Robustness

**Goal**: Fix known bugs that compound with later features.

### 1A: Big-M Scaling
- **File**: [milp_relaxation.py:504](python/discopt/_jax/milp_relaxation.py#L504)
- **Bug**: `M_k = max(abs(c) for c in corners) + 1.0` -- hardcoded `+1.0` is insufficient for large-coefficient problems (nlp3 has coefficients ~10000)
- **Fix**: `M_k = max(abs(c) for c in corners) * (1 + 1e-4) + 1e-2`
- **Test**: Existing tests must still pass; add test with large coefficients

### 1B: Original Variable Integrality in MILP
- **File**: [milp_relaxation.py:370](python/discopt/_jax/milp_relaxation.py#L370)
- **Bug**: `integrality_flags: list[int] = [0] * n_orig` -- integer/binary vars treated as continuous in MILP relaxation
- **Fix**: Walk `model._variables`, set flags for `VarType.BINARY` and `VarType.INTEGER`
- **Test**: Add test verifying MILP respects integrality of original variables

### 1C: OA Cut Feasibility Recovery + Constraint Check Fix
- **File**: [amp.py:332-362](python/discopt/solvers/amp.py#L332-L362), [amp.py:146](python/discopt/solvers/amp.py#L146)
- **Issue 1**: OA cuts at feasible NLP points are valid cutting planes, but accumulated cuts can make MILP infeasible. Add recovery: if MILP is infeasible and `oa_cuts` is non-empty, drop oldest half and retry.
- **Issue 2**: `_check_constraints` returns `True` on evaluation failure (line 146) -- can accept infeasible points as incumbents. Fix: return `False` on exception.
- **Use**: `convexity.py:classify_constraint()` exists and can optionally tag convex vs non-convex OA cuts for prioritized dropping.
- **Test**: Verify MILP recovers from infeasibility caused by aggressive OA cuts

### 1D: Integer Rounding Robustness
- **File**: [amp.py:302-303](python/discopt/solvers/amp.py#L302-L303)
- **Bug**: `_round_integers` rounds to nearest; if infeasible, NLP fails
- **Fix**: Try nearest first; if NLP fails, try floor/ceil alternatives
- **Test**: Add test where nearest-integer rounding is infeasible but floor/ceil works

### 1E: Maximize Objective Support
- **File**: [amp.py](python/discopt/solvers/amp.py) (entire `solve_amp()`)
- **Bug**: AMP loop assumes minimization -- LB/UB tracking, gap computation, incumbent comparison all break for maximize. The MILP builder handles it (negates at line 698), but `solve_amp` does not.
- **Fix**: Detect `ObjectiveSense.MAXIMIZE`, negate LB/UB semantics (MILP gives UB, NLP gives LB), adjust gap formula.
- **Test**: Add test with maximize objective that verifies correct gap certification.

**Verification**: `pytest python/tests/test_amp.py -m "not slow"` -- all 52 tests pass + new tests.
**Effort**: 1 session.

---

## Phase 2: OBBT Presolve Integration

**Goal**: Tighten variable bounds before AMP loop for tighter McCormick envelopes.

- **Files**: [amp.py](python/discopt/solvers/amp.py) (add OBBT call), [obbt.py](python/discopt/_jax/obbt.py) (reuse as-is)
- **Implementation**: After `classify_nonlinear_terms` and `_flat_bounds`, insert:
  ```python
  if presolve_bt:
      from discopt._jax.obbt import run_obbt
      obbt_result = run_obbt(model, lb=flat_lb.copy(), ub=flat_ub.copy(),
                             incumbent_cutoff=UB if UB < np.inf else None)
      if obbt_result.n_tightened > 0:
          flat_lb, flat_ub = obbt_result.tightened_lb, obbt_result.tightened_ub
  ```
- Add `presolve_bt: bool = True` parameter to `solve_amp()`
- Route through [solver.py:1104](python/discopt/solver.py#L1104)
- Alpine reference: `presolve.jl:1-120` (OBBT to fixed-point, `presolve_bt_max_iter=25`)

**Tests**: `test_obbt_presolve_tightens_bounds`, `test_amp_with_obbt_fewer_iterations`
**Effort**: 1 session.

---

## Phase 3: Trilinear Decomposition

**Goal**: Relax trilinear `x*y*z` by decomposing into nested bilinear: `w=x*y`, then `w*z`.

- **File**: [milp_relaxation.py](python/discopt/_jax/milp_relaxation.py) (add decomposition before bilinear processing)
- **Alpine reference**: `multilinear.jl:84-132`
- **Implementation**:
  1. For each `(i, j, k)` in `terms.trilinear`: create auxiliary `w_aux` with corner bounds from `x_i * x_j`
  2. Add `(i, j)` bilinear with `w_aux` as product variable
  3. Add `(w_aux_col, k)` as new bilinear term
  4. Generalize recursively for higher-order: `x_1*x_2*...*x_n -> (x_1*x_2=w_1), (w_1*x_3=w_2), ...`
- **Also update**: [term_classifier.py](python/discopt/_jax/term_classifier.py) if needed for higher-order detection

**Tests**: `test_trilinear_decomposition_valid_lb`, `test_trilinear_amp_converges`
**Effort**: 1 session.

---

## Phase 4: Convex Hull (Lambda) Formulation -- HIGHEST IMPACT

**Goal**: Implement Alpine's default, tightest relaxation. Strictly tighter than piecewise McCormick.

- **File**: [milp_relaxation.py](python/discopt/_jax/milp_relaxation.py) (add as alternative formulation)
- **Alpine reference**: `multilinear.jl:134-280` (SOS-2 lambda), `multilinear.jl:408-496` (monomial lambda)

### Theory (bilinear `w = x * y`, partition on x at `{x_0, ..., x_K}`):
```
Variables: lambda[i,j] >= 0 for grid {x_i} x {y_lb, y_ub}
           alpha[k] in {0,1} for k = 1..K-1

Constraints:
  sum(lambda) = 1                        (convexity)
  x = sum(lambda[i,j] * x_i)            (x-linking)  
  y = sum(lambda[i,j] * y_j)            (y-linking)
  w = sum(lambda[i,j] * x_i * y_j)      (bilinear product)
  sum(alpha) = 1                         (partition selection)
  SOS-2 linking: lambda_i <= alpha_{i-1} + alpha_i
```

### Implementation steps:
1. Add `convhull_formulation` parameter: `"mccormick"` (current), `"sos2"` (lambda), `"facet"` (Phase 7)
2. For `"sos2"`: create lambda variables at grid extreme points, alpha binaries, add all constraints above
3. For monomials: lambda at breakpoints with `w = sum(lambda_i * x_i^n)` values
4. Wire parameter through [amp.py](python/discopt/solvers/amp.py) and [solver.py](python/discopt/solver.py)

**Tests**: `test_lambda_lb_valid`, `test_lambda_tighter_than_mccormick`, `test_amp_lambda_nlp3_converges`
**Key benchmark**: nlp3 should converge within timeout with lambda formulation.
**Effort**: 2 sessions.

---

## Phase 5: Monomial n>2 Piecewise Relaxation

**Goal**: Fix incomplete monomial relaxation for `x^n` with `n > 2`.

- **File**: [milp_relaxation.py:637-654](python/discopt/_jax/milp_relaxation.py#L637-L654)
- **Bug**: Only ONE tangent cut at midpoint instead of piecewise at ALL breakpoints
- **Fix**:
  - Even n (convex): tangent underestimators at ALL breakpoints (same pattern as n=2), secant overestimator
  - Odd n, `lb >= 0`: convex, same as even
  - Odd n, `lb < 0 < ub`: piecewise -- tangent overestimators on `[lb, 0]`, tangent underestimators on `[0, ub]`

**Tests**: `test_cubic_monomial_valid`, `test_quartic_piecewise_tighter`
**Effort**: 1 session.

---

## Phase 6: Alpine-Compatible Solver Options

**Goal**: Expose all Alpine-compatible tuning parameters.

- **Files**: [amp.py](python/discopt/solvers/amp.py), [solver.py:1099-1126](python/discopt/solver.py#L1099-L1126)

| Parameter | Default | Alpine equivalent |
|-----------|---------|-------------------|
| `partition_scaling_factor` | 10 | `partition_scaling_factor` |
| `disc_uniform_rate` | 2 | `disc_uniform_rate` |
| `disc_var_pick` | `"auto"` | `disc_var_pick` (0/1/2/3) |
| `convhull_formulation` | `"sos2"` | `convhull_formulation` |
| `presolve_bt` | True | `presolve_bt` |
| `presolve_bt_max_iter` | 25 | `presolve_bt_max_iter` |
| `max_iter` | 99 | `max_iter` (currently 50) |
| `abs_gap` | 1e-6 | `tol` |

**Tests**: Parameter pass-through tests.
**Effort**: 1 session.

---

## Phase 7: Facet Formulation and Embedded Binary Encoding

### 7A: Facet Formulation
- **File**: [milp_relaxation.py](python/discopt/_jax/milp_relaxation.py)
- **Alpine reference**: `multilinear.jl:282-400`
- Facet-defining inequalities instead of SOS-2; strictly tighter for multilinear terms with 3+ variables

### 7B: Embedded Binary Encoding
- **New file**: `python/discopt/_jax/embedding.py`
- **Alpine reference**: `embedding.jl:1-237`
- Gray-code encoding: `O(log_2 K)` binary variables instead of `O(K)`
- Significant for large partition counts (K > 10)

**Tests**: Correctness + binary count is O(log K).
**Effort**: 2 sessions.

---

## Phase 8: Adaptive Variable Selection (disc_var_pick=3)

**Goal**: Re-select partition variables each iteration based on dual bound contribution.

- **File**: [partition_selection.py](python/discopt/_jax/partition_selection.py)
- **Alpine reference**: `heuristics.jl:1-60`
- After each MILP solve, compute `var_diffs[i] = |best_sol[i] - best_bound_sol[i]|`
- Use as weights in a weighted minimum vertex cover
- Re-compute `part_vars` at start of each iteration

**Tests**: Verify adaptive selection produces smaller MILPs on large problems.
**Effort**: 1 session.

---

## Phase 9: MINLPTests Integration

**Goal**: Validate AMP against 92 standardized test problems.

- **Infrastructure**: upstream MINLPTests files now present in this repository
  - [test_minlptests.py](python/tests/test_minlptests.py)
  - [known_failures.toml](python/tests/data/known_failures.toml)
  - [minlptests_problems.py](discopt_benchmarks/benchmarks/problems/minlptests_problems.py) (11 benchmark instances)

### Steps:
1. Benchmark AMP against the upstream MINLPTests suite on the non-convex instances
2. Create `TestMINLPTestsAMP` coverage where AMP-specific assertions belong in-tree
3. Register or refine AMP hooks in the MINLPTests infrastructure as needed
4. Track failures in `known_failures.toml`
5. Cross-validate: run Alpine.jl on the same problems, compare iteration counts and bounds

### Phased rollout:
- **9A**: Run AMP on 18 nonconvex NLP instances (`nlp-*`)
- **9B**: Run AMP on 16 MINLP instances (`nlp-mi-*`)  
- **9C**: Cross-validate against Alpine.jl on same 34 problems

**Effort**: 2 sessions.

---

## Phase 10: Performance and Polish

1. **MILP warm-starting**: Pass previous alpha solution as warm-start (Alpine `amp_warmstart_alpha`, `multilinear.jl:316-360`)
2. **Multilinear linking constraints**: For shared sub-expressions (Alpine `_add_multilinear_linking_constraints`, `multilinear.jl:617-717`)
3. **Sparse constraint matrix**: `scipy.sparse.csr_matrix` for large MILPs
4. **Structured logging**: Match Alpine's tabular iteration output
5. **Documentation**: `docs/notebooks/amp_global_optimization.ipynb` with citations

**Effort**: 2 sessions.

---

## Dependency Graph

```
Phase 1 (Bugs) ─┬─> Phase 2 (OBBT)  ──────────────────────┬─> Phase 9 (MINLPTests)
                 ├─> Phase 3 (Trilinear) ──────────────────┤
                 ├─> Phase 4 (Lambda) ─┬─> Phase 6 (Options)
                 ├─> Phase 5 (Mon n>2) ├─> Phase 7 (Facet/EBD)
                 │                     ├─> Phase 8 (Adaptive)
                 │                     └─> Phase 10 (Polish)
```

Phase 4 (lambda) does NOT depend on Phase 2 (OBBT) -- they are independent formulation vs presolve.
Phases 2, 3, 4, 5 are all independent of each other. Phase 4 is highest impact -- prioritize after Phase 1.

## Key Files

| File | Phases | Changes |
|------|--------|---------|
| `python/discopt/_jax/milp_relaxation.py` | 1A, 1B, 3, 4, 5, 7 | Lambda formulation, trilinear decomp, Big-M fix, monomial fix |
| `python/discopt/solvers/amp.py` | 1C, 1D, 2, 6, 8 | OBBT, OA guard, rounding, options, adaptive vars |
| `python/discopt/_jax/partition_selection.py` | 8 | Adaptive weighted MVC |
| `python/discopt/solver.py` | 6 | Route new AMP options (lines 1099-1126) |
| `python/tests/test_amp.py` | ALL | TDD tests for every phase |
| `python/discopt/_jax/embedding.py` | 7B | NEW: Gray-code binary encoding |

## Existing Code to Reuse

| Existing | Used In |
|----------|---------|
| `_jax/mccormick.py:relax_bilinear()` | Lambda formulation coefficient computation |
| `_jax/obbt.py:run_obbt()` | Phase 2 presolve |
| `_jax/convexity.py` | Phase 1C OA cut guard |
| `_jax/cutting_planes.py:generate_oa_cuts_from_evaluator()` | OA cut generation |
| `solver.py:_tighten_node_bounds()` | FBBT presolve |
| `solvers/milp_highs.py` | MILP model building |

## Verification per Phase

1. Write failing tests first (`test_amp.py`)
2. Implement feature
3. `JAX_ENABLE_X64=1 pytest python/tests/test_amp.py -m "not slow" --timeout=120`
4. Run slow nlp3 test to check convergence improvement
5. No regressions: `pytest python/tests/ -v --timeout=300`

## Alpine.jl Cross-Validation

For each phase, verify against Alpine on same problem:
```bash
cd /home/bernalde/repos/Alpine.jl && julia --project=. -e '
using JuMP, Alpine, Ipopt, HiGHS
# ... formulate same problem ...
optimize!(m)
# Compare: iterations, LB sequence, final gap
'
```
Target: discopt AMP iterations <= 2x Alpine iterations for same tolerance.

---

## Known Risks and Open Questions

### R1: Maximize objectives not handled in amp.py
`solve_amp()` has no check for `ObjectiveSense.MAXIMIZE`. The MILP relaxation builder negates the objective for maximization (line 698-700 of milp_relaxation.py), but the AMP loop's LB/UB tracking, gap computation, and incumbent comparison all assume minimization. **Must add maximize handling in Phase 1 or Phase 6.**

### R2: Trilinear auxiliary variables vs NLP subproblem
Phase 3 adds auxiliary variables `w = x*y` to the MILP. But the NLP subproblem solves the ORIGINAL model (no auxiliaries). The MILP-to-NLP mapping (`_extract_orig_solution`) only reads `x[:n_orig]`. This is correct as-is -- the auxiliaries are only in the MILP. However, partition refinement for auxiliary variables needs care: the `disc_state` must store partitions for ORIGINAL variables only, and trilinear aux bounds must be recomputed from updated original variable bounds.

### R3: Lambda formulation -- partition on one variable only
Alpine partitions only ONE variable per bilinear term (the one in `disc_vars`). The lambda grid is `{x_0,...,x_K} x {y_lb, y_ub}` (K+1 x 2 extreme points). The current piecewise McCormick also partitions only one variable. This is consistent, but should be documented clearly in the implementation to avoid confusion.

### R4: OA cut validity is more nuanced than Phase 1C suggests
OA cuts at feasible points are valid cutting planes even for non-convex constraints -- they cut off the linearization point from the feasible side. The issue is that for non-convex constraints, the OA cut may also cut off the global optimum. However, since AMP adds OA cuts only at NLP-feasible points (which are in the original feasible region), the cuts are valid as supporting hyperplanes. The real risk is MILP infeasibility from accumulated cuts. **Consider: keep all OA cuts but add a feasibility recovery mechanism (drop oldest cuts if MILP becomes infeasible) rather than skipping non-convex OA entirely.**

### R5: MINLPTests benchmark drift
The upstream MINLPTests files can change independently of the AMP stack. Phase 9 should benchmark against the upstream suite snapshot we intend to merge against and keep `known_failures.toml` aligned with that exact revision.

### R6: `_check_constraints` returns True on evaluation failure
[amp.py:146](python/discopt/solvers/amp.py#L146): `except Exception: return True` -- if constraint evaluation fails (e.g., NaN), the point is accepted as feasible. This can produce incorrect upper bounds. **Should return False on evaluation failure.**
