# AMP MINLPTests Remediation Plan

This note turns the current MINLPTests failures into implementation tracks for
discopt AMP. It is based on the current Phase 6 benchmark slice, targeted
single-instance diagnostics in discopt, and the corresponding Julia-side model
definitions in `MINLPTests.jl` plus Alpine's expression and presolve handling.

## What the Julia side already does

- `MINLPTests.jl` preserves per-instance start values for several hard cases,
  especially `nlp_001_010` and `nlp_002_010`.
- Alpine rejects unsupported nonlinear operators early in `src/nlexpr.jl`
  instead of trying to build an invalid relaxation.
- Alpine regulates solve time with remaining wall-clock budget through
  `set_mip_time_limit`.
- Alpine requires finite integer domains for some MOI bridges, which prevents it
  from starting on cases like `nlp_mi_007_010` without bound information.

discopt currently differs on all four points: translated MINLPTests models do
not carry starts, AMP can build numerically invalid relaxations from near-infinite
bounds, the NLP subproblem path does not use the same robust start logic as the
general NLP solve path, and unbounded integer domains are handed directly to AMP.

## Update after the current AMP patch

The translated feasible `nlp_mi` slice is no longer the main blocker, but the
latest full-suite benchmark does not yet count it as completely clean.

- In the refreshed full-suite benchmark, AMP passes `11/13` translated
  `nlp_mi` instances and the only remaining misses are `nlp_mi_003_014` and
  `nlp_mi_003_015`.
- Those two cases recover the expected incumbent objective `11.0`, but exit as
  `time_limit`, so they still fail the current benchmark pass criteria.
- The fixes were:
  - exhaustive enumeration of small finite integer domains for the incumbent
    NLP fallback
  - finite bound inference from simple quadratic norm constraints such as
    `x^2 + y^2 + z^2 <= c`
  - direct small-domain fallback when the first AMP relaxation errors out
  - returning `feasible` instead of `time_limit` when AMP already has a valid
    incumbent but runs out of proof budget
- Alpine still solves `0/13` on the same translated `nlp_mi` slice, so the
  Julia implementation is not the reference path for these repaired cases.

## Failure Families

| Family | Instances | discopt finding | Julia-side reference | Priority |
| --- | --- | --- | --- | --- |
| Unbounded-variable scaling and bad starts | `nlp_001_010`, `nlp_002_010`, `nlp_004_010`, `nlp_008_010`, `nlp_008_011`, `nlp_009_010`, `nlp_mi_007_010` | AMP relaxations or starts use values around `9.999e19`. For `nlp_008_010`, HiGHS rejects the relaxation because the matrix contains coefficients around `1.9998e20`. For `nlp_002_010` and `nlp_009_010`, the NLP subproblem starts from `-9.999e19` on unbounded variables. | MINLPTests provides sensible starts for some cases. Alpine clamps `Inf` bounds to a configured large bound and warns before proceeding. | P0 |
| Continuous upper-bound recovery fails from the MILP point | `nlp_003_014`, `nlp_003_015` | The MILP relaxation is fine, but `_solve_nlp_subproblem` fails from the MILP point `[4, 4]`. The same NLP succeeds from a start near `[3.2, 2.0]`. | MINLPTests records the solution near `[3.2565, 2.0131]`. Alpine separates `bounding_solve` and `local_solve`, so it always attempts a local feasible solve after each bound update. | P1 |
| Integer incumbent recovery from narrow neighborhoods | closed in the current branch for `nlp_mi_001_010` to `nlp_mi_005_010` | AMP now enumerates small finite integer boxes and falls back to direct fixed-integer NLP solves when the first relaxation fails. | Alpine still does not solve these translated mixed-integer cases because of unsupported operators or finite-domain bridge limits. | done |
| Infeasibility proof is missing on unbounded integer equalities | `nlp_mi_007_010` | AMP iterates until the wall clock expires. The current trace reached iteration `17` with lower bound `0.0` and no incumbent. | Alpine does not accept the instance without finite integer domains because the MOI integer-to-binary bridge requires them. | P1 |
| Operator-aware fallback is missing | `nlp_004_010`, `nlp_009_010` and similar future cases | AMP attempts the full relaxation path even when the relaxation is numerically unstable or when the operator mix is outside what the current relaxation builder handles robustly. | Alpine rejects unsupported operators early in `src/nlexpr.jl` with explicit messages. | P2 |

## Concrete Diagnostics Behind the Grouping

- `nlp_003_014`: MILP relaxation returns `[4, 4]` at the root, but the NLP upper-bound solve returns no feasible candidate from that point. The same subproblem solves to the correct objective when started near `[3.2, 2.0]`.
- `nlp_mi_003_012`: previously, with integer bounds fixed to the rounded MILP point `[4, 0]`, AMP found no candidate. The current branch fixes this by enumerating the finite box and recovering the `[3, 2]` assignment family-wide.
- `nlp_008_010`: the built relaxation is finite but numerically invalid for HiGHS because its constraint matrix contains coefficients above `1e15`, so HiGHS returns `kNotset` before solving.
- `nlp_002_010`: the MILP relaxation returns a nominally optimal zero objective, but the implied start is `[1e-5, -9.999e19]`, which is much worse than the original MINLPTests start `[1, -1.12]`.

## Workplan

### Track 1: Bound sanitation and start preservation

Target files:
- `python/discopt/solvers/amp.py`
- `python/discopt/_jax/milp_relaxation.py`
- `python/discopt/_jax/model_utils.py`
- `python/tests/test_minlptests.py`

Tasks:
- Add a finite-bound preprocessing step for AMP that replaces pseudo-infinite
  bounds with a safer working box before building the relaxation.
- Reuse the existing safe-start logic from the general NLP path instead of
  seeding AMP subproblems directly from raw `flat_lb`/`flat_ub` midpoints or
  pathological MILP solutions.
- Preserve MINLPTests start values in the translated instances and thread them
  into AMP as an `initial_solution` seed.
- Guard the relaxation builder against emitting HiGHS coefficients above the
  solver's numeric limit.

Success criteria:
- `nlp_008_010` and `nlp_008_011` build a solvable relaxation.
- `nlp_002_010` and `nlp_009_010` no longer start from `±9.999e19`.

### Track 2: Continuous upper-bound robustness

Target files:
- `python/discopt/solvers/amp.py`
- `python/discopt/_jax/primal_heuristics.py`
- `python/discopt/_jax/ipm_callbacks.py`

Tasks:
- Make `_solve_nlp_subproblem` accept a small multi-start set instead of a
  single MILP-derived seed.
- Include starts from:
  - the MILP point
  - a safe midpoint
  - any model-provided start values
  - the previous incumbent, when one exists
- Accept near-feasible IPM endpoints when the callback solver exits with
  iteration limit but the final point satisfies constraints within tolerance.

Success criteria:
- `nlp_003_014` and `nlp_003_015` produce feasible incumbents from AMP.

### Track 3: Integer incumbent search beyond nearest rounding

Target files:
- `python/discopt/solvers/amp.py`
- `python/discopt/_jax/primal_heuristics.py`

Tasks:
- Extend `_integer_rounding_candidates` to search a bounded neighborhood even
  when the MILP point is already integral.
- Prioritize candidates using constraint residuals and incumbent distance rather
  than nearest rounding only.
- Reuse the feasibility-pump code path or a simplified variant for AMP upper
  bounds after integer fixing.

Success criteria:
- At least the `nlp_mi_003_*` family reaches the feasible assignment `[3, 2]`
  or an equivalent optimum.

Status:
- completed on the current branch for the translated feasible `nlp_mi` slice

### Track 4: Nonlinear bound propagation for infeasibility

Target files:
- `python/discopt/solvers/amp.py`
- `python/discopt/_jax/obbt.py`
- a new nonlinear presolve helper under `python/discopt/_jax/`

Tasks:
- Add cheap nonlinear domain propagation for monotone expressions such as
  `exp(x)` and `y^2`.
- Use that propagation to derive finite domains for unbounded integer variables
  before building AMP relaxations.
- Add contradiction checks for equality systems such as `y = exp(x)` and
  `x = y^2`.

Success criteria:
- `nlp_mi_007_010` is proven infeasible without running to the wall clock.

### Track 5: Operator-aware routing and diagnostics

Target files:
- `python/discopt/solvers/amp.py`
- `python/discopt/_jax/term_classifier.py`
- `python/discopt/solvers/milp_highs.py`

Tasks:
- Detect when AMP is about to use an operator mix or bound regime that the
  current relaxation builder cannot handle numerically.
- Fail fast with a specific reason or route to a fallback solver path.
- Keep raw HiGHS model errors observable instead of collapsing them into an
  opaque `error`.

Success criteria:
- Failing instances report a stable, actionable reason.
- The benchmark table can distinguish numeric model-build failures from genuine
  infeasibility.

## Recommended Order

1. Track 1 first, because it removes the worst numerical pathologies and
   improves every remaining family.
2. Track 2 and Track 3 next, because they are the main blockers on the
   feasible `003_*` families.
3. Track 4 after that, because infeasibility proof is hard to debug while the
   relaxation and incumbent paths are still unstable.
4. Track 5 in parallel with the others for better observability.
