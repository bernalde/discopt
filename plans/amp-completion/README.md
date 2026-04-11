# AMP Completion PR Stack

This directory turns the AMP completion plan into a manageable draft-PR queue.

Base branch for this stack: `upstream-main-sync`

Draft PR order:
1. `feature/amp-completion/roadmap`
2. `feature/amp-completion/phase-1-robustness`
3. `feature/amp-completion/phase-2-lambda-options`
4. `feature/amp-completion/phase-3-obbt`
5. `feature/amp-completion/phase-4-relaxation-coverage`
6. `feature/amp-completion/phase-5-advanced-formulations`
7. `feature/amp-completion/phase-6-benchmarks-polish`

Grouping rationale:
- Phase 1 stands alone because it fixes correctness bugs that can poison later work.
- Phase 4 and Phase 6 are grouped because the lambda formulation needs exposed solver options.
- Phase 2 is independent presolve work and can be advanced separately.
- Phase 3 and Phase 5 are both relaxation-coverage fixes in `milp_relaxation.py`.
- Phases 7 and 8 both depend on the tighter formulation work and both change partitioning behavior.
- Phases 9 and 10 are validation, performance, and documentation work after the solver is feature-complete enough to benchmark.

Each draft PR should keep its checklist current and note any blockers discovered during implementation.
