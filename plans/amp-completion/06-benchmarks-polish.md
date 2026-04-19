# AMP Phase 6: MINLPTests and Alpine Status

The Phase 5 AMP base was first synced with `upstream/main` at `7c27be0`, and
the Phase 6 branch was then rebased on top of that merge so the numbers below
reflect current HEAD rather than the older stacked branch.

## Scope

This phase now has concrete outputs instead of a placeholder checklist.

- `python/tests/test_minlptests.py` provides the translated MINLPTests corpus.
- `python/tests/data/known_failures.toml` remains the tracker for default-solver
  gaps, not AMP-specific benchmark results.
- `discopt_benchmarks/benchmarks/problems/minlptests_problems.py` now exposes
  AMP in the nonconvex MINLPTests benchmark registry.
- `discopt_benchmarks/category_runner.py` now knows how to invoke `solver="amp"`
  rather than treating `amp` like an NLP local solver.
- `scripts/collect_minlptests_status.py` and
  `scripts/alpine_minlptests_status.jl` provide the reproducible comparison run.

The current benchmark slice is the full translated MINLPTests suite already
present in the repo:

- 91 convex `nlp_cvx`
- 15 feasible nonconvex `nlp`
- 13 feasible nonconvex `nlp_mi`
- 3 infeasible cases

## Reproduction

Full discopt + Alpine comparison:

```bash
source .venv/bin/activate
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="/tmp/minlptests-full-benchmark-${STAMP}"
UV_CACHE_DIR=/tmp/discopt-uv-cache maturin develop
FORCE_RERUN=1 OUTPUT_DIR="$OUT" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  bash ./scripts/run_full_minlptests_benchmark.sh
```

Alpine-only refresh against an existing discopt artifact:

```bash
source .venv/bin/activate
unset CONDA_DEFAULT_ENV CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_SHLVL
DISCOPT_DIR="/tmp/minlptests-discopt-run"
OUT="/tmp/minlptests-alpine-only-$(date +%Y%m%d-%H%M%S)"
FORCE_RERUN=1 OUTPUT_DIR="$OUT" RUN_DISCOPT=0 RUN_ALPINE=1 RUN_COMPARISON=1 \
  DISCOPT_INPUT_JSON="${DISCOPT_DIR}/minlptests-full-discopt.json" \
  JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  bash ./scripts/run_full_minlptests_benchmark.sh
```

## Benchmark Summary

| Solver | NLP-CVX pass | NLP-CVX fail | NLP pass | NLP fail | NLP-MI pass | NLP-MI fail | Infeasible pass | Infeasible fail | Total pass | Total fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| discopt AMP | 62 | 29 | 9 | 6 | 11 | 2 | 3 | 0 | 85 | 37 |
| Alpine.jl | 2 | 89 | 0 | 15 | 0 | 13 | 0 | 3 | 2 | 120 |

## Head-to-Head Outcome Split

| Outcome split | Count | Interpretation |
| --- | ---: | --- |
| `discopt_only_pass` | 83 | AMP solved the case to the MINLPTests expectation and Alpine did not. |
| `both_pass` | 2 | The shared solved subset is `nlp_cvx_001_010` and `nlp_cvx_002_010`. |
| `alpine_only_pass` | 0 | The refreshed full-suite run leaves no Alpine-only translated wins. |
| `both_fail` | 37 | AMP still misses the case and Alpine rejects it earlier. |

## Current AMP Failure Families

The remaining 37 AMP misses cluster into a few families:

- Convex false infeasibility: `nlp_cvx_108_010` to `nlp_cvx_108_013`,
  `nlp_cvx_203_010` to `nlp_cvx_206_010`, and the whole
  `nlp_cvx_501_011_{1d..20d}` family still return `infeasible`.
- Convex wrong objective: `nlp_cvx_106_010` returns a feasible point with
  objective `-1.149074984` instead of `-1.857215513`.
- Nonconvex NLP false infeasibility: `nlp_001_010`, `nlp_002_010`,
  `nlp_008_010`, `nlp_008_011`, and `nlp_009_010` still return `infeasible`.
- Nonconvex NLP wrong objective: `nlp_004_010` returns `optimal`, but at
  `-4.911509745` instead of `-4.872159041`.
- Mixed-integer proof-budget/status gap: `nlp_mi_003_014` and
  `nlp_mi_003_015` recover the expected objective `11.0`, but now report
  `time_limit` rather than `optimal` or `feasible`, so they remain benchmark
  failures under the current pass criteria.

Relative to the earlier Phase 6 checkpoints, the main change is that there are
no longer any Alpine-only wins to chase. The remaining work is now purely on
the discopt side.

## Alpine Failure Modes

| Failure mode | Count |
| --- | ---: |
| quadratic `>=` constraints unsupported | 37 |
| unsupported `sqrt` on variable | 23 |
| unsupported `exp` on variable | 22 |
| quadratic `<=` constraints unsupported | 16 |
| `Symbol.head` `FieldError` | 7 |
| integer finite-domain bridge required | 7 |
| unsupported `sin` on variable | 2 |
| unsupported `log` on variable | 2 |
| unsupported reciprocal denominator | 2 |
| unsupported non-integer exponent pattern | 1 |
| unsupported `tan` on variable | 1 |

The refreshed comparison still matters, but it is no longer an Alpine-gap
tracker. It now shows that AMP dominates the translated overlap numerically,
while Alpine still rejects most of the suite before a global-optimality
comparison is meaningful.

## What Is Missing for AMP

| Gap | Evidence | Affected problems | What remains to be done |
| --- | --- | --- | --- |
| Convex false infeasibility | AMP still returns `infeasible` on 28 translated convex cases, dominated by the `501_011` family and the `108_*` / `203_*` families. | `nlp_cvx_108_010` to `nlp_cvx_108_013`, `nlp_cvx_203_010` to `nlp_cvx_206_010`, `nlp_cvx_501_011_{1d..20d}` | Improve convex nonlinear feasibility recovery and avoid pruning valid roots when the relaxation or local solve is numerically weak. |
| Convex objective miss | AMP finds a feasible point but not the right one on one translated convex case. | `nlp_cvx_106_010` | Tighten the upper-bound/local-solve policy for convex nonlinear cases that do not take the direct fast path. |
| Nonconvex NLP robustness | Five nonconvex NLP cases still end in false infeasibility, and one still converges to the wrong objective. | `nlp_001_010`, `nlp_002_010`, `nlp_004_010`, `nlp_008_010`, `nlp_008_011`, `nlp_009_010` | Continue the start-quality and multi-start local solve work in the remediation plan. |
| Mixed-integer proof-budget/status gap | Two translated MINLP cases recover the right incumbent objective but still exit as `time_limit`, which the benchmark counts as a miss. | `nlp_mi_003_014`, `nlp_mi_003_015` | Decide whether AMP should convert incumbent-holding timeouts into `feasible` for benchmark purposes, or increase proof strength so the two cases certify before the wall clock expires. |
| Alpine parity tracker | The refreshed full run leaves no Alpine-only wins. | none | Close issue `#19`; remaining follow-up work belongs under issues `#15` to `#18` and the discopt-side remediation plan. |

## What Is Left To Be Done

| Item | Current state | Remaining work |
| --- | --- | --- |
| MINLPTests integration | Present in `python/tests/test_minlptests.py`; benchmark slice can be run reproducibly. | Keep the AMP report current as solver fixes land. |
| Benchmark runner support | `amp` is now wired into the nonconvex and global benchmark categories. | Add a dedicated CLI target or report artifact once the pass rate is high enough to compare runs over time. |
| Alpine comparison | Reproducible on local `../Alpine.jl` and `../MINLPTests.jl` using Julia `+release`; refreshed result is `alpine_only_pass = 0`. | Keep the full-suite comparison as a regression artifact, but move the remaining work to discopt-side issues rather than Alpine-gap tracking. |
| Notebook update | Not started in this phase. | Defer the notebook result table until the 37 remaining AMP failures are reduced enough to make the comparison more than a failure ledger. |

The instance-led remediation plan derived from these failures is tracked in
`plans/amp-completion/07-minlptests-remediation.md`.
