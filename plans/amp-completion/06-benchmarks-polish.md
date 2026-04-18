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

The current benchmark slice is the 31 translated nonconvex and infeasible
MINLPTests cases already present in the repo:

- 15 feasible `nlp`
- 13 feasible `nlp_mi`
- 3 infeasible cases

## Reproduction

discopt AMP only:

```bash
UV_CACHE_DIR=/tmp/uv-cache-pr14 PYTHONPATH=python \
~/.local/bin/uv run --extra dev python scripts/collect_minlptests_status.py \
  --skip-alpine \
  --output-json /tmp/discopt-amp-minlptests-phase6.json \
  --output-markdown /tmp/discopt-amp-minlptests-phase6.md
```

Alpine.jl only:

```bash
julia +release --project=. /tmp/discopt-pr14/scripts/alpine_minlptests_status.jl \
  /tmp/alpine-minlptests-request.tsv \
  /tmp/alpine-minlptests-results.jsonl \
  /home/bernalde/repos/MINLPTests.jl
```

Combined comparison:

```bash
UV_CACHE_DIR=/tmp/uv-cache-pr14 PYTHONPATH=python \
~/.local/bin/uv run --extra dev python scripts/collect_minlptests_status.py \
  --output-json /tmp/minlptests-phase6-comparison.json \
  --output-markdown /tmp/minlptests-phase6-comparison.md
```

## Benchmark Summary

| Solver | NLP pass | NLP fail | NLP-MI pass | NLP-MI fail | Infeasible pass | Infeasible fail | Total pass | Total fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| discopt AMP | 7 | 8 | 0 | 13 | 2 | 1 | 9 | 22 |
| Alpine.jl | 0 | 15 | 0 | 13 | 0 | 3 | 0 | 31 |

## Head-to-Head Outcome Split

| Outcome split | Count | Interpretation |
| --- | ---: | --- |
| `discopt_only_pass` | 9 | AMP solved the case to the MINLPTests expectation and Alpine did not. |
| `both_pass` | 0 | There is no shared solved subset yet. |
| `alpine_only_pass` | 0 | Alpine does not currently solve any case in this translated slice. |
| `both_fail` | 22 | AMP still misses the case and Alpine rejects it earlier. |

## Alpine Failure Modes

| Failure mode | Count | Affected problems |
| --- | ---: | --- |
| unsupported `exp` on variable | 16 | `nlp_001_010`, `nlp_003_010`, `nlp_003_012`, `nlp_003_013`, `nlp_003_014`, `nlp_003_015`, `nlp_003_016`, `nlp_008_010`, `nlp_008_011`, `nlp_mi_003_010`, `nlp_mi_003_012`, `nlp_mi_003_013`, `nlp_mi_003_014`, `nlp_mi_003_015`, `nlp_mi_003_016`, `nlp_007_010` |
| integer finite-domain bridge required | 7 | `nlp_mi_001_010`, `nlp_mi_002_010`, `nlp_mi_004_010`, `nlp_mi_004_011`, `nlp_mi_004_012`, `nlp_mi_005_010`, `nlp_mi_007_010` |
| `Symbol.head` `FieldError` | 3 | `nlp_009_010`, `nlp_009_011`, `nlp_mi_007_020` |
| unsupported `sqrt` on variable | 2 | `nlp_003_011`, `nlp_mi_003_011` |
| unsupported `log` on variable | 1 | `nlp_002_010` |
| unsupported `tan` on variable | 1 | `nlp_004_010` |
| unsupported reciprocal denominator | 1 | `nlp_005_010` |

These numbers mean the Alpine comparison is still useful, but not yet as a
head-to-head objective comparison. At the moment it is mostly a coverage
boundary: Alpine rejects the entire current translated slice before global
optimization quality can be compared.

## What Is Missing for AMP

| Gap | Evidence | Affected problems | What remains to be done |
| --- | --- | --- | --- |
| False infeasible on feasible NLP | AMP returns `infeasible` on more than half of the feasible nonconvex NLP slice. | `nlp_001_010`, `nlp_002_010`, `nlp_003_014`, `nlp_003_015`, `nlp_004_010`, `nlp_008_010`, `nlp_008_011`, `nlp_009_010` | Tighten the feasibility recovery path for transcendental continuous relaxations and stop pruning feasible incumbents as infeasible. |
| False infeasible on feasible MINLP | AMP returns `infeasible` on every feasible translated `nlp_mi` case in the current slice. | `nlp_mi_001_010`, `nlp_mi_002_010`, `nlp_mi_003_010`, `nlp_mi_003_011`, `nlp_mi_003_012`, `nlp_mi_003_013`, `nlp_mi_003_014`, `nlp_mi_003_015`, `nlp_mi_003_016`, `nlp_mi_004_010`, `nlp_mi_004_011`, `nlp_mi_004_012`, `nlp_mi_005_010` | Debug the mixed-integer OA and incumbent-validation path on the translated MINLPTests models before expanding the benchmark table further. |
| Incomplete infeasibility proof | AMP times out instead of proving one infeasible mixed-integer case. | `nlp_mi_007_010` | Strengthen infeasibility detection for the mixed-integer branch-and-bound path. |
| No shared solved subset with Alpine | Alpine is still at `0/31` on this translated scope. | whole comparison slice | Either narrow the published comparison to Alpine-supported operators or keep the full table and present Alpine as a coverage baseline, not as a quality baseline. |

## What Is Left To Be Done

| Item | Current state | Remaining work |
| --- | --- | --- |
| MINLPTests integration | Present in `python/tests/test_minlptests.py`; benchmark slice can be run reproducibly. | Keep the AMP report current as solver fixes land. |
| Benchmark runner support | `amp` is now wired into the nonconvex and global benchmark categories. | Add a dedicated CLI target or report artifact once the pass rate is high enough to compare runs over time. |
| Alpine comparison | Reproducible on local `../Alpine.jl` and `../MINLPTests.jl` using Julia `+release`. | Decide whether unsupported operators stay in-scope or whether the comparison should publish a smaller overlapping subset. |
| Notebook update | Not started in this phase. | Defer the notebook result table until there is a meaningful shared solved subset and the false-infeasible AMP gaps are reduced. |
