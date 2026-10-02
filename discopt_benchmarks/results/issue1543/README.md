# #1543 differential panel: PR #1547 vs main on real instances that reach MIQP-BB with s ≠ 0

Measured 2026-10-02. PR head is the merge of #1547 with `origin/main` `41f8eaf`; the main arm is `41f8eaf`.

## Selection, by instrumentation

`tscan.py` solves every in-repo real instance with spies on `_solve_miqp_bb` and
`_miqp_origin_shift`. That covers 131 distinct MINLPLib `.nl` instances from
`python/tests/data/minlplib{,_nl}/` and 9 QPLIB instances from
`python/tests/data/qplib/`. Each is solved under the #1537 translation
`x = y − c` (`_invariance.translate`, copied from #1546, `seed=0`) at offsets
10, 100, 1e3, 1e5 and 1e6 (`selection_scan.json`: 655 runs, 0 refused by the
transform).

- **Unshifted**, only `alan` and `QPLIB_3871` reach `_solve_miqp_bb`, both with
  a zero shift. The PR is a no-op on the corpus as shipped.
- **Translated**, 10 instances reach it with `s ≠ 0`, at every offset:
  `alan, meanvarx, st_miqp1..5, st_test1, st_testgr3, QPLIB_3871`. That gives
  50 cases (`hits_all.json`). No other instance reaches the route at any offset.

## Panel

`panel2.py pr|main` runs from the repo root with `time_limit=60`, one arm after
the other, on an otherwise idle 4-core box. `analyze2.py` produces
`analyze2.out`.

**Oracle.**
- `known_optima.toml`: st_miqp1–3, st_test1, st_testgr3.
- `qplib.solu`: QPLIB_3871. Its reference solution vector was verified feasible
  by `verify_point`, objective 197.33388.
- MINLPLib published values, not vendored: alan, meanvarx, st_miqp4, st_miqp5.

**Checks per row.** A row is incorrect if any of these holds:
- a certified objective lies outside `max(1e-6, 1e-4·|opt|)` of the oracle;
- `infeasible` is certified;
- a reported bound lies above the oracle;
- the incumbent, mapped back by `x = y − shift`, fails `verify_point` on the
  original model, or its verified objective differs from the reported one.

| | PR | main |
|---|---|---|
| rows | 50 | 50 |
| `incorrect_count` | **0** | 12 |
| raised (`RuntimeError`, #952 refusal) | 0 | 5 |
| certified | 45 | 32 |
| certificates lost vs main | 0 | n/a |
| certificates gained vs main | 13 | n/a |

Executed comparisons: **439**. Every PR row asserts `bb ≥ 1` and `shift_cols > 0`.

Rows where both arms certify the same optimum: 27. Node counts are equal on 22,
and the PR uses fewer on 3. It uses more on 2: st_miqp1@1e6 (9 vs 7) and
st_test1@1e6 (15 vs 3). Totals are 1575 (PR) vs 1585 (main). At the small
offsets (10, 100), which exercise "a box that merely excludes 0", PR and main
agree on status, objective and certification on every row. Node counts are
identical there except alan@10 (9 vs 13) and the time-limited QPLIB_3871.

The 5 PR rows that are not certified are all QPLIB_3871. No arm certifies that
instance within 60 s, shifted or unshifted. The PR's incumbent there (1139.53)
is feasible and its bound is below the oracle (197.33).

## Caveat on the extension

Both arms import the same compiled `discopt._rust` (the main worktree has no
built `.so`, so `discopt._rust` resolves to the PR tree's). This is valid only
because `crates/` is identical on both sides: #1547 touches no Rust, and
`crates/` last changed on 2026-09-28. Re-check this before reusing these
scripts on a Rust-touching change.

No wall times are compared. The arms ran sequentially, not interleaved (§9).
