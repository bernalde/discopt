# LP-node engine primal quality (#862) — the measurement, and two falsified levers

Issue #862 opened on the observation that the #844 no-incumbent fallback returns
incumbents *far* from optimal, and that nothing measured it:

| instance | incumbent | optimum | gap  |
|----------|-----------|---------|------|
| tln4     | 9.3       | 8.3     | +12% |
| tln5     | 32.2      | 10.3    | +213% |
| tln6     | 65.3      | 15.3    | +327% |

It asks for the work in a specific order — **(1) measure quality, then (2) fix it** —
because "without (1) there is no way to tell whether (2) helped". This document
records what (1) ships and what (2) has established so far, including the levers the
measurement killed.

---

## 1. What shipped: the quality axis

The #844 panel gated on `gains / lost_incumbents / cert_regressions / overshoots`
plus soundness — whether an incumbent **exists**, is sound, and stays in budget. It
never scored *how good* the incumbent was, so a change could have halved incumbent
quality with the panel still green.

- `discopt_benchmarks/utils/primal_quality.py` — the MIPLIB/Berthold **primal gap**
  (bounded in `[0,1]`, sign-safe, so it aggregates across a corpus whose objectives
  span orders of magnitude), the signed **relative excess** the issue's table quotes,
  a corpus **summary**, and `quality_regressions()` — the OFF-vs-ON comparison whose
  absence the issue is about.
- `discopt_benchmarks/utils/reference_optima.py` — one oracle accessor over
  `minlplib.solu` → `known_optima.toml` → `cert-optima.json`. The chain is the point:
  on a machine with the MINLPLib snapshot the panel scores tln4/5/6 automatically;
  in CI, where `.solu` does not exist, it still scores the vendored corpus instead of
  silently degrading to a no-op (which is what ~17 hand-rolled `.solu` parsers in
  `discopt_benchmarks/scripts/` do today).
- `discopt_benchmarks/scripts/issue844_primal_quality_panel.py` — the #844 gate,
  byte-for-byte, plus the quality axis.

Two design rules are load-bearing and are pinned by tests:

1. **Unscored is never reported as clean.** A missing incumbent or a missing oracle
   yields `None`, counted in `unscored`, never folded in as gap 0 (which would show a
   corpus as perfect exactly where nothing was measured) and never as gap 1.0 (which
   would let a change that trades incumbents for quality look neutral).
2. **Quality never absorbs soundness.** An incumbent *below* the reference optimum of
   a minimize is not a poor incumbent, it is a correctness failure (CLAUDE.md §1);
   `is_false_primal` scores it separately and the existing hard gate keeps it.

The panel's verdict is deliberately **unchanged** by default. #862 asks for the
measurement first, and tightening a graduation bar on the same commit that first
measures the thing would retroactively fail a flag graduated honestly under the bar
of its day. `--gate-quality` opts into failing on quality regressions.

---

## 2. Entry experiment: the class reproduces, on a different instance

> **Constraint on this work.** `tln4/5/6` are not vendored, and `minlplib.org` is
> policy-blocked from this environment, so the issue's own instances could not be
> re-run. Everything below is measured on the vendored pure-integer MINIMIZE corpus
> (24 in-scope instances), under the exact fallback configuration
> (`use_obbt=False`, `require_incremental=True`, bounded budget).

At a 10 s budget the engine splits cleanly into three groups:

| group | instances | outcome |
|---|---|---|
| certifies | nvs03, nvs10, nvs11, nvs12, st_miqp1/2/3, st_test1 | primal gap **0**, 4–98 nodes |
| declines | gear, nvs04, nvs06, nvs07, nvs09, nvs16, prob03 | no incremental structure → fallback correctly refuses |
| **fails the primal** | **nvs17, nvs19, nvs24** | **no incumbent at all** after 1100–2100 nodes |
| loose | st_testgr3 | −20.533 vs −20.59, primal gap 0.0028 |

So the vendored corpus reproduces the *class* — the LP-node engine's primal side is
weak — but in a sharper form than tln's: on `nvs17/19/24`, the family the engine was
**built for**, it returns nothing. (Those instances never reach the fallback in
practice, because the default path does find their incumbents; the failure is the
engine's, not a user-visible regression.)

Instrumented on nvs17 at 10 s:

```
dive_calls 38   dive_lp_infeas 38   (100% of dives die on an infeasible LP)
verify_calls 2562   verify_infeas 2562   (no rounded LP point is ever feasible)
pump_calls 5   pump_cycles 115   (the pump spends most iterations cycling)
fully_fixed 0   maxdepth 32   (best-first never reaches an exact leaf)
```

Three independent primal sources, all at zero.

---

## 3. Falsified lever A: rounding flip / backtracking

**Hypothesis.** `dive` fixes the chosen variable to `round(LP value)` and abandons
the descent the moment the LP turns infeasible. A one-step flip to the opposite
rounding — the standard diving backtrack — should rescue most dives.

**Kill criterion.** If the opposite rounding is also infeasible, the flip cannot help.

**Result: falsified.** Instrumenting the complement box at every dive failure, the
opposite rounding was infeasible **37/37** on nvs17 and **17/18** on nvs24. A rounding
flip has essentially nothing to rescue.

---

## 4. Falsified lever B: the narrowing dive

**Hypothesis (the strongest structural one).** `dive`'s "fix to the rounded LP value"
is a *binary*-domain step, where fixing and branching coincide. These are general
integers: on nvs17/19/24 **every domain is 200 wide**, so one step discards 200 of 201
values and snaps the McCormick rows of every product touching that variable onto an
exact line. Replacing it with the standard diving step (Berthold 2006; SCIP
`fracdiving`) — pick the *least*-fractional free integer rather than the coin-flip the
current `max` picks, tighten one bound to `floor`/`ceil`, and backtrack LIFO through
untried complements — should halve a domain instead of erasing it and keep the LP
feasible far longer.

**Kill criterion.** No new incumbents on the failing family, or a throughput loss not
paid for by primal gains.

**Result: falsified — not shipped.** Implemented and A/B'd at a 10 s budget:

| instance | dive (current) | narrowing dive |
|---|---|---|
| nvs17 | no incumbent, 2080 nodes | no incumbent, **1384 nodes** |
| nvs19 | no incumbent, 1766 nodes | no incumbent, **1280 nodes** |
| nvs24 | no incumbent, 1280 nodes | no incumbent, **896 nodes** |
| st_testgr3 | −20.533, 2961 nodes | −20.533, 2770 nodes |

Zero primal gain, **~30% throughput loss**. This is the `DISCOPT_CUT_INHERIT` pattern
(CLAUDE.md §5: sound ≠ helpful) — recorded and reverted rather than shipped as a
default-off flag nobody would ever turn on.

A solo narrowing descent from the root box with a 5000-LP cap confirms *why*: on
nvs17 it exhausts its entire backtrack trail in **11 LPs**. The LP-guided descent has
nowhere to go.

---

## 5. What the falsifications point at (the standing conclusion)

The two dive levers failed for the same underlying reason, and it is not a heuristic
detail. **The McCormick relaxation on this family is too weak to guide a primal
heuristic at all.** Fixing nvs17's variables one at a time to a *known optimal*
point walks the LP bound down through

```
root −278209  →  −175200  →  −78920  →  −29166  →  −12087  →  −3967  →  −2163  →  −1100.4
```

against a true optimum of **−1100.4**. The root relaxation is off by a factor of ~250.
An LP solution that far from the feasible set carries no usable signal about where
feasible points are, so *every* rounding-based heuristic reading it — one-shot
`verify`, `dive`, the pump — is reading noise. That is exactly what the instrumentation
shows: 2562/2562 infeasible roundings.

Corollaries, both checked:

- **The LP is not unsound.** Every box containing a true feasible point is reported
  feasible, and the fully-fixed box returns exactly −1100.4. The dives' Farkas-proven
  infeasibilities are genuine; the greedy choices really do exclude the feasible set.
- **The tree cannot substitute.** Best-first reaches depth 32 but `fully_fixed = 0`:
  no exact leaf in 2000+ nodes, so no incumbent arrives from the node loop either.

The productive direction is therefore **tightening the relaxation** (the cut/RLT work
in `certification-gap-plan.md`) or **plunging** so the node loop reaches exact leaves,
not a better rounding rule on top of a relaxation that is 250x loose. Both are
bound-changing changes needing the §5 differential panel — which, with §1 shipped,
can now score whether they moved incumbent quality at all.

---

## 6. Status

- (1) **done** — quality is measured, tested, and wired into the #844 panel.
- (2) **open** — investigated, two levers falsified and recorded, root cause
  identified as relaxation tightness rather than heuristic tuning. No fix ships:
  neither candidate met the net-positive bar, and shipping a measured-negative
  change would violate CLAUDE.md §5. Re-running the panel against `tln4/5/6` on a
  machine with the MINLPLib snapshot is the next step, and now costs one command.
