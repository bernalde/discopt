"""Score the #860 panel (see panel860.py) against the two CLAUDE.md §5 bars."""

from __future__ import annotations

import argparse
import json
import math

TOL = 1e-4


def rel(a, b):
    return abs(a - b) / (1.0 + abs(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="scratchpad/panel860_results.json")
    a = ap.parse_args()
    rows = json.load(open(a.results))

    # Best independently-verified feasible point per instance, in the model's own
    # sense, gathered across every run. A dual bound may never cross it.
    best_min, best_max, sense = {}, {}, {}
    for name, r in rows.items():
        for key in ("engine", "off", "on"):
            run = r.get(key) or {}
            if run.get("x_feasible") and run.get("x_true_obj") is not None:
                v = float(run["x_true_obj"])
                best_min[name] = min(best_min.get(name, math.inf), v)
                best_max[name] = max(best_max.get(name, -math.inf), v)

    # Objective sense: the engine reports a bound on the same side as the sense, so
    # infer it from the run that has both. (The panel worker does not record it
    # directly; ``bound <= obj`` => minimize, ``bound >= obj`` => maximize.)
    for name, r in rows.items():
        e = r.get("engine") or {}
        if e.get("obj") is not None and e.get("bound") is not None:
            sense[name] = "max" if e["bound"] > e["obj"] + TOL else "min"

    unsound, false_opt, cert_reg, drift, ivf = [], [], [], [], []
    scope_new, engine_inc, engine_declined = [], [], []
    gains, losses, wall_off, wall_on = [], [], 0.0, 0.0

    for name, r in rows.items():
        e = r.get("engine") or {}
        if e.get("in_scope") and not e.get("in_scope_legacy"):
            scope_new.append(name)
        if e.get("in_scope"):
            (engine_declined if e.get("declined") else engine_inc).append(name)

        # --- Panel A: engine soundness --------------------------------------- #
        if e.get("obj") is not None:
            if not e.get("x_feasible"):
                unsound.append((name, "engine incumbent NOT independently feasible"))
            elif e.get("x_true_obj") is not None and rel(e["x_true_obj"], e["obj"]) > TOL:
                unsound.append(
                    (name, f"engine objective {e['obj']} != re-evaluated {e['x_true_obj']}")
                )
        if e.get("bound") is not None:
            s = sense.get(name, "min")
            lo, hi = best_min.get(name), best_max.get(name)
            if s == "min" and lo is not None and e["bound"] > lo + TOL * (1 + abs(lo)):
                unsound.append((name, f"engine bound {e['bound']} > verified point {lo}"))
            if s == "max" and hi is not None and e["bound"] < hi - TOL * (1 + abs(hi)):
                unsound.append((name, f"engine bound {e['bound']} < verified point {hi}"))
        if e.get("status") == "optimal" and e.get("obj") is not None:
            s = sense.get(name, "min")
            lo, hi = best_min.get(name), best_max.get(name)
            if s == "min" and lo is not None and e["obj"] > lo + TOL * (1 + abs(lo)):
                false_opt.append((name, f"claimed optimal {e['obj']}, better point {lo}"))
            if s == "max" and hi is not None and e["obj"] < hi - TOL * (1 + abs(hi)):
                false_opt.append((name, f"claimed optimal {e['obj']}, better point {hi}"))

        # --- Panel B: flag off vs on ----------------------------------------- #
        off, on = r.get("off") or {}, r.get("on") or {}
        if off.get("error") or on.get("error"):
            continue
        wall_off += float(off.get("wall") or 0.0)
        wall_on += float(on.get("wall") or 0.0)
        if on.get("ivf"):
            ivf.append(name)
        if off.get("gapc") and not on.get("gapc"):
            cert_reg.append(name)
        if off.get("obj") is None and on.get("obj") is not None:
            gains.append((name, on["obj"], bool(on.get("x_feasible"))))
        if off.get("obj") is not None and on.get("obj") is None:
            losses.append(name)
        if off.get("obj") is not None and on.get("obj") is not None:
            if rel(on["obj"], off["obj"]) > 1e-3:
                drift.append((name, off["obj"], on["obj"]))
        for key, run in (("off", off), ("on", on)):
            if run.get("obj") is not None and run.get("x_feasible") is False:
                unsound.append((name, f"{key}: reported incumbent NOT feasible"))

    print(f"instances                      : {len(rows)}")
    print(f"newly in scope (#860)          : {len(scope_new)}")
    print(f"  engine returned a result     : {len(engine_inc)}")
    print(f"  engine declined              : {len(engine_declined)}")
    print()
    print("── Panel A: engine soundness (bar 1) ──")
    print(f"  unsound bounds / incumbents  : {len(unsound)}")
    for n, why in unsound:
        print(f"      {n}: {why}")
    print(f"  false optimality certificates: {len(false_opt)}")
    for n, why in false_opt:
        print(f"      {n}: {why}")
    print()
    print("── Panel B: DISCOPT_LP_SPATIAL_MIXED off vs on ──")
    print(f"  cert regressions (True->False): {len(cert_reg)} {cert_reg}")
    print(f"  incumbent-verification failed : {len(ivf)} {ivf}")
    print(f"  objective drift > 1e-3        : {len(drift)}")
    for n, o, w in drift:
        print(f"      {n}: {o} -> {w}")
    print(f"  incumbents GAINED             : {len(gains)}")
    for n, o, ok in gains:
        print(f"      {n}: {o} (independently feasible: {ok})")
    print(f"  incumbents LOST               : {len(losses)} {losses}")
    ratio = wall_on / max(wall_off, 1e-9)
    print(f"  total wall  off={wall_off:.1f}s  on={wall_on:.1f}s  ratio={ratio:.3f}")


if __name__ == "__main__":
    main()
