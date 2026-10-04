"""Panel: the convex-MINLP route's OA master on HiGHS vs the in-house simplex.

Arms are ``DISCOPT_CONVEX_ROUTE_OA_MASTER=auto`` (in-house) and ``=highs`` (the
default since performance-plan §25.13), run interleaved per instance through a
plain ``Model.solve(time_limit=T)`` so the #1066 guard, the budget reserve and
the spatial fallback all participate. Instances are every model in the chosen
corpus that the router diverts, optionally with each row multiplied by
``10**U(-s, s)`` (seeded by file name) to probe row-scale sensitivity (#1537).

Soundness: every arm's incumbent is checked with ``verify_point``; every
arm's bound is checked against the ``minlplib.solu`` value and against every
arm's *verified* incumbent. ``--kernel-off`` sets ``DISCOPT_CONVEX_KERNEL=0`` in
both arms -- the native convex kernel runs before the route and, when it cannot
certify, leaves ``solve_model`` no time, so with the kernel on the route never
fires on most of ``syn``/``rsyn`` and the arms measure nothing.

Exits non-zero when no arm's route fired, when any bound check fails, or when
any solve raised.

    python -u discopt_benchmarks/scripts/convex_route_oa_master_panel.py \\
        --corpus repo --spans 0,3,6 --time-limit 30 --out panel_repo.json
    python -u discopt_benchmarks/scripts/convex_route_oa_master_panel.py \\
        --corpus minlplib --pattern '^r?syn' --kernel-off --time-limit 30 --out panel_syn.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import warnings
import zlib

import numpy as np

warnings.simplefilter("ignore")
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python" / "tests"))

import discopt  # noqa: E402
import discopt.solver as _solver  # noqa: E402
from _invariance import _rebuild  # noqa: E402
from discopt.modeling.core import Constraint, ObjectiveSense, from_nl  # noqa: E402
from discopt.validation.feasibility import verify_point  # noqa: E402

BENCH = pathlib.Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark"))
CORPORA = {
    "repo": [REPO / "python/tests/data/minlplib_nl", REPO / "python/tests/data/minlplib"],
    "minlplib": [BENCH / "minlplib/nl"],
}
ARMS = {"auto": "in-house", "highs": "highs"}


def _solu() -> dict[str, float]:
    out: dict[str, float] = {}
    with open(BENCH / "minlplib.solu") as fh:
        for line in fh:
            p = line.split()
            if p and (p[0] == "=opt=" or (p[0] == "=best=" and p[1] not in out)):
                out[p[1]] = float(p[2])
    return out


def _build(path: pathlib.Path, span: float):
    base = from_nl(str(path))
    if span == 0:
        return base
    new = _rebuild(base, lambda v: np.zeros(v.lb.shape), 1.0, f"{path.stem}_pr{span:g}")
    rng = np.random.default_rng(zlib.crc32(path.name.encode()))
    new._constraints = [
        Constraint(
            body=float(10.0 ** rng.uniform(-span, span)) * c.body,
            sense=c.sense,
            rhs=0.0,
            name=c.name,
        )
        for c in new._constraints
    ]
    return new


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=sorted(CORPORA), default="repo")
    ap.add_argument("--pattern", default=".*")
    ap.add_argument("--spans", default="0")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--kernel-off", action="store_true")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()

    print("discopt from", discopt.__file__, flush=True)
    assert hasattr(_solver, "_convex_route_oa_master"), "loaded a discopt without the selector"
    if a.kernel_off:
        os.environ["DISCOPT_CONVEX_KERNEL"] = "0"
    spans = [float(s) for s in a.spans.split(",")]
    solu = _solu()

    paths: dict[str, pathlib.Path] = {}
    for d in CORPORA[a.corpus]:
        for p in sorted(d.glob("*.nl")):
            if re.match(a.pattern, p.stem):
                paths.setdefault(p.stem, p)
    os.environ["DISCOPT_CONVEX_ROUTE_OA_MASTER"] = "auto"
    routed = [n for n, p in paths.items() if _solver._convex_minlp_auto_route(from_nl(str(p)))[0]]
    print(f"routed {len(routed)}/{len(paths)}: {routed}", flush=True)
    if not routed:
        print("PANEL MEASURED NOTHING")
        return 1
    print("load before", os.getloadavg(), flush=True)

    rows: list[dict] = []
    checks, viol, errors = 0, [], []
    fired = dict.fromkeys(ARMS, 0)
    for name in routed:
        for span in spans:
            rec: dict = {"instance": name, "span": span}
            for arm, tag in ARMS.items():
                os.environ["DISCOPT_CONVEX_ROUTE_OA_MASTER"] = arm
                try:
                    m = _build(paths[name], span)
                    rec["sense"] = 1.0 if m._objective.sense == ObjectiveSense.MINIMIZE else -1.0
                    t = time.perf_counter()
                    r = m.solve(time_limit=a.time_limit)
                    w = time.perf_counter() - t
                except Exception as e:  # recorded, counted, and fails the run
                    rec[arm] = {"error": f"{type(e).__name__}: {e}"[:300]}
                    errors.append(f"{name} s={span} [{arm}] {rec[arm]['error']}")
                    print(f"{name:22s} s={span:g} {arm:6s} ERROR {rec[arm]['error']}", flush=True)
                    continue
                ok = None
                if r.x is not None:
                    flat = np.concatenate(
                        [np.ravel(np.asarray(r.x[v.name], float)) for v in m._variables]
                    )
                    ok = bool(verify_point(m, flat).ok)
                did = f"master={tag}" in (r.algorithm_route or "")
                fired[arm] += did
                rec[arm] = {
                    "status": str(r.status),
                    "cert": bool(r.gap_certified),
                    "obj": r.objective,
                    "bound": r.bound,
                    "wall": w,
                    "feas_ok": ok,
                    "fired": did,
                    "route": r.algorithm_route,
                }
                print(
                    f"{name:22s} s={span:g} {arm:6s} {rec[arm]['status']:11s} "
                    f"cert={rec[arm]['cert']!s:5s} obj={r.objective} bound={r.bound} "
                    f"wall={w:.2f} feas={ok} fired={did}",
                    flush=True,
                )
            s = rec.get("sense", 1.0)
            # Row scaling leaves the feasible set and the objective unchanged, so
            # the oracle applies at every span.
            refs = [("solu", solu[name])] if name in solu else []
            for arm in ARMS:
                d = rec.get(arm, {})
                if d.get("obj") is not None and d.get("feas_ok"):
                    refs.append((f"inc[{arm}]", float(d["obj"])))
            for arm in ARMS:
                d = rec.get(arm, {})
                b = d.get("bound")
                if b is not None:
                    for lab, ref in refs:
                        checks += 1
                        if s * b > s * ref + 1e-6 * max(1.0, abs(ref)):
                            viol.append(f"{name} s={span} [{arm}] bound {b} past {lab} {ref}")
                if d.get("cert") and d.get("feas_ok") is False:
                    viol.append(f"{name} s={span} [{arm}] certified an unverified point")
            rows.append(rec)
            a.out.write_text(json.dumps(rows, indent=1, default=str))

    print("load after", os.getloadavg())
    for span in spans:
        rr = [r for r in rows if r["span"] == span]
        for arm in ARMS:
            cert = sum(1 for r in rr if r.get(arm, {}).get("cert"))
            tw = sum(r.get(arm, {}).get("wall", 0.0) for r in rr)
            print(f"span={span:g} {arm:6s} certified {cert:3d}/{len(rr)} wall {tw:8.1f}s")
        gain = [r["instance"] for r in rr if r["highs"].get("cert") and not r["auto"].get("cert")]
        lost = [r["instance"] for r in rr if r["auto"].get("cert") and not r["highs"].get("cert")]
        print(f"span={span:g} highs vs auto: +{len(gain)} {gain} / -{len(lost)} {lost}")
    print("route fired per arm:", fired)
    print(f"bound checks executed: {checks}; violations: {len(viol)}; errors: {len(errors)}")
    for v in viol + errors:
        print("FLAG", v)
    return 1 if (checks == 0 or not all(fired.values()) or viol or errors) else 0


if __name__ == "__main__":
    sys.exit(main())
