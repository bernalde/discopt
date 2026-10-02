"""#1547 differential panel: PR vs main on real instances that reach _solve_miqp_bb with s != 0."""

import os

OUT = os.path.dirname(os.path.abspath(__file__))
import sys, json, time, tomllib, faulthandler

sys.path.insert(0, OUT)
import numpy as np
import discopt.solver as S
from discopt.modeling.core import from_nl
from discopt.interfaces.qplib import from_qplib
from discopt.validation.feasibility import verify_point
from _invariance import translate

arm = sys.argv[1]  # "pr" or "main"
has = hasattr(S, "_miqp_origin_shift")
assert has == (arm == "pr"), (arm, has)
print("loaded", S.__file__, "arm", arm, flush=True)
hits = json.load(open(f"{OUT}/hits_all.json"))
rec = {}
real_bb = S._solve_miqp_bb


def bb(*a, **k):
    rec["bb"] += 1
    return real_bb(*a, **k)


S._solve_miqp_bb = bb
if has:
    real_sh = S._miqp_origin_shift

    def sh(*a, **k):
        o = real_sh(*a, **k)
        if o is not None:
            rec["shift_cols"] = max(rec["shift_cols"], int(np.count_nonzero(o[0])))
        return o

    S._miqp_origin_shift = sh
out = []
for path, off in hits:
    base = from_qplib(path) if path.endswith(".qplib") else from_nl(path)
    m = translate(base, off, seed=0)
    shift = m._invariance_shift
    rec.update(bb=0, shift_cols=0)
    faulthandler.dump_traceback_later(900, exit=False)
    t = time.perf_counter()
    try:
        r = m.solve(time_limit=60.0)
        row = dict(
            status=r.status,
            obj=r.objective,
            bound=r.bound,
            nodes=r.node_count,
            cert=bool(r.gap_certified),
        )
        if r.x is not None and r.objective is not None:
            y = np.concatenate(
                [np.ravel(np.asarray(r.x[v.name], dtype=np.float64)) for v in m._variables]
            )
            vr = verify_point(base, y - shift, with_objective=True)
            row["verify_ok"] = bool(vr.ok)
            row["verify_obj"] = vr.objective
            row["verify_why"] = getattr(vr, "reason", None)
    except Exception as e:
        row = dict(
            status=f"RAISE {type(e).__name__}: {str(e)[:100]}",
            obj=None,
            bound=None,
            nodes=None,
            cert=False,
        )
    faulthandler.cancel_dump_traceback_later()
    row.update(
        path=path,
        offset=off,
        wall=round(time.perf_counter() - t, 2),
        bb=rec["bb"],
        shift_cols=rec["shift_cols"],
    )
    out.append(row)
    print(
        path.split("/")[-1],
        off,
        {
            k: row[k]
            for k in ("status", "obj", "bound", "nodes", "cert", "bb", "shift_cols", "wall")
        },
        row.get("verify_ok"),
        flush=True,
    )
print("solves", len(out), flush=True)
assert out
json.dump(out, open(f"{OUT}/panel2_{arm}.json", "w"))
