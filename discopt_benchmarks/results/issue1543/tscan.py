"""Translated real instances: which reach _solve_miqp_bb with a NONZERO origin shift?"""

import os

OUT = os.path.dirname(os.path.abspath(__file__))
import sys, glob, os, json, time, faulthandler

sys.path.insert(0, OUT)
import numpy as np
import discopt.solver as S
from discopt.modeling.core import from_nl
from discopt.interfaces.qplib import from_qplib
from _invariance import translate

assert hasattr(S, "_miqp_origin_shift"), "marker absent"
shard, nshard = int(sys.argv[1]), int(sys.argv[2])
OFFS = tuple(float(v) for v in sys.argv[3].split(","))
TAG = sys.argv[4]
print("loaded", S.__file__, "shard", shard, flush=True)
rec = {}
real_bb, real_sh = S._solve_miqp_bb, S._miqp_origin_shift


def bb(*a, **k):
    rec["bb"] += 1
    return real_bb(*a, **k)


def sh(*a, **k):
    o = real_sh(*a, **k)
    if o is not None:
        rec["shift_cols"] = max(rec["shift_cols"], int(np.count_nonzero(o[0])))
    return o


S._solve_miqp_bb, S._miqp_origin_shift = bb, sh
files = sorted(
    {os.path.basename(f): f for f in glob.glob("python/tests/data/minlplib*/*.nl")}.values()
) + sorted(glob.glob("python/tests/data/qplib/qplib/*.qplib"))
files = files[shard::nshard]
out = {}
for f in files:
    for off in OFFS:
        key = f"{os.path.basename(f)}@{off:g}"
        rec.update(bb=0, shift_cols=0)
        t = time.time()
        faulthandler.dump_traceback_later(300, exit=False)
        try:
            base = from_qplib(f) if f.endswith(".qplib") else from_nl(f)
            m = translate(base, off, seed=0)
        except Exception as e:
            out[key] = dict(skip=f"{type(e).__name__}: {str(e)[:80]}")
            print(key, out[key], flush=True)
            faulthandler.cancel_dump_traceback_later()
            continue
        try:
            r = m.solve(time_limit=20.0)
            st = r.status
        except Exception as e:
            st = f"RAISE {type(e).__name__}: {str(e)[:80]}"
        faulthandler.cancel_dump_traceback_later()
        out[key] = dict(rec, status=st, wall=round(time.time() - t, 1), path=f, off=off)
        print(key, out[key], flush=True)
print("done", len(out), flush=True)
assert out
json.dump(out, open(f"{OUT}/tscan_{TAG}_{shard}.json", "w"))
