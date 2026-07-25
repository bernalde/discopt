"""Establish the wall-overrun facts precisely.

Question: is the overrun (a) reproducible, (b) attributable to the lp_spatial
engine specifically, (c) present on SMALL models -- which would place it outside
#845's accepted "very large single NLP factorization" residual (closed NOT_PLANNED).
"""
import os, sys, time
os.environ.setdefault("JAX_PLATFORMS","cpu"); os.environ.setdefault("JAX_ENABLE_X64","1")
import warnings; warnings.filterwarnings("ignore")
from discopt.modeling.core import from_nl
BM = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")

def probe(name, tl, kw, label):
    t0=time.perf_counter(); m=from_nl(f"{BM}/{name}.nl"); load=time.perf_counter()-t0
    t1=time.perf_counter()
    try:
        r=m.solve(time_limit=tl, **kw); w=time.perf_counter()-t1
        print(f"  {name:22s} {label:12s} limit={tl:5.0f}s load={load:5.1f}s wall={w:7.1f}s "
              f"OVERRUN={w/tl:5.2f}x status={r.status} obj={r.objective}", flush=True)
    except Exception as e:
        print(f"  {name:22s} {label:12s} EXC {type(e).__name__}: {str(e)[:60]}", flush=True)

if __name__=="__main__":
    which=sys.argv[1]
    if which=="ball":
        for rep in (1,2):
            probe("ball_mk2_30", 30, {}, f"control#{rep}")
            probe("ball_mk2_30", 30, {"lp_spatial":True}, f"lp_spatial#{rep}")
    elif which=="water":
        probe("watercontamination0202", 30, {}, "control")
