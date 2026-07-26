#!/bin/bash
# #844 graduation panel: DISCOPT_LP_SPATIAL_FALLBACK off vs on, 60 s budget.
#
# Load-gated (two earlier measurement rounds were invalidated by concurrent CPU
# load) and interleaved A/B per instance. Set PANEL_REPS to repeat for spread.
cd /Users/jkitchin/projects/discopt || exit 1
for i in $(seq 1 120); do          # up to ~60 min
  L=$(uptime | sed 's/.*averages*: *//' | awk '{print $1}' | tr -d ',')
  BUSY=$(ps aux | awk '$3>50 {c++} END{print c+0}')
  if awk "BEGIN{exit !($L < 4.0)}" && [ "$BUSY" -eq 0 ]; then
    echo "QUIET at attempt $i (load=$L, busy procs=$BUSY) -- measuring"
    break
  fi
  [ $((i % 10)) -eq 1 ] && echo "  waiting: load=$L busy=$BUSY"
  sleep 30
done
L=$(uptime | sed 's/.*averages*: *//' | awk '{print $1}' | tr -d ',')
echo "=== starting measurement at load=$L ==="
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python - <<'PY'
import os, time
os.environ["JAX_PLATFORMS"]="cpu"; os.environ["JAX_ENABLE_X64"]="1"
import warnings; warnings.filterwarnings("ignore")
from discopt.modeling.core import from_nl
BM=os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")
TL=float(os.environ.get("PANEL_TL","60"))
REPS=int(os.environ.get("PANEL_REPS","1"))
CASES=[("tln4",8.3),("tln5",10.3),("tln6",15.3),("ball_mk2_30",0.0),
       ("nvs04",0.72),("nvs06",1.7703125),("nvs09",-43.134),("nvs15",1.0)]
res={}
for rep in range(REPS):
    print(f"--- rep {rep+1}/{REPS} ---", flush=True)
    for nm,opt in CASES:
        for fb in ("0","1"):
            os.environ["DISCOPT_LP_SPATIAL_FALLBACK"]=fb
            t=time.perf_counter(); r=from_nl(f"{BM}/{nm}.nl").solve(time_limit=TL)
            w=time.perf_counter()-t
            res.setdefault((nm,fb),[]).append(
                (r.objective, r.bound, bool(r.gap_certified), w))
            print(f"  {nm:12s} FB={fb} obj={r.objective} bound={r.bound} "
                  f"gapc={r.gap_certified} wall={w:6.1f}s ratio={w/TL:4.2f}x opt={opt}",
                  flush=True)
print()
print("=== VERDICT ===")
gains=regr=cert=over=unsound=false_primal=0
def last(k):  # worst-case wall across reps, last-run objective/bound
    v=res[k]; return v[-1][0], v[-1][1], v[-1][2], max(x[3] for x in v)
for nm,opt in CASES:
    o0,b0,c0,w0=last((nm,"0")); o1,b1,c1,w1=last((nm,"1"))
    if o0 is None and o1 is not None: gains+=1
    if o0 is not None and o1 is None: regr+=1
    if c0 and not c1: cert+=1
    if w1 > TL*1.25:
        over+=1; print(f"  OVERSHOOT {nm}: {w1:.1f}s vs {TL}s ({w1/TL:.2f}x)")
    # soundness: dual bound must never cross the incumbent, and (minimize) an
    # incumbent must never sit BELOW the reference optimum.
    if o1 is not None and b1 is not None and b1 > o1 + 1e-6*(1+abs(o1)):
        unsound+=1; print(f"  UNSOUND {nm}: bound {b1} > incumbent {o1}")
    if o1 is not None and o1 < opt - 1e-4*(1+abs(opt)):
        false_primal+=1; print(f"  FALSE PRIMAL {nm}: {o1} < reference optimum {opt}")
    if b1 is not None and b1 > opt + 1e-4*(1+abs(opt)):
        unsound+=1; print(f"  BOUND ABOVE OPT {nm}: {b1} > {opt}")
print(f"  gains={gains} lost_incumbents={regr} cert_regressions={cert} "
      f"overshoots={over} unsound={unsound} false_primals={false_primal}")
_ok = gains>0 and regr==0 and cert==0 and over==0 and unsound==0 and false_primal==0
print(f"  DEFAULT-ON OK: {_ok}")
print()
print("=== per-instance wall spread (FB=1) ===")
for nm,_ in CASES:
    ws=[f"{x[3]:.1f}" for x in res[(nm,"1")]]
    objs=[str(x[0]) for x in res[(nm,"1")]]
    print(f"  {nm:12s} walls={ws} objs={objs}")
PY
