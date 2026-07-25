"""#844 §5 differential panel: DISCOPT_LP_SPATIAL_AUTO off vs on.

Graduation requires BOTH:
  (1) cert-clean -- no incumbent above/below its reference optimum beyond tolerance,
      no bound crossing the reference optimum, no gap_certified=True -> False
      regression, and every reported incumbent feasibility-verified by discopt's own
      final guard (incumbent_verification_failed must stay False);
  (2) net-positive -- measurably helpful broadly (incumbents gained, bound improved),
      not merely sound.

Out-of-scope models must be BIT-IDENTICAL (the router skips them before dispatch).
"""
import json, os, subprocess, sys, time
BM = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib")
CORPUS = "python/tests/data/minlplib_nl"

WORKER = r'''
import os, sys, json, time
os.environ["JAX_PLATFORMS"]="cpu"; os.environ["JAX_ENABLE_X64"]="1"
os.environ["DISCOPT_LP_SPATIAL_AUTO"]=sys.argv[2]
import warnings; warnings.filterwarnings("ignore")
from discopt.modeling.core import from_nl
p=sys.argv[1]
try:
    m=from_nl(p); t0=time.perf_counter()
    r=m.solve(time_limit=float(sys.argv[3]))
    print("RESULT"+json.dumps({"obj":r.objective,"bound":r.bound,"status":r.status,
        "gapc":bool(getattr(r,"gap_certified",False)),"nodes":getattr(r,"node_count",None),
        "wall":time.perf_counter()-t0,
        "ivf":bool(getattr(r,"incumbent_verification_failed",False))}))
except Exception as e:
    print("RESULT"+json.dumps({"error":f"{type(e).__name__}: {str(e)[:80]}"}))
'''

def opts():
    d={}
    for line in open(f"{BM}.solu"):
        parts=line.split()
        if len(parts)>=3 and parts[0] in ("=opt=","=best="):
            d.setdefault(parts[1], float(parts[2]))
    return d

def run(path, auto, tl):
    try:
        out=subprocess.run([sys.executable,"-c",WORKER,path,auto,str(tl)],
                           capture_output=True,text=True,timeout=tl+120).stdout
        for ln in out.splitlines():
            if ln.startswith("RESULT"): return json.loads(ln[6:])
    except subprocess.TimeoutExpired:
        return {"error":"harness_timeout"}
    return {"error":"no_result"}

if __name__=="__main__":
    tl=float(os.environ.get("PANEL_TL","40"))
    O=opts()
    files=sorted(__import__("glob").glob(f"{CORPUS}/*.nl"))
    extra=[f"{BM}/nl/{n}.nl" for n in ("tln4","tln5","tln6","ball_mk2_30","gastrans040","portfol_robust050_34")]
    files=files+[e for e in extra if os.path.exists(e)]
    rows=[]
    for i,p in enumerate(files):
        name=os.path.basename(p)[:-3]
        a=run(p,"0",tl); b=run(p,"1",tl)
        rows.append({"name":name,"off":a,"on":b,"opt":O.get(name)})
        print(f"[{i+1}/{len(files)}] {name:28s} off={a.get('obj')} on={b.get('obj')} "
              f"| gapc {a.get('gapc')}->{b.get('gapc')} | opt={O.get(name)}", flush=True)
    json.dump(rows,open("scratchpad/panel844_results.json","w"),indent=1)
    print("WROTE scratchpad/panel844_results.json")
