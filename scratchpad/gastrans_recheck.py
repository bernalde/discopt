"""Was gastrans040's lp_spatial 'regression' systematic, or a marginal-timing
artifact? control hit status=optimal at wall 90.7s against a 90s limit -- i.e. it
barely made it, so ~3s of lp_spatial attempt overhead could explain the miss.
Re-run both at a generous 240s limit."""
import os, time
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("JAX_ENABLE_X64", "1")
from discopt.modeling.core import from_nl
BM = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")
for label, kw in (("control", {}), ("lp_spatial", {"lp_spatial": True})):
    m = from_nl(f"{BM}/gastrans040.nl"); t0 = time.perf_counter()
    r = m.solve(time_limit=240, **kw)
    print(f"  {label:12s} status={r.status:12s} obj={r.objective} bound={getattr(r,'bound',None)} "
          f"wall={time.perf_counter()-t0:6.1f}s", flush=True)
