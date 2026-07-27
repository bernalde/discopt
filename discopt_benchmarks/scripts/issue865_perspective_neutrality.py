#!/usr/bin/env python
"""Issue #865 bound-neutrality probe: every instance routed BEFORE the change must
marshal and solve identically AFTER it (node_count + certified objective exact)."""

import json
import os
import sys

os.environ.setdefault("DISCOPT_CONVEX_KERNEL", "1")

import discopt.modeling as dm  # noqa: E402
from discopt.solvers._convex_kernel import build_convex_spec, solve_convex_tree  # noqa: E402

PANEL = [
    ("syn05m", "python/tests/data/minlplib/syn05m.nl"),
    ("cvxnonsep_psig40r", "python/tests/data/minlplib_nl/cvxnonsep_psig40r.nl"),
    ("syn05hfsg", "python/tests/data/minlplib_nl/syn05hfsg.nl"),
]

out = {}
for name, path in PANEL:
    if not os.path.exists(path):
        out[name] = "MISSING"
        continue
    spec = build_convex_spec(dm.from_nl(path))
    if spec is None:
        out[name] = "DECLINED"
        continue
    r = solve_convex_tree(spec, initial_incumbent=None, time_limit_s=120.0)
    out[name] = {
        "status": r["status"],
        "node_count": int(r["node_count"]),
        "incumbent": r["incumbent"],
        "bound": r["bound"],
        "n_nl_rows": len(spec["nl_rhs"]),
        "n_terms": len(spec["term_coeff"]),
        "n_le": len(spec["le_rhs"]),
        "n_eq": len(spec["eq_rhs"]),
    }

with open(sys.argv[1], "w") as fh:
    json.dump(out, fh, indent=2, sort_keys=True)
print(json.dumps(out, indent=2, sort_keys=True))
