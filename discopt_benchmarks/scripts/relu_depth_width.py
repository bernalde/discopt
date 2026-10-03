"""Depth vs width at a FIXED hidden-neuron budget, ReLU big-M (discopt default route).

For each budget N and depth L (width N // L), over several seeds, records:

* structure (no solve): per-layer interval width of the pre-activations (from the
  same ``discopt.ml.bounds.propagate_bounds`` the big-M formulation uses), the
  number of UNSTABLE neurons (bounds straddle 0 -> one binary each; a stable
  neuron gets none), and the largest big-M;
* solve: status, certified, objective, bound, root bound, nodes, wall.

Each solve runs in a fresh process after an untimed warm-up (one-off import and
first-call setup excluded). Prints per-row progress and an executed count; exits
non-zero when nothing ran (CLAUDE.md §6, §10).

    python -u relu_depth_width.py --tl 60 --out depth_width.tsv
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

BUDGETS = (60, 120)
DEPTHS = (1, 2, 3, 4, 5, 6)
SEEDS = (0, 1, 2)
# input half-widths: the default [-1, 1] box everywhere, plus a narrower box on a
# subset to isolate the effect of input bounds
BOXES = (1.0,)
NARROW = ((60, 3), (60, 4), (120, 3))
NARROW_BOX = 0.25


def structure(layers, box: float) -> dict:
    from discopt.ml.bounds import propagate_bounds
    from discopt.ml.network import DenseLayer, NetworkDefinition

    net = NetworkDefinition(
        [DenseLayer(w, b, a) for w, b, a in layers],
        input_bounds=(np.full(2, -box), np.full(2, box)),
    )
    lb_list = propagate_bounds(net)
    widths, unstable, big_m = [], [], 0.0
    for lb in lb_list[:-1]:  # hidden layers only (the last is the linear output)
        lo, hi = np.asarray(lb.pre_lb), np.asarray(lb.pre_ub)
        widths.append(float(np.median(hi - lo)))
        unstable.append(int(np.sum((lo < 0) & (hi > 0))))
        big_m = max(big_m, float(np.max(np.maximum(np.abs(lo), np.abs(hi)))))
    return {
        "layer_width_median": widths,
        "unstable_per_layer": unstable,
        "unstable": int(sum(unstable)),
        "big_m": big_m,
    }


def one(n: int, depth: int, seed: int, box: float, tl: float) -> dict:
    sys.path.insert(0, HERE)
    import discopt.modeling as dm
    from discopt.ml import add_predictor
    from discopt.ml.network import DenseLayer, NetworkDefinition
    from omlt_scaling_panel import make_net

    def build(layers):
        m = dm.Model("nn")
        x = m.continuous("x", shape=(2,), lb=-box, ub=box)
        net = NetworkDefinition(
            [DenseLayer(w, b, a) for w, b, a in layers],
            input_bounds=(np.full(2, -box), np.full(2, box)),
        )
        y, _ = add_predictor(m, x, net, method="relu_bigm")
        m.minimize(y[0])
        return m

    build(make_net(2, 4, 1, "relu", 99)).solve(time_limit=10)  # untimed warm-up
    width = n // depth
    layers = make_net(2, width, depth, "relu", seed)
    s = structure(layers, box)
    m = build(layers)
    t0 = time.perf_counter()
    r = m.solve(time_limit=tl)
    wall = time.perf_counter() - t0
    return {
        "budget": n,
        "depth": depth,
        "width": width,
        "seed": seed,
        "box": box,
        "binaries": int(m.num_integer),
        "status": r.status,
        "cert": bool(r.gap_certified),
        "obj": r.objective,
        "bound": r.bound,
        "root_bound": r.root_bound,
        "nodes": r.node_count,
        "wall": wall,
        **s,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl", type=float, default=60.0)
    ap.add_argument("--out", default="depth_width.tsv")
    ap.add_argument("--one")
    a = ap.parse_args()
    if a.one:
        n, d, s, box = a.one.split(":")
        print(json.dumps(one(int(n), int(d), int(s), float(box), a.tl)))
        return 0

    jobs = [(n, d, s, b) for n in BUDGETS for d in DEPTHS for s in SEEDS for b in BOXES]
    jobs += [(n, d, s, NARROW_BOX) for (n, d) in NARROW for s in SEEDS]
    cols = [
        "budget",
        "depth",
        "width",
        "seed",
        "box",
        "binaries",
        "unstable",
        "big_m",
        "unstable_per_layer",
        "layer_width_median",
        "status",
        "cert",
        "obj",
        "bound",
        "root_bound",
        "nodes",
        "wall",
        "load",
    ]
    print(f"load {os.getloadavg()}; {len(jobs)} jobs", flush=True)
    done = 0
    with open(a.out, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for n, d, s, b in jobs:
            p = subprocess.run(
                [sys.executable, "-u", __file__, "--one", f"{n}:{d}:{s}:{b}", "--tl", str(a.tl)],
                capture_output=True,
                text=True,
                timeout=a.tl * 4 + 180,
            )
            if p.returncode != 0:
                raise RuntimeError(f"{n}:{d}:{s}:{b} failed\n{p.stderr[-2000:]}")
            r = json.loads(p.stdout.strip().splitlines()[-1])
            r["load"] = os.getloadavg()[0]
            fh.write(
                "\t".join(json.dumps(r[c]) if isinstance(r[c], list) else str(r[c]) for c in cols)
                + "\n"
            )
            fh.flush()
            done += 1
            print(
                f"N={n:3d} L={d} w={r['width']:3d} s={s} box={b:g} bins={r['binaries']:3d} "
                f"unstable={r['unstable']:3d} M={r['big_m']:.3g} {r['status']:9s} "
                f"cert={r['cert']!s:5s} root={r['root_bound']} bound={r['bound']} "
                f"wall={r['wall']:.1f} load={r['load']:.2f}",
                flush=True,
            )
    print(f"executed solves {done}")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
