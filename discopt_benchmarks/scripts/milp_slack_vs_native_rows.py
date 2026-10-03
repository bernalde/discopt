"""Entry experiment: does HiGHS solve discopt's MILPs faster WITHOUT slack columns?

discopt's LP/MILP route hands HiGHS its standard form ``A x = b, l <= x <= u``,
where every inequality row carries a logical (slack) column. For each model this
script builds that exact ``StdForm`` (``solver._highs_std_form``) and an
algebraically identical model with the slacks folded into ranged row bounds:

    row  A_i x + a s = b_i,  s in [l_s, u_s]   ->   b_i - a*u_s <= A_i x <= b_i - a*l_s   (a > 0)

Both are passed to HiGHS with the same options, timed inside ``Highs.run`` only,
interleaved and repeated. Same objective is asserted when both finish.

Hypothesis: native rows are faster. Kill criterion: median (native / std) wall
ratio over finished instances >= 0.95, or no consistent sign.

Prints an executed-run count; exits non-zero when nothing ran (CLAUDE.md §6).
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics as st
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def std_and_native(model):
    import scipy.sparse as sp
    from discopt.solver import _highs_std_form

    _, n_orig, sf = _highs_std_form(model)
    a = sf.A.tocsc()
    n_log = sf.n - n_orig
    rlo, rhi = sf.b.copy(), sf.b.copy()
    keep = np.ones(sf.m, dtype=bool)
    for j in range(n_orig, sf.n):
        rows = a.indices[a.indptr[j] : a.indptr[j + 1]]
        vals = a.data[a.indptr[j] : a.indptr[j + 1]]
        if len(rows) != 1 or sf.c[j] != 0.0:
            raise ValueError(f"column {j} is not a pure logical (rows={len(rows)}, c={sf.c[j]})")
        i, coef = int(rows[0]), float(vals[0])
        lo, hi = sf.xl[j], sf.xu[j]
        if coef > 0:
            rlo[i], rhi[i] = sf.b[i] - coef * hi, sf.b[i] - coef * lo
        else:
            rlo[i], rhi[i] = sf.b[i] - coef * lo, sf.b[i] - coef * hi
    native = {
        "c": sf.c[:n_orig],
        "A": sp.csc_matrix(a[:, :n_orig]),
        "rlo": np.where(rlo <= -1e20, -np.inf, rlo),
        "rhi": np.where(rhi >= 1e20, np.inf, rhi),
        "xl": np.where(sf.xl[:n_orig] <= -1e20, -np.inf, sf.xl[:n_orig]),
        "xu": np.where(sf.xu[:n_orig] >= 1e20, np.inf, sf.xu[:n_orig]),
        "int": [j for j in sf.int_idx if j < n_orig],
        "const": sf.obj_const,
    }
    std = {
        "c": sf.c,
        "A": a,
        "rlo": sf.b,
        "rhi": sf.b,
        "xl": np.where(sf.xl <= -1e20, -np.inf, sf.xl),
        "xu": np.where(sf.xu >= 1e20, np.inf, sf.xu),
        "int": list(sf.int_idx),
        "const": sf.obj_const,
    }
    assert keep.all()
    return std, native, n_log


def solve(form, tl):
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(tl))
    lp = highspy.HighsLp()
    n = len(form["c"])
    lp.num_col_, lp.num_row_ = n, form["A"].shape[0]
    lp.col_cost_ = np.asarray(form["c"], float)
    lp.offset_ = float(form["const"])
    lp.col_lower_, lp.col_upper_ = np.asarray(form["xl"], float), np.asarray(form["xu"], float)
    lp.row_lower_, lp.row_upper_ = np.asarray(form["rlo"], float), np.asarray(form["rhi"], float)
    mat = form["A"].tocsc()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = mat.indptr.astype(np.int32)
    lp.a_matrix_.index_ = mat.indices.astype(np.int32)
    lp.a_matrix_.value_ = mat.data.astype(float)
    integ = np.zeros(n, dtype=np.uint8)
    integ[form["int"]] = 1
    lp.integrality_ = [highspy.HighsVarType(int(v)) for v in integ]
    h.passModel(lp)
    t0 = time.perf_counter()
    h.run()
    dt = time.perf_counter() - t0
    info = h.getInfo()
    status = h.modelStatusToString(h.getModelStatus())
    return dt, status, info.objective_function_value, info.mip_dual_bound, info.mip_node_count


def _knapsack(n, k, seed):
    import discopt.modeling as dm

    rng = np.random.default_rng(seed)
    m = dm.Model(f"mknap_{n}x{k}_s{seed}")
    x = [m.binary(f"x{i}") for i in range(n)]
    w = rng.integers(5, 60, size=(k, n))
    v = rng.integers(10, 100, size=n)
    for r in range(k):
        m.subject_to(dm.sum([float(w[r, i]) * x[i] for i in range(n)]) <= float(w[r].sum()) / 3)
    m.maximize(dm.sum([float(v[i]) * x[i] for i in range(n)]))
    return m


def _facility(nf, nc, seed):
    import discopt.modeling as dm

    rng = np.random.default_rng(seed)
    m = dm.Model(f"cfl_{nf}x{nc}_s{seed}")
    y = [m.binary(f"y{i}") for i in range(nf)]
    x = [[m.continuous(f"x{i}_{j}", lb=0, ub=1) for j in range(nc)] for i in range(nf)]
    d = rng.uniform(5, 35, nc)
    cap = rng.uniform(1.5, 4.0, nf) * d.sum() / nf
    f = rng.uniform(100, 300, nf)
    cost = rng.uniform(1, 40, (nf, nc))
    for j in range(nc):
        m.subject_to(dm.sum([x[i][j] for i in range(nf)]) == 1)
    for i in range(nf):
        m.subject_to(dm.sum([float(d[j]) * x[i][j] for j in range(nc)]) <= float(cap[i]) * y[i])
        for j in range(nc):
            m.subject_to(x[i][j] <= y[i])
    m.minimize(
        dm.sum([float(f[i]) * y[i] for i in range(nf)])
        + dm.sum([float(cost[i, j] * d[j]) * x[i][j] for i in range(nf) for j in range(nc)])
    )
    return m


def _setcover(ne, ns, seed):
    import discopt.modeling as dm

    rng = np.random.default_rng(seed)
    m = dm.Model(f"setcover_{ne}x{ns}_s{seed}")
    x = [m.binary(f"x{s}") for s in range(ns)]
    member = rng.random((ne, ns)) < 0.05
    for e in range(ne):
        if not member[e].any():
            member[e, rng.integers(ns)] = True
        m.subject_to(dm.sum([x[s] for s in range(ns) if member[e, s]]) >= 1)
    c = rng.integers(1, 20, ns)
    m.minimize(dm.sum([float(c[s]) * x[s] for s in range(ns)]))
    return m


def _lotsizing(t_len, seed):
    import discopt.modeling as dm

    rng = np.random.default_rng(seed)
    m = dm.Model(f"lotsize_{t_len}_s{seed}")
    dem = rng.uniform(10, 60, t_len)
    big = float(dem.sum())
    p = [m.continuous(f"p{t}", lb=0, ub=big) for t in range(t_len)]
    s = [m.continuous(f"s{t}", lb=0, ub=big) for t in range(t_len)]
    y = [m.binary(f"y{t}") for t in range(t_len)]
    for t in range(t_len):
        prev = s[t - 1] if t else 0.0
        m.subject_to(prev + p[t] - s[t] == float(dem[t]))
        m.subject_to(p[t] <= float(min(big, 3 * dem.mean())) * y[t])
    setup = rng.uniform(80, 200, t_len)
    m.minimize(dm.sum([float(setup[t]) * y[t] + 1.0 * s[t] + 0.5 * p[t] for t in range(t_len)]))
    return m


def instances(corpus_dir):
    """ReLU big-M nets plus four classic MILP families, all built as discopt models
    so they go through discopt's own standard form. The in-repo ``.nl`` corpus is
    scanned too, but holds no pure MILP (measured: 0 of 66), and MIPLIB is not
    reachable from the benchmark host -- hence the generated families."""
    sys.path.insert(0, HERE)
    import discopt.modeling as dm
    from discopt.solver import _milp_is_exactly_linear
    from omlt_scaling_panel import build_discopt, make_net

    # sizes that finish within the limit (the timing comparison needs both arms done)
    for w, d in ((10, 1), (25, 1), (50, 1), (100, 1), (10, 2), (25, 2), (10, 3)):
        for s in range(5):
            yield (
                f"relu_{w}x{d}_s{s}",
                (lambda w=w, d=d, s=s: build_discopt(make_net(2, w, d, "relu", s), 2, "bigm")[0]),
            )
    for s in range(3):
        for n, k in ((60, 5), (100, 8)):
            yield f"mknap_{n}x{k}_s{s}", (lambda n=n, k=k, s=s: _knapsack(n, k, s))
        for nf, nc in ((15, 40), (25, 60)):
            yield f"cfl_{nf}x{nc}_s{s}", (lambda nf=nf, nc=nc, s=s: _facility(nf, nc, s))
        for ne, ns in ((300, 150), (500, 250)):
            yield f"setcover_{ne}x{ns}_s{s}", (lambda ne=ne, ns=ns, s=s: _setcover(ne, ns, s))
        for t_len in (40, 80):
            yield f"lotsize_{t_len}_s{s}", (lambda t=t_len, s=s: _lotsizing(t, s))
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.nl"))):
        m = dm.from_nl(path)
        if m.num_integer and _milp_is_exactly_linear(m):
            yield os.path.basename(path), (lambda p=path: dm.from_nl(p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl", type=float, default=60.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--corpus", default=os.path.join(HERE, "..", "..", "python", "tests", "data", "minlplib_nl")
    )
    ap.add_argument("--out", default="slack_vs_native.tsv")
    a = ap.parse_args()
    print(f"load {os.getloadavg()}", flush=True)
    ratios, runs, n_inst = [], 0, 0
    with open(a.out, "w") as fh:
        fh.write("name\tform\trep\twall\tstatus\tobj\tbound\tnodes\tlogicals\tload\n")
        for name, build in instances(a.corpus):
            std, native, n_log = std_and_native(build())
            if n_log == 0:
                print(f"{name}: no logical columns, skipped", flush=True)
                continue
            n_inst += 1
            res = {"std": [], "native": []}
            for rep in range(a.reps):
                order = ("std", "native") if rep % 2 == 0 else ("native", "std")
                for f in order:
                    r = solve(std if f == "std" else native, a.tl)
                    res[f].append(r)
                    runs += 1
                    fh.write(
                        f"{name}\t{f}\t{rep}\t{r[0]:.4f}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]}\t"
                        f"{n_log}\t{os.getloadavg()[0]:.2f}\n"
                    )
                    fh.flush()
            ws, wn = (st.median(x[0] for x in res[f]) for f in ("std", "native"))
            fin = all(x[1] == "Optimal" for f in res for x in res[f])
            if fin:
                o1, o2 = res["std"][0][2], res["native"][0][2]
                assert abs(o1 - o2) <= 1e-6 * (1 + abs(o1)), f"{name}: objectives differ {o1} {o2}"
                ratios.append(wn / ws)
            if fin:
                verdict = f"ratio {wn / ws:.3f}"
            else:
                verdict = f"not finished: bounds {res['std'][0][3]} / {res['native'][0][3]}"
            print(
                f"{name:22s} logicals={n_log:5d} std {ws:8.3f}s n={res['std'][0][4]:<7d} "
                f"native {wn:8.3f}s n={res['native'][0][4]:<7d} {verdict} "
                f"load={os.getloadavg()[0]:.2f}",
                flush=True,
            )
    if ratios:
        faster = sum(r < 1 for r in ratios)
        print(
            f"\nfinished-in-both instances {len(ratios)}: median native/std ratio "
            f"{st.median(ratios):.3f}, geomean {float(np.exp(np.mean(np.log(ratios)))):.3f}, "
            f"native faster on {faster}/{len(ratios)}"
        )
    print(f"instances {n_inst}; executed HiGHS runs {runs}")
    return 0 if runs else 1


if __name__ == "__main__":
    sys.exit(main())
