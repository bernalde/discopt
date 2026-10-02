import os

OUT = os.path.dirname(os.path.abspath(__file__))
import json, tomllib

P = json.load(open(f"{OUT}/panel2_pr.json"))
M = json.load(open(f"{OUT}/panel2_main.json"))
assert len(P) == len(M) == 50
K = tomllib.load(open("python/tests/data/known_optima.toml", "rb"))
solu = {
    l.split()[1]: float(l.split()[2])
    for l in open("python/tests/data/qplib/qplib.solu")
    if l.startswith("=")
}
# Not vendored in-repo; MINLPLib published optimum, equal to discopt's certified unshifted solve.
PUB = {"alan": 2.925, "meanvarx": 14.36923211, "st_miqp4": -4574.0, "st_miqp5": -333.8888889}


def oracle(path):
    n = path.split("/")[-1].rsplit(".", 1)[0]
    if n in K:
        return K[n]["optimum"], "known_optima.toml"
    if n in solu:
        return solu[n], "qplib.solu"
    return PUB[n], "MINLPLib published"


cmp = 0


def judge(r, opt):
    """list of defects; every check executed is counted"""
    global cmp
    tol = max(1e-6, 1e-4 * max(1.0, abs(opt)))
    bad = []
    if r["cert"]:
        cmp += 1
        if r["status"] == "infeasible":
            bad.append("certified-infeasible")
        elif r["obj"] is None or abs(r["obj"] - opt) > tol:
            bad.append(f"false-optimum {r['obj']}")
    if r["bound"] is not None and r["status"] != "infeasible":
        cmp += 1
        if r["bound"] > opt + tol:
            bad.append(f"bound>opt {r['bound']}")
    if r.get("verify_ok") is not None:
        cmp += 1
        if not r["verify_ok"]:
            bad.append(f"incumbent-infeasible({r.get('verify_why')})")
        else:
            cmp += 2
            if abs(r["verify_obj"] - r["obj"]) > tol:
                bad.append(f"obj-mismatch {r['verify_obj']} vs {r['obj']}")
            if r["verify_obj"] < opt - tol:
                bad.append(f"incumbent-below-opt {r['verify_obj']}")
    if r["status"].startswith("RAISE"):
        bad.append("raise")
    return bad


inc = {"pr": 0, "main": 0}
raise_ = {"pr": 0, "main": 0}
cert = {"pr": 0, "main": 0}
lost = 0
gained = 0
same = 0
rows = 0
bbcheck = 0
for p, m in zip(P, M):
    assert p["path"] == m["path"] and p["offset"] == m["offset"]
    assert p["bb"] >= 1 and p["shift_cols"] > 0, p  # the change was exercised on every PR row
    bbcheck += 1
    opt, src = oracle(p["path"])
    bp, bm = judge(p, opt), judge(m, opt)
    inc["pr"] += any(b != "raise" for b in bp)
    inc["main"] += any(b != "raise" for b in bm)
    raise_["pr"] += "raise" in bp
    raise_["main"] += "raise" in bm
    cert["pr"] += p["cert"]
    cert["main"] += m["cert"]
    lost += m["cert"] and not bm and not p["cert"]
    gained += p["cert"] and not m["cert"]
    same += all(p[k] == m[k] for k in ("status", "obj", "bound", "nodes", "cert"))
    rows += 1
    f = lambda r: (
        f"{r['status'][:9]:9} {r['obj'] if r['obj'] is None else round(r['obj'], 6)!s:>11} n={r['nodes']!s:>6}"
        + (" C" if r["cert"] else "  ")
    )
    print(
        f"{p['path'].split('/')[-1]:17}{p['offset']:>8g} opt={opt:<11.6g} | PR {f(p)} {','.join(bp) or 'ok':14} | main {f(m)} {','.join(bm) or 'ok'}"
    )
print(f"\nrows={rows} (PR rows asserted bb>=1 & shift>0: {bbcheck})  executed comparisons={cmp}")
print(
    f"incorrect_count: PR={inc['pr']} main={inc['main']}   RAISE: PR={raise_['pr']} main={raise_['main']}"
)
print(
    f"certified: PR={cert['pr']} main={cert['main']}   certificates lost (main sound+cert, PR not)={lost}   gained={gained}   rows identical={same}"
)
assert cmp > 0
