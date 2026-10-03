"""#1593 panel: the primal heuristics' implied-integer predicate on its own.

``primal_heuristics._is_free_integer`` leaves #1544 INTEGER auxes (``w == y - c``
over integer ``y``) to the continuous repair instead of rounding / pinning /
moving them as free integers. This A/Bs that predicate against main's pre-#1588
one (every INTEGER/BINARY column free), with everything else held:

* arm A -- legacy predicate;
* arm B -- with the lift ON (``PANEL_LIFT=1``, the default path) the shipped
  predicate; with ``PANEL_LIFT=0`` an "ungated" predicate that excludes every
  INTEGER ``_fr_aux_*`` column (what the ``=0`` path would do if ungated).

Rows: every in-repo MINLPLib instance x {as written, 1e3, 1e6 translation} whose
reformulation creates at least one #1544 INTEGER aux. Interleaved per row,
``PANEL_REPS`` reps with the arm order alternated. Every published incumbent is
re-verified with ``verify_point``; every certificate is checked against the
unshifted certified base and ``minlplib.solu`` (``PANEL_SOLU``). Exits non-zero
when it compared nothing (CLAUDE.md §6).

    PANEL_LIFT=1 PANEL_TL=20 PANEL_REPS=2 \\
        python -u discopt_benchmarks/scripts/implied_integer_predicate_panel.py
"""

from __future__ import annotations

import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "python", "tests"))
sys.path.insert(0, os.path.join(ROOT, "python"))
for _k in list(os.environ):
    if _k.startswith("DISCOPT_"):
        del os.environ[_k]
os.environ["DISCOPT_LIFT_AFFINE_MONOMIALS"] = os.environ.get("PANEL_LIFT", "1")

import discopt  # noqa: E402
import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402

assert discopt.__file__.startswith(os.path.join(ROOT, "python")), discopt.__file__
from _invariance import certified_answer_changed, translate  # noqa: E402
from discopt._relax import factorable_reform as fr  # noqa: E402
from discopt._relax import primal_heuristics as ph  # noqa: E402
from discopt.modeling.core import VarType  # noqa: E402
from discopt.validation.feasibility import verify_point  # noqa: E402

TL = float(os.environ.get("PANEL_TL", "20"))
REPS = int(os.environ.get("PANEL_REPS", "2"))
SOLU_PATH = os.environ.get(
    "PANEL_SOLU", os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu")
)
SHIPPED = ph._is_free_integer


def legacy(v, model) -> bool:
    """main before #1588: every INTEGER/BINARY column is free."""
    return v.var_type in (VarType.BINARY, VarType.INTEGER)


def ungated(v, model) -> bool:
    """Every #1544 INTEGER ``_fr_aux_*`` excluded, whatever the flag."""
    if v.var_type not in (VarType.BINARY, VarType.INTEGER):
        return False
    return not (v.var_type == VarType.INTEGER and v.name.startswith("_fr_aux_"))


def _int_auxes(model) -> set[str]:
    return {
        v.name
        for v in model._variables
        if v.var_type == VarType.INTEGER and v.name.startswith("_fr_aux_")
    }


def main() -> int:
    lift = fr._lift_affine_monomials_enabled()
    arms = {"A": legacy, "B": SHIPPED if lift else ungated}
    print(f"lift={lift} TL={TL} REPS={REPS} discopt={discopt.__file__} load={os.getloadavg()}")
    data = os.path.join(ROOT, "python", "tests", "data", "minlplib_nl")
    only = os.environ.get("PANEL_ONLY", "")
    rows = []
    for p in sorted(glob.glob(os.path.join(data, "*.nl"))):
        name = os.path.basename(p)[:-3]
        for sh, seed in ((0, 0), (1e3, 1), (1e6, 2)):
            label = f"{name}@{sh:g}"
            if only and label not in only.split(","):
                continue

            def mk(p=p, sh=sh, seed=seed):
                m = dm.from_nl(p)
                return translate(m, sh, seed=seed) if sh else m

            red = fr.factorable_reformulate(mk())
            auxes = _int_auxes(red)
            if lift:  # the recording covers every #1544 aux, lift-made or not
                assert red._implied_integer_auxes == auxes, label
            if auxes:
                rows.append((label, name, mk, len(auxes)))
    print(f"ROWS {len(rows)}: {[r[0] for r in rows]}", flush=True)

    solu: dict[str, float] = {}
    with open(SOLU_PATH) as fh:
        for line in fh:
            f = line.split()
            if len(f) >= 3 and f[0] == "=opt=":
                solu[f[1]] = float(f[2])
    solu_checks = compared = 0
    bases = {}
    tally = {a: {"false": 0, "bad": 0, "cert": 0, "wall": 0.0} for a in arms}
    for label, name, mk, n in rows:
        if name not in bases:
            ph._is_free_integer = SHIPPED
            bases[name] = dm.from_nl(os.path.join(data, name + ".nl")).solve(time_limit=TL)
        base = bases[name]
        res: dict[str, list] = {a: [] for a in arms}
        for rep in range(REPS):
            for arm in ("A", "B") if rep % 2 == 0 else ("B", "A"):
                ph._is_free_integer = arms[arm]
                m = mk()
                t0 = time.perf_counter()
                r = m.solve(time_limit=TL)  # a raise is a finding: let it propagate
                w = time.perf_counter() - t0
                t = tally[arm]
                t["wall"] += w
                bad = False
                if r.x is not None and r.status in ("optimal", "feasible"):
                    flat = np.concatenate(
                        [np.ravel(np.asarray(r.x[v.name], float)) for v in m._variables]
                    )
                    bad = not verify_point(m, flat).ok
                t["bad"] += bad
                why = certified_answer_changed(base, r) if base.gap_certified else ""
                if r.gap_certified and name in solu:
                    solu_checks += 1
                    opt = solu[name]
                    if abs(r.objective - opt) > 1e-4 * max(1.0, abs(opt)) + 1e-6:
                        why = why or f"certified {r.objective} vs solu {opt}"
                if why:
                    t["false"] += 1
                    print(f"FALSE {arm} {label}: {why}", flush=True)
                t["cert"] += bool(r.gap_certified)
                res[arm].append(
                    (
                        f"{r.status}/{'C' if r.gap_certified else '-'} obj={r.objective} "
                        f"bd={r.bound} n={r.node_count}{' BAD' if bad else ''}",
                        w,
                    )
                )
                compared += 1
        ph._is_free_integer = SHIPPED
        for arm in ("A", "B"):
            ws = [w for _, w in res[arm]]
            sd = np.std(ws, ddof=1) if len(ws) > 1 else 0.0
            print(
                f"{label:22s} aux={n:<3d} {arm} wall={np.mean(ws):6.2f}+-{sd:5.2f}  "
                f"{res[arm][0][0]}",
                flush=True,
            )
            for s, _ in res[arm][1:]:
                if s != res[arm][0][0]:
                    print(f"{'':34s} rep differs: {s}", flush=True)

    print(f"\nCOMPARED {compared} runs over {len(rows)} rows; solu checks {solu_checks}")
    for arm, t in tally.items():
        print(
            f"{arm}: false={t['false']} bad_point={t['bad']} certified={t['cert']} "
            f"wall={t['wall']:.0f}s"
        )
    return 0 if compared else 1


if __name__ == "__main__":
    sys.exit(main())
