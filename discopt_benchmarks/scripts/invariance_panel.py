#!/usr/bin/env python3
"""Translation / rescaling invariance panel over ``.nl`` instances (#1537, workstream B).

A certified answer must not depend on the representation of an identical model.
For every instance this solves the model as given, then each exact change of
representation from :mod:`discopt.validation.nl_invariance`:

* ``shift<c>``  -- every variable written as ``y = x + c`` (``c`` integral);
* ``rows<k>``   -- every algebraic row multiplied by ``k``;
* ``obj<k>``    -- the objective multiplied by ``k`` (the optimum scales by ``k``).

and compares. Two failure kinds are counted separately, as the issue asks:

* **false** -- a transformed solve certifies (``gap_certified``) an objective that
  disagrees with the reference: the as-given certified objective, or the
  ``--solu`` oracle (``minlplib.solu``) when given, which also catches a
  certified bound above the oracle. This is a soundness failure; the script
  exits 1.
* **lost** -- the as-given solve certifies and the transformed one does not. Not a
  soundness failure, but the measurement that exposed the #1295 guard's scale
  dependence; reported, never fatal.

Results are bit-for-bit comparable only within one discopt build; numerics
legitimately differ between representations, so objectives are compared at
``abs 1e-6 + rel 1e-4`` (the solver's own tolerances), not for equality.

The repository policy is that CI runs no ``schedule:`` lane (see ``ci.yml``), so
this is a dispatch-and-before-release instrument. The seeded generated half of the
panel runs on every PR in ``python/tests/test_1537_translation_rescaling_invariance.py``.

Usage::

    python -u -m discopt_benchmarks.scripts.invariance_panel \\
        --instances python/tests/data/minlplib_nl --time-limit 60 \\
        --shifts 1e3,1e6 --rowscales 1e-3,1e3,1e6 --objscales 1e6 \\
        --solu ~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu \\
        --out invariance.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import tempfile
import time
import warnings
from pathlib import Path

ABS_TOL = 1e-6
REL_TOL = 1e-4


def _floats(s: str) -> list[float]:
    return [float(t) for t in s.split(",") if t.strip()] if s else []


def _read_solu(path: Path) -> dict[str, float]:
    """``=opt= name value`` lines of a MINLPLib ``.solu`` file (``=best=`` is not an
    optimum and is not used as an oracle)."""
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) >= 3 and p[0] == "=opt=":
            out[p[1]] = float(p[2])
    return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= ABS_TOL + REL_TOL * max(abs(a), abs(b))


def _solve(path: Path, time_limit: float) -> dict:
    from discopt.modeling.core import ObjectiveSense, from_nl

    t0 = time.perf_counter()
    m = from_nl(str(path))
    r = m.solve(time_limit=time_limit)
    maximize = m._objective is not None and m._objective.sense == ObjectiveSense.MAXIMIZE
    return {
        "status": r.status,
        "certified": bool(r.gap_certified),
        "objective": None if r.objective is None else float(r.objective),
        "bound": None if r.bound is None else float(r.bound),
        "maximize": bool(maximize),
        "wall": time.perf_counter() - t0,
    }


def _judge(base: dict, res: dict, scale: float, oracle: float | None) -> tuple[str, str]:
    """``(verdict, why)``; verdict in ``ok``/``false``/``lost``/``open``."""
    if res["certified"]:
        obj = res["objective"]
        if obj is None or not math.isfinite(obj):
            return "false", "certified with no finite objective"
        if oracle is not None:
            ref = oracle * scale
            if not _close(obj, ref):
                return "false", f"certified {obj!r}, oracle {ref!r}"
            b = res["bound"]
            if b is not None:
                above = b > ref + ABS_TOL + REL_TOL * abs(ref)
                below = b < ref - ABS_TOL - REL_TOL * abs(ref)
                if (above and not res["maximize"]) or (below and res["maximize"]):
                    return "false", f"bound {b!r} crosses the oracle {ref!r}"
        if base["certified"] and base["objective"] is not None:
            ref = base["objective"] * scale
            if not _close(obj, ref):
                return "false", f"certified {obj!r}, as-given certified {ref!r}"
        return "ok", ""
    if base["certified"]:
        return "lost", f"{res['status']} (as given: certified)"
    return "open", ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--instances", required=True, help="directory of .nl files, or a file list")
    ap.add_argument("--names", default="", help="comma-separated subset of instance names")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--shifts", default="1e3,1e6")
    ap.add_argument("--rowscales", default="1e-3,1e3,1e6")
    ap.add_argument("--objscales", default="1e6")
    ap.add_argument("--solu", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore")
    import discopt.validation.nl_invariance as nli

    print(f"discopt from {nli.__file__}", flush=True)

    src = Path(args.instances).expanduser()
    files = sorted(src.glob("*.nl")) if src.is_dir() else [Path(p) for p in src.read_text().split()]
    if args.names:
        keep = set(args.names.split(","))
        files = [f for f in files if f.stem in keep]
    oracle = _read_solu(args.solu.expanduser()) if args.solu else {}
    transforms: list[tuple[str, object, float, float]] = []
    for c in _floats(args.shifts):
        transforms.append((f"shift{c:g}", nli.translate_nl, c, 1.0))
    for k in _floats(args.rowscales):
        transforms.append((f"rows{k:g}", nli.scale_rows_nl, k, 1.0))
    for k in _floats(args.objscales):
        transforms.append((f"obj{k:g}", nli.scale_objective_nl, k, k))

    tally = {
        name: {"ok": 0, "false": 0, "lost": 0, "open": 0, "refused": 0} for name, *_ in transforms
    }
    rows: list[dict] = []
    executed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            text = f.read_text(encoding="latin-1")
            try:
                base = _solve(f, args.time_limit)
            except Exception as exc:  # noqa: BLE001 - recorded, never hidden
                print(f"{f.stem}: as-given solve raised {type(exc).__name__}: {exc}", flush=True)
                rows.append({"instance": f.stem, "transform": "as-given", "error": repr(exc)})
                continue
            print(
                f"{f.stem}: as-given {base['status']} cert={base['certified']} "
                f"obj={base['objective']} ({base['wall']:.1f}s)",
                flush=True,
            )
            for name, fn, arg, scale in transforms:
                try:
                    new = fn(text, arg)  # type: ignore[operator]
                except NotImplementedError as exc:
                    tally[name]["refused"] += 1
                    rows.append({"instance": f.stem, "transform": name, "refused": str(exc)})
                    continue
                p = Path(tmp) / f"{f.stem}__{name}.nl"
                p.write_text(new, encoding="latin-1")
                res = _solve(p, args.time_limit)
                executed += 1
                verdict, why = _judge(base, res, scale, oracle.get(f.stem))
                tally[name][verdict] += 1
                rows.append({"instance": f.stem, "transform": name, "verdict": verdict,
                             "why": why, "as_given": base, "result": res})  # fmt: skip
                if verdict in ("false", "lost"):
                    print(f"  {name}: {verdict.upper()} {why}", flush=True)

    print("\ntransform         ok false  lost  open  refsd")
    for name, t in tally.items():
        cells = (t["ok"], t["false"], t["lost"], t["open"], t["refused"])
        print(f"{name:<14} " + " ".join(f"{v:>5}" for v in cells))
    n_false = sum(t["false"] for t in tally.values())
    n_lost = sum(t["lost"] for t in tally.values())
    print(f"executed {executed} transformed solves over {len(files)} instances; "
          f"false {n_false}, lost {n_lost}")  # fmt: skip
    if args.out:
        args.out.write_text(json.dumps({"tally": tally, "rows": rows}, indent=1, default=str))
    if executed == 0:
        print("no transformed solve ran: this panel measured nothing", file=sys.stderr)
        return 2
    return 1 if n_false else 0


if __name__ == "__main__":
    sys.exit(main())
