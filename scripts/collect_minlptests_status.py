#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
TEST_FILE = REPO_ROOT / "python" / "tests" / "test_minlptests.py"
ALPINE_HELPER = REPO_ROOT / "scripts" / "alpine_minlptests_status.jl"


def load_test_module():
    sys.path.insert(0, str(PYTHON_ROOT))
    if "pytest" not in sys.modules:
        class _ParameterSet:
            def __init__(self, value):
                self.values = (value,)

        class _MarkProxy:
            def __getattr__(self, _name):
                def decorator(*args, **kwargs):
                    if args and callable(args[0]) and len(args) == 1 and not kwargs:
                        return args[0]

                    def wrapper(obj):
                        return obj

                    return wrapper

                return decorator

        pytest_stub = types.SimpleNamespace()
        pytest_stub.mark = _MarkProxy()
        pytest_stub.param = lambda value, **_kwargs: _ParameterSet(value)
        pytest_stub.xfail = lambda *, reason="": (_ for _ in ()).throw(
            RuntimeError(f"Unexpected pytest.xfail call: {reason}")
        )
        sys.modules["pytest"] = pytest_stub

    spec = importlib.util.spec_from_file_location("discopt_test_minlptests", TEST_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {TEST_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def unwrap_case(case):
    return case.values[0] if hasattr(case, "values") else case


def case_catalog(
    mod,
    *,
    include_convex: bool,
    per_instance_time_limit: float,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    if include_convex:
        for raw in mod.NLP_CVX_INSTANCES:
            inst = unwrap_case(raw)
            cases.append(
                {
                    "category": "nlp_cvx",
                    "directory": "nlp-cvx",
                    "symbol": inst.problem_id,
                    "instance": inst,
                    "time_limit": per_instance_time_limit,
                    "gap_tolerance": 1e-6,
                    "expected_status": "optimal",
                }
            )

    for raw in mod.NLP_INSTANCES:
        inst = unwrap_case(raw)
        cases.append(
            {
                "category": "nlp",
                "directory": "nlp",
                "symbol": inst.problem_id,
                "instance": inst,
                "time_limit": per_instance_time_limit,
                "gap_tolerance": 1e-6,
                "expected_status": "optimal",
            }
        )

    for raw in mod.NLP_MI_INSTANCES:
        inst = unwrap_case(raw)
        cases.append(
            {
                "category": "nlp_mi",
                "directory": "nlp-mi",
                "symbol": inst.problem_id,
                "instance": inst,
                "time_limit": per_instance_time_limit,
                "gap_tolerance": 1e-6,
                "expected_status": "optimal",
            }
        )

    for raw in mod.INFEASIBLE_INSTANCES:
        inst = unwrap_case(raw)
        directory = "nlp-mi" if inst.problem_id.startswith("nlp_mi_") else "nlp"
        cases.append(
            {
                "category": "infeasible",
                "directory": directory,
                "symbol": inst.problem_id,
                "instance": inst,
                "time_limit": per_instance_time_limit,
                "gap_tolerance": None,
                "expected_status": "infeasible",
            }
        )

    return cases


def validate_discopt_result(mod, case: dict[str, Any], result) -> None:
    inst = case["instance"]
    if case["expected_status"] == "infeasible":
        mod.assert_infeasible(result, inst.problem_id)
        return

    mod.assert_optimal(result, inst.expected_obj, inst.problem_id)
    if inst.is_convex and getattr(result, "convex_fast_path", False) is not True:
        raise AssertionError(f"[{inst.problem_id}] Expected discopt convex fast path")


def run_discopt_cases(
    mod,
    cases: list[dict[str, Any]],
    *,
    solver_mode: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total_cases = len(cases)

    for index, case in enumerate(cases, start=1):
        inst = case["instance"]
        print(
            f"[discopt_{solver_mode} {index}/{total_cases}] starting {inst.problem_id}",
            flush=True,
        )

        t0 = time.perf_counter()
        result = None
        try:
            model = inst.build_fn()
            solve_kwargs: dict[str, Any] = {"time_limit": case["time_limit"]}
            if solver_mode == "amp":
                solve_kwargs["solver"] = "amp"
                solve_kwargs["nlp_solver"] = "ipm"
                if case["gap_tolerance"] is not None:
                    solve_kwargs["gap_tolerance"] = 1e-3
            elif case["gap_tolerance"] is not None:
                solve_kwargs["gap_tolerance"] = case["gap_tolerance"]
            result = model.solve(**solve_kwargs)
            validate_discopt_result(mod, case, result)
            outcome = "pass"
            note = ""
        except AssertionError as err:
            outcome = "fail"
            note = str(err)
        except Exception as err:  # pragma: no cover - exercised in real benchmark runs
            outcome = "error"
            note = f"{type(err).__name__}: {err}"

        wall_time = time.perf_counter() - t0
        records.append(
            {
                "solver": f"discopt_{solver_mode}",
                "category": case["category"],
                "problem_id": inst.problem_id,
                "outcome": outcome,
                "status": getattr(result, "status", None),
                "objective": getattr(result, "objective", None),
                "wall_time_sec": wall_time,
                "note": note,
            }
        )
        print(
            f"[discopt_{solver_mode} {index}/{total_cases}] finished {inst.problem_id} "
            f"outcome={outcome} status={getattr(result, 'status', None)} "
            f"time={wall_time:.3f}s",
            flush=True,
        )

    return records


def run_alpine_cases(
    cases: list[dict[str, Any]],
    alpine_project: Path,
    minlptests_path: Path,
    julia_bin: str,
    julia_channel: str,
    per_instance_time_limit: float,
    mip_solver: str = "highs",
) -> list[dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="alpine-minlptests-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        request_path = tmpdir_path / "request.tsv"
        output_path = tmpdir_path / "results.jsonl"

        request_lines = [
            "\t".join((case["instance"].problem_id, case["category"], case["symbol"]))
            for case in cases
        ]
        request_path.write_text("\n".join(request_lines) + "\n", encoding="utf-8")

        cmd = [julia_bin]
        if julia_channel:
            cmd.append(julia_channel)
        cmd.append(f"--project={alpine_project}")
        cmd.extend(
            [
                str(ALPINE_HELPER),
                str(request_path),
                str(output_path),
                str(minlptests_path),
                str(per_instance_time_limit),
            ]
        )
        env = dict(os.environ)
        env["ALPINE_MIP_SOLVER"] = mip_solver
        subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)

        records: list[dict[str, Any]] = []
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
                    payload["solver"] = f"alpine_{mip_solver}"
                    records.append(payload)
        return records


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        summary[record["category"]][record["outcome"]] += 1
        summary[record["category"]]["total"] += 1
    return {category: dict(counter) for category, counter in summary.items()}


def compare_outcomes(
    discopt_records: list[dict[str, Any]],
    alpine_records: list[dict[str, Any]],
) -> dict[str, int]:
    counts = compare_solver_pair(discopt_records, alpine_records)
    return {
        "both_pass": counts["both_pass"],
        "discopt_only_pass": counts["left_only_pass"],
        "alpine_only_pass": counts["right_only_pass"],
        "both_fail": counts["both_fail"],
    }


def compare_solver_pair(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
) -> dict[str, int]:
    right_by_problem = {record["problem_id"]: record for record in right_records}
    counts = Counter(
        {
            "both_pass": 0,
            "left_only_pass": 0,
            "right_only_pass": 0,
            "both_fail": 0,
        }
    )
    for left in left_records:
        right = right_by_problem.get(left["problem_id"])
        left_pass = left["outcome"] == "pass"
        right_pass = right is not None and right["outcome"] == "pass"
        if left_pass and right_pass:
            counts["both_pass"] += 1
        elif left_pass:
            counts["left_only_pass"] += 1
        elif right_pass:
            counts["right_only_pass"] += 1
        else:
            counts["both_fail"] += 1
    return dict(counts)


def _positive_wall_times(records: list[dict[str, Any]]) -> list[float]:
    return [
        float(record["wall_time_sec"])
        for record in records
        if record["outcome"] == "pass" and float(record.get("wall_time_sec", 0.0)) > 0.0
    ]


def geometric_median(values: list[float]) -> float | None:
    positive_values = [float(value) for value in values if float(value) > 0.0]
    if not positive_values:
        return None
    return float(math.exp(statistics.median(math.log(value) for value in positive_values)))


def summarize_timing(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    wall_times = _positive_wall_times(records)
    if not wall_times:
        return {
            "pass_count": 0,
            "median_sec": None,
            "geometric_median_sec": None,
        }

    return {
        "pass_count": len(wall_times),
        "median_sec": float(statistics.median(wall_times)),
        "geometric_median_sec": geometric_median(wall_times),
    }


def compare_pairwise_timing(
    left_records: list[dict[str, Any]],
    right_records: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    right_by_problem = {
        record["problem_id"]: record
        for record in right_records
        if record["outcome"] == "pass" and float(record.get("wall_time_sec", 0.0)) > 0.0
    }
    shared_left: list[float] = []
    shared_right: list[float] = []
    shared_ratios: list[float] = []

    for left in left_records:
        if left["outcome"] != "pass":
            continue
        left_time = float(left.get("wall_time_sec", 0.0))
        if left_time <= 0.0:
            continue
        right = right_by_problem.get(left["problem_id"])
        if right is None:
            continue
        right_time = float(right.get("wall_time_sec", 0.0))
        if right_time <= 0.0:
            continue
        shared_left.append(left_time)
        shared_right.append(right_time)
        shared_ratios.append(left_time / right_time)

    return {
        "shared_pass_count": len(shared_ratios),
        "left_geometric_median_sec": geometric_median(shared_left),
        "right_geometric_median_sec": geometric_median(shared_right),
        "geometric_median_ratio_left_over_right": geometric_median(shared_ratios),
    }


def build_solver_comparison_payload(
    solver_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    solver_order = list(solver_records)
    payload: dict[str, Any] = {
        "solver_order": solver_order,
        "solvers": solver_records,
        "solver_summaries": {
            solver: summarize(records) for solver, records in solver_records.items()
        },
        "timing_summaries": {
            solver: summarize_timing(records) for solver, records in solver_records.items()
        },
        "pairwise_outcomes": {},
        "pairwise_timing": {},
    }

    for idx, left_solver in enumerate(solver_order):
        for right_solver in solver_order[idx + 1 :]:
            pair_key = f"{left_solver}__vs__{right_solver}"
            left_records = solver_records[left_solver]
            right_records = solver_records[right_solver]
            payload["pairwise_outcomes"][pair_key] = compare_solver_pair(
                left_records,
                right_records,
            )
            payload["pairwise_timing"][pair_key] = compare_pairwise_timing(
                left_records,
                right_records,
            )

    return payload


def build_solver_comparison_markdown(
    solver_records: dict[str, list[dict[str, Any]]],
) -> str:
    payload = build_solver_comparison_payload(solver_records)
    lines: list[str] = [
        "# MINLPTests Status",
        "",
        "Generated from translated MINLPTests runs on the same problem IDs.",
        "",
        "Geometric median timing is computed as `exp(median(log(time_sec)))` on solved cases.",
        "",
    ]
    lines.extend(
        [
            "## Solver Summary By Category",
            "",
            "| Solver | Category | Pass | Fail | Error | Total |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for solver in payload["solver_order"]:
        summary = payload["solver_summaries"][solver]
        for category in ("nlp", "nlp_mi", "infeasible", "nlp_cvx"):
            row = summary.get(category, {})
            lines.append(
                (
                    f"| {solver} | {category} | {row.get('pass', 0)} | "
                    f"{row.get('fail', 0)} | {row.get('error', 0)} | {row.get('total', 0)} |"
                )
            )
    lines.append("")

    if payload["pairwise_outcomes"]:
        lines.extend(
            [
                "## Pairwise Outcome Comparison",
                "",
                (
                    "| Left solver | Right solver | Both pass | Left only pass | "
                    "Right only pass | Both fail |"
                ),
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for pair_key, counts in payload["pairwise_outcomes"].items():
            left_solver, right_solver = pair_key.split("__vs__")
            lines.append(
                (
                    f"| {left_solver} | {right_solver} | {counts['both_pass']} | "
                    f"{counts['left_only_pass']} | {counts['right_only_pass']} | "
                    f"{counts['both_fail']} |"
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Timing Summary On Passes",
            "",
            "| Solver | Passes | Median sec | Geometric median sec |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for solver in payload["solver_order"]:
        timing = payload["timing_summaries"][solver]
        median_sec = timing["median_sec"]
        geom_sec = timing["geometric_median_sec"]
        lines.append(
            (
                f"| {solver} | {timing['pass_count']} | "
                f"{'' if median_sec is None else f'{median_sec:.3f}'} | "
                f"{'' if geom_sec is None else f'{geom_sec:.3f}'} |"
            )
        )
    lines.append("")

    if payload["pairwise_timing"]:
        lines.extend(
            [
                "## Pairwise Geometric Timing Comparison",
                "",
                "A ratio below `1.0` means the left solver is faster on the shared solved cases.",
                "",
                (
                    "| Left solver | Right solver | Shared passes | Left geom median sec | "
                    "Right geom median sec | Geom median ratio left/right |"
                ),
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for pair_key, timing in payload["pairwise_timing"].items():
            left_solver, right_solver = pair_key.split("__vs__")
            left_geom = timing["left_geometric_median_sec"]
            right_geom = timing["right_geometric_median_sec"]
            ratio = timing["geometric_median_ratio_left_over_right"]
            lines.append(
                (
                    f"| {left_solver} | {right_solver} | {timing['shared_pass_count']} | "
                    f"{'' if left_geom is None else f'{left_geom:.3f}'} | "
                    f"{'' if right_geom is None else f'{right_geom:.3f}'} | "
                    f"{'' if ratio is None else f'{ratio:.3f}'} |"
                )
            )
        lines.append("")

    for solver in payload["solver_order"]:
        lines.extend(
            [
                f"## Failures: {solver}",
                "",
                "| Problem | Category | Outcome | Status | Note |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        shown = False
        for record in solver_records[solver]:
            if record["outcome"] == "pass":
                continue
            shown = True
            status = "" if record.get("status") is None else str(record["status"])
            note = str(record.get("note", "")).replace("|", "\\|")
            lines.append(
                (
                    f"| {record['problem_id']} | {record['category']} | "
                    f"{record['outcome']} | {status} | {note} |"
                )
            )
        if not shown:
            lines.append("| none | - | - | - | - |")
        lines.append("")

    return "\n".join(lines)


def build_markdown(
    discopt_records: list[dict[str, Any]],
    alpine_records: list[dict[str, Any]],
) -> str:
    solver_records = {"discopt": discopt_records}
    if alpine_records:
        solver_records["alpine"] = alpine_records
    return build_solver_comparison_markdown(solver_records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run discopt and Alpine MINLPTests status sweeps."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to write the merged JSON result payload.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        help="Optional path to write a Markdown summary table.",
    )
    parser.add_argument(
        "--skip-alpine",
        action="store_true",
        help="Run only the translated discopt suite and skip the Alpine comparison.",
    )
    parser.add_argument(
        "--include-convex",
        action="store_true",
        help=(
            "Include the convex nlp-cvx cases. By default the runner uses the "
            "nonconvex Phase 6 scope only."
        ),
    )
    parser.add_argument(
        "--per-instance-time-limit",
        type=float,
        default=300.0,
        help="Wall-clock time limit in seconds for each discopt or Alpine case.",
    )
    parser.add_argument(
        "--discopt-mode",
        choices=("amp", "default"),
        default="amp",
        help=(
            "Solve the translated MINLPTests cases with the AMP solver or the "
            "default solve path."
        ),
    )
    parser.add_argument(
        "--alpine-project",
        type=Path,
        default=REPO_ROOT.parent / "Alpine.jl",
        help="Path to the local Alpine.jl checkout.",
    )
    parser.add_argument(
        "--minlptests-path",
        type=Path,
        default=REPO_ROOT.parent / "MINLPTests.jl",
        help="Path to the local MINLPTests.jl checkout.",
    )
    parser.add_argument("--julia-bin", default="julia", help="Julia executable to use.")
    parser.add_argument(
        "--julia-channel",
        default="+release",
        help=(
            "Optional juliaup channel argument, e.g. '+release'. Use an empty "
            "string to disable it."
        ),
    )
    parser.add_argument(
        "--alpine-mip-solver",
        choices=("highs", "gurobi"),
        default="highs",
        help="MIP backend for the Alpine.jl run.",
    )
    args = parser.parse_args()

    mod = load_test_module()
    cases = case_catalog(
        mod,
        include_convex=args.include_convex,
        per_instance_time_limit=args.per_instance_time_limit,
    )
    discopt_records = run_discopt_cases(mod, cases, solver_mode=args.discopt_mode)
    alpine_records: list[dict[str, Any]] = []

    if not args.skip_alpine:
        alpine_records = run_alpine_cases(
            cases,
            args.alpine_project.resolve(),
            args.minlptests_path.resolve(),
            args.julia_bin,
            args.julia_channel,
            args.per_instance_time_limit,
            mip_solver=args.alpine_mip_solver,
        )

    payload = {
        "discopt": discopt_records,
        "alpine": alpine_records,
        "discopt_summary": summarize(discopt_records),
        "alpine_summary": summarize(alpine_records),
        "comparison_summary": compare_outcomes(discopt_records, alpine_records),
        "discopt_timing_summary": summarize_timing(discopt_records),
        "alpine_timing_summary": summarize_timing(alpine_records),
        "pairwise_timing_summary": compare_pairwise_timing(discopt_records, alpine_records),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(
            build_markdown(discopt_records, alpine_records),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
