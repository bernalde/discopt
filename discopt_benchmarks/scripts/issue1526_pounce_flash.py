#!/usr/bin/env python3
"""Build the shared POUNCE/DiscOpt Gate-1 flash comparison artifact.

The input is a real ``pounce-flash-results/1`` file emitted by POUNCE's full
harness.  This script preserves that evidence, appends the reduced DiscOpt
comparison, validates the result against POUNCE's packaged version-2 schema,
and writes one ``pounce-flash-results/2`` file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from discopt.benchmarks.problems.pounce_flash import (
    FULL_TEMPERATURES,
    SMOKE_TEMPERATURES,
    augment_pounce_artifact,
    pounce_seed,
    solve_flash_temperature,
    validate_comparison_artifact,
)


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pounce_artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--full",
        action="store_true",
        help="run all 34 temperatures (default: the five-point PR path)",
    )
    parser.add_argument("--time-limit", type=float, default=30.0, help="seconds per exact solve")
    parser.add_argument(
        "--schema",
        type=Path,
        default=None,
        help="explicit POUNCE v2 schema (default: packaged resource)",
    )
    args = parser.parse_args()

    with args.pounce_artifact.open("r", encoding="utf-8") as handle:
        pounce_artifact = json.load(handle)

    temperatures = FULL_TEMPERATURES if args.full else SMOKE_TEMPERATURES
    records = []
    failures = []
    for temperature in temperatures:
        seed = pounce_seed(pounce_artifact, temperature)
        if seed is None:
            raise RuntimeError(
                f"the POUNCE artifact has no successful five-variable leg record at "
                f"{temperature:g} K; refusing to substitute an unstamped seed"
            )
        for method in ("gdp", "sos1", "scholtes"):
            print(f"T={temperature:6.1f} K  {method:8s}", flush=True)
            solved = solve_flash_temperature(
                temperature,
                method,
                seed=seed,
                time_limit=args.time_limit,
            )
            records.append(solved.record)
            if method != "scholtes" and not solved.warm_start_accepted:
                failures.append(f"T={temperature:g} {method}: POUNCE seed was not accepted")
            if solved.record["state"] == "failed":
                failures.append(
                    f"T={temperature:g} {method}: {solved.record.get('error') or 'failed'}"
                )

    artifact = augment_pounce_artifact(
        pounce_artifact,
        records,
        discopt_commit=_git_revision(),
    )
    validate_comparison_artifact(artifact, args.schema)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=1, allow_nan=False)
        handle.write("\n")
    print(f"wrote schema-valid {args.output}")
    if failures:
        print("comparison completed with explicit failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
