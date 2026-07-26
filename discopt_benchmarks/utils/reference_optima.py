"""Reference optima ("oracles") for benchmark instances, from one accessor.

A quality metric is only as good as the reference it scores against, and the
reference lived in ~17 hand-rolled copies: every panel script re-implemented the
same six-line ``minlplib.solu`` parse, or pasted a private ``OPT = {...}`` dict.
That is how a panel ends up silently unscored — the ``.solu`` snapshot is a local
artifact that does not exist in CI, so a script keyed to it measures nothing there
and says nothing about it.

This module resolves an instance name against every source the repo actually has,
in decreasing order of authority:

1. **``minlplib.solu``** — the upstream oracle, when a snapshot is present. Located
   via ``$DISCOPT_MINLPLIB_SOLU``, else the standard checkout path (CLAUDE.md
   "Benchmark instance corpus"). Only ``=opt=`` and ``=best=`` are read; ``=best=``
   is a published incumbent, not a proven optimum, and is tagged as such.
2. **``python/tests/data/known_optima.toml``** — the vendored registry, the repo's
   declared single source of truth for the certification suites. Present in CI.
3. **``docs/dev/data/cert-optima.json``** — the certification baseline's oracle map,
   already used as a cross-check by ``t21_root_loop_replay.py``.

The point of the chain is that one panel runs everywhere: on the owner's machine it
scores the full corpus (tln4/5/6 included) from the upstream ``.solu``; in CI it
still scores the vendored instances instead of degrading to a no-op.

Sense: every value is stored in the instance's own sense, exactly as the source
records it. The ``.nl`` reader normalizes MINLPLib instances to MINIMIZE, so for
those the value is directly a lower-bound oracle.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KNOWN_OPTIMA_TOML = _REPO_ROOT / "python" / "tests" / "data" / "known_optima.toml"
_CERT_OPTIMA_JSON = _REPO_ROOT / "docs" / "dev" / "data" / "cert-optima.json"
_DEFAULT_SOLU = "~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu"


@dataclass(frozen=True)
class Oracle:
    """A reference objective value and where it came from.

    ``proven`` is False for a ``=best=`` entry: the value is the best *known*
    incumbent, so an incumbent matching it is not proof of optimality and a dual
    bound above it is not automatically unsound. Callers that gate on soundness must
    respect this — treating an unproven value as ground truth manufactures false
    violations.
    """

    value: float
    source: str
    proven: bool = True


def solu_path() -> Path | None:
    """Path to a ``minlplib.solu`` snapshot, or None when no snapshot is installed."""
    raw = os.environ.get("DISCOPT_MINLPLIB_SOLU") or _DEFAULT_SOLU
    p = Path(os.path.expanduser(raw))
    return p if p.is_file() else None


def _parse_solu(path: Path) -> dict[str, Oracle]:
    """Parse ``=opt=`` / ``=best=`` lines; ``=opt=`` wins where both exist."""
    best: dict[str, Oracle] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[0] not in ("=opt=", "=best="):
            continue
        try:
            value = float(parts[2])
        except ValueError:
            continue
        proven = parts[0] == "=opt="
        prior = best.get(parts[1])
        if prior is None or (proven and not prior.proven):
            best[parts[1]] = Oracle(value, f"minlplib.solu {parts[0]}", proven)
    return best


def _parse_known_optima(path: Path) -> dict[str, Oracle]:
    import tomllib

    with path.open("rb") as fh:
        data = tomllib.load(fh)
    data.pop("schema", None)
    out: dict[str, Oracle] = {}
    for name, entry in data.items():
        if isinstance(entry, dict) and isinstance(entry.get("optimum"), (int, float)):
            out[name] = Oracle(
                float(entry["optimum"]),
                f"known_optima.toml ({entry.get('source', 'unknown')})",
                bool(entry.get("proven", True)),
            )
    return out


def _parse_cert_optima(path: Path) -> dict[str, Oracle]:
    data = json.loads(path.read_text())
    return {
        name: Oracle(float(v), "cert-optima.json")
        for name, v in data.items()
        if isinstance(v, (int, float))
    }


@lru_cache(maxsize=1)
def oracle_table() -> dict[str, Oracle]:
    """Merged oracle map, highest-authority source last-wins per name."""
    table: dict[str, Oracle] = {}
    if _CERT_OPTIMA_JSON.is_file():
        table.update(_parse_cert_optima(_CERT_OPTIMA_JSON))
    if _KNOWN_OPTIMA_TOML.is_file():
        table.update(_parse_known_optima(_KNOWN_OPTIMA_TOML))
    solu = solu_path()
    if solu is not None:
        table.update(_parse_solu(solu))
    return table


def reference_optimum(name: str) -> float | None:
    """Reference objective for ``name``, or None when no source records one.

    Returns None rather than raising: a panel sweeping a corpus must be able to run
    past unknown instances, and it reports them as *unscored* (see
    ``primal_quality.summarize``) rather than skipping them silently.
    """
    entry = oracle_table().get(name)
    return None if entry is None else entry.value


def reference_oracle(name: str) -> Oracle | None:
    """Full :class:`Oracle` record for ``name`` (value + provenance), or None."""
    return oracle_table().get(name)
