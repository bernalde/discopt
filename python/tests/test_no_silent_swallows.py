"""Guard: no exception handler in ``discopt`` may be a pure silent no-op (#864).

``except Exception: pass`` is not a style question — a failure becomes an
invisible no-op, and the *absence* of a capability then reads as a measurement
of it. Issue #864 records three wrong conclusions produced by this shape in a
single session on #844: a swallowed ``TypeError`` (``copy.deepcopy`` on the
PyO3 ``_nl_repr``) disabled a whole fallback, so "the fallback doesn't help"
was an artifact of code that never ran; the engine's root OBBT call could skip
without a trace; and the same shape hid the ``_solve_deadline`` staleness later
fixed in PR #859.

Every swallow must therefore leave *evidence*. This test walks the AST of the
package and fails on any handler whose entire body is a no-op (``pass`` /
``...`` / a bare ``continue`` or ``break``) with nothing logged. The fix is
never to silence the test:

* ``logger.debug("<what was skipped>: %s: %s", type(exc).__name__, exc)`` when
  the failure is genuinely optional (cut separation, telemetry, optional
  imports) — behaviour is unchanged, the skip is merely observable; or
* a narrower ``except (ImportError, OSError, ...)`` when only specific failures
  were ever intended, or when logging itself is unsafe (a forked child must not
  take the logging lock — see ``_daemon_core._DeadlineWatchdog``); or
* an actual handler, when the swallow is hiding something that matters.

Prefer the first: a blind catch is often deliberate robustness (a memoization
store must not break a solve on a model with an exotic ``__setattr__``), and
narrowing it trades an invisible skip for a new crash. The defect this test
targets is the *silence*, not the width of the catch.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "discopt"

# Names whose call marks a handler as leaving evidence. ``warnings.warn`` and
# ``print`` count: the point is observability, not the logging module.
_EVIDENCE_NAMES = frozenset(
    {
        "debug",
        "info",
        "warning",
        "warn",
        "error",
        "exception",
        "critical",
        "log",
        "print",
    }
)

# Handlers that catch these are already narrow enough to be self-documenting:
# the exception type states which failure was anticipated. Only the *blind*
# catches (``Exception``/``BaseException``/bare ``except``) must leave evidence.
_BLIND = frozenset({"Exception", "BaseException"})


def _iter_python_files() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _is_blind(handler: ast.ExceptHandler) -> bool:
    """True when the handler catches everything (bare, ``Exception``, or a tuple with one)."""
    caught = handler.type
    if caught is None:  # bare ``except:``
        return True
    nodes = caught.elts if isinstance(caught, ast.Tuple) else [caught]
    return any(isinstance(n, ast.Name) and n.id in _BLIND for n in nodes)


def _is_noop_body(body: list[ast.stmt]) -> bool:
    """True when the handler body does nothing observable."""
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        # ``...`` as a statement
        if (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and stmt.value.value is Ellipsis
        ):
            continue
        if isinstance(stmt, (ast.Continue, ast.Break)):
            continue
        return False
    return True


def _leaves_evidence(handler: ast.ExceptHandler) -> bool:
    """True when the handler logs, warns, prints, re-raises, or otherwise acts."""
    if not _is_noop_body(handler.body):
        return True
    for node in ast.walk(handler):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name in _EVIDENCE_NAMES:
                return True
    return False


def _silent_swallows(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if not _is_blind(node):
            continue
        if _leaves_evidence(node):
            continue
        caught = "except:" if node.type is None else f"except {ast.unparse(node.type)}:"
        found.append((node.lineno, caught))
    return found


@pytest.mark.smoke
def test_no_silent_exception_swallows() -> None:
    """No blind handler in ``discopt`` may skip work without leaving a trace."""
    offenders: list[str] = []
    for path in _iter_python_files():
        for lineno, caught in _silent_swallows(path):
            rel = path.relative_to(PACKAGE_ROOT.parent)
            offenders.append(f"{rel}:{lineno}: {caught} <no-op>")

    assert not offenders, (
        "Blind exception handlers that skip silently (#864) — a failure here is "
        "invisible, so a disabled capability reads as a measurement of it.\n"
        "Log the exception (logger.debug with type+message), narrow the caught "
        "type, or handle it:\n  " + "\n  ".join(offenders)
    )


def test_guard_detects_a_silent_swallow() -> None:
    """The guard itself must catch the shape it exists to forbid."""
    src = "try:\n    f()\nexcept Exception:\n    pass\n"
    tree = ast.parse(src)
    handler = next(n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler))
    assert _is_blind(handler)
    assert not _leaves_evidence(handler)


@pytest.mark.parametrize(
    "src",
    [
        # logged
        "try:\n    f()\nexcept Exception as e:\n    logger.debug('x: %s', e)\n",
        # narrowed
        "try:\n    f()\nexcept AttributeError:\n    pass\n",
        # re-raised
        "try:\n    f()\nexcept Exception:\n    raise\n",
        # actually handled
        "try:\n    f()\nexcept Exception:\n    x = None\n",
    ],
)
def test_guard_accepts_acceptable_handlers(src: str) -> None:
    tree = ast.parse(src)
    handler = next(n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler))
    assert not (_is_blind(handler) and not _leaves_evidence(handler))
