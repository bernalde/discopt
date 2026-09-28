"""Once-per-solve reporting for documented sound-fallback sites (#1514, #1520).

A handful of solve-time sites keep a broad ``except`` because the failure they
absorb is a genuine external one (a POUNCE IPM call that raises from native
code, a user ``dm.custom`` callable the evaluator cannot differentiate) and
falling back is sound (the node stays open, the bound is not tightened, the
certificate is withheld). What made the old handlers a defect was that they
logged at DEBUG, so a *broken* backend read as a *declining* one. Each such site
reports through :func:`warn_fallback_once`: the first failure per site per solve
is a WARNING carrying the exception, later ones are DEBUG so a per-node failure
cannot flood the log.

This lives under ``_relax`` (no solver imports) so relaxation-layer modules can
report through the same record as ``solver.py`` without a circular import. The
record is scoped per top-level solve by ``solver._scoped_fallback_warnings``.
Messages go to the ``discopt.solver`` logger whichever module reports, so one
logger carries every fallback in a solve.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger("discopt.solver")


class FallbackWarnings(threading.local):
    """Per-thread record of which sound-fallback sites already warned in this solve.

    ``threading.local`` for the same reason as ``solver._CallbackFailures``: two
    solves on two threads keep separate records.
    """

    def __init__(self):
        self.seen: set[str] = set()
        self.active = False


FALLBACK_WARNINGS = FallbackWarnings()


def warn_fallback_once(site: str, exc: BaseException, fallback: str) -> None:
    """Report a failure absorbed at a documented sound-fallback site (#1514).

    ``site`` names the call that failed, ``fallback`` what the solve does
    instead. The first occurrence of ``site`` in a solve is logged at WARNING with
    the exception type and message; repeats are logged at DEBUG.
    """
    if site in FALLBACK_WARNINGS.seen:
        logger.debug("%s failed again (%s: %s); %s", site, type(exc).__name__, exc, fallback)
        return
    FALLBACK_WARNINGS.seen.add(site)
    logger.warning(
        "%s failed (%s: %s); %s. Further failures at this site in this solve are "
        "logged at DEBUG (#1514).",
        site,
        type(exc).__name__,
        exc,
        fallback,
    )
