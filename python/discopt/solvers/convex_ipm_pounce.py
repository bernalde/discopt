"""POUNCE's dedicated convex LP/QP interior-point method (``lp-ipm`` / ``qp-ipm``).

The other POUNCE matrix backends (:mod:`discopt.solvers.lp_pounce`,
:mod:`discopt.solvers.qp_pounce`) hand an LP or QP to POUNCE's general
**NLP** engine — the filter line-search interior-point method — through
LP/QP-shaped callbacks. This module instead calls :func:`pounce.qp.solve_qp`,
the specialised convex solver (Mehrotra predictor-corrector with presolve) that
POUNCE's own ``solver_selection=lp-ipm`` / ``qp-ipm`` reach. It backs
``Model.solve(solver="pounce")`` (#1533), whose purpose is that a pure LP or a
convex QP is answered by that algorithm and by nothing else.

Both entry points follow the shared matrix contracts
(:func:`discopt.solvers.lp_simplex.solve_lp` / ``qp_pounce.solve_qp``): same
signature, same :class:`~discopt.solvers.LPResult` / ``QPResult`` with
HiGHS-convention duals, so ``solver._solve_lp_matrix`` and
``solver._solve_qp_matrix`` apply their primal-feasibility and KKT guards to the
returned point unchanged.

Certificates
------------
POUNCE's convex engine reports ``primal_infeasible`` and ``dual_infeasible``
(unbounded) from its own detection. Neither is turned into a certificate on that
say-so alone, by the same rule the NLP-engine routes follow (#1309, #940):

* ``primal_infeasible`` becomes ``INFEASIBLE`` only when the exact Rust simplex
  (:func:`lp_pounce._simplex_feasibility_verdict`) proves the linear system
  empty; otherwise ``ERROR``.
* ``dual_infeasible`` becomes ``UNBOUNDED`` only when an improving recession ray
  is verified exactly (:func:`lp_pounce._certify_unbounded_ray`) **and** the
  simplex exhibits a feasible point; otherwise ``ERROR``.

Both checks only *withhold* a verdict; neither can produce an answer the IPM did
not, so the route never silently becomes a simplex solve.

Bounds
------
A declared bound at or beyond :func:`lp_pounce.finite_bound_threshold` is handed
to the engine as infinite, exactly as on the other POUNCE routes. That includes
the ``±9.999e19`` box a column declared without bounds receives, which, kept
finite, costs the IPM several times the iterations. The caller passes
``relaxes_huge_bounds=True`` to the matrix route so an ``UNBOUNDED`` verdict over
such a relaxed box is not certified (#850).
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp

from discopt.solvers import LPResult, QPResult, SolveStatus
from discopt.solvers.lp_pounce import (
    _INF,
    PHASE1_FEASIBLE,
    PHASE1_INFEASIBLE,
    POUNCE_AVAILABLE,
    _certify_unbounded_ray,
    _simplex_feasibility_verdict,
    _stack_constraints,
    finite_bound_threshold,
)

#: Options the convex engine understands. Anything else is refused by
#: :func:`convex_engine_options` rather than dropped: the NLP engine's options
#: (``mu_strategy``, ``linear_solver``, ...) name machinery this algorithm does
#: not have, and silently ignoring one would leave the caller believing it was
#: set (the M6 rule ``Model.solve`` applies to unknown keyword names).
CONVEX_OPTION_KEYS: frozenset[str] = frozenset(
    {"tol", "max_iter", "max_wall_time", "print_level", "tau", "tau_max"}
)


class IndefiniteQPError(ValueError):
    """The QP's Hessian is not positive semidefinite, so the convex IPM refuses it.

    POUNCE's own PSD check decides this, before any iteration runs. The
    ``solver="pounce"`` route catches it and solves the QP with the NLP engine
    instead, reporting a local result.
    """


def convex_engine_options(options: Optional[dict]) -> dict:
    """Validate ``options`` for the convex engine and return a copy.

    Raises:
        ValueError: on a key outside :data:`CONVEX_OPTION_KEYS`.
    """
    opts = dict(options or {})
    unknown = sorted(k for k in opts if k not in CONVEX_OPTION_KEYS)
    if unknown:
        raise ValueError(
            f"pounce_options {unknown} are not options of POUNCE's convex LP/QP "
            f"interior-point method (lp-ipm / qp-ipm), which this model was routed "
            f"to. That engine accepts {sorted(CONVEX_OPTION_KEYS)}. Options such as "
            f"mu_strategy or linear_solver belong to the NLP engine, which "
            f"solver='pounce' uses for nonlinear and nonconvex models. Refused rather "
            f"than ignored, so a setting never appears to apply when it does not."
        )
    return opts


def _engine_box(
    bounds: Optional[List[Tuple[float, float]]], n: int, default_lb: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Box with every ``|b| >= finite_bound_threshold()`` mapped to ``±inf``."""
    if bounds is not None:
        if len(bounds) != n:
            raise ValueError(f"bounds has {len(bounds)} entries but c has {n} elements")
        lb = np.array([b[0] for b in bounds], dtype=np.float64)
        ub = np.array([b[1] for b in bounds], dtype=np.float64)
    else:
        lb = np.full(n, default_lb, dtype=np.float64)
        ub = np.full(n, np.inf, dtype=np.float64)
    thr = finite_bound_threshold()
    lb = np.where(lb <= -thr, -np.inf, lb)
    ub = np.where(ub >= thr, np.inf, ub)
    return lb, ub


def _sentinel(v: np.ndarray) -> np.ndarray:
    """``±inf`` -> the ``±1e20`` sentinel the discopt LP helpers expect."""
    return np.asarray(np.clip(v, -_INF, _INF), dtype=np.float64)


def _print_trace(selector: str, res: Any) -> None:
    """The per-iteration convergence trace, in the spirit of Ipopt's log."""
    print(f"POUNCE {selector} (convex interior-point method)")
    print(
        f"{'iter':>4}  {'objective':>14}  {'inf_pr':>9}  {'inf_du':>9}  "
        f"{'mu':>9}  {'alpha_pr':>8}  {'alpha_du':>8}"
    )
    for it in res.iterates:
        print(
            f"{int(it['iter']):>4}  {float(it['objective']):>14.7e}  "
            f"{float(it['primal_infeasibility']):>9.2e}  "
            f"{float(it['dual_infeasibility']):>9.2e}  {float(it['mu']):>9.2e}  "
            f"{float(it['alpha_primal']):>8.2e}  {float(it['alpha_dual']):>8.2e}"
        )
    print(f"status: {res.status}   iterations: {int(res.iters)}   objective: {res.obj!r}")


def _solve(
    P: Optional[np.ndarray],
    c: np.ndarray,
    A_ub,
    b_ub,
    A_eq,
    b_eq,
    lb: np.ndarray,
    ub: np.ndarray,
    time_limit: Optional[float],
    options: Optional[dict],
) -> Tuple[str, Any, float, np.ndarray, np.ndarray, np.ndarray]:
    """Run :func:`pounce.qp.solve_qp` once; map nothing yet.

    Returns ``(raw_status, result, wall, A, cl, cu)`` where ``A, cl, cu`` is the
    stacked row system the certificate checks need.
    """
    from pounce.qp import solve_qp as _pounce_solve_qp

    opts = convex_engine_options(options)
    n = len(c)
    print_level = int(opts.pop("print_level", 0) or 0)
    max_wall = opts.pop("max_wall_time", None)
    budgets = [float(t) for t in (time_limit, max_wall) if t is not None]
    limit = min(budgets) if budgets else None

    G = None if A_ub is None else (A_ub if sp.issparse(A_ub) else np.asarray(A_ub, float))
    A = None if A_eq is None else (A_eq if sp.issparse(A_eq) else np.asarray(A_eq, float))
    h = None if b_ub is None else np.asarray(b_ub, dtype=np.float64).ravel()
    b = None if b_eq is None else np.asarray(b_eq, dtype=np.float64).ravel()

    t0 = time.perf_counter()
    try:
        res = _pounce_solve_qp(
            P=P,
            c=c,
            A=A,
            b=b,
            G=G,
            h=h,
            lb=lb,
            ub=ub,
            tol=opts.get("tol"),
            max_iter=opts.get("max_iter"),
            time_limit=limit,
            collect_iterates=print_level > 0,
            method="ipm",
            tau=opts.get("tau"),
            tau_max=opts.get("tau_max"),
        )
    except ValueError as exc:
        # POUNCE's PSD guard: the convex engine refuses an indefinite P before
        # iterating. Re-raised as a distinct type so the route can tell it from
        # a malformed-input error, which must still propagate.
        if P is not None and "positive semidefinite" in str(exc):
            raise IndefiniteQPError(str(exc)) from exc
        raise
    wall = time.perf_counter() - t0
    if print_level > 0:
        _print_trace("lp-ipm" if P is None else "qp-ipm", res)

    A_rows, cl, cu = _stack_constraints(A_ub, b_ub, A_eq, b_eq, n)
    return res.status, res, wall, A_rows, cl, cu


def _verdict_status(
    raw: str,
    c: np.ndarray,
    A: np.ndarray,
    cl: np.ndarray,
    cu: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    Q: Optional[np.ndarray],
) -> SolveStatus:
    """Map a non-optimal engine status, verifying any certificate it implies."""
    lbs, ubs = _sentinel(lb), _sentinel(ub)
    if raw == "primal_infeasible":
        if _simplex_feasibility_verdict(A, cl, cu, lbs, ubs) == PHASE1_INFEASIBLE:
            return SolveStatus.INFEASIBLE
        return SolveStatus.ERROR
    if raw == "dual_infeasible":
        from discopt.solvers import pounce_option_defaults

        ray = _certify_unbounded_ray(c, A, cl, cu, lbs, ubs, pounce_option_defaults(), Q=Q)
        if ray and _simplex_feasibility_verdict(A, cl, cu, lbs, ubs) == PHASE1_FEASIBLE:
            return SolveStatus.UNBOUNDED
        return SolveStatus.ERROR
    if raw == "time_limit":
        return SolveStatus.TIME_LIMIT
    if raw in ("iteration_limit", "optimal_inaccurate"):
        # ``optimal_inaccurate`` met only the engine's relaxed tolerance; it is not
        # reported as ``optimal`` (the matrix route would certify it).
        return SolveStatus.ITERATION_LIMIT
    return SolveStatus.ERROR


def _kkt_parts(res: Any, n_ub: int) -> Tuple[np.ndarray, np.ndarray]:
    """HiGHS-convention row duals and reduced costs from a POUNCE ``QpResult``.

    POUNCE's Lagrangian is ``½x'Px + c'x + y'(Ax-b) + z'(Gx-h) - z_lb'(x-lb) +
    z_ub'(x-ub)``; HiGHS reports ``∂obj/∂rhs``, which is ``-z`` for the ``G``
    (``A_ub``) rows and ``-y`` for the equality rows, stacked inequality rows
    first. Reduced costs are ``z_lb - z_ub`` (``c + P x - A'y_h``).
    """
    z = np.asarray(res.z, dtype=np.float64).ravel()[:n_ub]
    y = np.asarray(res.y, dtype=np.float64).ravel()
    dual = -np.concatenate([z, y])
    rc = np.asarray(res.z_lb, dtype=np.float64) - np.asarray(res.z_ub, dtype=np.float64)
    return dual, rc


def solve_lp(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    warm_basis: Optional[object] = None,
    time_limit: Optional[float] = None,
    options: Optional[dict] = None,
) -> LPResult:
    """Solve ``min c'x`` s.t. ``A_ub x <= b_ub``, ``A_eq x = b_eq`` with POUNCE's lp-ipm.

    ``bounds`` default to ``(0, +inf)`` (the shared LP contract). ``warm_basis``
    is accepted for signature compatibility and ignored: an IPM has no basis.
    ``options`` must be a subset of :data:`CONVEX_OPTION_KEYS`.
    """
    del warm_basis
    if not POUNCE_AVAILABLE:
        raise ImportError("pounce is required. Install it with:\n  pip install pounce-solver")
    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = len(c_arr)
    lb, ub = _engine_box(bounds, n, default_lb=0.0)
    raw, res, wall, A, cl, cu = _solve(
        None, c_arr, A_ub, b_ub, A_eq, b_eq, lb, ub, time_limit, options
    )
    iters = int(res.iters)
    if raw != "optimal":
        status = _verdict_status(raw, c_arr, A, cl, cu, lb, ub, None)
        return LPResult(
            status=status,
            iterations=iters,
            wall_time=wall,
            ray_verified=True if status == SolveStatus.UNBOUNDED else None,
        )
    n_ub = 0 if A_ub is None else int(A_ub.shape[0])
    dual, rc = _kkt_parts(res, n_ub)
    x = np.asarray(res.x, dtype=np.float64)
    return LPResult(
        status=SolveStatus.OPTIMAL,
        x=x,
        objective=float(c_arr @ x),
        dual_values=dual,
        reduced_costs=rc,
        rc_absum=np.abs(np.asarray(res.z_lb)) + np.abs(np.asarray(res.z_ub)),
        iterations=iters,
        wall_time=wall,
    )


def solve_qp(
    Q: np.ndarray,
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_ub: Optional[np.ndarray] = None,
    A_eq: Optional[Union[np.ndarray, sp.spmatrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    bounds: Optional[List[Tuple[float, float]]] = None,
    integrality: Optional[np.ndarray] = None,
    time_limit: Optional[float] = None,
    gap_tolerance: float = 1e-4,
    options: Optional[dict] = None,
) -> QPResult:
    """Solve ``min ½x'Qx + c'x`` s.t. linear rows with POUNCE's qp-ipm.

    ``bounds`` default to free variables (the shared QP contract).

    Raises:
        IndefiniteQPError: ``Q`` is not PSD (POUNCE's own check).
        ValueError: any integer-marked variable, or an unknown option.
    """
    del gap_tolerance  # a continuous QP has no gap to close
    if not POUNCE_AVAILABLE:
        raise ImportError("pounce is required. Install it with:\n  pip install pounce-solver")
    if integrality is not None and np.any(np.asarray(integrality) == 1):
        raise ValueError("POUNCE's convex QP IPM is continuous; integrality is not supported.")
    Q_arr = np.asarray(Q, dtype=np.float64)
    c_arr = np.asarray(c, dtype=np.float64).ravel()
    n = len(c_arr)
    if Q_arr.shape != (n, n):
        raise ValueError(f"Q has shape {Q_arr.shape} but c has {n} elements")
    lb, ub = _engine_box(bounds, n, default_lb=-np.inf)
    raw, res, wall, A, cl, cu = _solve(
        Q_arr, c_arr, A_ub, b_ub, A_eq, b_eq, lb, ub, time_limit, options
    )
    iters = int(res.iters)
    if raw != "optimal":
        return QPResult(
            status=_verdict_status(raw, c_arr, A, cl, cu, lb, ub, Q_arr),
            iterations=iters,
            wall_time=wall,
        )
    n_ub = 0 if A_ub is None else int(A_ub.shape[0])
    dual, rc = _kkt_parts(res, n_ub)
    x = np.asarray(res.x, dtype=np.float64)
    return QPResult(
        status=SolveStatus.OPTIMAL,
        x=x,
        objective=float(0.5 * x @ Q_arr @ x + c_arr @ x),
        dual_values=dual,
        reduced_costs=rc,
        iterations=iters,
        wall_time=wall,
        kkt_error=res.kkt_error,
    )
