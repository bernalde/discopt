//! Persistent in-tree bound tightening (B3 of issue #51).
//!
//! Runs lightweight FBBT on a B&B node's local bounds, returning the
//! tightened intervals. Tightenings persist by virtue of the B&B
//! contract: any child node inherits its parent's bounds, so a
//! tightening applied at a node automatically propagates to its
//! subtree.
//!
//! ## Why this is cheap
//!
//! The expression DAG and constraint structure are identical at every
//! node — only the variable bounds change. So FBBT at a child node
//! re-uses all of the topology and only re-evaluates intervals on
//! shifted leaves. The marginal work per node is proportional to the
//! number of variables that *changed* relative to the parent (in
//! principle); this kernel runs the full pass for now and leaves the
//! incremental optimisation to a follow-up.
//!
//! ## Scheduling
//!
//! In-tree FBBT is gated by [`InTreePresolveOptions::depth_stride`] —
//! the pass runs only when `node_depth % depth_stride == 0`, so the
//! caller can amortise the cost over the tree without paying it at
//! every node. `depth_stride = 1` runs at every node;
//! `depth_stride = 0` disables the pass.

use crate::expr::{ExprArena, ExprId, ExprNode, IndexElem, IndexSpec, ModelRepr, VarInfo, VarType};
use crate::presolve::fbbt::{
    any_empty_beyond, fbbt_with_cutoff, repair_subtol_crossings, Interval, FEAS_TOL,
};
use crate::presolve::probing::probe_node_bounds;

/// Options controlling persistent in-tree bound tightening.
#[derive(Debug, Clone)]
pub struct InTreePresolveOptions {
    /// Run the pass at every `depth_stride`-th tree depth. `0` disables
    /// the pass entirely; `1` runs at every node.
    pub depth_stride: u32,
    /// FBBT inner-loop iteration cap.
    pub max_iter: usize,
    /// FBBT inner-loop convergence tolerance.
    pub tol: f64,
    /// Run per-node probing (P3 branch-and-reduce) after FBBT. Probing
    /// tentatively fixes each discrete variable at a bound and re-runs FBBT,
    /// contracting the domain on any proven-infeasible fixing. Off by default
    /// (it costs O(discrete) extra FBBT solves per node); sound when on.
    pub probing: bool,
    /// Cap on the number of discrete variables probed per node (budget).
    pub probe_max_vars: usize,
}

impl Default for InTreePresolveOptions {
    fn default() -> Self {
        Self {
            depth_stride: 4,
            max_iter: 8,
            tol: 1e-6,
            probing: false,
            probe_max_vars: 32,
        }
    }
}

/// Per-node tightening result.
#[derive(Debug, Clone, Default)]
pub struct InTreeDelta {
    /// Tightened lower bounds (one per variable).
    pub lb: Vec<f64>,
    /// Tightened upper bounds (one per variable).
    pub ub: Vec<f64>,
    /// Number of variables whose bounds tightened (either side).
    pub bounds_tightened: u32,
    /// True if the kernel detected infeasibility (empty interval).
    pub infeasible: bool,
    /// How many sub-`FEAS_TOL` bound crossings were repaired (#907).
    ///
    /// Surfaced rather than absorbed: a rising count is a numerical smell, and
    /// before #907 each of these events could have fathomed a live node.
    pub subtol_repaired: usize,
    /// True iff the schedule actually ran the pass at this node.
    pub ran: bool,
}

/// Run in-tree FBBT at a node with the given local bounds.
///
/// `model` is the **root** model (variable bounds inside it are
/// ignored — `node_lb`/`node_ub` override them). Returns an
/// [`InTreeDelta`] containing the post-tightening bounds.
///
/// `incumbent` is forwarded verbatim to [`fbbt_with_cutoff`] and
/// [`probe_node_bounds`], so it is in the **model's own objective space** —
/// `f(x_inc)`, positive-as-written for a maximize — and NOT the caller's
/// internal minimization space. See `fbbt_with_cutoff` for what the mismatch
/// costs: an emptied box that the caller consumes as a rigorous fathom, hence a
/// false `optimal` (issue #1373).
///
/// The pass is a no-op (returns `ran = false`, copies `node_lb` /
/// `node_ub` unchanged) when the schedule says to skip this depth.
pub fn run_in_tree_presolve(
    model: &ModelRepr,
    node_lb: &[f64],
    node_ub: &[f64],
    node_depth: usize,
    incumbent: Option<f64>,
    opts: &InTreePresolveOptions,
) -> InTreeDelta {
    assert_eq!(node_lb.len(), model.variables.len());
    assert_eq!(node_ub.len(), model.variables.len());

    if opts.depth_stride == 0 || (node_depth as u32) % opts.depth_stride != 0 {
        return InTreeDelta {
            lb: node_lb.to_vec(),
            ub: node_ub.to_vec(),
            bounds_tightened: 0,
            infeasible: false,
            subtol_repaired: 0,
            ran: false,
        };
    }

    // #907. Sanitize the INCOMING node box before anything reads it. A caller
    // upstream (or an earlier `in_tree_presolve` on a parent node) may hand us a
    // box already inverted by rounding noise; patching that straight onto
    // `VarInfo` would seed FBBT from an inverted domain and manufacture the very
    // emptiness we are trying not to over-read.
    let mut node_box: Vec<Interval> = (0..node_lb.len())
        .map(|i| Interval::new(node_lb[i], node_ub[i]))
        .collect();
    let mut subtol_repaired = repair_subtol_crossings(&mut node_box, FEAS_TOL);

    // Patch the model's variable bounds with the node-local bounds.
    // We clone only the lightweight `variables` Vec, not the arena.
    let mut patched = model.clone();
    for (i, vinfo) in patched.variables.iter_mut().enumerate() {
        if !vinfo.lb.is_empty() {
            vinfo.lb[0] = node_box[i].lo;
        }
        if !vinfo.ub.is_empty() {
            vinfo.ub[0] = node_box[i].hi;
        }
    }

    // #907. `infeasible` is consumed by the B&B loop as a RIGOROUS FATHOM — the
    // subtree is pruned outright — so it must never be set by floating-point
    // noise. Repair sub-`FEAS_TOL` crossings, then conclude infeasibility only
    // beyond the tolerance, exactly as `fbbt`, `fbbt_fp` and `probing` do.
    //
    // Loosening a fathom is the SOUND direction: the node is explored rather
    // than discarded. Genuine detections are unaffected — corpus instrumentation
    // found every real fathom carried either the `[inf, -inf]` empty sentinel or
    // a crossing of exactly 1.0 (binary domain wipeout), 6+ orders above
    // `FEAS_TOL`.
    let mut bounds: Vec<Interval> = fbbt_with_cutoff(&patched, opts.max_iter, opts.tol, incumbent);
    subtol_repaired += repair_subtol_crossings(&mut bounds, FEAS_TOL);
    let mut infeasible = any_empty_beyond(&bounds, FEAS_TOL);

    let mut new_lb: Vec<f64> = node_box.iter().map(|b| b.lo).collect();
    let mut new_ub: Vec<f64> = node_box.iter().map(|b| b.hi).collect();
    let mut tightened = 0u32;
    if !infeasible {
        for i in 0..bounds.len() {
            let iv = bounds[i];
            // Floor with the node's bounds — never relax.
            if iv.lo > new_lb[i] + opts.tol {
                new_lb[i] = iv.lo;
                tightened += 1;
            }
            if iv.hi < new_ub[i] - opts.tol {
                new_ub[i] = iv.hi;
                tightened += 1;
            }
        }
    }

    // P3 probing pass: contract discrete-variable domains by tentatively fixing
    // each at a bound and re-running FBBT (proven-infeasible fixings only).
    // Runs on the FBBT-tightened box; folds its (subset) result back, never
    // loosening. `patched` carries the node bounds; probing re-seeds fully from
    // the explicit interval box, so the two boxes agree.
    if opts.probing && !infeasible {
        let node_box: Vec<Interval> = (0..new_lb.len())
            .map(|i| Interval::new(new_lb[i], new_ub[i]))
            .collect();
        let pr = probe_node_bounds(
            &patched,
            &node_box,
            opts.probe_max_vars,
            opts.max_iter,
            opts.tol,
            incumbent,
            None,
        );
        if pr.infeasible {
            infeasible = true;
        } else {
            for i in 0..pr.tightened_bounds.len().min(new_lb.len()) {
                let iv = pr.tightened_bounds[i];
                if iv.lo > new_lb[i] + opts.tol {
                    new_lb[i] = iv.lo;
                    tightened += 1;
                }
                if iv.hi < new_ub[i] - opts.tol {
                    new_ub[i] = iv.hi;
                    tightened += 1;
                }
                // #907. NO infeasibility verdict here. This loop used to test
                // `new_lb[i] > new_ub[i] + opts.tol`, but `opts.tol` is the FBBT
                // *convergence* tolerance — independently settable and smaller
                // than `FEAS_TOL` in practice — so a crossing in
                // `(opts.tol, FEAS_TOL]` set the rigorous-fathom flag on exactly
                // the noise this fix exists to tolerate. The single exit below
                // repairs and then decides, at `FEAS_TOL`, for every path.
            }
        }
    }

    // #907. Final sanitation at the single exit. The probing branch above gates
    // its own emptiness test on `opts.tol` (a different, smaller tolerance), so it
    // can fold back a box that is inverted by a sub-`FEAS_TOL` amount WITHOUT
    // setting `infeasible`. Returning that inverted box would push `lo > hi` onto
    // an LP column bound downstream, reproducing the false infeasibility one layer
    // down — declining to *declare* emptiness is not enough on its own.
    let mut out: Vec<Interval> = (0..new_lb.len())
        .map(|i| Interval::new(new_lb[i], new_ub[i]))
        .collect();
    subtol_repaired += repair_subtol_crossings(&mut out, FEAS_TOL);
    if !infeasible && any_empty_beyond(&out, FEAS_TOL) {
        infeasible = true;
    }
    for (i, b) in out.iter().enumerate() {
        new_lb[i] = b.lo;
        new_ub[i] = b.hi;
    }
    debug_assert!(
        infeasible || new_lb.iter().zip(&new_ub).all(|(l, u)| l <= u),
        "#907: in_tree_presolve returned an inverted box without declaring infeasible"
    );

    InTreeDelta {
        lb: new_lb,
        ub: new_ub,
        bounds_tightened: tightened,
        infeasible,
        subtol_repaired,
        ran: true,
    }
}

// ─────────────────────────────────────────────────────────────
// Per-scalar node boxes (issue #1513)
// ─────────────────────────────────────────────────────────────
//
// `run_in_tree_presolve` takes one interval per variable BLOCK, but every B&B
// node box is one interval per SCALAR. For a model whose blocks are all size 1
// the two coincide; for a model with a `shape=(n,)` variable they do not, and
// both Python node loops used to skip the kernel on that length mismatch --
// silently -- so array models never saw in-tree FBBT, cutoff FBBT or
// branch-and-reduce.
//
// The per-block kernel cannot simply be handed a per-scalar box: FBBT seeds an
// array block from the hull of its elements, reads `x[i]` as that hull, and
// never tightens a size>1 block (`backward_propagate` writes only size-1
// `Variable` nodes). So the fix hands FBBT a model in which every *element* is
// its own size-1 block: [`ScalarFbbtView`].

/// A per-scalar rewrite of a [`ModelRepr`] for FBBT (issue #1513).
///
/// * Variable slot `j < n_scalar` is the model's flat scalar `j` (block `b`,
///   element `k` maps to `offset_b + k`), a size-1 block.
/// * `x[i]` / `x[i, j]` on an array variable -- an `Index` whose spec selects
///   exactly one element on every axis -- becomes a size-1 `Variable` node on
///   that scalar slot, so FBBT reads and tightens the element itself.
/// * A reference that stays array-valued (the whole array in `sum(x)` or
///   `A @ x`, a slice `x[1:3]`, a partial index `X[i]` of a matrix) points at a
///   **proxy** block, one per array variable so used, appended after the
///   scalars. A proxy is seeded with the HULL of its elements' node bounds,
///   keeps `size > 1` (so FBBT never tightens it) and is typed continuous (so
///   probing never fixes it: fixing a proxy would fix every element at once,
///   which proves nothing about any single element). That is exactly what the
///   per-block kernel did for the whole block, so array-valued uses lose
///   nothing relative to it.
///
/// Node ids are preserved one-for-one (each node is rewritten in place), so the
/// objective and constraint bodies need no remapping and the per-node shapes
/// that `shapes_of` derives are unchanged (a full-arity element `Index` and a
/// size-1 `Variable` are both rank 0).
///
/// Soundness: the view denotes the same function of the same scalars. A proxy
/// is a sound OUTER enclosure of every element it stands for at the start of
/// the pass; the elements only tighten during FBBT, so the stale hull stays a
/// superset. The kernel's result is a subset of the node box, as before.
#[derive(Debug, Clone)]
pub struct ScalarFbbtView {
    model: ModelRepr,
    n_scalar: usize,
    /// `(view variable index, flat offset, size)` per proxy block.
    proxies: Vec<(usize, usize, usize)>,
}

impl ScalarFbbtView {
    /// Number of scalar slots (the length of a node box).
    pub fn n_scalar(&self) -> usize {
        self.n_scalar
    }

    /// Number of proxy blocks (array variables referenced as arrays).
    pub fn n_proxies(&self) -> usize {
        self.proxies.len()
    }

    /// The rewritten model (scalar blocks, then proxies).
    pub fn model(&self) -> &ModelRepr {
        &self.model
    }
}

/// True when every block is a single scalar, i.e. the per-block and per-scalar
/// layouts are the SAME vector (block `i` is scalar `i`). On such a model
/// [`run_in_tree_presolve_scalar`] calls [`run_in_tree_presolve`] on the model
/// itself, so it is identical to the per-block kernel by construction (the
/// bound-neutral argument for every scalar-only model, e.g. anything parsed
/// from `.nl`). `offset` is not consulted: the per-block kernel never reads it,
/// and the pre-#1513 callers accepted exactly the models this accepts.
pub fn is_scalar_layout(model: &ModelRepr) -> bool {
    model.n_vars == model.variables.len() && model.variables.iter().all(|v| v.size == 1)
}

/// Flat row-major index of an `Index` spec that selects exactly one element on
/// every axis of `shape`; `None` for anything array-valued or out of range.
fn single_element_flat(spec: &IndexSpec, shape: &[usize]) -> Option<usize> {
    let idx: Vec<usize> = match spec {
        IndexSpec::Scalar(i) => vec![*i],
        IndexSpec::Tuple(v) => v.clone(),
        IndexSpec::Multi(elems) => {
            let mut v = Vec::with_capacity(elems.len());
            for e in elems {
                match e {
                    IndexElem::Scalar(i) => v.push(*i),
                    // A newaxis makes the result array-shaped; an ellipsis is
                    // resolved against the base rank elsewhere (#1516). Both
                    // take the conservative array-reference (proxy) route.
                    IndexElem::Slice { .. } | IndexElem::NewAxis | IndexElem::Ellipsis => {
                        return None
                    }
                }
            }
            v
        }
    };
    // Numpy semantics: an index of lower arity than the rank keeps the trailing
    // axes, i.e. it is array-valued. Only a full-arity index is one element.
    if shape.is_empty() || idx.len() != shape.len() {
        return None;
    }
    let mut flat = 0usize;
    for (i, d) in idx.iter().zip(shape) {
        if i >= d {
            return None;
        }
        flat = flat * d + i;
    }
    Some(flat)
}

/// Build the per-scalar view of `model` (issue #1513).
///
/// Refuses -- `Err` naming the reason, never a guess -- when the variable
/// layout is not the contiguous `offset_b = sum_{c<b} size_c` layout a node box
/// is indexed by, or when a `Variable` node disagrees with its block.
pub fn scalarize_for_fbbt(model: &ModelRepr) -> Result<ScalarFbbtView, String> {
    let mut run = 0usize;
    for (b, v) in model.variables.iter().enumerate() {
        if v.offset != run {
            return Err(format!(
                "variable block {b} ('{}') has offset {} but the contiguous layout puts it at {run}",
                v.name, v.offset
            ));
        }
        if v.lb.len() != v.size || v.ub.len() != v.size {
            return Err(format!(
                "variable block {b} ('{}') has size {} but {} lower / {} upper bounds",
                v.name,
                v.size,
                v.lb.len(),
                v.ub.len()
            ));
        }
        run += v.size;
    }
    if run != model.n_vars {
        return Err(format!(
            "variable blocks cover {run} scalars but the model declares n_vars = {}",
            model.n_vars
        ));
    }
    let n_scalar = run;

    let mut variables: Vec<VarInfo> = Vec::with_capacity(n_scalar);
    for v in &model.variables {
        for k in 0..v.size {
            variables.push(VarInfo {
                name: if v.size == 1 {
                    v.name.clone()
                } else {
                    format!("{}[{k}]", v.name)
                },
                var_type: v.var_type,
                offset: v.offset + k,
                size: 1,
                shape: vec![],
                lb: vec![v.lb[k]],
                ub: vec![v.ub[k]],
            });
        }
    }
    let mut proxy_of: Vec<Option<usize>> = vec![None; model.variables.len()];
    let mut proxies: Vec<(usize, usize, usize)> = Vec::new();

    let n_nodes = model.arena.len();
    let mut arena = ExprArena::with_capacity(n_nodes);
    for id in 0..n_nodes {
        let node = model.arena.get(ExprId(id));
        let rewritten = match node {
            ExprNode::Variable {
                name,
                index,
                size,
                shape,
            } => {
                let blk = model.variables.get(*index).ok_or_else(|| {
                    format!("Variable node {id} ('{name}') references missing block {index}")
                })?;
                if blk.size != *size {
                    return Err(format!(
                        "Variable node {id} ('{name}') has size {size}, block {index} has {}",
                        blk.size
                    ));
                }
                if *size == 1 {
                    ExprNode::Variable {
                        name: name.clone(),
                        index: blk.offset,
                        size: 1,
                        shape: shape.clone(),
                    }
                } else {
                    let p = match proxy_of[*index] {
                        Some(p) => p,
                        None => {
                            let p = variables.len();
                            let lo = blk.lb.iter().copied().fold(f64::INFINITY, f64::min);
                            let hi = blk.ub.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            variables.push(VarInfo {
                                name: blk.name.clone(),
                                // Never probed -- see the struct docs.
                                var_type: VarType::Continuous,
                                offset: blk.offset,
                                size: blk.size,
                                shape: blk.shape.clone(),
                                // One hull interval; the node patch overwrites it.
                                lb: vec![lo],
                                ub: vec![hi],
                            });
                            proxies.push((p, blk.offset, blk.size));
                            proxy_of[*index] = Some(p);
                            p
                        }
                    };
                    ExprNode::Variable {
                        name: name.clone(),
                        index: p,
                        size: *size,
                        shape: shape.clone(),
                    }
                }
            }
            ExprNode::Index { base, index } => {
                let element = match model.arena.get(*base) {
                    ExprNode::Variable {
                        name,
                        index: vi,
                        size,
                        shape,
                    } if *size > 1 => single_element_flat(index, shape).and_then(|f| {
                        let blk = model.variables.get(*vi)?;
                        (f < blk.size).then(|| ExprNode::Variable {
                            name: format!("{name}[{f}]"),
                            index: blk.offset + f,
                            size: 1,
                            shape: vec![],
                        })
                    }),
                    _ => None,
                };
                element.unwrap_or_else(|| node.clone())
            }
            other => other.clone(),
        };
        arena.add(rewritten);
    }

    Ok(ScalarFbbtView {
        model: ModelRepr {
            arena,
            objective: model.objective,
            objective_sense: model.objective_sense,
            constraints: model.constraints.clone(),
            n_vars: variables.len(),
            variables,
        },
        n_scalar,
        proxies,
    })
}

/// [`run_in_tree_presolve`] on a PER-SCALAR node box (issue #1513).
///
/// `node_lb` / `node_ub` have one entry per scalar variable (`model.n_vars`),
/// exactly a B&B node box, and so does the returned delta. On a scalar-layout
/// model ([`is_scalar_layout`]) this IS `run_in_tree_presolve(model, ..)`; on
/// any other it runs the same kernel on the [`ScalarFbbtView`].
///
/// Errors -- never a silent skip -- when the box length is not `model.n_vars`
/// or the model cannot be scalarized.
pub fn run_in_tree_presolve_scalar(
    model: &ModelRepr,
    node_lb: &[f64],
    node_ub: &[f64],
    node_depth: usize,
    incumbent: Option<f64>,
    opts: &InTreePresolveOptions,
) -> Result<InTreeDelta, String> {
    if node_lb.len() != model.n_vars || node_ub.len() != model.n_vars {
        return Err(format!(
            "node box has {} lower / {} upper bounds but the model has {} scalar variables",
            node_lb.len(),
            node_ub.len(),
            model.n_vars
        ));
    }
    if is_scalar_layout(model) {
        return Ok(run_in_tree_presolve(
            model, node_lb, node_ub, node_depth, incumbent, opts,
        ));
    }
    let view = scalarize_for_fbbt(model)?;
    Ok(run_in_tree_presolve_view(
        &view, node_lb, node_ub, node_depth, incumbent, opts,
    ))
}

/// Run the kernel on a prebuilt [`ScalarFbbtView`] (per-scalar box in and out).
pub fn run_in_tree_presolve_view(
    view: &ScalarFbbtView,
    node_lb: &[f64],
    node_ub: &[f64],
    node_depth: usize,
    incumbent: Option<f64>,
    opts: &InTreePresolveOptions,
) -> InTreeDelta {
    let n = view.n_scalar;
    assert_eq!(node_lb.len(), n);
    assert_eq!(node_ub.len(), n);
    let mut full_lb = node_lb.to_vec();
    let mut full_ub = node_ub.to_vec();
    // Proxy seeds: the hull of the node box over the proxy's elements.
    for &(_, off, size) in &view.proxies {
        let lo = node_lb[off..off + size]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let hi = node_ub[off..off + size]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        full_lb.push(lo);
        full_ub.push(hi);
    }
    debug_assert_eq!(full_lb.len(), view.model.variables.len());
    let mut d = run_in_tree_presolve(&view.model, &full_lb, &full_ub, node_depth, incumbent, opts);
    d.lb.truncate(n);
    d.ub.truncate(n);
    if d.ran && !d.infeasible {
        // Count scalar half-bounds only, so the count cannot include a proxy.
        let mut t = 0u32;
        for i in 0..n {
            if d.lb[i] > node_lb[i] + opts.tol {
                t += 1;
            }
            if d.ub[i] < node_ub[i] - opts.tol {
                t += 1;
            }
        }
        d.bounds_tightened = t;
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{
        BinOp, ConstraintRepr, ConstraintSense, ExprArena, ExprId, ExprNode, ModelRepr,
        ObjectiveSense, VarInfo, VarType,
    };

    fn scalar_var(arena: &mut ExprArena, name: &str, idx: usize) -> ExprId {
        arena.add(ExprNode::Variable {
            name: name.to_string(),
            index: idx,
            size: 1,
            shape: vec![],
        })
    }

    fn vinfo(name: &str, lb: f64, ub: f64) -> VarInfo {
        VarInfo {
            name: name.to_string(),
            var_type: VarType::Continuous,
            offset: 0,
            size: 1,
            shape: vec![],
            lb: vec![lb],
            ub: vec![ub],
        }
    }

    fn x_plus_y_le_5() -> ModelRepr {
        // x + y <= 5, x ∈ [0, 10], y ∈ [0, 10], min x+y
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let y = scalar_var(&mut arena, "y", 1);
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x,
            right: y,
        });
        ModelRepr {
            arena,
            objective: body,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 5.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0.0, 10.0), vinfo("y", 0.0, 10.0)],
            n_vars: 2,
        }
    }

    #[test]
    fn tightens_at_node_with_branching_bound() {
        // Branch: x ∈ [3, 10] in the node. The constraint x+y≤5 then
        // forces y ≤ 2.
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            ..Default::default()
        };
        let delta = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 10.0], 1, None, &opts);
        assert!(delta.ran);
        assert!(!delta.infeasible);
        assert!(delta.bounds_tightened >= 1);
        assert!((delta.ub[1] - 2.0).abs() <= 1e-6);
        // Lower bounds are not relaxed.
        assert_eq!(delta.lb[0], 3.0);
    }

    #[test]
    fn infers_indicator_binary_at_node() {
        // Guard x ≤ 10·b, x ∈ [0, 10], b binary. At a node where branching has
        // tightened x to [3, 10], FBBT infers b ≥ 0.3 and snaps it to b = 1 —
        // per-node indicator propagation (issue #230). This is the integration
        // the root-only probing pass cannot deliver inside the tree.
        let mut arena = ExprArena::new();
        let x = scalar_var(&mut arena, "x", 0);
        let b = scalar_var(&mut arena, "b", 1);
        let m = arena.add(ExprNode::Constant(10.0));
        let mb = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: m,
            right: b,
        });
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Sub,
            left: x,
            right: mb,
        });
        let mut bvar = vinfo("b", 0.0, 1.0);
        bvar.var_type = VarType::Binary;
        let model = ModelRepr {
            arena,
            objective: x,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 0.0,
                name: None,
            }],
            variables: vec![vinfo("x", 0.0, 10.0), bvar],
            n_vars: 2,
        };
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            ..Default::default()
        };
        let delta = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 1.0], 1, None, &opts);
        assert!(delta.ran);
        assert!(!delta.infeasible);
        assert!(
            (delta.lb[1] - 1.0).abs() <= 1e-6,
            "binary should be fixed to 1 at the node, got [{}, {}]",
            delta.lb[1],
            delta.ub[1]
        );
    }

    #[test]
    fn skips_when_depth_stride_zero() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 0,
            ..Default::default()
        };
        let delta = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 10.0], 1, None, &opts);
        assert!(!delta.ran);
        assert_eq!(delta.bounds_tightened, 0);
        assert_eq!(delta.lb, vec![3.0, 0.0]);
        assert_eq!(delta.ub, vec![10.0, 10.0]);
    }

    #[test]
    fn skips_off_schedule_depths() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 4,
            ..Default::default()
        };
        // depth=1 is not a multiple of 4 ⇒ skipped.
        let d = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 10.0], 1, None, &opts);
        assert!(!d.ran);
        // depth=4 ⇒ runs.
        let d4 = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 10.0], 4, None, &opts);
        assert!(d4.ran);
        assert!(d4.bounds_tightened >= 1);
    }

    #[test]
    fn detects_infeasibility() {
        // Branch: x ∈ [10, 10] AND y ∈ [10, 10]. x+y=20 > 5 — infeasible.
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            ..Default::default()
        };
        let delta = run_in_tree_presolve(&model, &[10.0, 10.0], &[10.0, 10.0], 1, None, &opts);
        assert!(delta.ran);
        assert!(delta.infeasible);
    }

    #[test]
    fn never_relaxes_input_bounds() {
        // Bounds tighter than what FBBT alone would derive must be kept.
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            ..Default::default()
        };
        // Caller-supplied tighter ub on x.
        let delta = run_in_tree_presolve(&model, &[0.0, 0.0], &[1.0, 10.0], 0, None, &opts);
        assert!(delta.ran);
        // The ub on x must remain at 1.0 (or tighter), never relax to 5.
        assert!(delta.ub[0] <= 1.0 + 1e-9);
    }

    #[test]
    fn probing_fixes_binary_at_node() {
        // x ≤ 10·b, x ∈ [0,10], b binary; node branch x ∈ [3,10] ⇒ b = 1.
        // The probing pass (opts.probing = true) must fix b to 1 at the node.
        let model = {
            let mut arena = ExprArena::new();
            let x = scalar_var(&mut arena, "x", 0);
            let b = scalar_var(&mut arena, "b", 1);
            let m = arena.add(ExprNode::Constant(10.0));
            let mb = arena.add(ExprNode::BinaryOp {
                op: BinOp::Mul,
                left: m,
                right: b,
            });
            let body = arena.add(ExprNode::BinaryOp {
                op: BinOp::Sub,
                left: x,
                right: mb,
            });
            let mut bvar = vinfo("b", 0.0, 1.0);
            bvar.var_type = VarType::Binary;
            ModelRepr {
                arena,
                objective: x,
                objective_sense: ObjectiveSense::Minimize,
                constraints: vec![ConstraintRepr {
                    body,
                    sense: ConstraintSense::Le,
                    rhs: 0.0,
                    name: None,
                }],
                variables: vec![vinfo("x", 0.0, 10.0), bvar],
                n_vars: 2,
            }
        };
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            probing: true,
            probe_max_vars: 32,
        };
        let delta = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 1.0], 1, None, &opts);
        assert!(delta.ran);
        assert!(!delta.infeasible);
        assert!(
            (delta.lb[1] - 1.0).abs() <= 1e-6,
            "b should be fixed to 1 at the node, got [{}, {}]",
            delta.lb[1],
            delta.ub[1]
        );
    }

    #[test]
    fn probing_off_by_default_is_byte_neutral() {
        // With probing disabled (default), the delta matches the FBBT-only path.
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            ..Default::default()
        };
        assert!(!opts.probing);
        let delta = run_in_tree_presolve(&model, &[3.0, 0.0], &[10.0, 10.0], 1, None, &opts);
        assert!(delta.ran);
        assert!(!delta.infeasible);
        assert!((delta.ub[1] - 2.0).abs() <= 1e-6);
    }

    // ── #907: a sub-FEAS_TOL crossing must not FATHOM a node ─────────────
    //
    // `InTreeDelta::infeasible` is consumed by the B&B loop as a *rigorous
    // fathom*: the subtree is pruned outright. Setting it from a crossing of
    // 8.5e-14 discards a region that may contain feasible points, violating the
    // zero-slack `incorrect_count <= 0` gate with no flag set.

    /// A node box inverted by rounding noise must be explored, not fathomed,
    /// and must not be returned inverted.
    #[test]
    fn subtol_inverted_node_box_is_not_fathomed() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            probing: false,
            probe_max_vars: 0,
        };
        // x fixed at 2.5 by two derivations disagreeing in the last ulps.
        let lo = [2.5, 0.0];
        let hi = [2.5 - 1e-14, 10.0];
        let d = run_in_tree_presolve(&model, &lo, &hi, 0, None, &opts);

        assert!(d.ran);
        assert!(
            !d.infeasible,
            "a 1e-14 crossing fathomed a live node — #907 regressed"
        );
        assert_eq!(d.subtol_repaired, 1);
        // The returned box must be well-formed: an inverted interval reaching an
        // LP column bound reproduces the false infeasibility one layer down.
        for i in 0..d.lb.len() {
            assert!(
                d.lb[i] <= d.ub[i],
                "returned an inverted box at var{i}: [{}, {}]",
                d.lb[i],
                d.ub[i]
            );
        }
        // Repair widens to contain both endpoints, so the feasible point x=2.5
        // survives.
        assert!(d.lb[0] <= 2.5 && 2.5 <= d.ub[0]);
    }

    /// ANTI-PERMISSIVENESS CONTROL: a genuinely empty node box must STILL
    /// fathom. Without this the change is a tolerance-tweak, not a fix.
    #[test]
    fn genuine_empty_node_box_still_fathoms() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            probing: false,
            probe_max_vars: 0,
        };
        // x >= 10 AND y >= 10 with x + y <= 5 — infeasible by 15, not by noise.
        let d = run_in_tree_presolve(&model, &[10.0, 10.0], &[10.0, 10.0], 0, None, &opts);
        assert!(d.ran);
        assert!(d.infeasible, "a genuine infeasibility stopped fathoming");
        assert_eq!(d.subtol_repaired, 0);
    }

    /// The repair must never cut a point the caller's box contained: sweep a
    /// feasible point through many noise-inverted boxes and assert containment
    /// survives. Prints nothing, but the assertion count is the point (§6).
    #[test]
    fn repair_never_cuts_a_contained_feasible_point() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            probing: false,
            probe_max_vars: 0,
        };
        let mut checked = 0usize;
        for k in 0..40 {
            let v = 0.1 * k as f64; // feasible x value in [0, 3.9]
            for eps in [1e-16, 1e-14, 1e-12, 1e-9, 1e-7] {
                let d = run_in_tree_presolve(&model, &[v, 0.0], &[v - eps, 10.0], 0, None, &opts);
                assert!(!d.infeasible, "fathomed a live node at eps={eps}");
                assert!(
                    d.lb[0] <= v && v <= d.ub[0],
                    "repair cut x={v} at eps={eps}: [{}, {}]",
                    d.lb[0],
                    d.ub[0]
                );
                checked += 1;
            }
        }
        assert_eq!(
            checked, 200,
            "probe did not execute the comparisons it claims"
        );
    }

    /// #907, probing path enabled: a node box inverted by sub-`FEAS_TOL` noise
    /// must not be fathomed, and must not come back inverted or with the point
    /// cut. Runs with `opts.tol` two orders BELOW `FEAS_TOL` so the two
    /// tolerances are distinguishable.
    ///
    /// SCOPE, stated honestly: this covers the incoming-box sanitation on the
    /// probing path (it fails on pre-#907 `main`). It does NOT isolate the
    /// fold-back verdict that used to read `new_lb[i] > new_ub[i] + opts.tol` —
    /// reverting that one line alone leaves this test green, because reaching it
    /// requires `probe_node_bounds` to itself return an interval inverted by
    /// `(opts.tol, FEAS_TOL]`, which no toy model here produces. That line is
    /// changed on consistency grounds — `opts.tol` is a convergence tolerance and
    /// must not gate an infeasibility verdict — and is UNCOVERED by a
    /// fail-before test.
    #[test]
    fn probing_path_does_not_fathom_subtol_inverted_box() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-8, // << FEAS_TOL (1e-6): the window the old test fathomed in
            probing: true,
            probe_max_vars: 8,
        };
        let mut checked = 0usize;
        for eps in [1e-7, 5e-7, 9e-7] {
            // Crossing strictly inside (opts.tol, FEAS_TOL] — noise, not a proof.
            let d = run_in_tree_presolve(&model, &[2.5, 0.0], &[2.5 - eps, 10.0], 0, None, &opts);
            assert!(d.ran);
            assert!(
                !d.infeasible,
                "probing path fathomed a live node at a {eps:e} crossing (< FEAS_TOL)"
            );
            assert!(d.lb[0] <= d.ub[0], "probing path returned an inverted box");
            assert!(d.lb[0] <= 2.5 && 2.5 <= d.ub[0], "probing path cut x=2.5");
            checked += 1;
        }
        assert_eq!(
            checked, 3,
            "probe did not execute the comparisons it claims"
        );
    }

    /// ANTI-PERMISSIVENESS CONTROL for the probing path: a genuine infeasibility
    /// must still fathom with probing on and a small `opts.tol`.
    #[test]
    fn probing_path_still_fathoms_genuine_infeasibility() {
        let model = x_plus_y_le_5();
        let opts = InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-8,
            probing: true,
            probe_max_vars: 8,
        };
        let d = run_in_tree_presolve(&model, &[10.0, 10.0], &[10.0, 10.0], 0, None, &opts);
        assert!(d.ran);
        assert!(
            d.infeasible,
            "probing path stopped fathoming a real infeasibility"
        );
    }

    // ── #1513: per-scalar node boxes (array variable blocks) ─────────────

    fn arr_var(arena: &mut ExprArena, name: &str, idx: usize, shape: Vec<usize>) -> ExprId {
        arena.add(ExprNode::Variable {
            name: name.to_string(),
            index: idx,
            size: shape.iter().product(),
            shape,
        })
    }

    fn elem(arena: &mut ExprArena, base: ExprId, i: usize) -> ExprId {
        arena.add(ExprNode::Index {
            base,
            index: IndexSpec::Scalar(i),
        })
    }

    fn arr_vinfo(name: &str, offset: usize, shape: Vec<usize>, lb: f64, ub: f64) -> VarInfo {
        let n: usize = shape.iter().product();
        VarInfo {
            name: name.to_string(),
            var_type: VarType::Continuous,
            offset,
            size: n,
            shape,
            lb: vec![lb; n],
            ub: vec![ub; n],
        }
    }

    fn opts1() -> InTreePresolveOptions {
        InTreePresolveOptions {
            depth_stride: 1,
            max_iter: 16,
            tol: 1e-9,
            probing: false,
            probe_max_vars: 0,
        }
    }

    /// x ∈ [0,10]^2 as ONE block of shape (2,); x[0] + x[1] <= 5.
    fn array_x0_plus_x1_le_5() -> ModelRepr {
        let mut arena = ExprArena::new();
        let x = arr_var(&mut arena, "x", 0, vec![2]);
        let x0 = elem(&mut arena, x, 0);
        let x1 = elem(&mut arena, x, 1);
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: x0,
            right: x1,
        });
        ModelRepr {
            arena,
            objective: body,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 5.0,
                name: None,
            }],
            variables: vec![arr_vinfo("x", 0, vec![2], 0.0, 10.0)],
            n_vars: 2,
        }
    }

    /// The array form of `tightens_at_node_with_branching_bound`: the per-scalar
    /// kernel derives x[1] <= 2 from x[0] >= 3. The per-block kernel on the same
    /// model sees only the block hull [0, 10] and derives nothing -- the root
    /// cause of #1513, pinned here as the control.
    #[test]
    fn per_scalar_box_tightens_array_element() {
        let model = array_x0_plus_x1_le_5();
        let d = run_in_tree_presolve_scalar(&model, &[3.0, 0.0], &[10.0, 10.0], 0, None, &opts1())
            .expect("array model must be accepted");
        assert!(d.ran && !d.infeasible);
        assert_eq!(d.lb.len(), 2);
        assert_eq!(d.lb[0], 3.0);
        assert!((d.ub[1] - 2.0).abs() <= 1e-6, "x[1] ub = {}", d.ub[1]);
        assert!(d.ub[0] <= 5.0 + 1e-6, "x[0] ub = {}", d.ub[0]);
        assert_eq!(d.bounds_tightened, 2);

        // Control: the per-block kernel, handed the block hull, gets nothing.
        let blk = run_in_tree_presolve(&model, &[0.0], &[10.0], 0, None, &opts1());
        assert_eq!(blk.bounds_tightened, 0);
    }

    /// An element-level infeasibility fathoms: x[0] >= 4 and x[1] >= 4.
    #[test]
    fn per_scalar_box_detects_array_infeasibility() {
        let model = array_x0_plus_x1_le_5();
        let d = run_in_tree_presolve_scalar(&model, &[4.0, 4.0], &[10.0, 10.0], 0, None, &opts1())
            .unwrap();
        assert!(d.ran && d.infeasible);
    }

    /// On a scalar-layout model the per-scalar entry point IS the per-block
    /// kernel: every output field is bit-identical over a sweep of boxes (the
    /// bound-neutral argument for scalar models).
    #[test]
    fn scalar_layout_is_identical_to_per_block() {
        let model = x_plus_y_le_5();
        assert!(is_scalar_layout(&model));
        let mut checked = 0usize;
        for &(l0, u0, l1, u1) in &[
            (3.0, 10.0, 0.0, 10.0),
            (0.0, 1.0, 0.0, 10.0),
            (10.0, 10.0, 10.0, 10.0),
            (2.5, 2.5 - 1e-14, 0.0, 10.0),
            (-1.0, 4.0, 1.5, 3.0),
        ] {
            for probing in [false, true] {
                let mut o = opts1();
                o.probing = probing;
                o.probe_max_vars = 8;
                let a = run_in_tree_presolve(&model, &[l0, l1], &[u0, u1], 0, Some(4.0), &o);
                let b = run_in_tree_presolve_scalar(&model, &[l0, l1], &[u0, u1], 0, Some(4.0), &o)
                    .unwrap();
                assert_eq!(
                    a.lb.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.lb.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(
                    a.ub.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.ub.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(a.infeasible, b.infeasible);
                assert_eq!(a.bounds_tightened, b.bounds_tightened);
                assert_eq!(a.subtol_repaired, b.subtol_repaired);
                assert_eq!(a.ran, b.ran);
                checked += 1;
            }
        }
        assert_eq!(
            checked, 10,
            "probe did not execute the comparisons it claims"
        );
    }

    /// A box of the wrong length is an error, not a silent skip.
    #[test]
    fn wrong_box_length_is_refused() {
        let model = array_x0_plus_x1_le_5();
        let e = run_in_tree_presolve_scalar(&model, &[0.0], &[10.0], 0, None, &opts1());
        assert!(
            e.is_err(),
            "a per-block box on an array model must be refused"
        );
    }

    /// A non-contiguous layout (block offsets that do not tile 0..n_vars) is
    /// refused by name rather than scalarized into the wrong slots.
    #[test]
    fn noncontiguous_layout_is_refused() {
        let mut model = array_x0_plus_x1_le_5();
        model.variables[0].offset = 1;
        model.n_vars = 3;
        let e = scalarize_for_fbbt(&model);
        assert!(e.is_err());
        assert!(e.unwrap_err().contains("offset"));
    }

    /// Full-arity element indices are rewritten to scalar slots; array-valued
    /// references (the whole matrix, a row `X[1]` of a rank-2 block, a slice)
    /// stay array-valued on the hull proxy.
    #[test]
    fn scalarize_rewrites_elements_and_keeps_arrays_on_proxy() {
        let mut arena = ExprArena::new();
        let t = scalar_var(&mut arena, "t", 0);
        let x = arr_var(&mut arena, "X", 1, vec![2, 2]);
        let x10 = arena.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Tuple(vec![1, 0]),
        });
        let row = arena.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Scalar(1),
        });
        let sl = arena.add(ExprNode::Index {
            base: x,
            index: IndexSpec::Multi(vec![IndexElem::Scalar(0), IndexElem::FULL]),
        });
        let model = ModelRepr {
            arena,
            objective: t,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![],
            variables: vec![
                vinfo("t", 0.0, 1.0),
                arr_vinfo("X", 1, vec![2, 2], -1.0, 2.0),
            ],
            n_vars: 5,
        };
        let view = scalarize_for_fbbt(&model).unwrap();
        assert_eq!(view.n_scalar(), 5);
        assert_eq!(view.n_proxies(), 1);
        let m = view.model();
        assert_eq!(m.variables.len(), 6);
        // t stays slot 0.
        assert!(matches!(
            m.arena.get(t),
            ExprNode::Variable {
                index: 0,
                size: 1,
                ..
            }
        ));
        // X[1, 0] -> flat 2 -> slot 1 + 2 = 3.
        assert!(matches!(
            m.arena.get(x10),
            ExprNode::Variable {
                index: 3,
                size: 1,
                ..
            }
        ));
        // The whole X -> the proxy (slot 5), still size 4.
        assert!(matches!(
            m.arena.get(x),
            ExprNode::Variable {
                index: 5,
                size: 4,
                ..
            }
        ));
        assert_eq!(m.variables[5].var_type, VarType::Continuous);
        // Row and slice stay Index nodes on the proxy.
        assert!(matches!(m.arena.get(row), ExprNode::Index { .. }));
        assert!(matches!(m.arena.get(sl), ExprNode::Index { .. }));
    }

    /// Probing an array binary block PER ELEMENT. z ∈ {0,1}^2 as one block,
    /// z[0] + z[1] == 1. Fixing the whole block (what a per-block hull box
    /// does) makes both fixings infeasible and "proves" a feasible node empty;
    /// per element, the root box survives and z[0] = 1 forces z[1] = 0.
    #[test]
    fn probing_array_binary_block_is_per_element() {
        let mut arena = ExprArena::new();
        let z = arr_var(&mut arena, "z", 0, vec![2]);
        let z0 = elem(&mut arena, z, 0);
        let z1 = elem(&mut arena, z, 1);
        let body = arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: z0,
            right: z1,
        });
        let mut zi = arr_vinfo("z", 0, vec![2], 0.0, 1.0);
        zi.var_type = VarType::Binary;
        let model = ModelRepr {
            arena,
            objective: body,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Eq,
                rhs: 1.0,
                name: None,
            }],
            variables: vec![zi],
            n_vars: 2,
        };
        let mut o = opts1();
        o.probing = true;
        o.probe_max_vars = 8;
        let root =
            run_in_tree_presolve_scalar(&model, &[0.0, 0.0], &[1.0, 1.0], 0, None, &o).unwrap();
        assert!(
            !root.infeasible,
            "a feasible array-binary node was fathomed"
        );
        let node =
            run_in_tree_presolve_scalar(&model, &[1.0, 0.0], &[1.0, 1.0], 0, None, &o).unwrap();
        assert!(!node.infeasible);
        assert!(
            node.ub[1] <= 1e-9,
            "z[1] should be forced to 0, got {}",
            node.ub[1]
        );
    }

    /// Feasible-point sampling on a model mixing element references and a
    /// whole-array reduction (the proxy path): x ∈ [0,4]^3 as one block,
    ///   sum(x) <= 5,  x[0] * x[1] >= 1,  x[2] - x[0] <= 1.
    /// For many random sub-boxes and random feasible points inside them, the
    /// kernel must neither fathom the box nor cut the point. Deterministic LCG.
    #[test]
    fn per_scalar_kernel_never_cuts_a_feasible_point() {
        let mut arena = ExprArena::new();
        let x = arr_var(&mut arena, "x", 0, vec![3]);
        let s = arena.add(ExprNode::Sum {
            operand: x,
            axis: None,
        });
        let x0 = elem(&mut arena, x, 0);
        let x1 = elem(&mut arena, x, 1);
        let x2 = elem(&mut arena, x, 2);
        let prod = arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: x0,
            right: x1,
        });
        let diff = arena.add(ExprNode::BinaryOp {
            op: BinOp::Sub,
            left: x2,
            right: x0,
        });
        let model = ModelRepr {
            arena,
            objective: s,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![
                ConstraintRepr {
                    body: s,
                    sense: ConstraintSense::Le,
                    rhs: 5.0,
                    name: None,
                },
                ConstraintRepr {
                    body: prod,
                    sense: ConstraintSense::Ge,
                    rhs: 1.0,
                    name: None,
                },
                ConstraintRepr {
                    body: diff,
                    sense: ConstraintSense::Le,
                    rhs: 1.0,
                    name: None,
                },
            ],
            variables: vec![arr_vinfo("x", 0, vec![3], 0.0, 4.0)],
            n_vars: 3,
        };
        let mut seed: u64 = 0x1513;
        let mut rnd = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut checked = 0usize;
        let mut tightened_any = 0usize;
        for _ in 0..4000 {
            let p = [4.0 * rnd(), 4.0 * rnd(), 4.0 * rnd()];
            let feasible = p[0] + p[1] + p[2] <= 5.0 && p[0] * p[1] >= 1.0 && p[2] - p[0] <= 1.0;
            if !feasible {
                continue;
            }
            let mut lb = [0.0; 3];
            let mut ub = [0.0; 3];
            for k in 0..3 {
                lb[k] = p[k] - (p[k]) * rnd();
                ub[k] = p[k] + (4.0 - p[k]) * rnd();
            }
            for probing in [false, true] {
                let mut o = opts1();
                o.probing = probing;
                let d = run_in_tree_presolve_scalar(&model, &lb, &ub, 0, None, &o).unwrap();
                assert!(!d.infeasible, "fathomed a box containing feasible {p:?}");
                for k in 0..3 {
                    assert!(
                        d.lb[k] <= p[k] + 1e-9 && p[k] <= d.ub[k] + 1e-9,
                        "cut feasible {p:?} at x[{k}]: [{}, {}]",
                        d.lb[k],
                        d.ub[k]
                    );
                }
                if d.bounds_tightened > 0 {
                    tightened_any += 1;
                }
                checked += 1;
            }
        }
        assert!(checked > 500, "only {checked} boxes checked");
        // The kernel must actually DO something on this class, else the
        // soundness check above is vacuous.
        assert!(tightened_any > 100, "only {tightened_any} tightenings");
    }
}
