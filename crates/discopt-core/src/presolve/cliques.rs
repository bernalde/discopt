//! Implied-clique extraction across binary variables (F2 of the
//! presolve roadmap).
//!
//! ## What this pass does
//!
//! Detects pairs of binary variables `(b_i, b_j)` that cannot
//! simultaneously equal 1 because some linear constraint forbids it.
//! Each such pair is a 2-clique (edge) in the binary conflict graph;
//! the orchestrator records them on the pass delta for downstream
//! consumers (relaxation compiler, branching, primal heuristic).
//!
//! For a `≤` constraint `Σ a_k x_k + c0  ≤  rhs`, where every term
//! involves a binary variable, two binaries `b_i` and `b_j` conflict
//! if assigning both to 1 (and every other variable at its activity-
//! minimising value) already violates the bound:
//!
//! ```text
//!     a_i + a_j + min_activity_of_rest + c0  >  rhs
//! ```
//!
//! For `≥` the same question is asked of the negated row: `b_i` and `b_j`
//! conflict if setting both to 1 (every other binary at its activity-
//! *maximising* value) already falls below the bound. `=` applies both
//! tests. Every edge, whatever the row sense, means exactly one thing:
//! **`b_i = b_j = 1` is infeasible**, so `b_i + b_j <= 1` is valid.
//!
//! #1603: the `≥` arm used to test "both at **0**" instead (a covering
//! row `b0 + b1 + b2 >= 2` produced three edges), and its consumer --
//! the root clique-cut separator -- reads every edge as "not both 1",
//! so it would have emitted `b0 + b1 + b2 <= 1`, a cut that excludes
//! every feasible point of that row.
//!
//! ## Index space
//!
//! Edges are reported in **flat column** indices (`VarInfo::offset`),
//! the index space of the solver's `x` vector. Only scalar (size-1)
//! binary blocks are inspected, so the block's offset IS its column.
//! #1603: they used to be reported as variable-*block* indices, which
//! coincide with flat columns only when every variable declared before
//! the binary is a scalar. One array variable declared first shifted
//! every edge onto the wrong columns, and the clique cut
//! `sum x_C <= 1` landed on two continuous variables, cutting off the
//! true optimum and returning a wrong `optimal`.
//!
//! ## What this pass does *not* do (yet)
//!
//! - It does not enumerate maximal cliques. Only pairwise edges are
//!   produced. Maximal-clique extension over the edge graph is a
//!   future enhancement and is what gives the full reduction in LP
//!   relaxation strength reported by Achterberg et al. (2020).
//! - It does not rewrite or replace the source constraint. The clique
//!   data appears in the delta's `structure.cliques` field; the
//!   model is unchanged.
//! - It only inspects rows whose body is a polynomial of degree ≤ 1
//!   and whose only variable factors are scalar binary variables.
//!   Mixed continuous/binary rows are skipped to keep the activity
//!   computation simple.
//!
//! ## Determinism
//!
//! Constraints are scanned in `model.constraints` order; within each
//! constraint, binary leaves are sorted by variable-block index. Pair
//! detection runs over `(i, j)` with `i < j`, so the resulting clique
//! list is canonical. No `HashMap` iteration on the hot path.

use std::collections::BTreeSet;

use super::polynomial::try_polynomial;
use crate::expr::{ConstraintSense, ExprNode, ModelRepr, VarType};

/// Per-pass diagnostics for clique extraction.
#[derive(Debug, Clone, Default)]
pub struct CliqueStats {
    /// Number of constraints inspected.
    pub linear_rows_scanned: usize,
    /// Number of distinct conflict edges discovered (after dedup).
    pub edges_found: usize,
}

/// Result of [`extract_cliques`]: a list of conflict edges between
/// scalar binary variables, in FLAT COLUMN indices. Each edge is
/// `(i, j)` with `i < j`, meaning `x_i + x_j <= 1` is valid.
#[derive(Debug, Clone, Default)]
pub struct CliqueSet {
    /// Edges, sorted lexicographically by `(i, j)`.
    pub edges: Vec<(usize, usize)>,
}

/// Run binary clique extraction. Pure function; never modifies the
/// model.
pub fn extract_cliques(model: &ModelRepr) -> (CliqueSet, CliqueStats) {
    let mut stats = CliqueStats::default();
    let mut edge_set: BTreeSet<(usize, usize)> = BTreeSet::new();

    // Pre-build a map from variable-block index to "is binary".
    let is_binary: Vec<bool> = model
        .variables
        .iter()
        .map(|v| matches!(v.var_type, VarType::Binary))
        .collect();

    for c in &model.constraints {
        let poly = match try_polynomial(&model.arena, c.body) {
            Some(p) => p,
            None => continue,
        };
        if poly.max_total_degree() > 1 {
            continue;
        }
        // Collect (flat_column, coeff) pairs. All factors must resolve
        // to binary scalar variables; otherwise skip the row. The column
        // is the block's `offset` (a scalar block occupies exactly one
        // column), never the block index -- see "Index space" above.
        let mut row: Vec<(usize, f64)> = Vec::with_capacity(poly.monomials.len());
        let mut ok = true;
        for m in &poly.monomials {
            if m.factors.len() != 1 || m.factors[0].1 != 1 {
                ok = false;
                break;
            }
            let leaf = m.factors[0].0;
            let block = match model.arena.get(leaf) {
                ExprNode::Variable { index, size, .. } if *size == 1 => *index,
                _ => {
                    ok = false;
                    break;
                }
            };
            if !is_binary.get(block).copied().unwrap_or(false) {
                ok = false;
                break;
            }
            if m.coeff.abs() > 1e-15 {
                row.push((model.variables[block].offset, m.coeff));
            }
        }
        if !ok || row.len() < 2 {
            continue;
        }
        stats.linear_rows_scanned += 1;

        // Sort by column so output is canonical.
        row.sort_by_key(|(b, _)| *b);

        // Compute the activity-extremum baseline: each binary at the
        // value that *helps* the constraint (minimises LHS for Le,
        // maximises for Ge). We then test pairs by forcing two
        // binaries to 1 and checking whether the constraint can still
        // hold.
        scan_pairs(c.sense, c.rhs, poly.constant, &row, &mut edge_set);
    }

    let mut edges: Vec<(usize, usize)> = edge_set.into_iter().collect();
    edges.sort_unstable();
    stats.edges_found = edges.len();
    (CliqueSet { edges }, stats)
}

/// Add every pair of `(b_i, b_j)` from `row` that conflicts under
/// constraint `sense / rhs` with constant offset `c0` to `edges`.
///
/// `row` is sorted ascending by block index.
fn scan_pairs(
    sense: ConstraintSense,
    rhs: f64,
    c0: f64,
    row: &[(usize, f64)],
    edges: &mut BTreeSet<(usize, usize)>,
) {
    // For each binary k, the "helping" assignment minimises (Le case)
    // or maximises (Ge case) the contribution `a_k * b_k`. With
    // `b_k ∈ {0, 1}`:
    //   Le helping: 0 if a_k ≥ 0, 1 if a_k < 0.
    //   Ge helping: 1 if a_k ≥ 0, 0 if a_k < 0.
    // Eq sense triggers both checks. BOTH arms test the same event --
    // `b_i = b_j = 1` with everything else at its best -- because an
    // edge means "not both 1" to every consumer (#1603).
    let test_le = matches!(sense, ConstraintSense::Le | ConstraintSense::Eq);
    let test_ge = matches!(sense, ConstraintSense::Ge | ConstraintSense::Eq);

    if test_le {
        // Baseline LHS at "all helping for Le" assignment.
        let baseline_le: f64 = c0 + row.iter().map(|(_, a)| a.min(0.0)).sum::<f64>();
        for i in 0..row.len() {
            for j in (i + 1)..row.len() {
                let (bi, ai) = row[i];
                let (bj, aj) = row[j];
                // Switching i and j to 1 contributes (a_i - a_i.min(0)) + (a_j - a_j.min(0)).
                let delta_i = ai - ai.min(0.0); // = max(ai, 0)
                let delta_j = aj - aj.min(0.0);
                let test = baseline_le + delta_i + delta_j;
                if test > rhs + 1e-9 {
                    let (lo, hi) = if bi < bj { (bi, bj) } else { (bj, bi) };
                    if lo != hi {
                        edges.insert((lo, hi));
                    }
                }
            }
        }
    }
    if test_ge {
        // Baseline LHS at "all helping for Ge" assignment.
        let baseline_ge: f64 = c0 + row.iter().map(|(_, a)| a.max(0.0)).sum::<f64>();
        for i in 0..row.len() {
            for j in (i + 1)..row.len() {
                let (bi, ai) = row[i];
                let (bj, aj) = row[j];
                // Forcing b_k from its helping value to 1 changes the
                // activity by 0 when a_k >= 0 (it already helps at 1)
                // and by a_k when a_k < 0 (helping was 0): min(a_k, 0).
                //
                // #1603: this used to be `-max(a_k, 0)` -- forcing both
                // to **0** -- which reports "not both 0" pairs as edges.
                let delta_i = ai.min(0.0);
                let delta_j = aj.min(0.0);
                let test = baseline_ge + delta_i + delta_j;
                if test < rhs - 1e-9 {
                    let (lo, hi) = if bi < bj { (bi, bj) } else { (bj, bi) };
                    if lo != hi {
                        edges.insert((lo, hi));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{
        BinOp, ConstraintRepr, ConstraintSense, ExprArena, ExprId, ExprNode, ModelRepr,
        ObjectiveSense, VarInfo, VarType,
    };

    fn binary_var(arena: &mut ExprArena, name: &str, idx: usize) -> ExprId {
        arena.add(ExprNode::Variable {
            name: name.into(),
            index: idx,
            size: 1,
            shape: vec![],
        })
    }

    fn vinfo_bin(name: &str, offset: usize) -> VarInfo {
        VarInfo {
            name: name.into(),
            var_type: VarType::Binary,
            offset,
            size: 1,
            shape: vec![],
            lb: vec![0.0],
            ub: vec![1.0],
        }
    }

    fn vinfo_cont(name: &str, offset: usize) -> VarInfo {
        VarInfo {
            name: name.into(),
            var_type: VarType::Continuous,
            offset,
            size: 1,
            shape: vec![],
            lb: vec![0.0],
            ub: vec![1.0],
        }
    }

    fn lin(arena: &mut ExprArena, c: f64, var: ExprId) -> ExprId {
        let cn = arena.add(ExprNode::Constant(c));
        arena.add(ExprNode::BinaryOp {
            op: BinOp::Mul,
            left: cn,
            right: var,
        })
    }

    fn add(arena: &mut ExprArena, a: ExprId, b: ExprId) -> ExprId {
        arena.add(ExprNode::BinaryOp {
            op: BinOp::Add,
            left: a,
            right: b,
        })
    }

    /// Set-packing: `b0 + b1 + b2 ≤ 1` ⇒ all 3 pairs conflict.
    #[test]
    fn set_packing_three_edges() {
        let mut arena = ExprArena::new();
        let b0 = binary_var(&mut arena, "b0", 0);
        let b1 = binary_var(&mut arena, "b1", 1);
        let b2 = binary_var(&mut arena, "b2", 2);
        let body = {
            let s01 = add(&mut arena, b0, b1);
            add(&mut arena, s01, b2)
        };
        let model = ModelRepr {
            arena,
            objective: b0,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 1.0,
                name: None,
            }],
            variables: vec![vinfo_bin("b0", 0), vinfo_bin("b1", 1), vinfo_bin("b2", 2)],
            n_vars: 3,
        };
        let (c, s) = extract_cliques(&model);
        assert_eq!(s.edges_found, 3);
        assert_eq!(c.edges, vec![(0, 1), (0, 2), (1, 2)]);
    }

    /// Coefficient pair: `2 b0 + 2 b1 ≤ 3` forbids both = 1
    /// (4 > 3) but allows either alone.
    #[test]
    fn coeff_pair_forbidden() {
        let mut arena = ExprArena::new();
        let b0 = binary_var(&mut arena, "b0", 0);
        let b1 = binary_var(&mut arena, "b1", 1);
        let body = {
            let a = lin(&mut arena, 2.0, b0);
            let b = lin(&mut arena, 2.0, b1);
            add(&mut arena, a, b)
        };
        let model = ModelRepr {
            arena,
            objective: b0,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 3.0,
                name: None,
            }],
            variables: vec![vinfo_bin("b0", 0), vinfo_bin("b1", 1)],
            n_vars: 2,
        };
        let (c, _) = extract_cliques(&model);
        assert_eq!(c.edges, vec![(0, 1)]);
    }

    /// Loose constraint: `b0 + b1 ≤ 5` ⇒ no edges.
    #[test]
    fn loose_constraint_no_edges() {
        let mut arena = ExprArena::new();
        let b0 = binary_var(&mut arena, "b0", 0);
        let b1 = binary_var(&mut arena, "b1", 1);
        let body = add(&mut arena, b0, b1);
        let model = ModelRepr {
            arena,
            objective: b0,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 5.0,
                name: None,
            }],
            variables: vec![vinfo_bin("b0", 0), vinfo_bin("b1", 1)],
            n_vars: 2,
        };
        let (c, _) = extract_cliques(&model);
        assert!(c.edges.is_empty());
    }

    /// Mixed continuous/binary row is skipped (v0 scope).
    #[test]
    fn mixed_row_skipped() {
        let mut arena = ExprArena::new();
        let b0 = binary_var(&mut arena, "b0", 0);
        let x = arena.add(ExprNode::Variable {
            name: "x".into(),
            index: 1,
            size: 1,
            shape: vec![],
        });
        let body = add(&mut arena, b0, x);
        let model = ModelRepr {
            arena,
            objective: b0,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 1.0,
                name: None,
            }],
            variables: vec![vinfo_bin("b0", 0), vinfo_cont("x", 1)],
            n_vars: 2,
        };
        let (c, s) = extract_cliques(&model);
        assert!(c.edges.is_empty());
        assert_eq!(s.linear_rows_scanned, 0);
    }

    fn ge_model(coeffs: &[f64], rhs: f64) -> ModelRepr {
        let mut arena = ExprArena::new();
        let mut body: Option<ExprId> = None;
        let mut vars = Vec::new();
        for (k, &c) in coeffs.iter().enumerate() {
            let b = binary_var(&mut arena, &format!("b{k}"), k);
            let t = lin(&mut arena, c, b);
            body = Some(match body {
                None => t,
                Some(acc) => add(&mut arena, acc, t),
            });
            vars.push(vinfo_bin(&format!("b{k}"), k));
        }
        let body = body.unwrap();
        ModelRepr {
            arena,
            objective: body,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Ge,
                rhs,
                name: None,
            }],
            n_vars: coeffs.len(),
            variables: vars,
        }
    }

    /// #1603: a covering row `b0 + b1 + b2 ≥ 2` forbids two ZEROS, not two
    /// ones -- `(1, 1, 0)` is feasible. It has NO "not both 1" edge. The `≥`
    /// arm used to report all three pairs, which the clique-cut consumer turns
    /// into `b0 + b1 + b2 <= 1`, excluding every feasible point of the row.
    #[test]
    fn ge_covering_row_has_no_edges() {
        let (c, s) = extract_cliques(&ge_model(&[1.0, 1.0, 1.0], 2.0));
        assert_eq!(s.linear_rows_scanned, 1, "the row must actually be scanned");
        assert!(c.edges.is_empty(), "covering row reported {:?}", c.edges);
    }

    /// The `≥` arm still finds genuine "not both 1" pairs: `-b0 - b1 - b2 ≥ -1`
    /// is the set-packing row `b0 + b1 + b2 ≤ 1` written the other way round.
    #[test]
    fn ge_negated_packing_row_has_all_edges() {
        let (c, _) = extract_cliques(&ge_model(&[-1.0, -1.0, -1.0], -1.0));
        assert_eq!(c.edges, vec![(0, 1), (0, 2), (1, 2)]);
    }

    /// Exhaustive check of the edge contract on small `≤`/`≥`/`=` rows: every
    /// reported edge `(i, j)` must have NO feasible 0/1 point with
    /// `b_i = b_j = 1`. Counts the edges it checked and asserts it checked some.
    #[test]
    fn every_edge_is_a_genuine_not_both_one_conflict() {
        let rows: &[(&[f64], f64)] = &[
            (&[1.0, 1.0, 1.0], 2.0),
            (&[-1.0, -1.0, -1.0], -1.0),
            (&[2.0, -1.0, 3.0], 2.0),
            (&[-2.0, 1.0, -3.0], -2.0),
            (&[1.0, 2.0, -1.0], 1.0),
            (&[-3.0, -2.0, 1.0], -3.5),
        ];
        let mut checked = 0usize;
        for &(coeffs, rhs) in rows {
            for sense in [
                ConstraintSense::Le,
                ConstraintSense::Ge,
                ConstraintSense::Eq,
            ] {
                let mut model = ge_model(coeffs, rhs);
                model.constraints[0].sense = sense;
                let (c, _) = extract_cliques(&model);
                for &(i, j) in &c.edges {
                    for mask in 0u32..(1 << coeffs.len()) {
                        let x: Vec<f64> = (0..coeffs.len())
                            .map(|k| ((mask >> k) & 1) as f64)
                            .collect();
                        if x[i] != 1.0 || x[j] != 1.0 {
                            continue;
                        }
                        let act: f64 = coeffs.iter().zip(&x).map(|(a, v)| a * v).sum();
                        let feasible = match sense {
                            ConstraintSense::Le => act <= rhs + 1e-9,
                            ConstraintSense::Ge => act >= rhs - 1e-9,
                            ConstraintSense::Eq => (act - rhs).abs() <= 1e-9,
                        };
                        assert!(
                            !feasible,
                            "edge ({i},{j}) on {coeffs:?} {sense:?} {rhs} excludes feasible {x:?}"
                        );
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "no edge was checked");
    }

    /// #1603 index space: edges are FLAT COLUMNS (`VarInfo::offset`), not block
    /// indices. A 4-element array `d` declared first puts the scalar binaries
    /// `s0`, `s1` (blocks 1 and 2) at columns 4 and 5; `s0 + s1 ≤ 1` must report
    /// `(4, 5)`. Reporting `(1, 2)` puts the clique cut on `d[1] + d[2]`.
    #[test]
    fn edges_are_flat_columns_not_block_indices() {
        let mut arena = ExprArena::new();
        let s0 = binary_var(&mut arena, "s0", 1);
        let s1 = binary_var(&mut arena, "s1", 2);
        let body = add(&mut arena, s0, s1);
        let model = ModelRepr {
            arena,
            objective: body,
            objective_sense: ObjectiveSense::Minimize,
            constraints: vec![ConstraintRepr {
                body,
                sense: ConstraintSense::Le,
                rhs: 1.0,
                name: None,
            }],
            variables: vec![
                VarInfo {
                    name: "d".into(),
                    var_type: VarType::Continuous,
                    offset: 0,
                    size: 4,
                    shape: vec![4],
                    lb: vec![0.0; 4],
                    ub: vec![1.0; 4],
                },
                vinfo_bin("s0", 4),
                vinfo_bin("s1", 5),
            ],
            n_vars: 6,
        };
        let (c, _) = extract_cliques(&model);
        assert_eq!(c.edges, vec![(4, 5)]);
    }
}
