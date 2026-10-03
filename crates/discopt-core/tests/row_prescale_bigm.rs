//! #1537 review: the row pre-pass must not let a big-M row's scaled primal
//! tolerance swallow an O(1) violation in original units.
//!
//! `min -x + 3z` s.t. `x - M z <= 0`, `x` in [0, 10], `z` binary. A max-based
//! pre-pass divided the row by `2^40` at `M = 1e12`; the single-entry columns of
//! `x` and the slack then scaled back up by `2^40`, so `x`'s scaled box was
//! `[0, 9e-12]` -- below the primal tolerance -- and the driver returned
//! `x = 10, z = 0`, violating the row by 10. The geometric-mean pre-pass divides by
//! `2^20` and every arm returns a point on the row.
//!
//! One test in its own binary: the flag is process-wide and the test flips it.

use discopt_core::bnb::milp_driver::{solve_milp, MilpOptions};
use discopt_core::lp::crossover::LpView;
use discopt_core::lp::simplex::SimplexOptions;

const INF: f64 = 1e20;

fn options() -> MilpOptions {
    MilpOptions {
        n_struct: 2,
        integer_cols: vec![1],
        max_nodes: 10_000,
        time_limit_s: Some(30.0),
        gap_tol: 1e-9,
        abs_gap_tol: None,
        root_cuts: 0,
        cut_rounds: 0,
        gmi_cuts: false,
        cut_select: false,
        node_cuts: false,
        max_pool_cuts: 0,
        heuristics: true,
        presolve: true,
        strong_branch: false,
        node_propagation: true,
        reduced_cost_fixing: true,
        sb_max_cands: 0,
        sb_node_budget: 0,
        initial_incumbent: None,
        node_hook_rounds: 0,
        node_hook_cut_cap: 0,
        root_cut_time_s: None,
        root_cut_prune: true,
        simplex: SimplexOptions::default(),
    }
}

#[test]
fn big_m_row_is_satisfied_in_original_units_on_every_arm() {
    let mut checked = 0;
    for arm in ["0", "1"] {
        std::env::set_var("DISCOPT_LP_ROW_PRESCALE", arm);
        for big in [1e9, 1e12, 1e14] {
            // Standard form: x - M z + s = 0, s >= 0.
            let a = [1.0, -big, 1.0];
            let lp = LpView {
                a: &a,
                m: 1,
                n: 3,
                c: &[-1.0, 3.0, 0.0],
                l: &[0.0, 0.0, 0.0],
                u: &[10.0, 1.0, INF],
            };
            let r = solve_milp(&lp, &[0.0], 0.0, &options());
            assert!(r.x.len() >= 2, "arm {arm} M {big:e}: no point {r:?}");
            let viol = r.x[0] - big * r.x[1];
            assert!(
                viol <= 1e-6,
                "arm {arm} M {big:e}: row x - M z <= 0 violated by {viol} in original units: {r:?}"
            );
            // Never a bound above the true optimum (-7 at x = 10, z = 1).
            assert!(
                r.bound <= -7.0 + 1e-6,
                "arm {arm} M {big:e}: false bound {r:?}"
            );
            checked += 1;
        }
    }
    std::env::remove_var("DISCOPT_LP_ROW_PRESCALE");
    assert_eq!(checked, 6);
}
