//! Power-of-two geometric-mean equilibration scaling for the simplex.
//!
//! The revised simplex factorizes its basis with [`feral`]'s LU, which declares
//! a basis singular when its pivots are tiny relative to the matrix entries. On
//! lifted McCormick LPs whose coefficients span many orders of magnitude — FBBT
//! pushes product-aux bounds to ~1e9, so the secant-envelope slopes do too — the
//! *raw* constraint matrix is so ill-scaled that the LU breaks down after a few
//! dozen pivots and the solve returns [`LpStatus::Numerical`](super::LpStatus).
//! The LP is benign once rescaled (HiGHS solves the same system in ~0.01 s),
//! and a `Numerical` relaxation bound is uncertified, so a MILP B&B that relies
//! on it can never fathom and degenerates into full enumeration (issue #170).
//!
//! This module equilibrates the matrix so every nonzero is brought near
//! magnitude 1 before the simplex ever factorizes it, matching what HiGHS does.
//! Scaling replaces `A` with `R A C` (`R`, `C` positive diagonal), and `x` with
//! `C x̂`, giving the equivalent LP
//!
//! ```text
//!   min ĉᵀx̂  s.t.  (R A C) x̂ = R b,   C⁻¹l ≤ x̂ ≤ C⁻¹u,   ĉ = C c.
//! ```
//!
//! The objective is invariant (`ĉᵀx̂ = cᵀx`) and the optimal basis is unchanged
//! (scaling permutes nothing — a column is basic or not regardless of its
//! scale), so a warm-start basis stays valid across the B&B tree as long as the
//! factors come from `A` alone — they do; only bounds change between nodes.
//!
//! Factors are snapped to powers of two, so every scaled quantity is an exact
//! float (only the exponent shifts) and unscaling `x_j = c_j x̂_j` is exact.
//! Scaling is applied only when the matrix's dynamic range warrants it
//! ([`SCALE_TRIGGER`]); a well-conditioned LP is left untouched and bit-identical
//! to the unscaled solve, so existing behavior is preserved exactly.
//
// The equilibration sweeps index the row-major matrix and the row/col factor
// vectors by the same index, so range loops read clearer than zipping slices.
#![allow(clippy::needless_range_loop)]

use crate::lp::crossover::LpView;
use crate::lp::simplex::sparse::SparseCols;

const INF: f64 = 1e20;

/// Equilibrate only when the matrix's dynamic range (largest over smallest
/// nonzero magnitude) exceeds this. The fast simplex is reliable on raw matrices
/// up to a range of ~1e7; triggering at 1e6 leaves a margin while keeping every
/// well-conditioned LP an exact no-op.
const SCALE_TRIGGER: f64 = 1e6;

/// Alternating column/row sweeps. Power-of-two geometric-mean equilibration
/// converges within a few passes; the loop also stops early once no factor
/// changes (and guards against the occasional pow2 limit-cycle that never
/// fully settles).
const MAX_PASSES: usize = 4;

/// Entries smaller than this fraction of a line's maximum magnitude are treated
/// as numerical noise (not structure) when computing the equilibration factor,
/// so a spurious ~1e-16 coefficient cannot collapse the geometric mean and
/// over-scale the line. 1e-10 spans ten orders of magnitude — far wider than any
/// genuine per-line coefficient range, well above float noise.
pub(crate) const MAX_LINE_RANGE: f64 = 1e-10;

/// Whether [`equilibrate`] starts from a power-of-two **row-normalisation**
/// pre-pass (`DISCOPT_LP_ROW_PRESCALE`; default on, `0` is the opt-out) (#1537).
///
/// Without it the first sweep is a *column* sweep on the raw matrix, so the
/// [`MAX_LINE_RANGE`] noise filter judges an entry against entries of *other
/// rows*. Multiplying a row by a positive scalar changes no feasible point, yet a
/// column holding one entry from a row scaled by 1e5 and one from a row scaled by
/// 1e-6 then reads the small entry as noise, the row sweep cannot recover it (the
/// row's other entries, e.g. a slack's `1`, cap the row factor), and the #1296
/// guard in the MILP driver correctly withdraws every certificate: m3 and flay03m
/// with rows scaled in 10^[-6,6] lost theirs this way. The pre-pass first divides
/// every row by the power of two nearest its largest magnitude, which is an exact
/// reformulation (only exponents shift), so the column sweep compares entries on a
/// row-scale-invariant footing.
///
/// Bound-changing in the CLAUDE.md §5 sense — it changes the factors for every LP
/// whose range exceeds [`SCALE_TRIGGER`] — so the OFF arm is kept bit-identical to
/// the pre-#1537 factors. Read fresh on every call (a cached read pins the first
/// solve's arm for the whole process and makes a differential panel compare an arm
/// with itself), and an unrecognised value is a refusal rather than a silently
/// chosen arm.
pub(crate) fn row_prescale_enabled() -> bool {
    #[cfg(test)]
    if let Some(on) = ROW_PRESCALE_OVERRIDE.with(|c| c.get()) {
        return on;
    }
    match std::env::var("DISCOPT_LP_ROW_PRESCALE") {
        Err(_) => ROW_PRESCALE_DEFAULT,
        Ok(v) => match parse_row_prescale_flag(&v) {
            Ok(on) => on,
            Err(msg) => panic!("{msg}"),
        },
    }
}

#[cfg(test)]
thread_local! {
    static ROW_PRESCALE_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Test-only: pin [`row_prescale_enabled`] on this thread until the guard drops.
///
/// Unit-test fixtures captured under the pre-#1537 factors (the #1595 sub-tol
/// witnesses, the bchoco06 unstable-pivot LP, the #1296 gate grid) assert that a
/// specific code path is *reached*; the pre-pass changes which path they walk. A
/// thread-local, not `set_var`, so pinning one test never flips the arm under a
/// test running concurrently on another thread.
#[cfg(test)]
pub(crate) fn force_row_prescale(on: bool) -> RowPrescaleGuard {
    let prev = ROW_PRESCALE_OVERRIDE.with(|c| c.replace(Some(on)));
    RowPrescaleGuard(prev)
}

#[cfg(test)]
pub(crate) struct RowPrescaleGuard(Option<bool>);

#[cfg(test)]
impl Drop for RowPrescaleGuard {
    fn drop(&mut self) {
        ROW_PRESCALE_OVERRIDE.with(|c| c.set(self.0));
    }
}

/// Default arm of `DISCOPT_LP_ROW_PRESCALE` when the variable is unset or empty.
///
/// Graduated default-ON (CLAUDE.md §5, 2026-10-03, #1537): flag OFF vs ON over the
/// 66-file corpus as written and with rows scaled in 10^[-3,3] / 10^[-6,6] (198
/// interleaved pairs) gave 0 false bounds, 0 lost certificates, 0 failed point
/// verifications, certified 145 -> 147, total wall 1347 -> 1264 s; the pure LP/MILP
/// panel through this driver (49 pairs) certified 21 -> 38 with 0 false answers.
/// `DISCOPT_LP_ROW_PRESCALE=0` keeps the pre-#1537 factors bit-identical.
const ROW_PRESCALE_DEFAULT: bool = true;

/// Parse a `DISCOPT_LP_ROW_PRESCALE` value. Pure, so the refusal is testable
/// without touching the process environment.
fn parse_row_prescale_flag(raw: &str) -> Result<bool, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        // An empty value is how a shell spells "unset": it takes the default.
        "" => Ok(ROW_PRESCALE_DEFAULT),
        "0" | "false" | "no" => Ok(false),
        "1" | "true" | "yes" => Ok(true),
        other => Err(format!(
            "DISCOPT_LP_ROW_PRESCALE={other:?} is not recognized (expected 1/true/yes \
             or 0/false/no). Refusing rather than silently picking an arm."
        )),
    }
}

/// Power-of-two row-normalisation factors: `r_i = 2^k` nearest `1 / max_j |a_ij|`
/// (`1` for an empty row). Every factor is an exact power of two, so `r_i a_ij`
/// and `r_i b_i` are exact and the scaled row is the same constraint. Shared by
/// [`equilibrate`] and the MILP driver's #1296 noise-floor guard, which must judge
/// entries on the matrix the equilibration actually sees.
pub(crate) fn row_prescale_factors(sp: &SparseCols, m: usize) -> Vec<f64> {
    let (_, row_idx, vals) = sp.raw();
    let mut rmax = vec![0.0f64; m];
    for (&i, &v) in row_idx.iter().zip(vals) {
        let av = v.abs();
        if av > rmax[i] {
            rmax[i] = av;
        }
    }
    rmax.iter()
        .map(|&h| if h > 0.0 { nearest_pow2(1.0 / h) } else { 1.0 })
        .collect()
}

/// Round `v > 0` to the nearest power of two (an exact float scale factor).
#[inline]
fn nearest_pow2(v: f64) -> f64 {
    if !v.is_finite() || v <= 0.0 {
        return 1.0;
    }
    2f64.powi(v.log2().round() as i32)
}

/// Diagonal row/column equilibration factors for a fixed constraint matrix `A`,
/// derived from `A` alone. Because the factors depend only on the matrix — not on
/// the rhs or bounds — one `Scaling` can be reused to scale a whole family of LPs
/// that share `A` (the B&B tree, a multi-rhs or batched solve): compute it once,
/// then apply the cheap per-vector transforms below.
pub struct Scaling {
    row: Vec<f64>, // r_i, length m
    col: Vec<f64>, // c_j, length n  (also the x unscale factors)
    m: usize,
    n: usize,
}

impl Scaling {
    /// Equilibration factors for the dense row-major `m × n` matrix `a`, or
    /// `None` when its dynamic range is below [`SCALE_TRIGGER`] (well-conditioned
    /// — the caller should solve unscaled, which is then bit-identical to before).
    pub fn from_matrix(a: &[f64], m: usize, n: usize) -> Option<Scaling> {
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        for &v in a.iter() {
            let av = v.abs();
            if av > 0.0 {
                lo = lo.min(av);
                hi = hi.max(av);
            }
        }
        if hi == 0.0 || hi / lo <= SCALE_TRIGGER {
            return None;
        }
        let (row, col) = equilibrate(&SparseCols::from_dense(a, m, n), m, n);
        Some(Scaling { row, col, m, n })
    }

    /// Equilibration factors for an already-CSC matrix, or `None` when its dynamic
    /// range is below [`SCALE_TRIGGER`]. The sparse-native entry: identical factors
    /// to [`Self::from_matrix`] (zeros never affect a line's significant min/max),
    /// with no dense `m·n` materialization anywhere on the path.
    pub fn from_sparse(sp: &SparseCols, m: usize, n: usize) -> Option<Scaling> {
        let (lo, hi) = sp.value_range();
        if hi == 0.0 || hi / lo <= SCALE_TRIGGER {
            return None;
        }
        let (row, col) = equilibrate(sp, m, n);
        Some(Scaling { row, col, m, n })
    }

    /// Scale a CSC matrix in place by the same `R A C` factors — the sparse
    /// analogue of [`Self::scale_matrix`], `O(nnz)` with no dense copy.
    pub fn scale_cols(&self, sp: &mut SparseCols) {
        sp.scale_in_place(&self.row, &self.col);
    }

    /// Undo an in-place [`scale_cols`] on `sp`, recovering the original matrix
    /// **exactly**: multiply by the reciprocal `R⁻¹ · Â · C⁻¹` factors. Every
    /// factor is a power of two (see [`nearest_pow2`]), so its reciprocal and the
    /// product are exact floating-point operations — `unscale_cols(scale_cols(A))`
    /// is bit-identical to `A`. Used by the #649 unscaled-fallback to reconstruct
    /// the original matrix for a trusted cold retry with no clone on the success
    /// path (a scaled solve that succeeds never calls this).
    pub fn unscale_cols(&self, sp: &mut SparseCols) {
        let rinv: Vec<f64> = self.row.iter().map(|&r| 1.0 / r).collect();
        let cinv: Vec<f64> = self.col.iter().map(|&c| 1.0 / c).collect();
        sp.scale_in_place(&rinv, &cinv);
    }

    /// Scaled matrix `Â = R A C` (owned, row-major `m × n`).
    pub fn scale_matrix(&self, a: &[f64]) -> Vec<f64> {
        let (m, n) = (self.m, self.n);
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            let ri = self.row[i];
            for j in 0..n {
                out[i * n + j] = ri * a[i * n + j] * self.col[j];
            }
        }
        out
    }

    /// Scaled objective `ĉ = C c` (objective value is then invariant: `ĉᵀx̂ = cᵀx`).
    pub fn scale_c(&self, c: &[f64]) -> Vec<f64> {
        (0..self.n).map(|j| self.col[j] * c[j]).collect()
    }

    /// Scaled rhs `b̂ = R b`.
    pub fn scale_b(&self, b: &[f64]) -> Vec<f64> {
        (0..self.m).map(|i| self.row[i] * b[i]).collect()
    }

    /// Scaled lower bounds `l̂ = C⁻¹ l` (infinite bounds preserved).
    pub fn scale_lower(&self, l: &[f64]) -> Vec<f64> {
        (0..self.n)
            .map(|j| {
                if l[j] <= -INF {
                    l[j]
                } else {
                    l[j] / self.col[j]
                }
            })
            .collect()
    }

    /// Scaled upper bounds `û = C⁻¹ u` (infinite bounds preserved).
    pub fn scale_upper(&self, u: &[f64]) -> Vec<f64> {
        (0..self.n)
            .map(|j| {
                if u[j] >= INF {
                    u[j]
                } else {
                    u[j] / self.col[j]
                }
            })
            .collect()
    }

    /// Map a scaled primal point back to the original space in place: the first
    /// `n` entries are scaled `x_j ← c_j x̂_j` (exact — `c_j` is a power of two).
    /// Any trailing entries are left untouched.
    pub fn unscale_x(&self, x: &mut [f64]) {
        for (j, c) in self.col.iter().enumerate() {
            if j >= x.len() {
                break;
            }
            x[j] *= *c;
        }
    }

    /// Map scaled row duals (or a scaled Farkas ray) back to the original space in
    /// place: `y_i ← r_i ŷ_i`. The scaled constraint `i` was multiplied by `r_i`,
    /// so its multiplier scales by the same factor (equivalently, the safe dual
    /// bound `bᵀy + Σⱼ min_box((c−Aᵀy)ⱼ zⱼ)` is invariant under this map — see the
    /// module identity `b̂ᵀŷ = bᵀ(Rŷ)` and `ĉ − Âᵀŷ = C(c − Aᵀ(Rŷ))`).
    pub fn unscale_dual(&self, y: &mut [f64]) {
        for (i, r) in self.row.iter().enumerate() {
            if i >= y.len() {
                break;
            }
            y[i] *= *r;
        }
    }

    /// Map a scaled primal ray back to the original space: identical column
    /// transform to [`Self::unscale_x`] (the ray lives in the primal `x` space).
    pub fn unscale_ray(&self, d: &mut [f64]) {
        self.unscale_x(d);
    }
}

/// An owned, equilibrated copy of an [`LpView`] together with its RHS, for a
/// single solve. Thin convenience wrapper over [`Scaling`].
pub struct ScaledLp {
    scaling: Scaling,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    l: Vec<f64>,
    u: Vec<f64>,
}

impl ScaledLp {
    /// Equilibrate `lp` (with rhs `b`) if its dynamic range exceeds
    /// [`SCALE_TRIGGER`]; otherwise return `None` so the caller solves the
    /// original system unchanged (zero copy, bit-identical).
    pub fn maybe_new(lp: &LpView<'_>, b: &[f64]) -> Option<ScaledLp> {
        let scaling = Scaling::from_matrix(lp.a, lp.m, lp.n)?;
        Some(ScaledLp {
            a: scaling.scale_matrix(lp.a),
            b: scaling.scale_b(b),
            c: scaling.scale_c(lp.c),
            l: scaling.scale_lower(lp.l),
            u: scaling.scale_upper(lp.u),
            scaling,
        })
    }

    /// Borrow the scaled LP as an [`LpView`].
    pub fn view(&self) -> LpView<'_> {
        LpView {
            a: &self.a,
            m: self.scaling.m,
            n: self.scaling.n,
            c: &self.c,
            l: &self.l,
            u: &self.u,
        }
    }

    /// The scaled right-hand side.
    pub fn b(&self) -> &[f64] {
        &self.b
    }

    /// Map a scaled primal point back to the original space in place.
    pub fn unscale_x(&self, x: &mut [f64]) {
        self.scaling.unscale_x(x);
    }

    /// Map scaled row duals / a scaled Farkas ray back to the original space.
    pub fn unscale_dual(&self, y: &mut [f64]) {
        self.scaling.unscale_dual(y);
    }

    /// Map a scaled primal ray back to the original space.
    pub fn unscale_ray(&self, d: &mut [f64]) {
        self.scaling.unscale_ray(d);
    }
}

/// Geometric-mean equilibration of the `m × n` matrix `a`: alternating sweeps set
/// each column/row factor to `1/sqrt(min·max)` of the current scaled magnitudes in
/// that line, snapped to a power of two. Returns `(row, col)` factor vectors.
///
/// The sweeps read the matrix through a CSC view so they visit only the nonzeros,
/// in cache-friendly storage order, rather than striding the dense row-major
/// matrix per column — `O(nnz)` instead of `O(m·n)` cache-missing accesses, which
/// was the dominant cost on the 0.3%-dense lifted McCormick relaxations (a
/// zero-iteration solve of ex1252's 6908×7799 / 19k-nonzero relaxation spent
/// seconds here alone). The factors are *bit-identical* to the old dense sweep: a
/// zero entry never affects a line's significant min/max, so skipping the
/// structural zeros changes nothing.
fn equilibrate(sp: &SparseCols, m: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    equilibrate_with(sp, m, n, row_prescale_enabled())
}

/// [`equilibrate`] with the #1537 row pre-pass chosen explicitly (`prescale`), so
/// both arms are testable without touching the process environment.
fn equilibrate_with(sp: &SparseCols, m: usize, n: usize, prescale: bool) -> (Vec<f64>, Vec<f64>) {
    let (col_ptr, row_idx, vals) = sp.raw();
    // #1537: optionally start from the exact pow2 row normalisation so the first
    // column sweep's noise filter is row-scale invariant (see
    // [`row_prescale_enabled`]). OFF keeps the all-ones start, bit-identical.
    let mut row = if prescale {
        row_prescale_factors(sp, m)
    } else {
        vec![1.0f64; m]
    };
    let mut col = vec![1.0f64; n];
    for _ in 0..MAX_PASSES {
        let mut changed = false;
        // Column sweep: column-major storage makes each column's nonzeros contiguous.
        for j in 0..n {
            let (lo, hi) = line_lo_hi(
                (col_ptr[j]..col_ptr[j + 1]).map(|k| row[row_idx[k]] * vals[k] * col[j]),
            );
            if hi > 0.0 {
                let f = nearest_pow2(1.0 / (lo * hi).sqrt());
                if f != 1.0 {
                    col[j] *= f;
                    changed = true;
                }
            }
        }
        // Row sweep: accumulate per-row significant lo/hi over the same nonzeros.
        // Two O(nnz) passes mirror `line_lo_hi`'s noise floor — the per-row max
        // first, then the min over entries within `MAX_LINE_RANGE` of it.
        let mut rhi = vec![0.0f64; m];
        for j in 0..n {
            let cj = col[j];
            for k in col_ptr[j]..col_ptr[j + 1] {
                let i = row_idx[k];
                let av = (row[i] * vals[k] * cj).abs();
                if av > rhi[i] {
                    rhi[i] = av;
                }
            }
        }
        let mut rlo = vec![f64::INFINITY; m];
        for j in 0..n {
            let cj = col[j];
            for k in col_ptr[j]..col_ptr[j + 1] {
                let i = row_idx[k];
                let av = (row[i] * vals[k] * cj).abs();
                if av > 0.0 && av >= rhi[i] * MAX_LINE_RANGE && av < rlo[i] {
                    rlo[i] = av;
                }
            }
        }
        for i in 0..m {
            if rhi[i] > 0.0 {
                let f = nearest_pow2(1.0 / (rlo[i] * rhi[i]).sqrt());
                if f != 1.0 {
                    row[i] *= f;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    (row, col)
}

/// Smallest and largest *significant* magnitudes along a matrix line (row or
/// column). The equilibration factor is `1/sqrt(lo·hi)`, so a spurious
/// near-zero entry — e.g. a ~1e-16 cut coefficient that is numerical noise, not
/// structure — would otherwise collapse `lo` and blow the factor up to ~1e8,
/// over-scaling the column and pinning its variable's bounds to ~0 in scaled
/// space (then unscaling a within-tolerance scaled value into a gross
/// bound violation). Ignoring entries more than [`MAX_LINE_RANGE`] below the
/// line's max bounds the per-line dynamic range and keeps the factor sane;
/// genuinely structural small coefficients are unaffected.
fn line_lo_hi(vals: impl Iterator<Item = f64>) -> (f64, f64) {
    let mags: Vec<f64> = vals.map(f64::abs).filter(|v| *v > 0.0).collect();
    let hi = mags.iter().copied().fold(0.0f64, f64::max);
    if hi == 0.0 {
        return (f64::INFINITY, 0.0);
    }
    let floor = hi * MAX_LINE_RANGE;
    let lo = mags
        .iter()
        .copied()
        .filter(|v| *v >= floor)
        .fold(f64::INFINITY, f64::min);
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow2_rounding() {
        assert_eq!(nearest_pow2(1.0), 1.0);
        assert_eq!(nearest_pow2(3.0), 4.0); // log2(3)=1.58 → 2^2
        assert_eq!(nearest_pow2(1.4), 1.0); // log2(1.4)=0.49 → 2^0
        assert_eq!(nearest_pow2(0.2), 0.25); // log2(0.2)=-2.32 → 2^-2
        assert_eq!(nearest_pow2(0.0), 1.0);
        assert_eq!(nearest_pow2(f64::INFINITY), 1.0);
    }

    #[test]
    fn well_conditioned_is_no_op() {
        // Entries in [1,5]: dynamic range 5 ≪ trigger → no scaling.
        let a = [5.0, 5.0, 5.0, 5.0, 1.0];
        let lp = LpView {
            a: &a,
            m: 1,
            n: 5,
            c: &[0.0; 5],
            l: &[0.0; 5],
            u: &[1.0; 5],
        };
        assert!(ScaledLp::maybe_new(&lp, &[9.0]).is_none());
    }

    #[test]
    fn wide_range_triggers_and_equilibrates() {
        // One huge and one tiny coefficient: range 1e9 ≫ trigger → scaled, and
        // the equilibrated matrix has its nonzero magnitudes brought together.
        let a = [1e9, 1.0, 1.0, 1e-3];
        let lp = LpView {
            a: &a,
            m: 2,
            n: 2,
            c: &[1.0, 1.0],
            l: &[0.0, 0.0],
            u: &[1e12, 1e12],
        };
        let scaled = ScaledLp::maybe_new(&lp, &[1.0, 1.0]).expect("should scale");
        let sv = scaled.view();
        let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
        for &v in sv.a.iter() {
            let av = v.abs();
            if av > 0.0 {
                lo = lo.min(av);
                hi = hi.max(av);
            }
        }
        // Raw range was 1e9/1e-3 = 1e12; equilibration must shrink it sharply.
        assert!(hi / lo < 1e4, "post-scale range {} too wide", hi / lo);
    }

    #[test]
    fn noise_entry_does_not_overscale_column() {
        // Column 0 has a genuine unit entry and a ~1e-16 noise entry (e.g. a cut
        // coefficient that is float noise). The noise must not collapse the
        // column's geometric-mean factor: without the dynamic-range floor it
        // would scale the column by ~1e8, pinning the variable's scaled bounds to
        // ~0 so a feasible scaled point unscales far out of its box (a [0,1] var
        // returned at -1). The structural entry is 1.0, so the factor must stay
        // O(1).
        let a = [
            1.0, 1.0, // row 0
            1e-16, 1.0, // row 1: col 0's entry here is pure noise
        ];
        let scaling = Scaling::from_matrix(&a, 2, 2).expect("noise widens range past the trigger");
        let su = scaling.scale_upper(&[1.0, 1.0]);
        assert!(
            su[0] > 1e-3,
            "column 0 over-scaled by a noise entry: scaled upper bound {}",
            su[0]
        );
    }

    #[test]
    fn unscale_recovers_original_scale() {
        // Scaling derived from a wide-range matrix; x̂ maps back exactly by the
        // (power-of-two) column factors, and scale_lower/upper invert it.
        let a = [1e9, 0.0, 0.0, 1.0];
        let scaling = Scaling::from_matrix(&a, 2, 2).expect("should scale");
        let l = scaling.scale_lower(&[4.0, 8.0]);
        let mut x = l.clone();
        scaling.unscale_x(&mut x);
        // unscale ∘ scale_lower is the identity (exact for power-of-two factors).
        assert_eq!(x, vec![4.0, 8.0]);
        // Infinite bounds are preserved, not divided.
        assert_eq!(scaling.scale_upper(&[INF, 1e12])[0], INF);
    }

    #[test]
    fn shared_scaling_matches_single_solve_transform() {
        // The reusable Scaling applied piecewise must reproduce ScaledLp's combined
        // transform (the batch path and the single-solve path agree).
        let a = [1e8, 2.0, 3.0, 1e-2];
        let lp = LpView {
            a: &a,
            m: 2,
            n: 2,
            c: &[5.0, 7.0],
            l: &[0.0, 0.0],
            u: &[1e10, 1e10],
        };
        let b = [1.0, 2.0];
        let single = ScaledLp::maybe_new(&lp, &b).expect("should scale");
        let sc = Scaling::from_matrix(lp.a, lp.m, lp.n).expect("should scale");
        assert_eq!(single.a, sc.scale_matrix(lp.a));
        assert_eq!(single.b, sc.scale_b(&b));
        assert_eq!(single.c, sc.scale_c(lp.c));
        assert_eq!(single.l, sc.scale_lower(lp.l));
        assert_eq!(single.u, sc.scale_upper(lp.u));
    }

    // ---------------------------------------------------------------------- //
    // #1537: the row-normalisation pre-pass (`DISCOPT_LP_ROW_PRESCALE`).
    // ---------------------------------------------------------------------- //

    /// The witness shape from the m3 / flay03m masters with rows scaled in
    /// 10^[-6,6]: row 0 is `1e5·(x + y)`, row 1 is `1e-6·(x - y)`, and each row
    /// carries its engine slack with coefficient 1, which no row scaling touches.
    /// Column `x` holds `1e5` and `1e-6`: a ratio of 1e-11, under the noise floor.
    #[rustfmt::skip]
    const WITNESS: [f64; 8] = [
        1e5,  1e5,  1.0, 0.0,
        1e-6, -1e-6, 0.0, 1.0,
    ];

    /// Entries of the scaled matrix `R A C` that sit below [`MAX_LINE_RANGE`]
    /// times their row's or column's largest magnitude -- what the simplex treats
    /// as noise.
    fn noise_entries(a: &[f64], m: usize, n: usize, row: &[f64], col: &[f64]) -> usize {
        let s: Vec<f64> = (0..m * n)
            .map(|k| (row[k / n] * a[k] * col[k % n]).abs())
            .collect();
        let rmax: Vec<f64> = (0..m)
            .map(|i| (0..n).map(|j| s[i * n + j]).fold(0.0, f64::max))
            .collect();
        let cmax: Vec<f64> = (0..n)
            .map(|j| (0..m).map(|i| s[i * n + j]).fold(0.0, f64::max))
            .collect();
        (0..m * n)
            .filter(|&k| {
                s[k] > 0.0
                    && (s[k] < MAX_LINE_RANGE * rmax[k / n] || s[k] < MAX_LINE_RANGE * cmax[k % n])
            })
            .count()
    }

    #[test]
    fn row_prescale_flag_parses_and_refuses() {
        for on in ["1", "true", "YES", " 1 "] {
            assert_eq!(parse_row_prescale_flag(on), Ok(true), "{on:?}");
        }
        for off in ["0", "false", "no"] {
            assert_eq!(parse_row_prescale_flag(off), Ok(false), "{off:?}");
        }
        assert_eq!(parse_row_prescale_flag(""), Ok(ROW_PRESCALE_DEFAULT));
        assert!(parse_row_prescale_flag("on").is_err());
        assert!(parse_row_prescale_flag("2").is_err());
    }

    #[test]
    fn row_prescale_factors_are_exact_powers_of_two() {
        let sp = SparseCols::from_dense(&WITNESS, 2, 4);
        let r = row_prescale_factors(&sp, 2);
        for &ri in &r {
            let (mant, _) = frexp(ri);
            assert_eq!(mant, 0.5, "{ri} is not a power of two");
        }
        // Row 0's largest entry 1e5 lands within a factor sqrt(2) of 1; row 1's
        // largest entry is the slack's 1, so its factor is 1.
        assert!((r[0] * 1e5 - 1.0).abs() <= std::f64::consts::SQRT_2 - 1.0 + 1e-12);
        assert_eq!(r[1], 1.0);
        // An empty row keeps factor 1.
        let empty = SparseCols::from_dense(&[0.0, 0.0, 3.0, 0.0], 2, 2);
        assert_eq!(row_prescale_factors(&empty, 2)[0], 1.0);
    }

    fn frexp(v: f64) -> (f64, i32) {
        let e = v.abs().log2().floor() as i32 + 1;
        (v / 2f64.powi(e), e)
    }

    #[test]
    fn prescaled_unscale_is_bit_exact() {
        // The certificate is published in original units, so the round trip
        // through the pre-scaled factors must be the identity, bit for bit: the
        // matrix, the primal point, and the dual map `y = R ŷ` all go through
        // power-of-two products only.
        let (m, n) = (2, 4);
        let mut sp = SparseCols::from_dense(&WITNESS, m, n);
        let (row, col) = equilibrate_with(&sp, m, n, true);
        let sc = Scaling { row, col, m, n };
        let orig = sp.clone();
        sc.scale_cols(&mut sp);
        sc.unscale_cols(&mut sp);
        assert_eq!(sp.raw(), orig.raw());

        let x = [3.25, -7.0e-3, 1.5e6, 0.1];
        let l = sc.scale_lower(&x);
        let mut back = l.clone();
        sc.unscale_x(&mut back);
        assert_eq!(back.to_vec(), x.to_vec());

        // b̂ᵀŷ = bᵀ(Rŷ) exactly for a power-of-two R.
        let b = [3.5e5, 1e-6];
        let bh = sc.scale_b(&b);
        let yh = [0.37, -2.5];
        let mut y = yh;
        sc.unscale_dual(&mut y);
        assert_eq!(bh[0] * yh[0], b[0] * y[0]);
        assert_eq!(bh[1] * yh[1], b[1] * y[1]);
    }

    #[test]
    fn prescale_keeps_the_witness_entry_significant() {
        let (m, n) = (2, 4);
        let sp = SparseCols::from_dense(&WITNESS, m, n);
        // Anti-vacuity (CLAUDE.md §6): the legacy factors DO leave a noise entry
        // on this shape, so the ON assertion below is about the fix, not the data.
        let (r0, c0) = equilibrate_with(&sp, m, n, false);
        assert!(noise_entries(&WITNESS, m, n, &r0, &c0) > 0);
        let (r1, c1) = equilibrate_with(&sp, m, n, true);
        assert_eq!(noise_entries(&WITNESS, m, n, &r1, &c1), 0);
    }

    #[test]
    fn prescale_leaves_no_noise_entry_on_either_spelling() {
        // The same LP with its rows written at unit scale (the "as written" model).
        // After the pre-pass both spellings equilibrate to matrices with no noise
        // entry.
        #[rustfmt::skip]
        let unit = [
            1.0, 1.0, 1.0, 0.0,
            1.0, -1.0, 0.0, 1.0,
        ];
        let (m, n) = (2, 4);
        for a in [&unit[..], &WITNESS[..]] {
            let sp = SparseCols::from_dense(a, m, n);
            let (r, c) = equilibrate_with(&sp, m, n, true);
            assert_eq!(noise_entries(a, m, n, &r, &c), 0);
        }
    }
}
