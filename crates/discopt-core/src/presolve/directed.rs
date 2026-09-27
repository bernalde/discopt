//! Directed (outward) rounding primitives for rigorous interval arithmetic (#1504).
//!
//! # Why this exists
//!
//! FBBT's interval engine (`fbbt.rs`) computed every endpoint in IEEE
//! round-to-nearest. A single inward-rounded endpoint is harmless on its own --
//! it is one ulp -- but the backward pass then *inverts* functions whose inverse
//! is not Lipschitz, and an ulp-level error becomes a bound shift orders of
//! magnitude larger. Issue #1504 measured the chain on
//! `b**3 - 2*x0**3 == 5`, `x0 in [-2, 0]`, `b` binary:
//!
//! 1. backward through `x0**3 in [-2.5, -2]` gave `x0 >= -(2.5).powf(1/3)`
//!    `= -1.3572088082974532`, which lies ABOVE the true `-cbrt(2.5)`: in exact
//!    rational arithmetic `5 + 2*x0_lo**3 = +1.368e-15 > 0`;
//! 2. the next forward pass therefore gave `b**3 >= 1.776e-15`;
//! 3. the cube-root inverse turned that into `b >= 1.2110908904786693e-05`,
//!    above the integrality snap tolerance, so `b` was fixed to 1 and the true
//!    optimum `b = 0` was cut -- a certified wrong optimum.
//!
//! No fixed ulp or fixed absolute margin can close this: the error a root
//! amplifies is set by the magnitudes of the terms that cancelled upstream, and
//! a root of a residual `d` is `d^(1/n)`. The fix is to make every endpoint an
//! honest enclosure, so the residual that reaches a root is `<= 0` whenever the
//! true value is, and no inversion ever sees a positive phantom.
//!
//! # What the primitives guarantee
//!
//! * `+ - * /` and `sqrt` are correctly rounded in IEEE-754, and their rounding
//!   error is recovered EXACTLY by an error-free transform (TwoSum, FMA
//!   remainder). The `_down`/`_up` variants return the round-to-nearest result
//!   when it is exact, and the adjacent float in the outward direction when it
//!   is not -- i.e. true directed rounding. Exact results (integer data, `0`,
//!   `2 * 0.5`) are left untouched, so bounds on well-scaled models do not move.
//! * `powi_down`/`powi_up` chain the directed products (square-and-multiply on
//!   non-negative operands, where directed rounding is monotone).
//! * `root_down`/`root_up` return a float VERIFIED against `powi_up`/`powi_down`
//!   to lie on the correct side of the true real root; if verification cannot
//!   be achieved they fall back to the trivially sound `0` / `+inf`.
//! * Library transcendental functions are not correctly rounded; `lib_down` /
//!   `lib_up` widen a libm result outward by a relative margin
//!   ([`LIB_ULPS`] ulps) plus one ulp, which covers the < 1 ulp error bound of
//!   the platform libm with room to spare. `abs_down`/`abs_up` add an absolute
//!   component for expressions whose error is absolute rather than relative
//!   (an argument shift by a rounded `pi`, a `1 - p` subtraction).
//!
//! Non-finite values pass through unchanged: `+-inf` is already an outward
//! bound, and `NaN` is the callers' existing concern (e.g. `0 * inf` in
//! `interval_mul`).

/// Relative widening, in units of `f64::EPSILON`, applied to libm results.
pub const LIB_ULPS: f64 = 4.0;

/// Below this magnitude an FMA remainder may underflow and stop being exact, so
/// the exactness test is skipped and the result is widened unconditionally.
const TINY: f64 = 1.0e-290;

/// The next float above `x` (`f64::next_up`, which needs Rust 1.86; MSRV 1.84).
#[inline]
pub fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        // Covers -0.0 too: the smallest positive subnormal.
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// The next float below `x`.
#[inline]
pub fn next_down(x: f64) -> f64 {
    -next_up(-x)
}

/// Exact rounding error of `s = fl(a + b)` (TwoSum, Knuth): `a + b = s + err`.
#[inline]
fn two_sum_err(a: f64, b: f64, s: f64) -> f64 {
    let bb = s - a;
    (a - (s - bb)) + (b - bb)
}

/// `a + b` rounded toward `-inf`.
#[inline]
pub fn add_down(a: f64, b: f64) -> f64 {
    let s = a + b;
    if !s.is_finite() {
        return s;
    }
    if two_sum_err(a, b, s) < 0.0 {
        next_down(s)
    } else {
        s
    }
}

/// `a + b` rounded toward `+inf`.
#[inline]
pub fn add_up(a: f64, b: f64) -> f64 {
    let s = a + b;
    if !s.is_finite() {
        return s;
    }
    if two_sum_err(a, b, s) > 0.0 {
        next_up(s)
    } else {
        s
    }
}

/// `a - b` rounded toward `-inf`.
#[inline]
pub fn sub_down(a: f64, b: f64) -> f64 {
    add_down(a, -b)
}

/// `a - b` rounded toward `+inf`.
#[inline]
pub fn sub_up(a: f64, b: f64) -> f64 {
    add_up(a, -b)
}

/// `a * b` rounded toward `-inf`. A NaN (`0 * inf`) is returned as NaN.
#[inline]
pub fn mul_down(a: f64, b: f64) -> f64 {
    let p = a * b;
    if !p.is_finite() || a == 0.0 || b == 0.0 {
        return p;
    }
    if p.abs() < TINY {
        return next_down(p);
    }
    // a*b = p + e exactly (FMA error-free transform).
    if a.mul_add(b, -p) < 0.0 {
        next_down(p)
    } else {
        p
    }
}

/// `a * b` rounded toward `+inf`. A NaN (`0 * inf`) is returned as NaN.
#[inline]
pub fn mul_up(a: f64, b: f64) -> f64 {
    let p = a * b;
    if !p.is_finite() || a == 0.0 || b == 0.0 {
        return p;
    }
    if p.abs() < TINY {
        return next_up(p);
    }
    if a.mul_add(b, -p) > 0.0 {
        next_up(p)
    } else {
        p
    }
}

/// Sign of `a/b - fl(a/b)`: `-1`, `0` or `+1`; `None` when not decidable exactly.
#[inline]
fn div_err_sign(a: f64, b: f64, q: f64) -> Option<i8> {
    if q.abs() < TINY || a.abs() < TINY || b.abs() < TINY {
        return None;
    }
    // r = a - q*b is exact for q = RN(a/b) without underflow; a/b - q = r/b.
    let r = (-q).mul_add(b, a);
    if r == 0.0 {
        Some(0)
    } else if (r > 0.0) == (b > 0.0) {
        Some(1)
    } else {
        Some(-1)
    }
}

/// `a / b` rounded toward `-inf`.
#[inline]
pub fn div_down(a: f64, b: f64) -> f64 {
    let q = a / b;
    if !q.is_finite() || a == 0.0 || b.is_infinite() {
        return q;
    }
    match div_err_sign(a, b, q) {
        Some(s) if s >= 0 => q,
        _ => next_down(q),
    }
}

/// `a / b` rounded toward `+inf`.
#[inline]
pub fn div_up(a: f64, b: f64) -> f64 {
    let q = a / b;
    if !q.is_finite() || a == 0.0 || b.is_infinite() {
        return q;
    }
    match div_err_sign(a, b, q) {
        Some(s) if s <= 0 => q,
        _ => next_up(q),
    }
}

/// `sqrt(x)` rounded toward `-inf` (`x >= 0`).
#[inline]
pub fn sqrt_down(x: f64) -> f64 {
    let s = x.sqrt();
    if !s.is_finite() || x == 0.0 {
        return s;
    }
    if x < TINY {
        return next_down(s).max(0.0);
    }
    // x - s*s exact; negative means s overshoots the true root.
    if (-s).mul_add(s, x) < 0.0 {
        next_down(s)
    } else {
        s
    }
}

/// `sqrt(x)` rounded toward `+inf` (`x >= 0`).
#[inline]
pub fn sqrt_up(x: f64) -> f64 {
    let s = x.sqrt();
    if !s.is_finite() || x == 0.0 {
        return s;
    }
    if x < TINY {
        return next_up(s);
    }
    if (-s).mul_add(s, x) > 0.0 {
        next_up(s)
    } else {
        s
    }
}

/// `x^n` for `x >= 0`, every product rounded in one direction. Directed rounding
/// is monotone on non-negative operands, so this bounds the true power.
fn pow_nonneg_dir(x: f64, n: u64, up: bool) -> f64 {
    let mul = |a: f64, b: f64| if up { mul_up(a, b) } else { mul_down(a, b) };
    let mut result = 1.0_f64;
    let mut base = x;
    let mut k = n;
    while k > 0 {
        if k & 1 == 1 {
            result = mul(result, base);
        }
        k >>= 1;
        if k > 0 {
            base = mul(base, base);
        }
    }
    result
}

/// `x^n` (integer `n >= 0`) rounded toward `-inf`.
pub fn powi_down(x: f64, n: u64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x >= 0.0 {
        pow_nonneg_dir(x, n, false)
    } else if n % 2 == 0 {
        pow_nonneg_dir(-x, n, false)
    } else {
        -pow_nonneg_dir(-x, n, true)
    }
}

/// `x^n` (integer `n >= 0`) rounded toward `+inf`.
pub fn powi_up(x: f64, n: u64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x >= 0.0 {
        pow_nonneg_dir(x, n, true)
    } else if n % 2 == 0 {
        pow_nonneg_dir(-x, n, true)
    } else {
        -pow_nonneg_dir(-x, n, false)
    }
}

/// Round-to-nearest-ish candidate for the real `n`-th root of `t >= 0`.
fn root_candidate(t: f64, n: u64) -> f64 {
    match n {
        1 => t,
        2 => t.sqrt(),
        3 => t.cbrt(),
        _ => t.powf(1.0 / n as f64),
    }
}

/// Maximum verification steps before a root falls back to its trivial bound.
const ROOT_MAX_STEPS: u32 = 64;

/// A float `r >= 0` with `r <= t^(1/n)` (true real root), for `t >= 0`.
fn root_nonneg_down(t: f64, n: u64) -> f64 {
    if t.is_nan() || t <= 0.0 || n == 0 {
        return 0.0;
    }
    if t == f64::INFINITY {
        return f64::INFINITY;
    }
    let mut r = root_candidate(t, n);
    for i in 0..ROOT_MAX_STEPS {
        if !r.is_finite() || r <= 0.0 {
            return 0.0;
        }
        // r^n <= t  (upper-rounded power)  =>  r <= t^(1/n).
        if powi_up(r, n) <= t {
            return r;
        }
        let d = r * f64::EPSILON * f64::from(1u32 << i.min(30));
        r = next_down(r - d);
    }
    0.0
}

/// A float `r` with `r >= t^(1/n)` (true real root), for `t >= 0`.
fn root_nonneg_up(t: f64, n: u64) -> f64 {
    if t.is_nan() || n == 0 {
        return f64::INFINITY;
    }
    if t <= 0.0 {
        return 0.0;
    }
    if t == f64::INFINITY {
        return f64::INFINITY;
    }
    let mut r = root_candidate(t, n);
    for i in 0..ROOT_MAX_STEPS {
        if !r.is_finite() {
            return f64::INFINITY;
        }
        // r^n >= t  (lower-rounded power)  =>  r >= t^(1/n).
        if r > 0.0 && powi_down(r, n) >= t {
            return r;
        }
        let d = r.abs() * f64::EPSILON * f64::from(1u32 << i.min(30));
        r = next_up(r + d);
    }
    f64::INFINITY
}

/// Lower bound on the real `n`-th root of `t`. For odd `n` the root is signed;
/// for even `n` the caller must pass `t >= 0` (a negative `t` is clamped to 0).
pub fn root_down(t: f64, n: u64) -> f64 {
    if t < 0.0 && n % 2 == 1 {
        -root_nonneg_up(-t, n)
    } else {
        root_nonneg_down(t.max(0.0), n)
    }
}

/// Upper bound on the real `n`-th root of `t` (see [`root_down`]).
pub fn root_up(t: f64, n: u64) -> f64 {
    if t < 0.0 && n % 2 == 1 {
        -root_nonneg_down(-t, n)
    } else {
        root_nonneg_up(t.max(0.0), n)
    }
}

/// Widen a libm result `v` down by a relative margin: `LIB_ULPS` ulps + 1.
#[inline]
pub fn lib_down(v: f64) -> f64 {
    if !v.is_finite() {
        return v;
    }
    next_down(v - LIB_ULPS * f64::EPSILON * v.abs())
}

/// Widen a libm result `v` up by a relative margin: `LIB_ULPS` ulps + 1.
#[inline]
pub fn lib_up(v: f64) -> f64 {
    if !v.is_finite() {
        return v;
    }
    next_up(v + LIB_ULPS * f64::EPSILON * v.abs())
}

/// Widen `v` down by a margin relative to `|v| + scale`, for a quantity whose
/// error is absolute at magnitude `scale` (e.g. an argument shifted by a rounded
/// `pi`, or a `1 - p` that lost digits).
#[inline]
pub fn abs_down(v: f64, scale: f64) -> f64 {
    if !v.is_finite() {
        return v;
    }
    next_down(v - LIB_ULPS * f64::EPSILON * (v.abs() + scale.abs()))
}

/// Widen `v` up by a margin relative to `|v| + scale` (see [`abs_down`]).
#[inline]
pub fn abs_up(v: f64, scale: f64) -> f64 {
    if !v.is_finite() {
        return v;
    }
    next_up(v + LIB_ULPS * f64::EPSILON * (v.abs() + scale.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact rational value of a double, as (numerator, power-of-two exponent),
    /// is overkill here; instead use the error-free transforms' own identities
    /// and known-exact cases.
    #[test]
    fn exact_operations_are_not_widened() {
        assert_eq!(add_down(1.0, 2.0), 3.0);
        assert_eq!(add_up(1.0, 2.0), 3.0);
        assert_eq!(sub_down(5.0, 5.0), 0.0);
        assert_eq!(mul_down(2.0, 0.5), 1.0);
        assert_eq!(mul_up(-3.0, 4.0), -12.0);
        assert_eq!(div_down(6.0, 3.0), 2.0);
        assert_eq!(div_up(1.0, 4.0), 0.25);
        assert_eq!(sqrt_down(9.0), 3.0);
        assert_eq!(sqrt_up(9.0), 3.0);
        assert_eq!(powi_down(2.0, 3), 8.0);
        assert_eq!(powi_up(-2.0, 3), -8.0);
        assert_eq!(powi_down(-2.0, 2), 4.0);
        assert_eq!(root_down(8.0, 3), 2.0);
        assert_eq!(root_up(-8.0, 3), -2.0);
        assert_eq!(root_down(0.0, 5), 0.0);
    }

    #[test]
    fn inexact_operations_bracket_the_true_value() {
        // 0.1 + 0.2 is inexact: the two directions must differ by one ulp.
        let lo = add_down(0.1, 0.2);
        let hi = add_up(0.1, 0.2);
        assert!(lo < hi);
        assert_eq!(next_up(lo), hi);
        // 1/3.
        let lo = div_down(1.0, 3.0);
        let hi = div_up(1.0, 3.0);
        assert!(lo < hi && lo * 3.0 <= 1.0);
        // sqrt(2): lo^2 <= 2 <= hi^2 (checked with the FMA remainder).
        let lo = sqrt_down(2.0);
        let hi = sqrt_up(2.0);
        assert!((-lo).mul_add(lo, 2.0) >= 0.0);
        assert!((-hi).mul_add(hi, 2.0) <= 0.0);
    }

    #[test]
    fn roots_are_verified_outward() {
        // The #1504 root: -(2.5)^(1/3). The lower bound must satisfy
        // x^3 <= -2.5 under upward-rounded cubing, i.e. lie at/below the true root.
        let lo = root_down(-2.5, 3);
        let hi = root_up(-2.5, 3);
        assert!(lo <= hi);
        assert!(powi_up(lo, 3) <= -2.5, "lo={lo:e}");
        assert!(powi_down(hi, 3) >= -2.5, "hi={hi:e}");
        // round-to-nearest powf lands ABOVE the true root here (measured in #1504).
        assert!(lo <= -(2.5f64.powf(1.0 / 3.0)));
        for n in 2..=9u64 {
            for &t in &[1e-300, 1.7e-15, 0.3, 2.0, 5.0, 7.0, 1e10, 1e300] {
                let lo = root_down(t, n);
                let hi = root_up(t, n);
                assert!(lo <= hi, "n={n} t={t:e}");
                assert!(powi_up(lo, n) <= t, "n={n} t={t:e} lo={lo:e}");
                assert!(powi_down(hi, n) >= t, "n={n} t={t:e} hi={hi:e}");
            }
        }
    }

    #[test]
    fn non_finite_values_pass_through() {
        assert_eq!(add_down(f64::NEG_INFINITY, 1.0), f64::NEG_INFINITY);
        assert_eq!(add_up(f64::INFINITY, 1.0), f64::INFINITY);
        assert!(mul_down(0.0, f64::INFINITY).is_nan());
        assert_eq!(lib_down(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(root_up(f64::INFINITY, 3), f64::INFINITY);
        assert_eq!(next_up(-f64::INFINITY), f64::MIN);
    }
}
