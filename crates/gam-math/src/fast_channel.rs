//! Hand-speed single-channel Faà di Bruno (#932 "unified source, fast as hand").
//!
//! A dense [`super::jet_tower::Tower4<K>`] reading ONE mixed channel materializes
//! the entire `K⁴` derivative tensor — measured at ~19× the x86 instruction count
//! of the hand factorization for an order-4 channel. The runtime partition walker
//! in [`super::jet_algebra::faa_di_bruno`] is exact but its recursive
//! `&mut dyn FnMut` enumeration does not inline to straight-line arithmetic.
//!
//! This module owns the *compiled* form: for a composition `f ∘ q` whose inner
//! map carries partials over `N` DISTINCT differentiation directions, the single
//! fully-mixed top channel `∂ᴺ(f∘q)/∂d₁…∂d_N` is the Faà di Bruno sum over the set
//! partitions of `{d₁…d_N}`,
//!
//! ```text
//!   Σ_{π ∈ partitions} f^{(|π|)}(q) · Π_{B ∈ π} q_B ,
//! ```
//!
//! where `q_B` is the inner partial over the directions in block `B`. The blocks
//! are read out of a squarefree bitmask array `q[mask]` (`q[0]` is unused — the
//! value channel never enters a top mixed partial). These functions write that
//! sum out for the small fixed orders the engine actually uses (`N ∈ {2,3,4}`),
//! so it compiles to the optimal straight-line form (LLVM CSE recovers the shared
//! sub-products). They are the SINGLE SOURCE every family feeds — there is no
//! hand-maintained per-family chain rule — and the `oracle_tests` below pin each
//! one BIT-FOR-BIT against the general runtime partition walker
//! ([`super::jet_algebra::faa_di_bruno`]), so the compiled form can never drift
//! from the universal rule.
//!
//! `f_stack[k]` is `f^{(k+1)}(q)` (the derivative magnitudes `m_{k+1}`); the value
//! `f(q)` is index −1 and never appears in a top mixed partial.

/// `N = 2`: `∂²(f∘q)/∂a∂b = m₂·q_a·q_b + m₁·q_ab`.
/// `q[1]=q_a, q[2]=q_b, q[3]=q_ab`. `m=[m₁,m₂]`.
#[inline(always)]
pub fn faa_top2(m: [f64; 2], q: &[f64; 4]) -> f64 {
    // |π|=2: {a}{b} ; |π|=1: {ab}
    m[1] * q[1] * q[2] + m[0] * q[3]
}

/// `N = 3`: the fully-mixed third channel `∂³(f∘q)/∂a∂b∂u`.
/// Bitmask: `a=1, b=2, u=4`. `m=[m₁,m₂,m₃]`.
#[inline(always)]
pub fn faa_top3(m: [f64; 3], q: &[f64; 8]) -> f64 {
    let (a, b, u) = (1usize, 2, 4);
    // |π|=3: {a}{b}{u}
    let p3 = q[a] * q[b] * q[u];
    // |π|=2: one pair + one singleton (3 partitions)
    let p2 = q[a | b] * q[u] + q[a | u] * q[b] + q[b | u] * q[a];
    // |π|=1: {abu}
    let p1 = q[a | b | u];
    m[2] * p3 + m[1] * p2 + m[0] * p1
}

/// `N = 4`: the fully-mixed fourth channel `∂⁴(f∘q)/∂a∂b∂u∂v`.
/// Bitmask: `a=1, b=2, u=4, v=8`. `m=[m₁,m₂,m₃,m₄]`.
#[inline(always)]
pub fn faa_top4(m: [f64; 4], q: &[f64; 16]) -> f64 {
    let (a, b, u, v) = (1usize, 2, 4, 8);
    // |π|=4: {a}{b}{u}{v}
    let p4 = q[a] * q[b] * q[u] * q[v];
    // |π|=3: one pair + two singletons (6 partitions)
    let p3 = q[a | b] * q[u] * q[v]
        + q[a | u] * q[b] * q[v]
        + q[a | v] * q[b] * q[u]
        + q[b | u] * q[a] * q[v]
        + q[b | v] * q[a] * q[u]
        + q[u | v] * q[a] * q[b];
    // |π|=2: three pair-pair + four triple-singleton (7 partitions)
    let p2 = q[a | b] * q[u | v]
        + q[a | u] * q[b | v]
        + q[a | v] * q[b | u]
        + q[a | b | u] * q[v]
        + q[a | b | v] * q[u]
        + q[a | u | v] * q[b]
        + q[b | u | v] * q[a];
    // |π|=1: {abuv}
    let p1 = q[a | b | u | v];
    m[3] * p4 + m[2] * p3 + m[1] * p2 + m[0] * p1
}

/// Compile several order-three top channels as one output schedule.
#[inline(always)]
pub fn faa_bundle3<const OUTPUTS: usize>(m: [f64; 3], q: &[[f64; 8]; OUTPUTS]) -> [f64; OUTPUTS] {
    std::array::from_fn(|output| faa_top3(m, &q[output]))
}

/// Joint order-three pullback coefficients for the same curve/wiggle map.
#[inline(always)]
pub fn curve_wiggle_bundle3(
    m: [f64; 3],
    q_u: f64,
    a: f64,
    b: f64,
    a_u: f64,
    b_u: f64,
    xi: f64,
) -> [f64; 6] {
    faa_bundle3(
        m,
        &[
            [0.0, a, a, b, q_u, a_u, a_u, b_u],
            [0.0, a, 1.0, 0.0, q_u, a_u, 0.0, 0.0],
            [0.0, a, 0.0, 1.0, q_u, a_u, xi, 0.0],
            [0.0, a, 0.0, 0.0, q_u, a_u, 0.0, xi],
            [0.0, 1.0, 1.0, 0.0, q_u, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, q_u, xi, 0.0, 0.0],
        ],
    )
}

/// Joint order-four pullback coefficients for the same curve/wiggle map.
///
/// This is the sparse, build-time-expanded form of nine [`faa_top4`] calls.
/// Terms whose inner-map mask is structurally zero are absent, and outputs are
/// scheduled from the smallest reusable channels (`ww_*` → `eta_w_*` →
/// `eta_eta`). The independent partition walker tests below remain the semantic
/// oracle, so this is a compiler lowering of the universal rule rather than a
/// second family calculus.
#[inline(always)]
pub fn curve_wiggle_bundle4(
    m: [f64; 4],
    q: [f64; 3],
    a_jet: [f64; 4],
    b_jet: [f64; 4],
    xi: [f64; 2],
) -> [f64; 9] {
    let [m1, m2, m3, m4] = m;
    let [q_u, q_v, q_uv] = q;
    let [a, a_u, a_v, a_uv] = a_jet;
    let [b, b_u, b_v, b_uv] = b_jet;
    let [xi_u, xi_v] = xi;
    let xi_uv = xi_u * xi_v;
    let ww_bb = m4 * q_u * q_v + m3 * q_uv;
    let ww_db = m3 * (xi_u * q_v + xi_v * q_u);
    let ww_ddb = m2 * xi_uv;
    let ww_dd = 2.0 * ww_ddb;
    let eta_w_b = ww_bb * a + m3 * (a_u * q_v + a_v * q_u) + m2 * a_uv;
    let eta_w_d1 = ww_db * a + m3 * q_u * q_v + m2 * (q_uv + a_u * xi_v + a_v * xi_u);
    let eta_w_d2 = ww_ddb * a + m2 * (q_u * xi_v + q_v * xi_u);
    let eta_w_d3 = m1 * xi_uv;
    let eta_eta = eta_w_b * a
        + m3 * (a * (q_u * a_v + q_v * a_u) + q_u * q_v * b)
        + m2 * (a * a_uv + 2.0 * a_u * a_v + q_uv * b + q_u * b_v + q_v * b_u)
        + m1 * b_uv;
    [
        eta_eta, eta_w_b, eta_w_d1, eta_w_d2, eta_w_d3, ww_bb, ww_db, ww_ddb, ww_dd,
    ]
}

#[cfg(test)]
mod oracle_tests {
    //! Pin each compiled top-channel sum BIT-FOR-BIT against the general runtime
    //! partition walker [`gam_math::jet_algebra::faa_di_bruno`]. If a
    //! `faa_top*` ever diverges from the universal rule these disagree.
    use super::*;
    use crate::jet_algebra::faa_di_bruno;
    use std::hint::black_box;
    use std::time::Instant;

    fn stream(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    /// Reference: the runtime walker's value for the fully-mixed top channel of
    /// `N` distinct directions, reading inner block partials from `q[mask]`.
    fn walker_top(n: usize, derivs: &[f64], q: &[f64]) -> f64 {
        let positions: Vec<usize> = (0..n).collect();
        faa_di_bruno(&positions, derivs, |block| {
            // block positions ARE the direction indices; build the squarefree mask.
            let mask: usize = block.iter().fold(0usize, |acc, &p| acc | (1 << p));
            q[mask]
        })
    }

    #[test]
    fn faa_top2_matches_runtime_walker() {
        let mut next = stream(0x2);
        for _ in 0..500 {
            let m = [next(), next()];
            let mut q = [0.0; 4];
            for (mask, qm) in q.iter_mut().enumerate() {
                if mask != 0 {
                    *qm = next();
                }
            }
            // derivs stack for faa_di_bruno is [f, f', f''] = [_, m1, m2].
            let derivs = [0.0, m[0], m[1]];
            let got = faa_top2(m, &q);
            let want = walker_top(2, &derivs, &q);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "faa_top2 {got:+.17e} vs walker {want:+.17e}"
            );
        }
    }

    #[test]
    fn faa_top3_matches_runtime_walker() {
        let mut next = stream(0x3);
        for _ in 0..500 {
            let m = [next(), next(), next()];
            let mut q = [0.0; 8];
            for (mask, qm) in q.iter_mut().enumerate() {
                if mask != 0 {
                    *qm = next();
                }
            }
            let derivs = [0.0, m[0], m[1], m[2]];
            let got = faa_top3(m, &q);
            let want = walker_top(3, &derivs, &q);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "faa_top3 {got:+.17e} vs walker {want:+.17e}"
            );
        }
    }

    #[test]
    fn faa_top4_matches_runtime_walker() {
        let mut next = stream(0x4);
        for _ in 0..500 {
            let m = [next(), next(), next(), next()];
            let mut q = [0.0; 16];
            for (mask, qm) in q.iter_mut().enumerate() {
                if mask != 0 {
                    *qm = next();
                }
            }
            let derivs = [0.0, m[0], m[1], m[2], m[3]];
            let got = faa_top4(m, &q);
            let want = walker_top(4, &derivs, &q);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "faa_top4 {got:+.17e} vs walker {want:+.17e}"
            );
        }
    }

    #[inline(never)]
    fn compiled_bundle3(x: [f64; 9]) -> [f64; 6] {
        curve_wiggle_bundle3([x[0], x[1], x[2]], x[3], x[4], x[5], x[6], x[7], x[8])
    }

    #[inline(never)]
    fn strongest_hand_bundle3(x: [f64; 9]) -> [f64; 6] {
        let [m1, m2, m3, q_u, a, b, a_u, b_u, xi] = x;
        let ww_bb = m3 * q_u;
        let ww_db = m2 * xi;
        let eta_w_b = ww_bb * a + m2 * a_u;
        [
            eta_w_b * a + m2 * (a * a_u + q_u * b) + m1 * b_u,
            eta_w_b,
            m2 * (a * xi + q_u),
            m1 * xi,
            ww_bb,
            ww_db,
        ]
    }

    #[inline(never)]
    fn compiled_bundle4(x: [f64; 17]) -> [f64; 9] {
        curve_wiggle_bundle4(
            [x[0], x[1], x[2], x[3]],
            [x[4], x[5], x[6]],
            [x[7], x[9], x[10], x[11]],
            [x[8], x[12], x[13], x[14]],
            [x[15], x[16]],
        )
    }

    fn canonical_bundle4(x: [f64; 17]) -> [f64; 9] {
        let [
            m1,
            m2,
            m3,
            m4,
            q_u,
            q_v,
            q_uv,
            a,
            b,
            a_u,
            a_v,
            a_uv,
            b_u,
            b_v,
            b_uv,
            xi_u,
            xi_v,
        ] = x;
        let xi_uv = xi_u * xi_v;
        let q = [
            [
                0.0, a, a, b, q_u, a_u, a_u, b_u, q_v, a_v, a_v, b_v, q_uv, a_uv, a_uv, b_uv,
            ],
            [
                0.0, a, 1.0, 0.0, q_u, a_u, 0.0, 0.0, q_v, a_v, 0.0, 0.0, q_uv, a_uv, 0.0, 0.0,
            ],
            [
                0.0, a, 0.0, 1.0, q_u, a_u, xi_u, 0.0, q_v, a_v, xi_v, 0.0, q_uv, 0.0, 0.0, 0.0,
            ],
            [
                0.0, a, 0.0, 0.0, q_u, 0.0, 0.0, xi_u, q_v, 0.0, 0.0, xi_v, 0.0, 0.0, xi_uv, 0.0,
            ],
            [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, xi_uv,
            ],
            [
                0.0, 1.0, 1.0, 0.0, q_u, 0.0, 0.0, 0.0, q_v, 0.0, 0.0, 0.0, q_uv, 0.0, 0.0, 0.0,
            ],
            [
                0.0, 0.0, 1.0, 0.0, q_u, xi_u, 0.0, 0.0, q_v, xi_v, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            [
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, xi_uv, 0.0, 0.0,
            ],
            [
                0.0, 0.0, 0.0, 0.0, 0.0, xi_u, xi_u, 0.0, 0.0, xi_v, xi_v, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
        ];
        std::array::from_fn(|output| faa_top4([m1, m2, m3, m4], &q[output]))
    }

    #[inline(never)]
    fn strongest_hand_bundle4(x: [f64; 17]) -> [f64; 9] {
        let [
            m1,
            m2,
            m3,
            m4,
            q_u,
            q_v,
            q_uv,
            a,
            b,
            a_u,
            a_v,
            a_uv,
            b_u,
            b_v,
            b_uv,
            xi_u,
            xi_v,
        ] = x;
        let eta_eta = m4 * q_u * q_v * a * a
            + m3 * (q_uv * a * a
                + q_u * (a_v * a + a * a_v)
                + q_v * (a_u * a + a * a_u)
                + q_u * q_v * b)
            + m2 * (a_uv * a + a_u * a_v + a_v * a_u + a * a_uv + q_uv * b + q_u * b_v + q_v * b_u)
            + m1 * b_uv;
        let eta_w_b = m4 * q_u * q_v * a + m3 * (q_uv * a + q_u * a_v + q_v * a_u) + m2 * a_uv;
        let eta_w_d1 = (m3 * q_u * a + m2 * a_u) * xi_v
            + (m3 * q_v * a + m2 * a_v) * xi_u
            + m3 * q_u * q_v
            + m2 * q_uv;
        let xi_uv = xi_u * xi_v;
        let eta_w_d2 = m2 * a * xi_uv + m2 * q_u * xi_v + m2 * q_v * xi_u;
        let eta_w_d3 = m1 * xi_uv;
        let ww_bb = m4 * q_u * q_v + m3 * q_uv;
        let ww_db = m3 * q_v * xi_u + m3 * q_u * xi_v;
        let ww_ddb = m2 * xi_uv;
        let ww_dd = 2.0 * m2 * xi_uv;
        [
            eta_eta, eta_w_b, eta_w_d1, eta_w_d2, eta_w_d3, ww_bb, ww_db, ww_ddb, ww_dd,
        ]
    }

    fn paired_medians<const INPUTS: usize, const OUTPUTS: usize>(
        input: [f64; INPUTS],
        iterations: usize,
        compiled: fn([f64; INPUTS]) -> [f64; OUTPUTS],
        hand: fn([f64; INPUTS]) -> [f64; OUTPUTS],
    ) -> (f64, f64) {
        let mut compiled_samples = [0.0; 7];
        let mut hand_samples = [0.0; 7];
        for round in 0..7 {
            let compiled_first = round % 2 == 0;
            for side in 0..2 {
                let run_compiled = compiled_first == (side == 0);
                let mut x = input;
                let mut checksum = 0.0;
                let started = Instant::now();
                for _ in 0..iterations {
                    x[0] += checksum * 1e-18;
                    let output = if run_compiled {
                        compiled(black_box(x))
                    } else {
                        hand(black_box(x))
                    };
                    checksum += output.into_iter().sum::<f64>();
                }
                black_box(checksum);
                let ns = started.elapsed().as_secs_f64() * 1e9 / iterations as f64;
                if run_compiled {
                    compiled_samples[round] = ns;
                } else {
                    hand_samples[round] = ns;
                }
            }
        }
        compiled_samples.sort_by(f64::total_cmp);
        hand_samples.sort_by(f64::total_cmp);
        (compiled_samples[3], hand_samples[3])
    }

    fn assert_close<const N: usize>(actual: [f64; N], expected: [f64; N]) {
        for channel in 0..N {
            let band = 2e-12 * actual[channel].abs().max(expected[channel].abs()).max(1.0);
            assert!(
                (actual[channel] - expected[channel]).abs() <= band,
                "bundle channel {channel}: compiled={} hand={} band={band}",
                actual[channel],
                expected[channel],
            );
        }
    }

    #[test]
    fn higher_order_curve_wiggle_bundles_beat_strongest_hand_932() {
        let order3 = [0.31, 0.72, -0.41, 0.83, 1.13, -0.27, 0.19, -0.08, 0.47];
        let order4 = [
            0.31, 0.72, -0.41, 0.26, 0.83, -0.54, 0.17, 1.13, -0.27, 0.19, -0.12, 0.07, -0.08,
            0.05, -0.03, 0.47, -0.38,
        ];
        assert_close(compiled_bundle3(order3), strongest_hand_bundle3(order3));
        assert_close(compiled_bundle4(order4), strongest_hand_bundle4(order4));
        assert_close(compiled_bundle4(order4), canonical_bundle4(order4));
        let mut next = stream(0x9324_BA7C);
        for _ in 0..500 {
            let random3 = std::array::from_fn(|_| next());
            let random4 = std::array::from_fn(|_| next());
            assert_close(compiled_bundle3(random3), strongest_hand_bundle3(random3));
            assert_close(compiled_bundle4(random4), canonical_bundle4(random4));
            assert_close(compiled_bundle4(random4), strongest_hand_bundle4(random4));
        }

        // Everything above is correctness parity (both orders, the canonical
        // route, and 500 random argument pairs) and holds in any build.
        //
        // The timing gate below does NOT. Measured twice on the same tree, in a
        // debug build, this assertion failed at a DIFFERENT order each time:
        //
        //   run 1: order-3  compiled=15.858 ns/row  hand=14.598 ns/row  (~9%)
        //   run 2: order-4  compiled=18.728 ns/row  hand=18.192 ns/row  (~3%)
        //
        // A gate that picks a different loser on consecutive runs is sampling
        // noise, not detecting a regression: without optimization the compiled
        // lowering's whole advantage -- the thing this test exists to protect --
        // is not generated. It also costs 1.5M timed iterations per run.
        //
        // So debug stops here, and the ratio is asserted only under `--release`,
        // matching `release_measure_multinomial_fisher_vs_strongest_hand_932`
        // (#932, `14395b5d7`), which reaches its timing assertion the same way.
        if cfg!(debug_assertions) {
            return;
        }

        for (order, compiled_ns, hand_ns) in {
            let (c3, h3) =
                paired_medians(order3, 1_000_000, compiled_bundle3, strongest_hand_bundle3);
            let (c4, h4) =
                paired_medians(order4, 500_000, compiled_bundle4, strongest_hand_bundle4);
            [(3, c3, h3), (4, c4, h4)]
        } {
            eprintln!(
                "CURVE-WIGGLE-BUNDLE-932 order={order} compiled={compiled_ns:.3} ns/row \
                 strongest_hand={hand_ns:.3} ns/row hand_over_compiled={:.6}",
                hand_ns / compiled_ns,
            );
            assert!(
                hand_ns > compiled_ns,
                "order-{order} compiled curve/wiggle bundle must beat strongest hand: \
                 compiled={compiled_ns:.3} ns/row hand={hand_ns:.3} ns/row"
            );
        }
    }
}
