//! One way to measure "does A beat B", for every speed gate in this workspace.
//!
//! Fifteen separate timing harnesses across ten files were doing this in three
//! different ways, and the differences decide whether a gate can tell a
//! regression from a busy machine (issue #932, and #2470 for the duplication).
//! Two of the fifteen interleaved the arms; thirteen did not.
//!
//! # Why the majority pattern cannot resolve what it asserts
//!
//! The `best_ns` family — thirteen copies — times each arm in a **separate
//! call**: A is measured to completion (five rounds, minimum taken), and only
//! then is B. Two things go wrong at once.
//!
//! * **The arms occupy different wall-clock windows.** Anything that drifts
//!   between them — a neighbour job starting, a frequency ramp, cache or
//!   branch-predictor state warmed by the first arm, first-touch page faults
//!   amortised by the first arm — lands entirely in the ratio. Taking a minimum
//!   rejects a transient *spike*; it does nothing about a systematic offset.
//! * **A minimum is the order statistic most exposed to exactly that.** It is
//!   the single most favourable draw for each arm, so a small constant advantage
//!   to whichever arm ran second survives the minimum intact rather than
//!   averaging out. And pairing two independently-minimised blocks throws away
//!   the pairing that would have cancelled the drift in the first place.
//!
//! The consequence is not hypothetical. On the same tree, one such gate PASSED
//! on a quiet node (whole suite 5.1 s) and FAILED at 1.62x against a 1.5x bar on
//! a loaded one (same suite 219.1 s); another picked a **different loser on
//! consecutive runs** of one tree, 9% then 3%. Both were then guarded off in
//! debug builds, which hid the symptom without touching the cause: the harness
//! cannot resolve a margin of a few percent, and most of these gates assert
//! margins of a few percent.
//!
//! # What this does instead
//!
//! * **Interleave per repetition, not per block.** Each repetition times A and B
//!   adjacent in time, so drift slower than one repetition is common to both
//!   sides of that repetition's ratio and divides out.
//! * **Randomise the order within each repetition.** If a first-versus-second
//!   advantage exists at all, randomisation makes it cancel in expectation
//!   instead of accruing to a fixed arm — and [`PairedTiming::first_position_bias`]
//!   reports the residual so it is measured rather than assumed away.
//! * **Report the distribution of PAIRED ratios**, not a ratio of aggregate
//!   extrema. The per-repetition ratio is the quantity the claim is about; a
//!   median of paired ratios is robust, and the spread of that distribution is
//!   the gate's own resolution — which is the number you need in order to know
//!   whether a 3% claim is assertable at all.
//! * **Feed each iteration from the previous result.** A dependence chain
//!   through `checksum` is what stops the optimizer hoisting or vectorising
//!   across iterations; `black_box` alone permits both, and one of the replaced
//!   harnesses relied on `black_box` with no data dependence.
//!
//! # Why not measure the arms separately and normalise afterwards
//!
//! Because it does not work, and it fails in **both** directions. `iperf2`
//! measured this directly on `gam-solve::inner_fit_core_scaling`, a gate whose
//! two arms genuinely cannot be interleaved — one fans out over the whole Rayon
//! pool and the other is held serial by a guard, so external load hurts them
//! unequally. They divided the ratio by the parallel headroom the machine was
//! delivering at that moment, measured on an embarrassingly parallel kernel:
//!
//! * **Normaliser sampled once.** Headroom on four saturated cores bounced
//!   `2.36 / 1.28 / 2.81` across three consecutive repetitions. At the `1.28`
//!   sample a genuinely serial solve scores `1.0 / 1.28 = 0.78` and **passes** a
//!   `0.5` bar — a false green in which the gate certifies the exact defect it
//!   exists to catch.
//! * **Normaliser as max over five repetitions** (the right estimator for a
//!   capability, since interference only pushes an observed speedup down). Fixes
//!   the false green, and then loaded runs score `0.44` and `0.52` against the
//!   same `0.5` bar — red on working code.
//!
//! The underlying reason generalises past that one gate:
//!
//! > **A ratio whose two arms are measured at different times, on a machine
//! > whose load moves on that timescale, cannot be normalised after the fact.**
//! > Interleaving per repetition is not tidiness — it is what makes the arms
//! > share machine state instead of sampling it twice.
//!
//! The same lane's control is the cleanest demonstration that the *measurement*
//! rather than the *code* is what breaks: on one node, same four cores, back to
//! back, the identical solve scored `2.91` / `3.43` idle and `1.94` under four
//! spinners — straddling its own bar with nothing about the solver changed. A
//! width sweep at 2/4/8/16/32 cores on dedicated allocations tracked the pool
//! width at every width.
//!
//! # When the arms cannot be interleaved at all
//!
//! Some comparisons are between configurations that *want different machines* —
//! different core counts, different memory pressure — and no amount of
//! interleaving makes them share state. For those, take the confound away from
//! the measurement instead of modelling it: `.config/nextest.toml` supports
//!
//! ```toml
//! [[profile.default.overrides]]
//! threads-required = 'num-test-threads'
//! ```
//!
//! which reserves every runner slot so the test runs alone. It is already in use
//! in this repository for exactly this reason. The general rule, which is worth
//! more than the mechanism: **before building something to cancel an
//! environmental confound, check whether the runner can remove the confound
//! instead.**
//!
//! # Using it as a gate
//!
//! Assert on [`PairedTiming::median_ratio`] together with
//! [`PairedTiming::wins_fraction`], and print [`PairedTiming::summary`] whatever
//! the outcome. A median ratio that clears the bar while `wins_fraction` sits
//! near 0.5 means the margin is inside the measurement's resolution and the
//! claim is not established, however good the point estimate looks.
//!
//! **Lead with `wins_fraction`, not the ratio.** It is the statistic that
//! survives someone disbelieving the rest of the output. `wins == 1.0` over `n`
//! repetitions is a sign test at `2⁻ⁿ` — 15 repetitions is `≈3e-5` — and it is
//! **distribution-free**: it does not depend on [`PairedTiming::ratio_resolution`]
//! being correctly characterised, which is the one number a skeptic can
//! reasonably question. The ratio says *how much*; `wins` says *whether*. When
//! the first real migration onto this harness reported `median_ratio=0.938934`
//! with `wins=0.00`, it was the `wins` that settled a question two lanes had
//! been arguing from opposite directions.
//!
//! # The design lesson, for the next gate
//!
//! [`PairedTiming::first_position_bias`] is here because of a specific failure:
//! a fixed-order harness cannot separate a real 6% margin from a 6%
//! first-versus-second offset, since **both** produce a stable ratio with noisy
//! absolutes. The pre-existing answer was to run the whole gate a second time
//! with the arms swapped and see whether the verdict flipped — which works, but
//! only ever yields yes/no, costs a full second measurement, and has to be
//! redone by hand every time anyone doubts it.
//!
//! Randomising the order and reporting the residual **apportions** the confound
//! instead: on the measurement above, ordering contributed 0.0001 and the code
//! contributed 0.061. Generalising:
//!
//! > **Report a confound as a measured field rather than eliminating it by
//! > argument.** An argument that a confound was controlled has to be re-made,
//! > and re-believed, by every later reader. A field in the output is checked
//! > once and then simply read.
//!
//! That is the property to copy when building the next gate here, more than any
//! particular statistic in this module.

use std::hint::black_box;
use std::time::Instant;

/// Paired per-repetition timings for two implementations of one computation.
///
/// `a_ns[i]` and `b_ns[i]` were measured adjacent in time within repetition `i`,
/// in an order chosen by the repetition's coin flip. `ratios[i]` is
/// `b_ns[i] / a_ns[i]`, so a value **above 1 means A is faster** — the same
/// orientation as the `hand_over_production` token these gates already print.
#[derive(Clone, Debug)]
pub struct PairedTiming {
    /// Nanoseconds per iteration for arm A, one entry per repetition.
    pub a_ns: Vec<f64>,
    /// Nanoseconds per iteration for arm B, one entry per repetition.
    pub b_ns: Vec<f64>,
    /// `b_ns[i] / a_ns[i]`, one entry per repetition. Above 1 ⇒ A faster.
    pub ratios: Vec<f64>,
    /// `true` when arm A was timed first in that repetition.
    pub a_went_first: Vec<bool>,
}

impl PairedTiming {
    /// Median of the paired ratios — the headline estimate.
    ///
    /// The median rather than the mean because a single descheduled repetition
    /// produces an arbitrarily large outlier in one direction only, and rather
    /// than a minimum because a minimum is the statistic that preserves a
    /// systematic offset (see the module docs).
    pub fn median_ratio(&self) -> f64 {
        median(&self.ratios)
    }

    /// Fraction of repetitions in which A was faster than B.
    ///
    /// This is the gate's honesty check. A median ratio of 1.06 with a wins
    /// fraction of 1.0 is a real effect; the same 1.06 with 0.55 means the
    /// repetitions disagree with each other and the point estimate is riding on
    /// a few draws.
    pub fn wins_fraction(&self) -> f64 {
        if self.ratios.is_empty() {
            return f64::NAN;
        }
        let wins = self
            .a_ns
            .iter()
            .zip(self.b_ns.iter())
            .filter(|(a, b)| a < b)
            .count();
        wins as f64 / self.a_ns.len() as f64
    }

    /// Half the central 90% span of the paired ratios, as a fraction of the
    /// median — the gate's own **resolution**.
    ///
    /// A claimed margin smaller than this is not measurable by this harness at
    /// this repetition count, and asserting it is asserting noise. Report it
    /// beside the margin so the comparison is visible.
    pub fn ratio_resolution(&self) -> f64 {
        if self.ratios.len() < 2 {
            return f64::NAN;
        }
        let mut sorted = self.ratios.clone();
        sorted.sort_by(f64::total_cmp);
        let lo = quantile_sorted(&sorted, 0.05);
        let hi = quantile_sorted(&sorted, 0.95);
        let med = median(&self.ratios);
        if !med.is_finite() || med == 0.0 {
            return f64::NAN;
        }
        ((hi - lo) / 2.0 / med).abs()
    }

    /// Median ratio among repetitions where A ran first, minus the median among
    /// those where B ran first.
    ///
    /// This is the diagnostic the old harnesses could not produce, because they
    /// never varied the order. A value near zero says position does not matter
    /// on this host; a large value says the measurement is dominated by
    /// whichever arm goes first, and **no ordering of a non-randomised harness
    /// would have been trustworthy**. Returns `NaN` if either group is empty.
    pub fn first_position_bias(&self) -> f64 {
        let a_first: Vec<f64> = self
            .ratios
            .iter()
            .zip(self.a_went_first.iter())
            .filter(|(_, first)| **first)
            .map(|(r, _)| *r)
            .collect();
        let b_first: Vec<f64> = self
            .ratios
            .iter()
            .zip(self.a_went_first.iter())
            .filter(|(_, first)| !**first)
            .map(|(r, _)| *r)
            .collect();
        if a_first.is_empty() || b_first.is_empty() {
            return f64::NAN;
        }
        median(&a_first) - median(&b_first)
    }

    /// One line carrying everything needed to audit the verdict, including the
    /// numbers that would reveal the verdict as unsupported.
    pub fn summary(&self, a_label: &str, b_label: &str) -> String {
        format!(
            "{a_label}={:.2} ns/iter {b_label}={:.2} ns/iter \
             median_ratio={:.6} wins={:.2} resolution={:.4} position_bias={:.4} reps={}",
            median(&self.a_ns),
            median(&self.b_ns),
            self.median_ratio(),
            self.wins_fraction(),
            self.ratio_resolution(),
            self.first_position_bias(),
            self.ratios.len(),
        )
    }
}

/// Time two implementations against each other, interleaved per repetition with
/// a randomised order.
///
/// Each arm is a closure taking a perturbation seeded from the running checksum
/// and returning a value folded back into it, so consecutive iterations carry a
/// data dependence the optimizer cannot hoist across. Both closures must compute
/// the SAME quantity — this measures speed and assumes agreement has already
/// been established by a separate parity assertion.
///
/// `seed` fixes the order sequence, so a run is reproducible; vary it to check a
/// verdict is not an artifact of one particular order sequence.
///
/// # Panics
///
/// If `reps` or `iterations` is zero, or if either arm's accumulated checksum is
/// not finite — a non-finite checksum means the timed body degenerated (NaN
/// short-circuits are often much faster) and the timing is meaningless.
pub fn paired_interleaved<A, B>(
    reps: usize,
    iterations: usize,
    seed: u64,
    mut arm_a: A,
    mut arm_b: B,
) -> PairedTiming
where
    A: FnMut(f64) -> f64,
    B: FnMut(f64) -> f64,
{
    assert!(reps > 0, "paired_interleaved needs at least one repetition");
    assert!(
        iterations > 0,
        "paired_interleaved needs at least one iteration per repetition"
    );

    let mut a_ns = Vec::with_capacity(reps);
    let mut b_ns = Vec::with_capacity(reps);
    let mut ratios = Vec::with_capacity(reps);
    let mut a_went_first = Vec::with_capacity(reps);
    let mut rng = SplitMix64::new(seed);

    for _ in 0..reps {
        let a_first = rng.next_bool();
        let (ta, tb) = if a_first {
            let ta = time_arm(iterations, &mut arm_a);
            let tb = time_arm(iterations, &mut arm_b);
            (ta, tb)
        } else {
            let tb = time_arm(iterations, &mut arm_b);
            let ta = time_arm(iterations, &mut arm_a);
            (ta, tb)
        };
        ratios.push(tb / ta);
        a_ns.push(ta);
        b_ns.push(tb);
        a_went_first.push(a_first);
    }

    PairedTiming {
        a_ns,
        b_ns,
        ratios,
        a_went_first,
    }
}

/// Nanoseconds per iteration for one arm of one repetition.
///
/// The `checksum * 1e-18` perturbation is a feedback barrier, not a nudge: it
/// makes iteration `n + 1` depend on the result of iteration `n`, which is what
/// prevents the loop being hoisted or vectorised into something that is not the
/// per-call cost being claimed. The scale is small enough that the arithmetic
/// stays in the intended regime.
fn time_arm<F: FnMut(f64) -> f64>(iterations: usize, arm: &mut F) -> f64 {
    let mut checksum = 0.0_f64;
    let started = Instant::now();
    for _ in 0..iterations {
        checksum += arm(black_box(checksum * 1e-18));
    }
    let elapsed = started.elapsed().as_secs_f64();
    assert!(
        black_box(checksum).is_finite(),
        "timed arm accumulated a non-finite checksum, so the loop it timed is \
         not the computation being compared"
    );
    elapsed * 1e9 / iterations as f64
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        0.5 * (sorted[mid - 1] + sorted[mid])
    }
}

/// Linear-interpolated quantile of an already-sorted slice.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// SplitMix64 — a deterministic order sequence, so the interleave is randomised
/// but a run is reproducible. Deliberately not a dependency: the harness must
/// not be able to perturb the timing through an allocation or a dynamic call.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two arms doing identical work must land near ratio 1, and the harness
    /// must say so through `wins_fraction` too: identical arms should win about
    /// half the time, which is the signature a gate uses to recognise "no
    /// measurable difference" rather than reading a point estimate of 1.02 as a
    /// 2% win.
    #[test]
    fn identical_arms_report_no_winner() {
        let work = |x: f64| {
            let mut acc = x;
            for i in 0..64 {
                acc = acc.mul_add(1.000_001, (i as f64) * 1e-9);
            }
            acc
        };
        let timing = paired_interleaved(21, 2_000, 0xA5A5_1234, work, work);
        let median = timing.median_ratio();
        assert!(
            (median - 1.0).abs() < 0.35,
            "identical arms should sit near ratio 1: {}",
            timing.summary("a", "b")
        );
        let wins = timing.wins_fraction();
        assert!(
            (0.15..=0.85).contains(&wins),
            "identical arms should win about half the repetitions: {}",
            timing.summary("a", "b")
        );
    }

    /// A genuinely faster arm must be detected with every repetition agreeing —
    /// the property that separates a real margin from one inside the noise.
    #[test]
    fn a_large_real_difference_is_detected_unanimously() {
        let fast = |x: f64| {
            let mut acc = x;
            for i in 0..16 {
                acc = acc.mul_add(1.000_001, (i as f64) * 1e-9);
            }
            acc
        };
        let slow = |x: f64| {
            let mut acc = x;
            for i in 0..256 {
                acc = acc.mul_add(1.000_001, (i as f64) * 1e-9);
            }
            acc
        };
        let timing = paired_interleaved(15, 2_000, 0x5EED, fast, slow);
        assert!(
            timing.median_ratio() > 2.0,
            "a 16x work difference must show as a large ratio: {}",
            timing.summary("fast", "slow")
        );
        assert_eq!(
            timing.wins_fraction(),
            1.0,
            "a large real difference must win EVERY repetition: {}",
            timing.summary("fast", "slow")
        );
    }

    /// The order must actually vary. A harness that believes it randomises but
    /// does not is indistinguishable from the ones being replaced, and the
    /// position-bias diagnostic would silently become `NaN`.
    #[test]
    fn both_orders_occur_and_position_bias_is_reportable() {
        let work = |x: f64| x.mul_add(1.000_001, 1e-9);
        let timing = paired_interleaved(20, 500, 7, work, work);
        let a_first = timing.a_went_first.iter().filter(|f| **f).count();
        assert!(
            a_first > 0 && a_first < timing.a_went_first.len(),
            "both arm orders must occur across repetitions, got {a_first} of {}",
            timing.a_went_first.len()
        );
        assert!(
            timing.first_position_bias().is_finite(),
            "position bias must be reportable once both orders occur"
        );
    }

    /// `ratio_resolution` is what tells a caller whether its bar is assertable.
    /// It must be finite and positive on a real measurement, or the gate has no
    /// way to know it is asserting inside its own noise.
    #[test]
    fn resolution_is_reported_and_positive() {
        let work = |x: f64| x.mul_add(1.000_001, 1e-9);
        let timing = paired_interleaved(15, 500, 99, work, work);
        let resolution = timing.ratio_resolution();
        assert!(
            resolution.is_finite() && resolution > 0.0,
            "resolution must be a usable number: {}",
            timing.summary("a", "b")
        );
    }

    /// The summary must carry the numbers that could overturn the verdict, not
    /// just the verdict. A gate that prints only the ratio is how a 3% claim
    /// with 6% resolution gets read as established.
    #[test]
    fn summary_carries_the_overturning_numbers() {
        let work = |x: f64| x.mul_add(1.000_001, 1e-9);
        let timing = paired_interleaved(9, 500, 3, work, work);
        let line = timing.summary("production", "hand");
        for field in [
            "production=",
            "hand=",
            "median_ratio=",
            "wins=",
            "resolution=",
            "position_bias=",
            "reps=",
        ] {
            assert!(line.contains(field), "summary is missing {field}: {line}");
        }
    }

    #[test]
    #[should_panic(expected = "at least one repetition")]
    fn zero_repetitions_is_refused_not_silently_empty() {
        let work = |x: f64| x;
        // Called as a bare statement. A discarding binding is banned in this
        // workspace, and it would be the wrong shape regardless: this call is
        // expected to panic, so there is no result to discard.
        paired_interleaved(0, 10, 1, work, work);
    }
}
