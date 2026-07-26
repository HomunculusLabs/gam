//! Dumps `gam-math` special-function values as TSV for an external
//! arbitrary-precision comparison. Not part of the build; run by hand:
//!
//! ```text
//! cargo run -p gam-math --release --example special_audit_dump > /tmp/dump.tsv
//! ```
//!
//! Each line is `channel<TAB>arg<TAB>value` (or `channel<TAB>arg<TAB>arg2<TAB>value`
//! for the two-argument channels), with every float printed at full `{:e}`
//! precision so the reader can round-trip it.

use gam_math::probability as prob;
use gam_math::special;

fn emit(channel: &str, arg: f64, value: f64) {
    println!("{channel}\t{arg:e}\t{value:e}");
}

fn geometric_grid(lo: f64, hi: f64, count: usize) -> Vec<f64> {
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    (0..count)
        .map(|i| (log_lo + (log_hi - log_lo) * (i as f64) / ((count - 1) as f64)).exp())
        .collect()
}

fn linear_grid(lo: f64, hi: f64, count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| lo + (hi - lo) * (i as f64) / ((count - 1) as f64))
        .collect()
}

fn main() {
    // ---- Bessel channels -------------------------------------------------
    for eta in geometric_grid(1e-6, 1e12, 400) {
        let (centered, ratio, d1) = special::bessel_i0_centered_terms(eta);
        emit("bessel_centered_log", eta, centered);
        emit("bessel_ratio", eta, ratio);
        emit("bessel_d1", eta, d1);
        emit(
            "bessel_d2",
            eta,
            special::bessel_i0_centered_second_log_derivative_from_log_abs(eta.ln()),
        );
    }
    // Dense sweep across the ascending/asymptotic seam at 20.
    for eta in linear_grid(15.0, 25.0, 201) {
        let (centered, ratio, d1) = special::bessel_i0_centered_terms(eta);
        emit("bessel_centered_log", eta, centered);
        emit("bessel_ratio", eta, ratio);
        emit("bessel_d1", eta, d1);
        emit(
            "bessel_d2",
            eta,
            special::bessel_i0_centered_second_log_derivative_from_log_abs(eta.ln()),
        );
    }

    // ---- Polygamma family ------------------------------------------------
    for x in geometric_grid(1e-8, 1e10, 400) {
        emit("digamma", x, special::digamma(x));
        emit("trigamma", x, special::trigamma(x));
        emit("tetragamma", x, special::tetragamma(x));
        emit("pentagamma", x, special::pentagamma(x));
    }
    for x in linear_grid(0.5, 30.0, 300) {
        emit("digamma", x, special::digamma(x));
        emit("trigamma", x, special::trigamma(x));
        emit("tetragamma", x, special::tetragamma(x));
        emit("pentagamma", x, special::pentagamma(x));
    }

    // ---- Normal-distribution channels ------------------------------------
    let mut normal_args: Vec<f64> = linear_grid(-40.0, 40.0, 801);
    normal_args.extend(geometric_grid(1e-8, 1e2, 200));
    normal_args.extend(geometric_grid(1e-8, 1e2, 200).into_iter().map(|v| -v));
    for x in normal_args {
        emit("normal_pdf", x, prob::normal_pdf(x));
        emit("normal_cdf", x, prob::normal_cdf(x));
        emit("normal_logcdf", x, prob::normal_logcdf(x));
        emit("normal_logsf", x, prob::normal_logsf(x));
        let (log_cdf, mills) = prob::signed_probit_logcdf_and_mills_ratio(x);
        emit("probit_logcdf", x, log_cdf);
        emit("probit_mills", x, mills);
        let derivatives = prob::normal_logcdf_derivatives(x);
        for (order, value) in derivatives.iter().enumerate() {
            emit(&format!("logcdf_d{order}"), x, *value);
        }
        if x >= 0.0 {
            emit("erfcx", x, prob::erfcx_nonnegative(x));
        }
    }
    for x in geometric_grid(1e-3, 1e6, 400) {
        emit("erfcx", x, prob::erfcx_nonnegative(x));
    }

    // ---- log1mexp --------------------------------------------------------
    for a in geometric_grid(1e-14, 50.0, 300) {
        emit("log1mexp", a, prob::log1mexp_positive(a));
    }

    // ---- Normal quantile -------------------------------------------------
    for p in geometric_grid(1e-300, 0.5, 400) {
        if let Ok(q) = prob::standard_normal_quantile(p) {
            emit("normal_quantile", p, q);
        }
    }
    for p in linear_grid(0.001, 0.999, 400) {
        if let Ok(q) = prob::standard_normal_quantile(p) {
            emit("normal_quantile", p, q);
        }
    }
    for log_p in linear_grid(-700.0, -1e-6, 400) {
        if let Ok(q) = prob::standard_normal_quantile_from_log_cdf(log_p) {
            emit("normal_quantile_from_log", log_p, q);
        }
    }

    // ---- Gauss-Legendre --------------------------------------------------
    for n in [
        3usize, 4, 5, 7, 8, 12, 15, 16, 20, 24, 31, 32, 40, 48, 63, 64, 80, 96, 100, 127, 128, 160,
        200, 256,
    ] {
        let (nodes, weights) = special::gauss_legendre(n);
        for (i, (node, weight)) in nodes.iter().zip(weights.iter()).enumerate() {
            println!("gl_node\t{n}\t{i}\t{node:e}");
            println!("gl_weight\t{n}\t{i}\t{weight:e}");
        }
    }

    // ---- Binomial coefficient --------------------------------------------
    for n in 0usize..=60 {
        for k in 0..=n {
            println!(
                "binomial\t{n}\t{k}\t{:e}",
                special::binomial_coefficient_f64(n, k)
            );
        }
    }

    // Beta quantiles. The shapes are the ones a beta-regression predictive
    // interval produces from a mean and a variance, plus a direct sweep of
    // small and large shape pairs. The lower tail here reaches quantiles far
    // below `f64::EPSILON`, which is exactly where a solver with an absolute
    // tolerance in `x` stalls rather than degrading (#2528), so the sweep is
    // deliberately weighted toward small `a`.
    for (mu, variance_fraction) in [
        (0.001_f64, 0.3_f64),
        (0.01, 0.2),
        (0.01, 0.5),
        (0.02, 0.3),
        (0.05, 0.5),
        (0.1, 0.5),
        (0.3, 0.5),
        (0.5, 0.5),
        (0.7, 0.5),
        (0.9, 0.2),
    ] {
        let bernoulli_variance = mu * (1.0 - mu);
        let precision = 1.0 / variance_fraction - 1.0;
        let (a, b) = (mu * precision, (1.0 - mu) * precision);
        for p in [0.001_f64, 0.025, 0.1, 0.5, 0.9, 0.975, 0.999] {
            println!(
                "beta_quantile\t{a:e}\t{b:e}\t{p:e}\t{:e}",
                prob::beta_quantile(p, a, b)
            );
        }
        // Keep the moment-matched variance in the record so a reader can see
        // which mean produced which shape pair.
        println!("beta_shape\t{mu:e}\t{bernoulli_variance:e}\t{a:e}\t{b:e}");
    }
    for (a, b) in [
        (0.1_f64, 0.1_f64),
        (0.5, 0.5),
        (0.5, 20.0),
        (20.0, 0.5),
        (1.0, 1.0),
        (2.0, 3.0),
        (2.5, 7.5),
        (100.0, 100.0),
        (1000.0, 5.0),
        (5.0, 1000.0),
    ] {
        for p in [
            1.0e-8_f64, 1.0e-4, 0.001, 0.01, 0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975, 0.999,
        ] {
            println!(
                "beta_quantile\t{a:e}\t{b:e}\t{p:e}\t{:e}",
                prob::beta_quantile(p, a, b)
            );
        }
    }
}
