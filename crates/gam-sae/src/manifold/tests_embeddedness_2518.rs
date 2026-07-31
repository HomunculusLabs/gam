//! #2518 item 2 — the decoded-image embeddedness certificate.
//!
//! The load-bearing pair is `plain_circle_is_certified_embedded` versus
//! `double_wrapped_circle_is_refused`: those two decoders are INDISTINGUISHABLE
//! to every other certificate in the crate (same image, same constant speed,
//! same zero isometry defect, same exact reconstruction), and differ only in
//! that the second one traverses the image twice. That is precisely the
//! multi-preimage hazard the encode path's uniqueness claim cannot see.

use ndarray::Array2;

use super::embeddedness::{certify_periodic_decoder_embeddedness, AtomEmbeddednessCertificate};

/// Build a periodic decoder in the `PeriodicHarmonicEvaluator` column layout:
/// row `0` is the constant, rows `2h−1` / `2h` are the `sin`/`cos` coefficients
/// of harmonic `h`. `harmonic_rows` supplies `(h, a_h, b_h)`.
fn decoder(p: usize, harmonics: usize, harmonic_rows: &[(usize, Vec<f64>, Vec<f64>)]) -> Array2<f64> {
    let mut b = Array2::<f64>::zeros((2 * harmonics + 1, p));
    for (h, a, bb) in harmonic_rows {
        assert_eq!(a.len(), p, "sin coefficient width");
        assert_eq!(bb.len(), p, "cos coefficient width");
        for j in 0..p {
            b[[2 * h - 1, j]] = a[j];
            b[[2 * h, j]] = bb[j];
        }
    }
    b
}

/// `m(t) = (cos 2πt, sin 2πt)` — the unit circle, traversed once.
fn plain_circle() -> Array2<f64> {
    decoder(2, 1, &[(1, vec![0.0, 1.0], vec![1.0, 0.0])])
}

/// `m(t) = (cos 4πt, sin 4πt)` — the SAME image, traversed twice. An immersion
/// with constant speed and zero isometry defect, but `m(t) = m(t + ½)`.
fn double_wrapped_circle() -> Array2<f64> {
    decoder(2, 2, &[(2, vec![0.0, 1.0], vec![1.0, 0.0])])
}

fn certify(b: &Array2<f64>) -> AtomEmbeddednessCertificate {
    certify_periodic_decoder_embeddedness(b.view()).expect("certificate")
}

#[test]
fn plain_circle_is_certified_embedded() {
    let cert = certify(&plain_circle());
    // `F(c, x) = (−sin 2πc, cos 2πc)` exactly, so `G ≡ 1` on the whole domain —
    // the grid minimum is not an approximation here, it is the analytic value.
    assert!(
        (cert.grid_min - 1.0).abs() < 1.0e-12,
        "G is identically 1 for the unit circle; grid_min={}",
        cert.grid_min
    );
    assert!(
        cert.embedded,
        "the once-traversed unit circle must certify: certified_min={} correction={}",
        cert.certified_min, cert.grid_correction
    );
    // The correction must be a small fraction of the margin it corrects, or the
    // grid resolution — not the geometry — would be deciding the verdict.
    assert!(
        cert.grid_correction < 0.1 * cert.grid_min,
        "grid correction {} should be well under the {} margin it corrects",
        cert.grid_correction,
        cert.grid_min
    );
}

#[test]
fn double_wrapped_circle_is_refused() {
    let cert = certify(&double_wrapped_circle());
    // `G(c, x) = 4x²`, which vanishes at `x = 0` — i.e. at `s = ½`, exactly the
    // half-turn where `m(u + ½) = m(u)`. The refusal is not a resolution
    // artifact: the coincidence is ON the grid and the sampled minimum is zero.
    assert!(
        cert.grid_min.abs() < 1.0e-20,
        "the double wrap has an exact coincidence at s=1/2; grid_min={}",
        cert.grid_min
    );
    assert!(
        !cert.embedded,
        "a 2-to-1 decoder must never certify as embedded: certified_min={}",
        cert.certified_min
    );
}

#[test]
fn cusped_curve_is_refused() {
    // `z(t) = e^{2πit} + ½ e^{4πit}` has `z′(t) = 2πi e^{2πit}(1 + e^{2πit})`,
    // which vanishes at `t = ½`: injective, but not an immersion. Embeddedness
    // must fail on the immersion half too, and that half lives at the `x = ±1`
    // edge of the domain rather than in its interior.
    let b = decoder(
        2,
        2,
        &[
            (1, vec![0.0, 1.0], vec![1.0, 0.0]),
            (2, vec![0.0, 0.5], vec![0.5, 0.0]),
        ],
    );
    let cert = certify(&b);
    assert!(
        !cert.embedded,
        "a cusped (non-immersed) curve must not certify: certified_min={} grid_min={}",
        cert.certified_min, cert.grid_min
    );
}

#[test]
fn wavy_embedded_circle_certifies_with_two_harmonics() {
    // `m(t) = (cos 2πt, sin 2πt, 0.2 sin 4πt)` — a genuinely embedded closed
    // curve that is NOT a round circle and carries a second harmonic, so the
    // Chebyshev block and the `∂_x` half of the correction are both exercised.
    let b = decoder(
        3,
        2,
        &[
            (1, vec![0.0, 1.0, 0.0], vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 0.0, 0.2], vec![0.0, 0.0, 0.0]),
        ],
    );
    let cert = certify(&b);
    assert!(
        cert.embedded,
        "an embedded wavy circle must certify: certified_min={} correction={} grid_min={}",
        cert.certified_min, cert.grid_correction, cert.grid_min
    );
}

#[test]
fn verdict_and_relative_margin_are_scale_invariant() {
    // `G` is homogeneous of degree 2 in the decoder, so the atom amplitude
    // cannot change the verdict — the property being certified is a property of
    // the IMAGE, and the image of a scaled decoder is a scaled image.
    let base = certify(&plain_circle());
    let scaled = certify(&(plain_circle() * 7.0));
    assert!(scaled.embedded);
    assert!(
        (scaled.relative_margin - base.relative_margin).abs() < 1.0e-12,
        "relative margin moved under rescaling: {} vs {}",
        scaled.relative_margin,
        base.relative_margin
    );
    assert!(
        (scaled.certified_min - 49.0 * base.certified_min).abs() < 1.0e-9 * 49.0,
        "certified_min must scale as the square of the decoder"
    );
}

#[test]
fn constant_decoder_reports_a_collapsed_atom_rather_than_certifying() {
    let b = Array2::<f64>::zeros((1, 4));
    let cert = certify(&b);
    assert_eq!(cert.harmonics, 0);
    assert!(!cert.embedded, "a point image is not an embedded circle");
}

#[test]
fn even_row_count_is_rejected_as_a_layout_error() {
    let b = Array2::<f64>::zeros((4, 3));
    let err = certify_periodic_decoder_embeddedness(b.view())
        .expect_err("even row counts are not the periodic harmonic layout");
    assert!(err.contains("odd row count"), "unexpected message: {err}");
}
