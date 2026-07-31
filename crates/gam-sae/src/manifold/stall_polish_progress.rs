/// Typed progress verdict for repeated exact-Hessian rescue at an inner
/// objective plateau.
///
/// The quasi-Laplace acceptance rule is disjunctive in two KKT currencies:
/// either the raw residual or its identified-quotient residual may establish
/// stationarity. Continuation therefore has to preserve those currencies
/// separately too. Collapsing them to one scalar can retire a productive
/// quotient contraction merely because the ambient residual rose along a
/// gauge/stiff direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum StallPolishProgressVerdict {
    /// The first plateau establishes the finite residual frontier.
    Baseline,
    /// At least one KKT currency improved beyond numerical resolution.
    Contracting { raw: bool, quotient: bool },
    /// Neither KKT currency improved beyond numerical resolution.
    NonContracting,
    /// A residual, tolerance, or resolution was not a valid certificate input.
    Invalid,
}

impl StallPolishProgressVerdict {
    pub(crate) fn permits_continuation(self) -> bool {
        matches!(self, Self::Baseline | Self::Contracting { .. })
    }
}

/// Cross-plateau contraction certificate for terminal exact-Newton polish.
///
/// #2653 — a raw eight-invocation cap retired a K=1 circle solve while the
/// exact-Hessian continuation was still contracting both KKT residuals. This
/// certificate replaces invocation counting with the property the count was
/// trying to approximate: each re-entry must move the Pareto frontier of the
/// two stationarity currencies by more than the caller's numerical-resolution
/// band. A non-contracting plateau is refused; a contracting one may continue.
///
/// This is globally bounded without an arbitrary iteration ration. Every
/// permitted re-entry after the baseline decreases at least one non-negative
/// frontier by `relative_resolution * tolerance`; neither frontier can
/// decrease below zero. Thus a finite starting residual and positive KKT
/// tolerance imply finitely many permitted re-entries, while a healthy
/// contraction is never cut off solely because it happened to be the ninth.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StallPolishProgressCertificate {
    relative_resolution: f64,
    best_raw: Option<f64>,
    best_quotient: Option<f64>,
}

impl StallPolishProgressCertificate {
    pub(crate) fn new(relative_resolution: f64) -> Self {
        Self {
            relative_resolution,
            best_raw: None,
            best_quotient: None,
        }
    }

    fn materially_contracts(&self, current: f64, best: f64, tolerance: f64) -> bool {
        if !(current.is_finite()
            && current >= 0.0
            && best.is_finite()
            && best >= 0.0
            && tolerance.is_finite()
            && tolerance > 0.0
            && self.relative_resolution.is_finite()
            && self.relative_resolution > 0.0
            && self.relative_resolution <= 1.0)
        {
            return false;
        }
        let resolution = self.relative_resolution * best.max(tolerance);
        best - current > resolution
    }

    pub(crate) fn observe(
        &mut self,
        raw: f64,
        quotient: f64,
        tolerance: f64,
    ) -> StallPolishProgressVerdict {
        if !(raw.is_finite()
            && raw >= 0.0
            && quotient.is_finite()
            && quotient >= 0.0
            && tolerance.is_finite()
            && tolerance > 0.0
            && self.relative_resolution.is_finite()
            && self.relative_resolution > 0.0
            && self.relative_resolution <= 1.0)
        {
            return StallPolishProgressVerdict::Invalid;
        }

        let (Some(best_raw), Some(best_quotient)) = (self.best_raw, self.best_quotient) else {
            self.best_raw = Some(raw);
            self.best_quotient = Some(quotient);
            return StallPolishProgressVerdict::Baseline;
        };

        let raw_contracting = self.materially_contracts(raw, best_raw, tolerance);
        let quotient_contracting = self.materially_contracts(quotient, best_quotient, tolerance);
        if !(raw_contracting || quotient_contracting) {
            return StallPolishProgressVerdict::NonContracting;
        }

        // Advance only the frontier that actually paid this continuation.
        // Recording a sub-resolution decrease would discard it without ever
        // letting accumulated movement become material, and would break the
        // finite-decrease argument above.
        if raw_contracting {
            self.best_raw = Some(raw);
        }
        if quotient_contracting {
            self.best_quotient = Some(quotient);
        }
        StallPolishProgressVerdict::Contracting {
            raw: raw_contracting,
            quotient: quotient_contracting,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contracting_kkt_tail_remains_eligible_after_eight_plateaus_2653() {
        let tolerance = 9.211_394e-5;
        let mut certificate = StallPolishProgressCertificate::new(f64::EPSILON.sqrt());
        let mut raw = 8.312_567e-4;
        let mut quotient = 7.389_493e-4;

        assert_eq!(
            certificate.observe(raw, quotient, tolerance),
            StallPolishProgressVerdict::Baseline
        );

        // A raw invocation cap at eight rejected the production tail even
        // though both stationarity currencies were still contracting. The
        // certificate has no ordinal boundary: all twelve later plateaus earn
        // continuation from measured KKT progress.
        for plateau in 1..=12 {
            raw *= 0.94;
            quotient *= 0.93;
            let verdict = certificate.observe(raw, quotient, tolerance);
            assert!(
                matches!(
                    verdict,
                    StallPolishProgressVerdict::Contracting {
                        raw: true,
                        quotient: true
                    }
                ),
                "plateau {plateau} must remain eligible while both KKT currencies contract: \
                 {verdict:?}"
            );
            assert!(verdict.permits_continuation());
        }

        assert!(
            raw > tolerance && quotient > tolerance,
            "the witness must still need continuation after the old eight-entry boundary"
        );
    }

    #[test]
    fn either_kkt_currency_can_pay_but_a_repeated_frontier_cannot_2653() {
        let tolerance = 1.0e-4;
        let mut certificate = StallPolishProgressCertificate::new(f64::EPSILON.sqrt());

        assert_eq!(
            certificate.observe(8.0e-4, 6.0e-4, tolerance),
            StallPolishProgressVerdict::Baseline
        );
        assert_eq!(
            certificate.observe(8.2e-4, 5.0e-4, tolerance),
            StallPolishProgressVerdict::Contracting {
                raw: false,
                quotient: true,
            },
            "quotient contraction is independently sufficient because quotient KKT \
             stationarity is an acceptance authority"
        );
        assert_eq!(
            certificate.observe(8.1e-4, 5.0e-4, tolerance),
            StallPolishProgressVerdict::NonContracting,
            "neither currency moved its best frontier"
        );
        assert!(!StallPolishProgressVerdict::NonContracting.permits_continuation());
        assert_eq!(
            certificate.observe(f64::NAN, 4.0e-4, tolerance),
            StallPolishProgressVerdict::Invalid
        );
    }
}
