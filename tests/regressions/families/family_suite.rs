use gam::families::family_runtime::{FamilyStrategy, strategy_for_spec};
use gam::families::survival::latent::fixed_latent_hazard_frailty;
use gam::families::survival::lognormal_kernel::{FrailtyScale, FrailtySpec, HazardLoading};
use gam::types::inverse_link_to_binomial_spec;
use gam::types::{InverseLink, LatentCLogLogState, LikelihoodSpec, ResponseFamily, StandardLink};

#[test]
fn bug_family_meta_binomial_inverse_links_round_trip_identity() {
    let links = vec![
        InverseLink::Standard(StandardLink::Logit),
        InverseLink::Standard(StandardLink::Probit),
        InverseLink::Standard(StandardLink::CLogLog),
        InverseLink::LatentCLogLog(LatentCLogLogState { latent_sd: 0.4 }),
    ];
    for link in links {
        let spec = inverse_link_to_binomial_spec(&link)
            .expect("Every supported binomial inverse link must resolve to a binomial likelihood specification.");
        assert_eq!(
            spec.response,
            ResponseFamily::Binomial,
            "Each supported binomial inverse link must map to ResponseFamily::Binomial."
        );
        assert_eq!(
            spec.link, link,
            "Round-tripping through inverse_link_to_binomial_spec must preserve inverse-link identity."
        );
    }
}

#[test]
fn bug_strategy_for_spec_preserves_family_marker_for_all_response_variants() {
    let specs = vec![
        LikelihoodSpec::new(
            ResponseFamily::Gaussian,
            InverseLink::Standard(StandardLink::Identity),
        ),
        LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Standard(StandardLink::Logit),
        ),
        LikelihoodSpec::new(
            ResponseFamily::Poisson,
            InverseLink::Standard(StandardLink::Log),
        ),
        LikelihoodSpec::new(
            ResponseFamily::Tweedie { p: 1.5 },
            InverseLink::Standard(StandardLink::Log),
        ),
        LikelihoodSpec::new(
            ResponseFamily::NegativeBinomial {
                theta: 2.0,
                theta_fixed: false,
            },
            InverseLink::Standard(StandardLink::Log),
        ),
        LikelihoodSpec::new(
            ResponseFamily::Beta { phi: 3.0 },
            InverseLink::Standard(StandardLink::Logit),
        ),
        LikelihoodSpec::new(
            ResponseFamily::Gamma,
            InverseLink::Standard(StandardLink::Log),
        ),
        LikelihoodSpec::new(
            ResponseFamily::RoystonParmar,
            InverseLink::Standard(StandardLink::Identity),
        ),
    ];
    for spec in specs {
        let strategy = strategy_for_spec(&spec);
        assert_eq!(
            strategy.family().response,
            spec.response,
            "strategy_for_spec must preserve the response-family marker for each LikelihoodSpec variant."
        );
    }
}

#[test]
fn bug_latent_survival_fixed_frailty_accepts_valid_hazard_multiplier_spec() {
    let frailty = FrailtySpec::HazardMultiplier {
        scale: FrailtyScale::Fixed { sigma: 0.7 },
        loading: HazardLoading::Full,
    };
    let (sigma, loading) = fixed_latent_hazard_frailty(&frailty, "latent-survival")
        .expect("fixed_latent_hazard_frailty must accept a finite non-negative fixed hazard-multiplier sigma.");
    assert!(
        (sigma - 0.7).abs() < 1e-12,
        "fixed_latent_hazard_frailty must return the configured sigma value exactly."
    );
    assert_eq!(
        loading,
        HazardLoading::Full,
        "fixed_latent_hazard_frailty must preserve hazard-loading identity."
    );
}
