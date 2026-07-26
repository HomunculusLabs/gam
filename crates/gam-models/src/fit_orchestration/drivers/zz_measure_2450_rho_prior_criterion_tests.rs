// #2450/#2463 — which criterion does a `gam-models` fit minimize?
//
// SPEC makes `Flat` the REML/LAML default. A non-Flat `FitOptions::rho_prior`
// is therefore an explicit caller request, and the fit must either minimize that
// configured criterion or refuse it — never rewrite it silently.
//
// This paired A/B/C gate holds data, seed, spec, and every other option fixed:
//
//   * REML: `Flat`;
//   * configured: `Normal { mean: 0, sd: 3 }`;
//   * discriminator: `Normal { mean: -6, sd: 0.25 }`, which pins λ near e^-6.
//
// It exercises both the relaxable P-spline route and the moving-κ Matérn control.
// Every measured cell must move under the discriminator. Bitwise-identical rho
// means the caller's prior was discarded before reaching the criterion (#2463).
//
#[cfg(test)]
mod zz_measure_2450_rho_prior_criterion_tests {
    use super::*;
    use gam_terms::basis::{
        BSplineBasisSpec, BSplineBoundaryConditions, BSplineIdentifiability, BSplineKnotSpec,
        MaternBasisSpec, MaternNu, OneDimensionalBoundary,
    };
    use ndarray::{Array1, Array2};

    /// One arm's realized numbers at one (cell, replicate).
    #[derive(Clone, Copy, Debug)]
    struct ArmReading {
        rho_hat: f64,
        edf: f64,
        mise: f64,
    }

    /// Deterministic standard-normal stream: a 64-bit SplitMix step feeding a
    /// Box–Muller pair. Self-contained so the draw is byte-reproducible from
    /// the seed alone and identical across arms.
    struct GaussStream {
        state: u64,
        spare: Option<f64>,
    }

    impl GaussStream {
        fn new(seed: u64) -> Self {
            Self {
                state: seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x1234_5678_9abc_def0,
                spare: None,
            }
        }

        fn next_uniform(&mut self) -> f64 {
            self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            (((z >> 11) as f64) + 0.5) / ((1u64 << 53) as f64)
        }

        fn next_normal(&mut self) -> f64 {
            if let Some(value) = self.spare.take() {
                return value;
            }
            let u1 = self.next_uniform();
            let u2 = self.next_uniform();
            let radius = (-2.0 * u1.ln()).sqrt();
            let angle = std::f64::consts::TAU * u2;
            self.spare = Some(radius * angle.sin());
            radius * angle.cos()
        }
    }

    /// The truth: a sinusoid whose frequency sets how much curvature the smooth
    /// must buy, hence where the REML optimum ρ* sits. Sweeping frequency
    /// sweeps ρ* — and with it the sign and size of a ρ-prior's pull.
    fn truth(x: f64, frequency: f64) -> f64 {
        (std::f64::consts::TAU * frequency * x).sin()
    }

    fn bspline_spec(num_internal_knots: usize) -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "s".to_string(),
                basis: SmoothBasisSpec::BSpline1D {
                    feature_col: 0,
                    spec: BSplineBasisSpec {
                        degree: 3,
                        penalty_order: 2,
                        knotspec: BSplineKnotSpec::Generate {
                            data_range: (0.0, 1.0),
                            num_internal_knots,
                        },
                        double_penalty: false,
                        // Centering removes the constant the smooth would
                        // otherwise alias against the collection's intercept.
                        identifiability: BSplineIdentifiability::WeightedSumToZero {
                            weights: None,
                        },
                        boundary_conditions: BSplineBoundaryConditions::default(),
                        boundary: OneDimensionalBoundary::Open,
                    },
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        }
    }

    fn matern_spec(num_centers: usize) -> TermCollectionSpec {
        TermCollectionSpec {
            linear_terms: vec![],
            random_effect_terms: vec![],
            smooth_terms: vec![SmoothTermSpec {
                name: "m".to_string(),
                basis: SmoothBasisSpec::Matern {
                    feature_cols: vec![0],
                    spec: MaternBasisSpec {
                        periodic: None,
                        center_strategy: CenterStrategy::FarthestPoint { num_centers },
                        length_scale: gam_terms::basis::MaternLengthScale::fixed(0.25),
                        nu: MaternNu::FiveHalves,
                        include_intercept: false,
                        double_penalty: true,
                        identifiability: MaternIdentifiability::CenterSumToZero,
                        aniso_log_scales: None,
                    },
                    input_scale: None,
                },
                shape: ShapeConstraint::None,
                joint_null_rotation: None,
            }],
        }
    }

    /// Fit one arm and read back `(ρ̂₀, edf, MISE-against-truth)`.
    ///
    /// MISE is against the NOISE-FREE truth on the training design, so it is
    /// estimation error directly rather than through a second sampling step.
    /// The realized predictor is `affine_offset + design·β` by definition of
    /// `TermCollectionDesign`.
    fn run_arm(
        data: &Array2<f64>,
        y: &Array1<f64>,
        truth_values: &Array1<f64>,
        spec: &TermCollectionSpec,
        prior: gam_problem::RhoPrior,
    ) -> Option<ArmReading> {
        let n = y.len();
        let weights = Array1::<f64>::ones(n);
        let offset = Array1::<f64>::zeros(n);
        let opts = FitOptions {
            rho_prior: prior,
            max_iter: 60,
            ..FitOptions::default()
        };
        let fitted = match fit_term_collection_forspec(
            data.view(),
            y.view(),
            weights.view(),
            offset.view(),
            spec,
            LikelihoodSpec::gaussian_identity(),
            &opts,
        ) {
            Ok(fitted) => fitted,
            Err(error) => {
                // An instrument that stops measuring must SAY so; swallowing
                // this is how a probe reports "not measured" with no reason.
                println!("[2450] arm refused: {error:?}");
                return None;
            }
        };
        let beta = fitted.fit.blocks.first()?.beta.clone();
        let dense = fitted.design.design.to_dense();
        if dense.ncols() != beta.len() || dense.nrows() != n {
            println!(
                "[2450] arm shape mismatch: design {}x{} vs beta {}",
                dense.nrows(),
                dense.ncols(),
                beta.len()
            );
            return None;
        }
        let mut mise = 0.0;
        for i in 0..n {
            let mut eta = fitted.design.affine_offset[i];
            for j in 0..beta.len() {
                eta += dense[[i, j]] * beta[j];
            }
            let residual = eta - truth_values[i];
            mise += residual * residual;
        }
        Some(ArmReading {
            rho_hat: *fitted.fit.log_lambdas.first()?,
            edf: fitted.fit.blocks.first()?.edf,
            mise: mise / n as f64,
        })
    }

    /// Paired mean and standard error of the difference `a − b`.
    fn paired_mean_se(a: &[f64], b: &[f64]) -> (f64, f64) {
        let d: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
        let m = d.len() as f64;
        let mean = d.iter().sum::<f64>() / m;
        if d.len() < 2 {
            return (mean, f64::NAN);
        }
        let variance = d.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (m - 1.0);
        (mean, (variance / m).sqrt())
    }

    const FLAT: fn() -> gam_problem::RhoPrior = || gam_problem::RhoPrior::Flat;
    const CONFIGURED: fn() -> gam_problem::RhoPrior = || gam_problem::RhoPrior::Normal {
        mean: 0.0,
        sd: 3.0,
    };
    /// A prior no data could shrug off: it pins λ at `e^{-6}` to a quarter of a
    /// log unit. Any criterion that honors its caller MUST move under it.
    const ABSURD: fn() -> gam_problem::RhoPrior = || gam_problem::RhoPrior::Normal {
        mean: -6.0,
        sd: 0.25,
    };

    #[test]
    fn zz_measure_2450_rho_prior_criterion_bias_ladder() {
        struct Cell {
            family: &'static str,
            n: usize,
            frequency: f64,
            spec: TermCollectionSpec,
        }
        const NOISE_SD: f64 = 0.2;
        const REPLICATES: usize = 8;

        let mut cells = Vec::<Cell>::new();
        for &frequency in &[0.5_f64, 1.0, 2.0, 4.0] {
            cells.push(Cell {
                family: "ps",
                n: 200,
                frequency,
                spec: bspline_spec(20),
            });
        }
        for &frequency in &[0.5_f64, 1.0, 2.0] {
            cells.push(Cell {
                family: "matern",
                n: 200,
                frequency,
                spec: matern_spec(12),
            });
        }

        println!(
            "[2450] paired A/B/C over FitOptions::rho_prior at one SHA. \
             REML = Flat; CONFIGURED = Normal{{0,3}}; \
             ABSURD = Normal{{-6,0.25}} (the discriminator)."
        );
        println!(
            "[2450] {:>7} {:>5} {:>5} | {:>9} {:>9} {:>16} | {:>8} {:>8} | {:>10} {:>10} | {:>9}",
            "family",
            "n",
            "freq",
            "rho_REML",
            "rho_MAP",
            "d_rho(se)",
            "edf_REML",
            "edf_MAP",
            "mise_REML",
            "mise_MAP",
            "d_rho_ABS"
        );

        let mut any_cell_measured = false;
        for cell in &cells {
            let (mut r_flat, mut r_map, mut r_abs) = (vec![], vec![], vec![]);
            let (mut e_flat, mut e_map) = (vec![], vec![]);
            let (mut m_flat, mut m_map) = (vec![], vec![]);
            for replicate in 0..REPLICATES {
                let seed = 0x2450_0000_0000_0000u64
                    ^ ((cell.n as u64) << 32)
                    ^ ((cell.frequency.to_bits() >> 40) << 16)
                    ^ ((cell.family.len() as u64) << 8)
                    ^ replicate as u64;
                let mut stream = GaussStream::new(seed);
                let mut data = Array2::<f64>::zeros((cell.n, 1));
                let mut y = Array1::<f64>::zeros(cell.n);
                let mut truth_values = Array1::<f64>::zeros(cell.n);
                for i in 0..cell.n {
                    let x = i as f64 / (cell.n as f64 - 1.0);
                    data[[i, 0]] = x;
                    truth_values[i] = truth(x, cell.frequency);
                    y[i] = truth_values[i] + NOISE_SD * stream.next_normal();
                }
                let flat = run_arm(&data, &y, &truth_values, &cell.spec, FLAT());
                let map = run_arm(&data, &y, &truth_values, &cell.spec, CONFIGURED());
                let absurd = run_arm(&data, &y, &truth_values, &cell.spec, ABSURD());
                // A replicate counts only when every arm produced a fit;
                // dropping one arm alone would break the pairing.
                let (Some(flat), Some(map), Some(absurd)) = (flat, map, absurd) else {
                    continue;
                };
                r_flat.push(flat.rho_hat);
                r_map.push(map.rho_hat);
                r_abs.push(absurd.rho_hat);
                e_flat.push(flat.edf);
                e_map.push(map.edf);
                m_flat.push(flat.mise);
                m_map.push(map.mise);
            }
            if r_flat.is_empty() {
                println!(
                    "[2450] {:>7} {:>5} {:>5.1} | no replicate produced all three arms",
                    cell.family, cell.n, cell.frequency
                );
                continue;
            }
            any_cell_measured = true;
            let count = r_flat.len() as f64;
            let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / count;
            let (d_rho, d_rho_se) = paired_mean_se(&r_map, &r_flat);
            let (d_abs, _) = paired_mean_se(&r_abs, &r_flat);
            let (d_mise, d_mise_se) = paired_mean_se(&m_map, &m_flat);
            println!(
                "[2450] {:>7} {:>5} {:>5.1} | {:>9.4} {:>9.4} {:>+9.4}({:.4}) | {:>8.3} {:>8.3} | {:>10.3e} {:>10.3e} | {:>+9.4}",
                cell.family,
                cell.n,
                cell.frequency,
                mean(&r_flat),
                mean(&r_map),
                d_rho,
                d_rho_se,
                mean(&e_flat),
                mean(&e_map),
                mean(&m_flat),
                mean(&m_map),
                d_abs
            );
            // The discriminator, stated as a verdict rather than left to the
            // reader: an ABSURD prior that moves nothing means the caller's
            // prior never reached the criterion on this path.
            let honored = r_abs
                .iter()
                .zip(r_flat.iter())
                .any(|(a, f)| a.to_bits() != f.to_bits());
            println!(
                "[2450]   reps={} d_mise={:+.4e} se={:.4e} t={:+.2} | caller's rho_prior {}",
                r_flat.len(),
                d_mise,
                d_mise_se,
                d_mise / d_mise_se,
                if honored {
                    "REACHES the criterion"
                } else {
                    "IS OVERRIDDEN before the fit (bitwise identical under an absurd prior)"
                }
            );
            assert!(
                honored,
                "#2463: {family} n={n} frequency={frequency} accepted an explicit rho prior \
                 but returned bitwise-identical rho under Normal{{-6,0.25}} and Flat",
                family = cell.family,
                n = cell.n,
                frequency = cell.frequency,
            );
        }
        assert!(
            any_cell_measured,
            "#2450 probe took no measurement: no cell produced all three arms"
        );
    }
}
