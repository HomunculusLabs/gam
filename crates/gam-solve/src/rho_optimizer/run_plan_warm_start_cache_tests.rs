// Child module of `run_plan::run_plan_tests` (see the `#[path]` declaration
// there): the warm-start cache lane — the checkpointing objective, iterate
// payload round-trips, cached-beta binding and seeding, cache-entry
// classification, and cached-rho resume/recertification. Scope comes from the
// parent via `use super::*`; the split is purely physical.

use super::*;
use ndarray::array;

#[test]
fn checkpointing_objective_persists_finite_evals() {
    let (_d, session) = tmp_cache_session("ckpt-persist");
    let problem = OuterProblem::new(1).with_gradient(Derivative::Unavailable);
    let mut inner: ClosureObjective<_, _, _> = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
        |_: &mut (), _: &Array1<f64>| Err(EstimationError::InvalidInput("eval not used".into())),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut wrapped = CheckpointingObjective::new(&mut inner, Arc::clone(&session), Vec::new());
    // Initial: nothing on disk.
    assert!(session.try_load().is_none());
    // First eval persists.
    let v0 = wrapped.eval_cost(&array![3.0]).unwrap();
    assert!((v0 - 9.0).abs() < 1e-12);
    let on_disk = session.try_load().expect("first eval should checkpoint");
    let payload = decode_iterate(&on_disk.payload, 1).expect("payload decodes");
    assert!((payload.cost - 9.0).abs() < 1e-12);
    assert_eq!(payload.rho, vec![3.0]);
    // Strictly improving eval must bypass the 2-second rate limit.
    let v1 = wrapped.eval_cost(&array![0.5]).unwrap();
    assert!((v1 - 0.25).abs() < 1e-12);
    let on_disk = session
        .try_load()
        .expect("improving eval should checkpoint");
    let payload = decode_iterate(&on_disk.payload, 1).expect("payload decodes");
    assert!((payload.cost - 0.25).abs() < 1e-12);
    assert_eq!(payload.rho, vec![0.5]);
    // Non-finite values must not corrupt the on-disk best-known iterate.
    let v_inf = wrapped.eval_cost(&array![f64::NAN]);
    match v_inf {
        Ok(value) => assert!(!value.is_finite()),
        Err(err) => assert!(!err.to_string().is_empty()),
    }
    let on_disk = session.try_load().expect("prior best preserved");
    let payload = decode_iterate(&on_disk.payload, 1).expect("payload decodes");
    assert!((payload.cost - 0.25).abs() < 1e-12);
}

#[test]
fn checkpointing_objective_never_persists_reactive_initialization_waypoints() {
    let (_d, session) = tmp_cache_session("ckpt-reactive-phase");
    let mut inner =
        ReactiveDomainObjective::new(0.125, ReactiveDomainMode::FiniteAtColdSeed);
    let mut wrapped = CheckpointingObjective::new(&mut inner, Arc::clone(&session), Vec::new());

    wrapped
        .begin_reactive_domain_waypoint()
        .expect("begin typed reactive waypoint");
    let entry_cost = wrapped
        .eval_cost(&array![2.5])
        .expect("finite initialization waypoint");
    assert!(entry_cost.is_finite());
    assert!(
        session.try_load().is_none(),
        "a reactive initialization waypoint is not an outer candidate and must not become a restart seed",
    );
    wrapped
        .commit_reactive_domain_waypoint(&array![2.5])
        .expect("commit typed reactive waypoint");

    wrapped
        .eval_cost(&array![0.125])
        .expect("literal target evaluation");
    let payload = decode_iterate(
        &session
            .try_load()
            .expect("literal target should checkpoint")
            .payload,
        1,
    )
    .expect("literal checkpoint decodes");
    assert_eq!(payload.rho, vec![0.125]);
}

#[test]
fn checkpointing_objective_rejects_wrong_dim_on_decode() {
    // A payload from a 3-dim fit is invalid input for a 5-dim resume.
    let bytes = encode_iterate(&array![1.0, 2.0, 3.0], None, None, 0.5, 0).expect("encode");
    assert!(decode_iterate(&bytes, 3).is_some());
    assert!(decode_iterate(&bytes, 5).is_none());
}

#[test]
fn schema_two_iterate_is_rejected_after_hessian_provenance_break_2253() {
    let obsolete = serde_json::json!({
        "schema": 2,
        "rho": [0.5],
        "beta": [1.0],
        "hessian": [4.0],
        "hessian_dim": 1,
        "cost": 2.0,
        "eval_id": 7
    });
    let bytes = serde_json::to_vec(&obsolete).expect("serialize obsolete payload");
    assert!(decode_iterate(&bytes, 1).is_none());
}

#[test]
fn transferred_hessian_requires_current_analytic_capability_2253() {
    let hessian = array![[2.0_f64, 0.0], [0.0, 3.0]];
    assert!(
        eligible_transferred_outer_hessian(
            Some(&hessian),
            DeclaredHessianForm::Unavailable,
            2,
        )
        .is_none()
    );
    assert!(
        eligible_transferred_outer_hessian(Some(&hessian), DeclaredHessianForm::Dense, 2)
            .is_some()
    );
}

#[test]
fn iterate_payload_round_trips_beta() {
    // Every persisted entry that comes with an inner-β hint round-trips
    // (ρ, β) together — that pair lets a resume open inner PIRLS in the
    // basin of quadratic attraction regardless of where ρ sits.
    let rho = array![10.0, -10.0, 5.0];
    let beta = array![0.12, -0.34, 0.56, 7.89];
    let bytes = encode_iterate(&rho, Some(&beta), None, 1.0, 7).expect("encode");
    let decoded = decode_iterate(&bytes, rho.len()).expect("decode");
    assert_eq!(decoded.rho, rho.to_vec());
    assert_eq!(decoded.beta, beta.to_vec());
    // ρ-only writes (β = None) still encode but with an empty beta slot.
    let ro_bytes = encode_iterate(&rho, None, None, 1.0, 7).expect("encode-rho-only");
    let ro = decode_iterate(&ro_bytes, rho.len()).expect("decode-rho-only");
    assert!(ro.beta.is_empty());
}

#[test]
fn iterate_payload_round_trips_converged_outer_hessian() {
    // The converged outer curvature persists alongside (ρ, β) so the next
    // structurally-matching fit can seed BFGS with H⁻¹ for a quasi-Newton
    // first step instead of restarting from an unscaled identity metric.
    let rho = array![0.5, -1.5];
    let h = array![[4.0, 1.0], [1.0, 3.0]];
    let bytes = encode_iterate(&rho, None, Some(&h), 1.0, 0).expect("encode");
    let decoded = decode_iterate(&bytes, rho.len()).expect("decode");
    assert_eq!(decoded.hessian_dim, 2);
    assert_eq!(decoded.hessian, vec![4.0, 1.0, 1.0, 3.0]);

    // The classifier surfaces the square Hessian as a (dim, flat) pair on the
    // Seed decision so the resume path can reconstruct and invert it.
    let loaded = gam_runtime::warm_start::LoadedEntry {
        entry: gam_runtime::warm_start::WarmStartEntry {
            payload: bytes,
            objective: Some(1.0),
            iteration: Some(0),
            kind: gam_runtime::warm_start::EntryKind::Checkpoint,
            written_unix_secs: 0,
        },
        source: gam_runtime::warm_start::LoadSource::Preloaded,
    };
    let CacheSeedDecision::Seed {
        hessian: decoded_h, ..
    } = classify_cache_entry_for_outer(&loaded, 2)
    else {
        panic!("expected Seed decision");
    };
    let (dim, flat) = decoded_h.expect("Seed must carry the persisted Hessian");
    assert_eq!(dim, 2);
    assert_eq!(flat, vec![4.0, 1.0, 1.0, 3.0]);
}

#[test]
fn iterate_payload_scrubs_non_finite_or_non_square_hessian() {
    // A malformed curvature must never reach the warm-start metric: a
    // non-square or non-finite Hessian is scrubbed to "no Hessian" while the
    // ρ/β seed is preserved, so the resume degrades to the scalar metric
    // rather than corrupting the first BFGS step.
    let rho = array![0.0];
    let nan_h = array![[f64::NAN]];
    let bytes = encode_iterate(&rho, None, Some(&nan_h), 1.0, 0).expect("encode");
    // encode_iterate itself drops a non-finite Hessian before serialization.
    let decoded = decode_iterate(&bytes, 1).expect("decode");
    assert_eq!(decoded.hessian_dim, 0);
    assert!(decoded.hessian.is_empty());
}

#[test]
fn note_persists_inner_beta_hint_from_eval() {
    // Write-side proof of the principled fix: when the inner solver
    // surfaces β via OuterEval::inner_beta_hint, CheckpointingObjective
    // captures it on every accepted eval AND exposes it for finalize.
    let (_d, session) = tmp_cache_session("note-persists-beta");
    let problem = OuterProblem::new(1).with_gradient(Derivative::Unavailable);
    let mut inner: ClosureObjective<_, _, _> = problem.build_objective(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(1.0),
        |_: &mut (), theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: theta[0] * theta[0],
                gradient: array![2.0 * theta[0]],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: Some(array![1.5, 2.5, 3.5]),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut wrapped = CheckpointingObjective::new(&mut inner, Arc::clone(&session), Vec::new());
    let eval = wrapped.eval(&array![0.5]).expect("eval ok");
    assert!((eval.cost - 0.25).abs() < 1e-12);
    let on_disk = session
        .try_load()
        .expect("eval with finite β must persist a (ρ,β) checkpoint");
    let payload = decode_iterate(&on_disk.payload, 1).expect("payload decodes");
    assert_eq!(payload.beta, vec![1.5, 2.5, 3.5]);
    let captured = wrapped
        .inner_beta_for(&array![0.5])
        .expect("β was captured at its producing rho");
    assert_eq!(captured.to_vec(), vec![1.5, 2.5, 3.5]);
    assert!(
        wrapped.inner_beta_for(&array![0.25]).is_none(),
        "β from rho=0.5 must not be exposed as state for another outer coordinate",
    );
}

#[test]
fn note_rejects_nonfinite_inner_beta() {
    // A divergent inner state must NOT poison the cache: persisting a
    // non-finite β would re-create the inner-PIRLS budget-exhaustion
    // failure mode at boundary ρ where the cached β is supposed to
    // place the resume inside Newton's quadratic basin.
    let (_d, session) = tmp_cache_session("note-rejects-bad-beta");
    let problem = OuterProblem::new(1).with_gradient(Derivative::Unavailable);
    let mut inner: ClosureObjective<_, _, _> = problem.build_objective(
        (),
        |_: &mut (), _: &Array1<f64>| Ok(1.0),
        |_: &mut (), theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: theta[0] * theta[0],
                gradient: array![2.0 * theta[0]],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: Some(array![f64::NAN, 0.5]),
            })
        },
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut wrapped = CheckpointingObjective::new(&mut inner, Arc::clone(&session), Vec::new());
    let eval = wrapped.eval(&array![0.5]).expect("eval ok");
    assert!((eval.cost - 0.25).abs() < 1e-12);
    assert!(
        session.try_load().is_none(),
        "non-finite β must abort the checkpoint write, not poison the cache",
    );
    assert!(
        wrapped.inner_beta_for(&array![0.5]).is_none(),
        "non-finite β must not be exposed as exact outer/inner state",
    );
}

#[test]
fn classify_extracts_beta_from_v2_payload() {
    // The classifier propagates `beta` from the v2 payload onto its
    // Seed/ExactFinal decisions so the dispatcher can hand it to
    // OuterObjective::seed_inner_state. Without this, the (ρ, β) payload
    // would write β but never resurface it on resume.
    let rho = array![1.0, 2.0];
    let beta = array![10.0, 20.0, 30.0];
    let payload = encode_iterate(&rho, Some(&beta), None, 1.0, 0).expect("encode");
    let loaded = gam_runtime::warm_start::LoadedEntry {
        entry: gam_runtime::warm_start::WarmStartEntry {
            payload,
            objective: Some(1.0),
            iteration: Some(0),
            kind: gam_runtime::warm_start::EntryKind::Checkpoint,
            written_unix_secs: 0,
        },
        source: gam_runtime::warm_start::LoadSource::Preloaded,
    };
    let CacheSeedDecision::Seed {
        beta: decoded_beta, ..
    } = classify_cache_entry_for_outer(&loaded, 2)
    else {
        panic!("expected Seed decision");
    };
    assert_eq!(decoded_beta, beta.to_vec());

    // ρ-only payload (legacy or family-without-β) decodes to empty beta.
    let payload = encode_iterate(&rho, None, None, 1.0, 0).expect("encode");
    let loaded = gam_runtime::warm_start::LoadedEntry {
        entry: gam_runtime::warm_start::WarmStartEntry {
            payload,
            objective: Some(1.0),
            iteration: Some(0),
            kind: gam_runtime::warm_start::EntryKind::Checkpoint,
            written_unix_secs: 0,
        },
        source: gam_runtime::warm_start::LoadSource::Preloaded,
    };
    let CacheSeedDecision::Seed {
        beta: decoded_beta, ..
    } = classify_cache_entry_for_outer(&loaded, 2)
    else {
        panic!("expected Seed decision");
    };
    assert!(
        decoded_beta.is_empty(),
        "ρ-only payload must produce an empty beta so the dispatcher skips seed_inner_state"
    );
}

#[test]
fn cached_beta_binds_only_to_its_bitwise_matching_generated_seed() {
    struct ReplayObj {
        installed: Option<Array1<f64>>,
        seed_calls: usize,
    }
    impl OuterObjective for ReplayObj {
        fn capability(&self) -> OuterCapability {
            OuterCapability {
                gradient: Derivative::Analytic,
                hessian: DeclaredHessianForm::Unavailable,
                n_params: 2,
                psi_dim: 0,
                fixed_point_available: false,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            }
        }

        fn eval_cost(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
            Ok(theta.dot(theta))
        }

        fn eval(&mut self, theta: &Array1<f64>) -> Result<OuterEval, EstimationError> {
            Ok(OuterEval {
                cost: theta.dot(theta),
                gradient: 2.0 * theta,
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        }

        fn reset(&mut self) {
            self.installed = None;
        }

        fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
            self.seed_calls += 1;
            self.installed = Some(beta.clone());
            Ok(SeedOutcome::Installed)
        }
    }

    let owner = array![1.0, 2.0];
    let beta = array![7.0, 8.0, 9.0];
    let config = OuterConfig {
        initial_inner_seed: Some(BoundInnerSeed {
            theta: owner.clone(),
            beta: beta.clone(),
        }),
        ..OuterConfig::default()
    };
    let one_ulp_away = array![f64::from_bits(1.0_f64.to_bits() + 1), 2.0];
    let candidates = [one_ulp_away, owner, array![-1.0, 2.0]];
    let mut objective = ReplayObj {
        installed: None,
        seed_calls: 0,
    };

    for (index, candidate) in candidates.iter().enumerate() {
        objective.reset();
        install_matching_initial_inner_seed(
            &mut objective,
            &config,
            candidate,
            "bitwise seed ownership",
        )
        .expect("cache replay decision");
        if index == 1 {
            assert_eq!(objective.installed, Some(beta.clone()));
        } else {
            assert!(
                objective.installed.is_none(),
                "cached beta leaked into non-owning generated seed {index}",
            );
        }
    }
    assert_eq!(objective.seed_calls, 1);
}

#[test]
fn run_calls_seed_inner_state_with_cached_beta() {
    // End-to-end read-side wiring: a cache hit carrying β must call
    // OuterObjective::seed_inner_state(&beta) *before* the first BFGS
    // eval. We verify this by routing through a custom OuterObjective
    // that records the β it was seeded with.
    struct RecordingObj {
        seeded: Arc<Mutex<Option<Array1<f64>>>>,
        first_eval_seeded: Arc<Mutex<Option<Array1<f64>>>>,
        eval_count: Arc<Mutex<usize>>,
    }
    impl OuterObjective for RecordingObj {
        fn capability(&self) -> OuterCapability {
            // Analytic gradient AND analytic Hessian so the planner picks
            // the same Hessian-bearing path a real fit takes; using
            // Unavailable here would test a degenerate plan.
            OuterCapability {
                gradient: Derivative::Analytic,
                hessian: DeclaredHessianForm::Dense,
                n_params: 2,
                psi_dim: 0,
                fixed_point_available: false,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            }
        }
        fn eval_cost(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
            Ok(theta.dot(theta))
        }
        fn eval(&mut self, theta: &Array1<f64>) -> Result<OuterEval, EstimationError> {
            let mut eval_count = self.eval_count.lock().unwrap();
            if *eval_count == 0 {
                *self.first_eval_seeded.lock().unwrap() = self.seeded.lock().unwrap().clone();
            }
            *eval_count += 1;
            // f(θ) = ‖θ‖² → ∇f = 2θ, ∇²f = 2I.
            Ok(OuterEval {
                cost: theta.dot(theta),
                gradient: 2.0 * theta,
                hessian: HessianValue::Dense(2.0 * Array2::<f64>::eye(theta.len())),
                inner_beta_hint: None,
            })
        }
        fn reset(&mut self) {
            *self.seeded.lock().unwrap() = None;
        }
        fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
            *self.seeded.lock().unwrap() = Some(beta.clone());
            Ok(SeedOutcome::Installed)
        }
    }

    let (_d, session) = tmp_cache_session("seed-inner-state-call");
    let bytes = encode_iterate(
        &array![1.0, 2.0],
        Some(&array![7.5, 8.5, 9.5]),
        None,
        5.0,
        3,
    )
    .expect("encode");
    session.checkpoint(&bytes, Some(5.0), Some(3));

    let seeded: Arc<Mutex<Option<Array1<f64>>>> = Arc::new(Mutex::new(None));
    let first_eval_seeded: Arc<Mutex<Option<Array1<f64>>>> = Arc::new(Mutex::new(None));
    let eval_count: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let mut obj = RecordingObj {
        seeded: Arc::clone(&seeded),
        first_eval_seeded: Arc::clone(&first_eval_seeded),
        eval_count: Arc::clone(&eval_count),
    };

    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_max_iter(1)
        .with_cache_session(Arc::clone(&session));
    match problem.run(&mut obj, "seed-inner-state-call") {
        Ok(result) => assert!(result.final_value.is_finite()),
        Err(err) => assert!(!err.to_string().is_empty()),
    }

    let observed = first_eval_seeded.lock().unwrap().clone();
    assert_eq!(
        observed,
        Some(array![7.5, 8.5, 9.5]),
        "the first exact evaluation after the per-seed reset must observe the cached β",
    );
}

#[test]
fn run_skips_seed_inner_state_when_payload_has_no_beta() {
    // Symmetric guard: a ρ-only warm-start entry must NOT invoke
    // seed_inner_state — calling it with an empty / zero / garbage β
    // would silently degrade a family that has a non-trivial inner
    // default into one started at zeros.
    struct CountingObj {
        seed_calls: Arc<Mutex<usize>>,
    }
    impl OuterObjective for CountingObj {
        fn capability(&self) -> OuterCapability {
            // Analytic gradient AND analytic Hessian so the planner picks
            // the same Hessian-bearing path a real fit takes; using
            // Unavailable here would test a degenerate plan.
            OuterCapability {
                gradient: Derivative::Analytic,
                hessian: DeclaredHessianForm::Dense,
                n_params: 2,
                psi_dim: 0,
                fixed_point_available: false,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: false,
            }
        }
        fn eval_cost(&mut self, theta: &Array1<f64>) -> Result<f64, EstimationError> {
            Ok(theta.dot(theta))
        }
        fn eval(&mut self, theta: &Array1<f64>) -> Result<OuterEval, EstimationError> {
            // f(θ) = ‖θ‖² → ∇f = 2θ, ∇²f = 2I.
            Ok(OuterEval {
                cost: theta.dot(theta),
                gradient: 2.0 * theta,
                hessian: HessianValue::Dense(2.0 * Array2::<f64>::eye(theta.len())),
                inner_beta_hint: None,
            })
        }
        fn reset(&mut self) {}
        fn seed_inner_state(&mut self, beta: &Array1<f64>) -> Result<SeedOutcome, EstimationError> {
            *self.seed_calls.lock().unwrap() += beta.len().max(1);
            Ok(SeedOutcome::Installed)
        }
    }

    let (_d, session) = tmp_cache_session("seed-inner-state-skip");
    // ρ-only payload — no β.
    let bytes = encode_iterate(&array![1.0, 2.0], None, None, 5.0, 3).expect("encode");
    session.checkpoint(&bytes, Some(5.0), Some(3));

    let seed_calls: Arc<Mutex<usize>> = Arc::new(Mutex::new(0));
    let mut obj = CountingObj {
        seed_calls: Arc::clone(&seed_calls),
    };

    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_max_iter(1)
        .with_cache_session(Arc::clone(&session));
    match problem.run(&mut obj, "seed-inner-state-skip") {
        Ok(result) => assert!(result.final_value.is_finite()),
        Err(err) => assert!(!err.to_string().is_empty()),
    }

    assert_eq!(
        *seed_calls.lock().unwrap(),
        0,
        "seed_inner_state must not fire when the cached payload carries no β",
    );
}

#[test]
fn cache_entry_classifier_honors_finite_seeds_regardless_of_saturation() {
    // The classifier no longer reshapes ρ based on shape. Any finite,
    // correctly-dimensioned payload is honored as the next run's seed.
    // Boundary-saturated entries written under the v2 (ρ, β) invariant
    // are a *legitimate* finding — the smoothness wants to be near-null
    // — and the persisted β puts the next inner solve at zero-gradient,
    // making the cold-β failure mode impossible to re-create from cache.
    for rho_seed in [array![9.0, 0.0], array![10.0, -10.0], array![-10.0, 10.0]] {
        let payload = encode_iterate(&rho_seed, None, None, 1.0, 0).expect("encode");
        let loaded = gam_runtime::warm_start::LoadedEntry {
            entry: gam_runtime::warm_start::WarmStartEntry {
                payload,
                objective: Some(1.0),
                iteration: Some(0),
                kind: gam_runtime::warm_start::EntryKind::Checkpoint,
                written_unix_secs: 0,
            },
            source: gam_runtime::warm_start::LoadSource::Preloaded,
        };

        assert!(cache_entry_would_help_outer(&loaded, 2));
        let CacheSeedDecision::Seed { rho, .. } = classify_cache_entry_for_outer(&loaded, 2) else {
            panic!(
                "finite seed {:?} must be honored unchanged; the read-side clamp / \
                     all-saturated-discard branches were band-aids over the missing β cache",
                rho_seed
            );
        };
        assert_eq!(rho, rho_seed, "ρ must round-trip without reshaping");
    }
}

#[test]
fn cache_entry_classifier_rejects_only_structural_failures() {
    // Only structural failures discard: payload shape (wrong rho_dim,
    // non-finite payload internals → decode None → "payload-shape-mismatch")
    // and non-finite warm-start metadata → "non-finite-payload". Saturation
    // and β presence are NOT discards here: saturation is honored, and
    // ρ-only payloads decode cleanly with an empty β slot.

    // Non-finite metadata objective: decode succeeds (finite payload
    // cost), but the entry-level objective is NaN — discard as
    // non-finite-payload.
    let payload = encode_iterate(&array![0.5, 0.5], None, None, 1.0, 0).expect("encode");
    let loaded = gam_runtime::warm_start::LoadedEntry {
        entry: gam_runtime::warm_start::WarmStartEntry {
            payload,
            objective: Some(f64::NAN),
            iteration: Some(0),
            kind: gam_runtime::warm_start::EntryKind::Checkpoint,
            written_unix_secs: 0,
        },
        source: gam_runtime::warm_start::LoadSource::Preloaded,
    };
    assert!(matches!(
        classify_cache_entry_for_outer(&loaded, 2),
        CacheSeedDecision::Discard {
            reason: "non-finite-payload",
            ..
        }
    ));

    // Dimension mismatch: 2-D payload viewed as a 3-D problem → decode
    // rejects shape → "payload-shape-mismatch".
    let payload = encode_iterate(&array![0.5, 0.5], None, None, 1.0, 0).expect("encode");
    let loaded = gam_runtime::warm_start::LoadedEntry {
        entry: gam_runtime::warm_start::WarmStartEntry {
            payload,
            objective: Some(1.0),
            iteration: Some(0),
            kind: gam_runtime::warm_start::EntryKind::Checkpoint,
            written_unix_secs: 0,
        },
        source: gam_runtime::warm_start::LoadSource::Preloaded,
    };
    assert!(matches!(
        classify_cache_entry_for_outer(&loaded, 3),
        CacheSeedDecision::Discard {
            reason: "payload-shape-mismatch",
            ..
        }
    ));
}

#[test]
fn exact_final_warm_start_hit_is_helpful_even_at_boundary() {
    let payload = encode_iterate(&array![10.0, -10.0], None, None, 1.0, 3).expect("encode");
    let loaded = gam_runtime::warm_start::LoadedEntry {
        entry: gam_runtime::warm_start::WarmStartEntry {
            payload,
            objective: Some(1.0),
            iteration: Some(3),
            kind: gam_runtime::warm_start::EntryKind::Final,
            written_unix_secs: 0,
        },
        source: gam_runtime::warm_start::LoadSource::Exact,
    };

    assert!(cache_entry_would_help_outer(&loaded, 2));
    assert!(matches!(
        classify_cache_entry_for_outer(&loaded, 2),
        CacheSeedDecision::ExactFinal { iterations: 3, .. }
    ));
}

#[test]
fn checkpointing_objective_mirrors_checkpoints() {
    let (_primary_dir, primary) = tmp_cache_session("ckpt-primary");
    let (_mirror_dir, mirror) = tmp_cache_session("ckpt-mirror");
    let problem = OuterProblem::new(1).with_gradient(Derivative::Unavailable);
    let mut inner: ClosureObjective<_, _, _> = problem.build_objective(
        (),
        |_: &mut (), theta: &Array1<f64>| Ok(theta[0] * theta[0]),
        |_: &mut (), _: &Array1<f64>| Err(EstimationError::InvalidInput("eval not used".into())),
        None::<fn(&mut ())>,
        None::<fn(&mut (), &Array1<f64>) -> Result<EfsEval, EstimationError>>,
    );
    let mut wrapped =
        CheckpointingObjective::new(&mut inner, Arc::clone(&primary), vec![Arc::clone(&mirror)]);

    let value = wrapped.eval_cost(&array![4.0]).unwrap();
    assert_eq!(value, 16.0);

    let primary_payload =
        decode_iterate(&primary.try_load().expect("primary checkpoint").payload, 1)
            .expect("primary decode");
    let mirror_payload = decode_iterate(&mirror.try_load().expect("mirror checkpoint").payload, 1)
        .expect("mirror decode");
    assert_eq!(primary_payload.rho, vec![4.0]);
    assert_eq!(mirror_payload.rho, vec![4.0]);
    assert_eq!(primary_payload.cost, mirror_payload.cost);
}

#[test]
fn cached_rho_is_prepended_as_first_seed() {
    // Whitebox: pre-seed the session with a known iterate, then run
    // an OuterProblem with a deliberately-different `initial_rho`.
    // The runner must visit the cached rho before the configured
    // `initial_rho` because `try_load` overrode it.
    let (_d, session) = tmp_cache_session("seed-prepend");
    // Hand-write the cached checkpoint: rho = [2.5], cost = 0.25.
    // Final exact hits return immediately; checkpoints still exercise the
    // regular seed-prepend path.
    let payload = encode_iterate(&array![2.5], None, None, 0.25, 0).expect("encode");
    session.checkpoint(&payload, Some(0.25), Some(0));
    assert!(
        session.try_load().is_some(),
        "precondition: cache populated"
    );

    let seen: Arc<Mutex<Vec<Array1<f64>>>> = Arc::new(Mutex::new(Vec::new()));
    // A gradient-bearing BFGS problem. Bounds must contain the cached rho
    // so the projector doesn't snap it away.
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_bounds(array![-5.0], array![5.0])
        .with_initial_rho(array![-3.0]) // deliberately not 2.5
        .with_max_iter(8)
        .with_cache_session(Arc::clone(&session));
    let mut obj = problem.build_objective(
        seen.clone(),
        |seen: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
            seen.lock().unwrap().push(theta.clone());
            Ok((theta[0] - 2.5).powi(2))
        },
        {
            let seen = seen.clone();
            move |_: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
                // Record full-eval points too: startup now performs a DIRECT full
                // evaluation at the seed (the `run_starts_solver_with_direct_startup_eval`
                // contract) routed through THIS eval path, not a separate cost-screening
                // pass. The cached rho is consumed by that startup eval, so a recorder
                // that watched only the cost closure never saw the exact cached ρ — it
                // only caught the later line-search cost probes clustered near it.
                seen.lock().unwrap().push(theta.clone());
                Ok(OuterEval {
                    cost: (theta[0] - 2.5).powi(2),
                    gradient: array![2.0 * (theta[0] - 2.5)],
                    hessian: HessianValue::Unavailable,
                    inner_beta_hint: None,
                })
            }
        },
        None::<fn(&mut Arc<Mutex<Vec<Array1<f64>>>>)>,
        None::<
            fn(&mut Arc<Mutex<Vec<Array1<f64>>>>, &Array1<f64>) -> Result<EfsEval, EstimationError>,
        >,
    );
    match problem.run(&mut obj, "seed-prepend") {
        Ok(result) => assert!(result.final_value.is_finite()),
        Err(err) => assert!(!err.to_string().is_empty()),
    }
    // The cached rho (2.5) must appear in the eval trace, and it must
    // appear no later than the configured initial_rho (−3.0). Both
    // are inside the bounds so the projector cannot rewrite them.
    let evals = seen.lock().unwrap();
    let pos_cached = evals.iter().position(|r| (r[0] - 2.5).abs() < 1e-9);
    let pos_initial = evals.iter().position(|r| (r[0] + 3.0).abs() < 1e-9);
    assert!(
        pos_cached.is_some(),
        "cached rho must be evaluated; saw {:?}",
        *evals
    );
    if let (Some(c), Some(i)) = (pos_cached, pos_initial) {
        assert!(
            c <= i,
            "cached rho (idx {c}) must precede initial_rho (idx {i})",
        );
    }
}

#[test]
fn all_saturated_cached_rho_is_honored_as_seed() {
    // Inverse of the prior `all_saturated_cached_rho_is_discarded_before_seed_validation`
    // test. Under v1 the cache stored ρ-only, so resuming at boundary ρ
    // forced PIRLS to cold-start β against a Hessian with condition
    // number `≈ e^{2·rho_bound}` — Newton degraded to O(1/k) descent
    // that exhausted the cycle budget. The "discard if all-saturated"
    // branch was a read-side band-aid; it suppressed a legitimate
    // resume signal in exchange for tolerating the broken contract.
    //
    // Under v2 the iterate payload carries (ρ, β). When β is persisted
    // alongside boundary ρ the next inner solve opens at zero gradient,
    // and the conditioning is no longer a barrier. Therefore the
    // classifier no longer reshapes ρ based on saturation: every
    // finite, correctly-dimensioned entry is used as the seed. This
    // test pins that contract.
    let (_d, session) = tmp_cache_session("all-saturated-honored");
    let payload = encode_iterate(&array![10.0, -10.0], None, None, 1.0, 0).expect("encode");
    session.checkpoint(&payload, Some(1.0), Some(0));
    assert!(
        session.try_load().is_some(),
        "precondition: cache populated"
    );

    let seen: Arc<Mutex<Vec<Array1<f64>>>> = Arc::new(Mutex::new(Vec::new()));
    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.max_seeds = 4;
    seed_config.seed_budget = 1;
    let problem = OuterProblem::new(2)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_seed_config(seed_config)
        .with_initial_rho(array![0.0, 0.0])
        .with_rho_bound(10.0)
        .with_max_iter(1)
        .with_cache_session(Arc::clone(&session));

    let mut obj = problem.build_objective(
        seen.clone(),
        |_: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| Ok(theta.dot(theta)),
        |seen: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
            seen.lock().unwrap().push(theta.clone());
            Ok(OuterEval {
                cost: theta.dot(theta),
                gradient: theta.clone(),
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut Arc<Mutex<Vec<Array1<f64>>>>)>,
        None::<
            fn(&mut Arc<Mutex<Vec<Array1<f64>>>>, &Array1<f64>) -> Result<EfsEval, EstimationError>,
        >,
    );

    match problem.run(&mut obj, "all-saturated-honored") {
        Ok(result) => assert!(result.final_value.is_finite()),
        Err(err) => assert!(!err.to_string().is_empty()),
    }
    let evals = seen.lock().unwrap();
    assert!(
        evals.iter().any(|rho| rho == array![10.0, -10.0]),
        "cached saturated ρ must be evaluated unchanged under v2 (ρ, β) invariant; saw {:?}",
        *evals
    );
}

#[test]
fn exact_final_cache_hit_resumes_and_recertifies_without_resolving() {
    let (_d, session) = tmp_cache_session("final-skip");
    let payload = encode_iterate(&array![2.5], None, None, 0.25, 7).expect("encode");
    session.finalize(&payload, Some(0.25), Some(7));

    let seen: Arc<Mutex<Vec<Array1<f64>>>> = Arc::new(Mutex::new(Vec::new()));
    // An exact final cache hit seeds the solver AT the cached rho and re-runs
    // certification of the CURRENT criterion (run.rs resume-and-recertify): the
    // cache donates only the starting point, never the shipped value, so a stale
    // or version-drifted cache can never change the outcome (#2363). These trivial
    // derivatives make the recertify pass converge in a single step.
    let problem = OuterProblem::new(1)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Unavailable)
        .with_bounds(array![-5.0], array![5.0])
        .with_initial_rho(array![-3.0])
        .with_max_iter(8)
        .with_cache_session(Arc::clone(&session));
    let mut obj = problem.build_objective(
        seen.clone(),
        |seen: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
            seen.lock().unwrap().push(theta.clone());
            Ok((theta[0] - 2.5).powi(2))
        },
        |_: &mut Arc<Mutex<Vec<Array1<f64>>>>, theta: &Array1<f64>| {
            Ok(OuterEval {
                cost: (theta[0] - 2.5).powi(2),
                gradient: array![2.0 * (theta[0] - 2.5)],
                hessian: HessianValue::Unavailable,
                inner_beta_hint: None,
            })
        },
        None::<fn(&mut Arc<Mutex<Vec<Array1<f64>>>>)>,
        None::<
            fn(&mut Arc<Mutex<Vec<Array1<f64>>>>, &Array1<f64>) -> Result<EfsEval, EstimationError>,
        >,
    );

    let result = problem
        .run(&mut obj, "final-skip")
        .expect("exact final cache hit must resume-and-recertify from the cached rho");
    // Resumed FROM the cached rho (not the -3.0 initial): the recertify lands on
    // the cached optimum.
    assert_eq!(result.rho, array![2.5]);
    // The shipped value is the RECOMPUTED true objective at rho=2.5, which is
    // (2.5-2.5)^2 = 0, NOT the fictional stored 0.25. Outcome-invariance: the
    // cache donates the seed, the current criterion decides the value.
    assert_eq!(result.final_value, 0.0);
    assert!(result.converged());
    // Accelerator half AND proof the run RESUMED from the cached rho: the recertify
    // must certify in ~0-1 outer iterations. The Hessian-free gradient solve here
    // could not reach the 2.5 optimum from the -3.0 initial in a single step, so a
    // bound of 1 is only reachable if the solver was SEEDED at the cached rho
    // (screen_initial_rho = false). A regression that cold-solved from -3.0 on
    // every cache hit would blow this bound -- skipping that work is the whole
    // point of the cache -- and no other test would catch it.
    assert!(
        result.iterations <= 1,
        "resume-at-cached-optimum must not cold-solve; got {} iterations",
        result.iterations,
    );
    // The cache donates only the SEED: the resume path optimizes through the
    // gradient objective, so `cost_fn` is never used to SOLVE. The load-bearing
    // statement is therefore about WHERE it is called, not how often — every
    // call must price the CACHED rho. A regression that cold-solved from the
    // -3.0 initial, or that priced a search iterate, shows up here as a rho
    // that is not 2.5, whatever the count.
    let seen = seen.lock().unwrap();
    assert!(
        !seen.is_empty(),
        "a certificate minted without ever pricing the scalar lane is not audited",
    );
    assert!(
        seen.iter().all(|rho| *rho == array![2.5]),
        "the audit must price the CACHED rho, never a cold-solve or search iterate; saw {seen:?}",
    );
    // Count, derived rather than observed. On the ANALYTIC route the same-rho
    // value audit runs at BOTH certification fidelities and `8086f0f50` says why:
    // it is a refusal gate on `final_value`, the very number
    // `retain_best_outer_checkpoint` ranks multistart candidates by, so screening
    // cannot skip it without letting an unvalidated value win the ranking. (The
    // EFS route gates it to the mint instead — there screening and mint are
    // otherwise identical work, which is what `run_efs_skips_global_cost_screening`
    // pins.) One screening audit plus one mint audit is two. The bound matters
    // because a regression that moved the audit back to per-CANDIDATE would scale
    // it with the seed budget instead of staying at one per fidelity.
    assert_eq!(
        seen.len(),
        2,
        "the analytic route audits the scalar lane once per certification fidelity \
         (screening, then mint), not once per candidate; saw {seen:?}",
    );
}
