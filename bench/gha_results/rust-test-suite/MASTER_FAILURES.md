# MASTER_FAILURES

- Compile failures: **3**
- Runtime test failures (FAIL/TIMEOUT/TERMINATING/LEAK): **NOT MEASURED** (14 seen in the shards that did run)
- Python test failures: **NOT MEASURED — at least 6** (LOWER BOUND, not a count: Python API tests (job `cancelled`); Python populations, slow + torch (job `cancelled`) did not run to completion, so the tests they never reached are unmeasured, not passing)
- Forbidden runtime signatures seen: **NOT MEASURED** (0 seen in the shards that did run)
- Slow/timeout notices (#1393): **NOT MEASURED** (0 seen in the shards that did run)

Coverage:
- workspace shards: **NOT MEASURED** (build `success`, matrix `cancelled`)
- gam-pyffi unit tests: **MEASURED** (job `failure`)
- Python API tests: **NOT MEASURED** (job `cancelled`)
- Python populations (slow + torch): **NOT MEASURED** (job `cancelled`)

> NOTE: the Python failure count above is a LOWER BOUND, not a total — it sums over jobs and these did not run to completion: Python API tests (job `cancelled`); Python populations, slow + torch (job `cancelled`). Everything those jobs had not reached when they stopped is unmeasured; do not read the number as "that is how many Python tests are red".

> NOTE: the Python surface was NOT measured — the Python job reported `cancelled`. The Python counter above is not a result.

> NOTE: the runtime surface was NOT measured — the shard matrix reported `cancelled`; only 7 of 10 planned shard logs were collected; a shard reported ARCHIVE_MISSING. Runtime counters above are not results. Fix the build first; the runtime surface will then be exercised.

## Compile failures

- `crates/gam-models/src/fit_orchestration/drivers/constant_curvature_kappa_box_probe_tests.rs:602:14` — [E0277] `drivers::ConstantCurvatureProfile<'_>` doesn't implement `std::fmt::Debug`
- `?` — could not compile `gam-models` (lib test) due to 1 previous error
- `?` — command `/home/runner/.rustup/toolchains/1.97.1-x86_64-unknown-linux-gnu/bin/cargo test --no-run --message-format json-render-diagnostics --workspace --exclude gam-pyffi --all-features --config 'profile.test.debug=0'` exited with code 101

## Runtime test failures

- **FAIL** `gam-pyffi` :: `batch_tests::circle_latent_recovers_circle_not_collapse`
- **FAIL** `gam-pyffi` :: `inference::inference_instruments::tests::matched_controls_do_not_promote_circle_on_seeded_isotropic_noise_2262`
- **FAIL** `gam-pyffi` :: `inference::inference_instruments::tests::ring_of_clusters_owns_discrete_cyclic_verdict_2262`
- **FAIL** `gam-pyffi` :: `manifold::manifold_sae_payload::manifold_sae_payload_serde_tests::crosscoder_layout_and_reports_round_trip_in_v6`
- **FAIL** `gam-pyffi` :: `manifold::manifold_sae_payload::manifold_sae_payload_serde_tests::golden_full_is_a_serde_round_trip_fixed_point`
- **FAIL** `gam-pyffi` :: `manifold::manifold_sae_payload::manifold_sae_payload_serde_tests::invalid_geometry_tuple_and_assignment_alias_are_rejected`
- **FAIL** `gam-pyffi` :: `manifold::manifold_sae_payload::manifold_sae_payload_serde_tests::missing_optional_valued_field_is_rejected`
- **FAIL** `gam-pyffi` :: `manifold::manifold_sae_payload::manifold_sae_payload_serde_tests::runtime_diagnostics_round_trip_in_schema`
- **FAIL** `gam-pyffi` :: `manifold::manifold_sae_payload::manifold_sae_payload_serde_tests::unknown_field_is_rejected`
- **FAIL** `gam-pyffi` :: `manifold::manifold_sae_payload::manifold_sae_payload_serde_tests::v1_payload_is_rejected_instead_of_guessing_crosscoder_layout`
- **FAIL** `gam-pyffi` :: `manifold::manifold_sae_payload::manifold_sae_payload_serde_tests::wrong_schema_tag_is_rejected`
- **FAIL** `gam-pyffi` :: `tests::blocks_negative_reml_score_backward_sign_matches_profile_perturbations`
- **FAIL** `gam-pyffi` :: `tests::manifold_sae_structured_metric_without_behavior_shard_is_loadable`
- **FAIL** `gam-pyffi` :: `tests::shared_tangent_fit_is_output_rotation_equivariant`

## Python test failures

_Lower bound: 6 recorded before the run stopped. Unmeasured: Python API tests (job `cancelled`); Python populations, slow + torch (job `cancelled`)._

- **FAIL** `python::tests/test_python_api` :: `test_transformation_normal_pgs_conditional_mean_tracks_response`
- **FAIL** `python::tests/test_python_api` :: `test_survival_marginal_slope_weibull_n3000_returns_under_60s`
- **FAIL** `python::tests/test_bug_hunt_curv_smooth_hyperbolic_recovered_as_spherical` :: `test_curv_recovers_constant_curvature_sign[-2.0]`
- **FAIL** `python::tests/test_survival_api_regressions` :: `test_joint_competing_risks_survival_is_reachable_from_fit`
- **FAIL** `python::tests/test_survival_api_regressions` :: `test_survival_marginal_slope_fit_returns`
- **FAIL** `python::tests/test_survival_save_load_roundtrip` :: `test_survival_marginal_slope_save_load_predict_roundtrips`

## Forbidden runtime-error signatures

_None._

## Slow / timeout attribution (#1393)

_No test crossed the 300s slow period._

