# MASTER_FAILURES

- Compile failures: **3**
- Runtime test failures (FAIL/TIMEOUT/TERMINATING/LEAK): **NOT MEASURED** (14 seen in the shards that did run)
- Python test failures: **NOT MEASURED — at least 147** (LOWER BOUND, not a count: Python populations, slow + torch (job `cancelled`) did not run to completion, so the tests they never reached are unmeasured, not passing)
- Forbidden runtime signatures seen: **NOT MEASURED** (0 seen in the shards that did run)
- Slow/timeout notices (#1393): **NOT MEASURED** (0 seen in the shards that did run)

Coverage:
- workspace shards: **NOT MEASURED** (build `success`, matrix `success`)
- gam-pyffi unit tests: **MEASURED** (job `failure`)
- Python API tests: **MEASURED** (job `failure`)
- Python populations (slow + torch): **NOT MEASURED** (job `cancelled`)

> NOTE: the Python failure count above is a LOWER BOUND, not a total — it sums over jobs and these did not run to completion: Python populations, slow + torch (job `cancelled`). Everything those jobs had not reached when they stopped is unmeasured; do not read the number as "that is how many Python tests are red".

> NOTE: the runtime surface was NOT measured — a shard reported ARCHIVE_MISSING. Runtime counters above are not results. Fix the build first; the runtime surface will then be exercised.

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

_Lower bound: 147 recorded before the run stopped. Unmeasured: Python populations, slow + torch (job `cancelled`)._

- **FAIL** `python::tests/test_python_api` :: `test_transformation_normal_pgs_conditional_mean_tracks_response`
- **FAIL** `python::tests/test_python_api` :: `test_survival_marginal_slope_weibull_n3000_returns_under_60s`
- **FAIL** `python::tests/test_bug_hunt_curv_smooth_hyperbolic_recovered_as_spherical` :: `test_curv_recovers_constant_curvature_sign[-2.0]`
- **FAIL** `python::tests/bug_hunt_curv_smooth_aborts_nan_covariance_low_signal_test` :: `test_peer_2d_smooths_fit_the_same_low_signal_data`
- **FAIL** `python::tests/bug_hunt_spline_scan_model_summary_unavailable_test` :: `test_scan_model_predicts_and_summarizes[quintic-order3]`
- **FAIL** `python::tests/bug_hunt_left_truncated_survival_predicts_degenerate_covariate_independent_survival_test` :: `test_left_truncated_survival_is_nondegenerate_and_covariate_dependent`
- **FAIL** `python::tests/bug_hunt_spline_scan_model_summary_unavailable_test` :: `test_scan_predict_point_and_interval_still_work`
- **FAIL** `python::tests/bug_hunt_spline_scan_model_summary_unavailable_test` :: `test_scan_predictions_intervals_and_summary_replay_exactly_after_save_load`
- **FAIL** `python::tests/bug_hunt_double_penalty_inflates_edf_instead_of_shrinking_test` :: `test_bspline_double_penalty_does_not_inflate_linear_edf`
- **FAIL** `python::tests/bug_hunt_te_tensor_smooth_edf_depends_on_row_order_test` :: `test_te_tensor_smooth_edf_and_se_invariant_to_row_order`
- **FAIL** `python::tests/bug_hunt_thinplate_single_penalty_overfits_linear_data_test` :: `test_pspline_single_penalty_linear_data_control`
- **FAIL** `python::tests/bug_hunt_2299_affine_design_link_wiggle_test` :: `test_link_wiggle_affine_design_offset_separation_red_gate`
- **FAIL** `python::tests/bug_hunt_2299_affine_design_link_wiggle_test` :: `test_link_wiggle_affine_design_flex_link_joint_newton_blowup_red_gate`
- **FAIL** `python::tests/bug_hunt_matern_periodic_option_rejected_despite_builder_support_test` :: `test_matern_periodic_smooth_is_accepted_and_fits`
- **FAIL** `python::tests/bug_hunt_transformation_normal_generate_draws_on_latent_scale_wrong_direction_test` :: `test_ctm_generate_conditional_mean_increases_with_covariate`
- **FAIL** `python::tests/bug_hunt_flexible_link_engages_and_predicts_test` :: `test_flexible_link_engages_predicts_and_is_monotone_on_probit_data`
- **FAIL** `python::tests/bug_hunt_convex_concave_smooth_collapses_to_flat_at_moderate_snr_test` :: `test_concave_smooth_recovers_clean_concave_signal`
- **FAIL** `python::tests/bug_hunt_flexible_link_predict_deviance_matches_fit_2141_test` :: `test_flexible_link_predict_deviance_matches_fit_2141`
- **FAIL** `python::tests/bug_hunt_univariate_matern_gp_smooth_degenerate_range_penalty_test` :: `test_univariate_matern_smooth_fits_ordinary_1d_data`
- **FAIL** `python::tests/bug_hunt_univariate_matern_gp_smooth_degenerate_range_penalty_test` :: `test_univariate_gp_basis_alias_fits_ordinary_1d_data`
- **FAIL** `python::tests/bug_hunt_posterior_sampling_ignores_box_coefficient_constraints_test` :: `test_posterior_respects_nonnegative_coefficient_bound`
- **FAIL** `python::tests/bug_hunt_posterior_sampling_ignores_box_coefficient_constraints_test` :: `test_posterior_respects_two_sided_linear_coefficient_bounds`
- **FAIL** `python::tests/bug_hunt_sae_fit_avg_active_atoms_structurally_zero_test` :: `test_sae_fit_avg_active_atoms_is_consistent_with_reconstruction`
- **FAIL** `python::tests/test_binomial_posterior_mean_uncertainty` :: `test_rare_event_observation_band_is_informative`
- **FAIL** `python::tests/test_bug_hunt_2026_tweedie_estimated_power_conditional_coverage` :: `test_bare_tweedie_low_mean_coverage_is_near_nominal_on_p_neq_1p5_data`
- **FAIL** `python::tests/bug_hunt_flexible_loglog_cauchit_link_accepted_then_crashes_test` :: `test_flexible_cloglog_positive_control_fits`
- **FAIL** `python::tests/bug_hunt_flexible_loglog_cauchit_link_accepted_then_crashes_test` :: `test_flexible_advertised_link_is_handled_gracefully[flexible(cauchit)]`
- **FAIL** `python::tests/bug_hunt_gamma_dispersion_location_scale_unpredictable_test` :: `test_gamma_dispersion_location_scale_is_predictable`
- **FAIL** `python::tests/bug_hunt_gamma_dispersion_location_scale_unpredictable_test` :: `test_negbin_dispersion_location_scale_is_predictable`
- **FAIL** `python::tests/test_bug_hunt_cyclic_smooth_overfits_null_space_signal` :: `test_cyclic_pspline_collapses_to_constant_on_flat_data`
- **FAIL** `python::tests/test_bug_hunt_dispersion_location_scale_observation_interval_symmetric` :: `test_dispersion_location_scale_observation_band_is_skewed_not_symmetric`
- **FAIL** `python::tests/test_bug_hunt_dispersion_location_scale_observation_interval_symmetric` :: `test_location_scale_band_matches_standard_path_skew_on_identical_data`
- **FAIL** `python::tests/bug_hunt_sae_manifold_fit_tiny_ordered_beta_bernoulli_circle_process_kill_2089_test` :: `test_tiny_ordered_beta_bernoulli_circle_fit_does_not_kill_the_process`
- **FAIL** `python::tests/test_fit_model_spec_kwargs_match_config` :: `test_rust_flexible_link_fit_kwarg_matches_config`
- **FAIL** `python::tests/bug_hunt_gamma_dispersion_location_scale_unpredictable_test` :: `test_tweedie_dispersion_location_scale_is_predictable`
- **FAIL** `python::tests/test_bug_hunt_2026_tweedie_estimated_power_conditional_coverage` :: `test_estimated_power_strictly_beats_pinned_1p5_in_low_mean_stratum`
- **FAIL** `python::tests/test_bug_hunt_marginal_slope_predict_interval` :: `test_marginal_slope_predict_emits_interval_columns`
- **FAIL** `python::tests/test_bug_hunt_marginal_slope_predict_interval` :: `test_marginal_slope_interval_matches_transform_eta_construction`
- **FAIL** `python::tests/test_bug_hunt_marginal_slope_predict_interval` :: `test_marginal_slope_interval_covers_truth_at_nominal_rate`
- **FAIL** `python::tests/test_bug_hunt_marginal_slope_predict_interval` :: `test_marginal_slope_sample_predict_returns_posterior_bands`
- **FAIL** `python::tests/test_bug_hunt_marginal_slope_predict_interval` :: `test_marginal_slope_posterior_predict_draws_matrix_shape`
- **FAIL** `python::tests/test_issue_247_latent_duchon_jet_mismatch` :: `test_gaussian_reml_fit_latent_duchon_d1_default_repro`
- **FAIL** `python::tests/test_issue_627_latent_coordinate_optimization` :: `test_optimizer_recovers_coordinate_from_random_init`
- **FAIL** `python::tests/test_issue_627_latent_coordinate_optimization` :: `test_recovery_is_initialization_independent`
- **FAIL** `python::tests/test_issue_627_latent_coordinate_optimization` :: `test_recovered_latent_is_returned_with_shape`
- **FAIL** `python::tests/test_issue_627_latent_coordinate_optimization` :: `test_caller_init_is_a_pure_local_solve`
- **FAIL** `python::tests/test_manifold_sae_golden_roundtrip` :: `test_to_dict_from_dict_is_a_fixed_point`
- **FAIL** `python::tests/test_manifold_sae_golden_roundtrip` :: `test_all_fields_are_explicit_and_runtime_diagnostics_round_trip`
- **FAIL** `python::tests/test_manifold_sae_golden_roundtrip` :: `test_deprecated_score_alias_is_rejected`
- **FAIL** `python::tests/test_manifold_sae_golden_roundtrip` :: `test_obsolete_projected_model_payloads_are_rejected[top_k_projection]`
- **FAIL** `python::tests/test_manifold_sae_golden_roundtrip` :: `test_obsolete_projected_model_payloads_are_rejected[pre_topk]`
- **FAIL** `python::tests/test_manifold_sae_golden_roundtrip` :: `test_noncanonical_assignment_is_rejected`
- **FAIL** `python::tests/test_manifold_sae_golden_roundtrip` :: `test_covariance_bearing_fixture_round_trips`
- **FAIL** `python::tests/test_manifold_sae_pyclass_getters` :: `test_to_dict_is_a_fixed_point`
- **FAIL** `python::tests/test_manifold_sae_pyclass_getters` :: `test_dense_array_getters`
- **FAIL** `python::tests/test_manifold_sae_pyclass_getters` :: `test_fisher_and_selected_getters`
- **FAIL** `python::tests/test_manifold_sae_pyclass_getters` :: `test_scalar_and_list_getters`
- **FAIL** `python::tests/test_manifold_sae_pyclass_getters` :: `test_atoms_is_an_object_surface`
- **FAIL** `python::tests/test_manifold_sae_pyclass_getters` :: `test_atom_dense_covariance_is_reconstructed`
- **FAIL** `python::tests/test_manifold_sae_pyclass_getters` :: `test_report_block_getters`
- **FAIL** `python::tests/test_manifold_sae_pyclass_getters` :: `test_native_summary_and_description_length_use_the_fitted_artifact`
- **FAIL** `python::tests/test_manifold_sae_pyclass_steer_equiv` :: `test_public_type_is_the_pyo3_model`
- **FAIL** `python::tests/test_manifold_sae_pyclass_steer_equiv` :: `test_steer_reuses_the_resident_metric`
- **FAIL** `python::tests/test_manifold_sae_pyclass_steer_equiv` :: `test_target_dose_probe_is_wired_through_the_public_model`
- **FAIL** `python::tests/test_manifold_sae_pyclass_steer_equiv` :: `test_target_dose_rejects_scalar_and_malformed_probe_results`
- **FAIL** `python::tests/test_manifold_sae_pyclass_steer_equiv` :: `test_target_dose_local_decrease_does_not_claim_unreachable`
- **FAIL** `python::tests/test_manifold_sae_pyclass_steer_equiv` :: `test_target_dose_unreachable_requires_a_global_envelope_certificate`
- **FAIL** `python::tests/test_manifold_sae_pyclass_steer_equiv` :: `test_public_steer_rebuilds_every_analytic_coordinate_action[geometry_plan0-3-t_from0-t_to0]`
- **FAIL** `python::tests/test_manifold_sae_pyclass_steer_equiv` :: `test_public_steer_rebuilds_every_analytic_coordinate_action[geometry_plan1-9-t_from1-t_to1]`
- **FAIL** `python::tests/test_posterior_box_coefficient_constraints` :: `test_nonnegative_active_bound_gaussian`
- **FAIL** `python::tests/test_posterior_box_coefficient_constraints` :: `test_nonnegative_active_bound_binomial`
- **FAIL** `python::tests/test_posterior_box_coefficient_constraints` :: `test_linear_min_max_active_lower_bound`
- **FAIL** `python::tests/test_posterior_box_coefficient_constraints` :: `test_nonpositive_active_bound`
- **FAIL** `python::tests/test_posterior_monotone_shape_constraint` :: `test_convex_posterior_curves_are_convex`
- **FAIL** `python::tests/test_pyffi_bug_hunt4` :: `test_bug_custom_family_coefficient_group_labels_are_stably_routed`
- **FAIL** `python::tests/test_manifold_sae_pyclass_steer_equiv` :: `test_public_steer_rebuilds_every_analytic_coordinate_action[geometry_plan2-6-t_from2-t_to2]`
- **FAIL** `python::tests/test_manifold_sae_pyclass_steer_equiv` :: `test_public_steer_rebuilds_every_analytic_coordinate_action[geometry_plan3-3-t_from3-t_to3]`
- **FAIL** `python::tests/test_manifold_sae_pyclass_steer_equiv` :: `test_public_steer_respects_periodic_seams_in_product_manifolds[geometry_plan0-9-1.0-t_from0-t_wrapped0-t_unwrapped0]`
- **FAIL** `python::tests/test_manifold_sae_pyclass_steer_equiv` :: `test_public_steer_respects_periodic_seams_in_product_manifolds[geometry_plan1-6-1.0-t_from1-t_wrapped1-t_unwrapped1]`
- **FAIL** `python::tests/test_manifold_sae_pyclass_steer_equiv` :: `test_public_mobius_steer_identifies_deck_twins`
- **FAIL** `python::tests/test_manifold_sae_pyclass_steer_equiv` :: `test_public_target_dose_rebuilds_cylinder_metadata`
- **FAIL** `python::tests/test_manifold_sae_pyclass_steer_equiv` :: `test_attach_fisher_is_atomic_and_builds_once`
- **FAIL** `python::tests/test_manifold_sae_pyclass_steer_equiv` :: `test_detach_is_explicit_not_attach_none`
- **FAIL** `python::tests/test_response_geometry_constant_curvature_e2e` :: `test_constant_curvature_response_recovers_spherical_sign`
- **FAIL** `python::tests/test_response_geometry_constant_curvature_e2e` :: `test_constant_curvature_response_does_not_reject_flat_truth`
- **FAIL** `python::tests/test_sae_coordinate_fidelity_public_api_2081` :: `test_coordinate_fidelity_report_and_gated_angle_reader`
- **FAIL** `python::tests/test_sae_coordinate_fidelity_public_api_2081` :: `test_coordinate_fidelity_round_trips_through_dict`
- **FAIL** `python::tests/test_sae_hybrid_split_public_field_1204` :: `test_manifoldsae_has_hybrid_split_field_defaulting_none`
- **FAIL** `python::tests/test_sae_hybrid_split_public_field_1204` :: `test_to_dict_emits_hybrid_split`
- **FAIL** `python::tests/test_sae_hybrid_split_public_field_1204` :: `test_to_dict_emits_none_when_absent`
- **FAIL** `python::tests/test_sae_hybrid_split_public_field_1204` :: `test_hybrid_split_round_trips_through_from_dict`
- **FAIL** `python::tests/test_sae_manifold_certify_external_2266` :: `test_certify_external_round_trips_a_genuinely_converged_native_fit`
- **FAIL** `python::tests/bug_hunt_smooth_significance_ref_df_tracks_edf_not_basis_dim_test` :: `test_nonconverged_flat_fit_is_not_flagged_significant`
- **FAIL** `python::tests/test_cross_frame_dispatch` :: `test_sphere_numpy_frame`
- **FAIL** `python::tests/test_curvature_estimand_surface_wired` :: `test_summary_surfaces_fitted_kappa_with_no_refit`
- **FAIL** `python::tests/test_curvature_estimand_surface_wired` :: `test_curvature_method_reports_ci_and_flatness`
- **FAIL** `python::tests/test_sae_manifold_determinism` :: `test_sae_fit_random_state_changes_output_topk_path`
- **FAIL** `python::tests/test_sae_manifold_certify_external_2266` :: `test_certify_external_returns_typed_nonfit_for_perturbed_state`
- **FAIL** `python::tests/test_sae_manifold_certify_external_2266` :: `test_certify_external_requires_matching_per_atom_metadata_lengths`
- **FAIL** `python::tests/test_sae_manifold_k_multi_heterogeneous_atoms` :: `test_heterogeneous_mixed_topology_atoms_reconstruct`
- **FAIL** `python::tests/test_bug_hunt_spline_scan_predict_observation_interval` :: `test_scan_observation_interval_covers_about_95pct[y ~ s(x, bs="ps", degree=5, penalty_order=3, double_penalty=false)]`
- **FAIL** `python::tests/test_sae_manifold_converged_latents_issue_357` :: `test_converged_latents_exposes_t_star_and_a_star`
- **FAIL** `python::tests/test_sae_manifold_olmo_real_recon_ev` :: `test_olmo_real_heldout_reconstruction_ev_meets_linear_parity`
- **FAIL** `python::tests/test_bug_hunt_tweedie_observation_interval_equal_tailed` :: `test_tweedie_observation_upper_edge_is_above_the_symmetric_band`
- **FAIL** `python::tests/test_sae_manifold_oos_reencode_issue_2132` :: `test_oos_reencode_of_training_rows_matches_native_reconstruction`
- **FAIL** `python::tests/test_sae_manifold_converged_latents_issue_357` :: `test_standalone_per_atom_projection`
- **FAIL** `python::tests/test_sae_manifold_oos_reencode_issue_2132` :: `test_selected_rho_survives_save_load_roundtrip`
- **FAIL** `python::tests/test_sae_manifold_regularizer_noops_issue_240` :: `test_primitive_names_metadata_is_not_a_substitute_for_effect`
- **FAIL** `python::tests/test_sae_manifold_converged_latents_issue_357` :: `test_warm_start_accepted_and_refines`
- **FAIL** `python::tests/test_sae_manifold_public_api_shapes` :: `test_per_atom_uncertainty_shape_band_shapes_are_sane`
- **FAIL** `python::tests/test_sae_manifold_shape_band_save_load` :: `test_shape_band_survives_save_load_to_tight_tolerance`
- **FAIL** `python::tests/test_sae_manifold_converged_latents_issue_357` :: `test_oos_solve_returns_converged_latents_not_post_solve_reseed`
- **FAIL** `python::tests/test_sae_manifold_regularizer_noops_issue_240` :: `test_isometry_weight_is_not_a_silent_noop`
- **FAIL** `python::tests/test_sae_manifold_shape_band_save_load` :: `test_restored_covariance_reproduces_analytic_band`
- **FAIL** `python::tests/test_sae_manifold_determinism` :: `test_sae_fit_is_deterministic_for_fixed_seed`
- **FAIL** `python::tests/test_sae_manifold_regularizer_noops_issue_240` :: `test_ard_per_atom_is_not_a_silent_noop`
- **FAIL** `python::tests/test_sae_manifold_regularizer_noops_issue_240` :: `test_block_orthogonality_weight_is_not_a_silent_noop`
- **FAIL** `python::tests/test_sae_manifold_shape_band_save_load` :: `test_on_disk_format_is_compact_per_channel_not_dense_joint`
- **FAIL** `python::tests/test_sae_manifold_determinism` :: `test_sae_fit_random_state_changes_output`
- **FAIL** `python::tests/test_sae_manifold_regularizer_noops_issue_240` :: `test_decoder_feature_sparsity_groups_is_not_a_silent_noop`
- **FAIL** `python::tests/test_sae_manifold_regularizer_noops_issue_240` :: `test_decoder_feature_sparsity_groups_produces_nontrivial_gradient`
- **FAIL** `python::tests/test_sae_manifold_shape_band_save_load` :: `test_reconstruction_and_band_are_physical_under_heterogeneous_column_scale`
- **FAIL** `python::tests/test_sae_manifold_synthetic_quality_ground_truth` :: `test_fit_learns_disjoint_periodic_atoms_without_inactive_leakage`
- **FAIL** `python::tests/test_sphere_descriptor_issue_224` :: `test_sphere_evaluate_more_centers_than_rows_numpy`
- **FAIL** `python::tests/test_sphere_descriptor_issue_224` :: `test_sphere_basis_size_then_evaluate_consistent`
- **FAIL** `python::tests/test_structure_certificate_1058` :: `test_structure_certificate_matches_independent_e_bh`
- **FAIL** `python::tests/test_sae_manifold_synthetic_quality_ground_truth` :: `test_fit_oos_quality_matches_training_on_planted_oracle_distribution`
- **FAIL** `python::tests/test_sae_manifold_shape_uncertainty` :: `test_shape_uncertainty_fields_present_and_well_shaped`
- **FAIL** `python::tests/test_structure_certificate_1058` :: `test_stricter_alpha_never_grows_confirmed_set`
- **FAIL** `python::tests/test_sae_manifold_synthetic_quality_ground_truth` :: `test_isometry_on_circle_recovers_planted_geometry_normalized_reference`
- **FAIL** `python::tests/test_sae_manifold_shape_uncertainty` :: `test_band_sd_matches_analytic_phi_cov_phi_propagation`
- **FAIL** `python::tests/test_structure_certificate_1058` :: `test_contested_entries_are_the_unconfirmed_complement`
- **FAIL** `python::tests/test_sae_manifold_top_k_issue` :: `test_topk_fit_uses_exact_fixed_support[1]`
- **FAIL** `python::tests/test_sae_manifold_shape_uncertainty` :: `test_posterior_shape_band_is_tighter_than_data_deviation`
- **FAIL** `python::tests/test_structure_certificate_1058` :: `test_certificate_round_trips_through_native_payload`
- **FAIL** `python::tests/test_sae_manifold_top_k_issue` :: `test_topk_fit_uses_exact_fixed_support[2]`
- **FAIL** `python::tests/test_sae_manifold_shape_uncertainty` :: `test_more_data_tightens_the_posterior_band`
- **FAIL** `python::tests/test_survival_api_regressions` :: `test_joint_competing_risks_survival_is_reachable_from_fit`
- **FAIL** `python::tests/test_sae_manifold_top_k_issue` :: `test_topk_payload_is_one_unprojected_model`
- **FAIL** `python::tests/test_sae_manifold_single_circle_default_795` :: `test_single_circle_quickstart_converges_with_default_regularizers`
- **FAIL** `python::tests/test_survival_api_regressions` :: `test_survival_marginal_slope_fit_returns`
- **FAIL** `python::tests/test_survival_marginal_slope_clustered_pc_808` :: `test_survival_marginal_slope_clustered_pc_converges_808`
- **FAIL** `python::tests/test_sae_manifold_single_circle_default_795` :: `test_single_circle_positive_isometry_recovers_honest_chart_span`
- **FAIL** `python::tests/test_survival_nonlinear_baseline_fits_issue_392_369` :: `test_weibull_survival_with_timewiggle_fits`
- **FAIL** `python::tests/test_survival_save_load_roundtrip` :: `test_survival_marginal_slope_save_load_predict_roundtrips`
- **FAIL** `python::tests/test_sae_manifold_speed` :: `test_periodic_atom_fit_recovers_one_harmonic_within_iteration_cap`
- **FAIL** `python::tests/test_validate_formula_operator_family_issue_219` :: `test_identity_wrapper_pass_through`

## Forbidden runtime-error signatures

_None._

## Slow / timeout attribution (#1393)

_No test crossed the 300s slow period._

