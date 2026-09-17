// Child module of `run_plan::run_plan_tests` (see the `#[path]` declaration there): an outer solver's
// `ObjectiveFailed` read against the objective error its bridge published (#1561). Scope comes from the
// parent via `use super::*`.
//
// The defect these pin. The ARC and BFGS arms of `run_outer_with_plan` took the publication slot with
// `.expect(..)` and compared the solver's message with the published one using `assert_eq!`. So a solver that
// reported `ObjectiveFailed` without having evaluated anything (a refused seed or initial metric, a
// solver-internal exit), or a publication that disagreed with it, crashed the fit instead of returning an
// error. A lane probe hit exactly that: a synthetic ARC exit panicked at run_plan.rs:2076 (job 1150334).

use super::*;
use gam_problem::estimation_error::OuterObjectiveErrorSource;

const SOLVER_CONTEXTS: [&str; 2] = ["outer ARC evaluation", "outer BFGS evaluation"];

fn engine_source_rendering(context: &str, error: &EstimationError) -> String {
    assert!(
        matches!(
            error,
            EstimationError::OuterObjectiveEvaluationFailed {
                context: named,
                source: OuterObjectiveErrorSource::Estimation(_),
            } if named == context
        ),
        "{context}: expected a fatal outer evaluation over an engine error, got `{error}`"
    );
    match error {
        EstimationError::OuterObjectiveEvaluationFailed {
            source: OuterObjectiveErrorSource::Estimation(inner),
            ..
        } => inner.to_string(),
        other => other.to_string(),
    }
}

#[test]
fn an_unpublished_objective_failure_is_a_typed_fatal_outer_error_1561() {
    for context in SOLVER_CONTEXTS {
        let publication: Mutex<Option<ObjectiveEvalError>> = Mutex::new(None);
        let error = objective_failure_from_publication(
            context,
            "saturated cubic model at an unchanged iterate".to_string(),
            &publication,
        );
        assert!(
            error.is_fatal_outer_evaluation(),
            "{context}: an unpublished failure routes as a fatal outer evaluation"
        );
        let rendered = engine_source_rendering(context, &error);
        assert!(
            rendered.contains("saturated cubic model at an unchanged iterate")
                && rendered.contains("no published objective evaluation"),
            "{context}: the typed error carries the solver's message: {rendered}"
        );
    }
}

#[test]
fn a_matching_fatal_publication_maps_exactly_as_before_1561() {
    for context in SOLVER_CONTEXTS {
        let published = ObjectiveEvalError::fatal("inner solve refused at this rho");
        let publication = Mutex::new(Some(published.clone()));
        let mapped = objective_failure_from_publication(
            context,
            published.message().to_string(),
            &publication,
        );
        let direct = EstimationError::fatal_objective_evaluation(context, published.clone());
        assert_eq!(
            mapped.to_string(),
            direct.to_string(),
            "{context}: a published failure renders as it always has"
        );
        assert!(
            matches!(
                &mapped,
                EstimationError::OuterObjectiveEvaluationFailed {
                    context: named,
                    source: OuterObjectiveErrorSource::Objective(source),
                } if named == context && source.is_fatal() && source.message() == published.message()
            ),
            "{context}: a published failure keeps the producer's typed objective error as its source"
        );
        assert!(
            publication
                .lock()
                .expect("an unpoisoned publication slot locks")
                .is_none(),
            "{context}: the publication is consumed"
        );
    }
}

#[test]
fn a_disagreeing_or_recoverable_publication_is_a_typed_fatal_outer_error_1561() {
    for context in SOLVER_CONTEXTS {
        let disagreeing = Mutex::new(Some(ObjectiveEvalError::fatal("the bridge's message")));
        let error = objective_failure_from_publication(
            context,
            "the solver's message".to_string(),
            &disagreeing,
        );
        assert!(error.is_fatal_outer_evaluation());
        let rendered = engine_source_rendering(context, &error);
        assert!(
            rendered.contains("the solver's message")
                && rendered.contains("the bridge's message")
                && rendered.contains("a fatal"),
            "{context}: a disagreement names both messages: {rendered}"
        );

        let recoverable = Mutex::new(Some(ObjectiveEvalError::recoverable("probe infeasible")));
        let error = objective_failure_from_publication(
            context,
            "probe infeasible".to_string(),
            &recoverable,
        );
        assert!(error.is_fatal_outer_evaluation());
        let rendered = engine_source_rendering(context, &error);
        assert!(
            rendered.contains("probe infeasible") && rendered.contains("a recoverable"),
            "{context}: a recoverable publication is named, not handed to the fatal constructor: {rendered}"
        );
    }
}

#[test]
fn a_poisoned_publication_slot_is_the_typed_unpublished_error_1561() {
    for context in SOLVER_CONTEXTS {
        let publication = Arc::new(Mutex::new(Some(ObjectiveEvalError::fatal(
            "held while the slot was poisoned",
        ))));
        let holder = Arc::clone(&publication);
        let joined = std::thread::spawn(move || {
            let held = holder.lock().expect("an unpoisoned publication slot locks");
            assert!(held.is_some(), "the slot holds a publication before it is poisoned");
            panic!("poison the publication slot");
        })
        .join();
        assert!(joined.is_err(), "the holder thread panicked while holding the lock");
        assert!(publication.is_poisoned(), "positive control: the slot is poisoned");
        let error = objective_failure_from_publication(
            context,
            "solver message".to_string(),
            &publication,
        );
        assert!(error.is_fatal_outer_evaluation());
        let rendered = engine_source_rendering(context, &error);
        assert!(
            rendered.contains("solver message")
                && rendered.contains("no published objective evaluation"),
            "{context}: a poisoned slot reads as nothing published: {rendered}"
        );
    }
}
