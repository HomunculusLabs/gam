use super::*;

#[test]
fn a_text_column_of_numbers_is_continuous_and_passes_through() {
    let (codes, levels) = CovariateCells::from_text(&["1", "2.5", "-3"]).encode();
    assert_eq!(codes, vec![1.0, 2.5, -3.0]);
    assert!(levels.is_empty());
}

#[test]
fn a_text_column_with_one_label_is_categorical_coded_by_sorted_levels() {
    let cells = CovariateCells::from_text(&["b", "a", "b", "c"]);
    assert_eq!(
        cells,
        CovariateCells::Labels(vec!["b".into(), "a".into(), "b".into(), "c".into()])
    );
    let (codes, levels) = cells.encode();
    assert_eq!(levels, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
    assert_eq!(codes, vec![1.0, 0.0, 1.0, 2.0]);
}

#[test]
fn numeric_labels_beside_a_missing_marker_stay_labels() {
    let (codes, levels) = CovariateCells::from_text(&["1", "NA", "2"]).encode();
    assert_eq!(levels, vec!["1".to_string(), "2".to_string(), "NA".to_string()]);
    assert_eq!(codes, vec![0.0, 2.0, 1.0]);
}

#[test]
fn labels_supplied_as_labels_are_categorical_even_when_they_spell_numbers() {
    let (codes, levels) = CovariateCells::Labels(vec!["2".into(), "1".into()]).encode();
    assert_eq!(levels, vec!["1".to_string(), "2".to_string()]);
    assert_eq!(codes, vec![1.0, 0.0]);
}

#[test]
fn record_values_are_coded_against_the_fitted_levels() {
    let levels = vec!["a".to_string(), "b".to_string()];
    assert_eq!(
        code_covariate_value("g", &levels, CovariateValue::Label("b".into())),
        Ok(1.0)
    );
    assert_eq!(code_covariate_value("x", &[], CovariateValue::Number(0.25)), Ok(0.25));
    let unknown = code_covariate_value("g", &levels, CovariateValue::Label("z".into()))
        .expect_err("a label outside the levels is refused");
    assert!(unknown.to_string().contains("unknown level \"z\""), "{unknown}");
    assert!(code_covariate_value("g", &levels, CovariateValue::Number(1.0)).is_err());
    assert!(code_covariate_value("x", &[], CovariateValue::Label("a".into())).is_err());
}

#[test]
fn the_observed_mark_vocabulary_is_sorted_distinct_and_recurrent() {
    let (names, kinds) = observed_mark_vocabulary(["relapse", "death", "relapse"]);
    assert_eq!(names, vec!["death".to_string(), "relapse".to_string()]);
    assert_eq!(kinds, vec![MarkKind::Recurrent, MarkKind::Recurrent]);
}
