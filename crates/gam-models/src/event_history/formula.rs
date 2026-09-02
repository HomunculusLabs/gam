//! The covariate/time formula of an event-history model: the right-hand side
//! of the ordinary gam formula language, resolved against the node data
//! matrix (the covariate table's columns followed by `time`).

use super::cohort::{CohortNodes, EventHistoryError};
use gam_data::{ColumnKindTag, DataSchema, EncodedDataset, SchemaColumn};
use gam_terms::inference::formula_dsl::parse_formula;
use gam_terms::smooth::TermCollectionSpec;
use gam_terms::term_builder::build_termspec;

/// Name of the node-time column visible to the formula.
pub const TIME_COLUMN: &str = "time";

/// The node data matrix as an encoded dataset every column of which is
/// continuous: the covariate columns under their names, then `time`.
pub fn node_dataset(nodes: &CohortNodes, covariate_names: &[String]) -> Result<EncodedDataset, EventHistoryError> {
    if covariate_names.len() + 1 != nodes.node_data.ncols() {
        return Err(EventHistoryError::InvalidInput {
            reason: format!(
                "{} covariate names for a node data matrix with {} covariate columns",
                covariate_names.len(),
                nodes.node_data.ncols() - 1
            ),
        });
    }
    let mut headers: Vec<String> = covariate_names.to_vec();
    headers.push(TIME_COLUMN.to_string());
    if headers.iter().any(|h| h == TIME_COLUMN && covariate_names.contains(h)) {
        return Err(EventHistoryError::InvalidInput {
            reason: format!("a covariate may not be named {TIME_COLUMN:?}"),
        });
    }
    let columns = headers
        .iter()
        .map(|name| SchemaColumn {
            name: name.clone(),
            kind: ColumnKindTag::Continuous,
            levels: Vec::new(),
        })
        .collect();
    Ok(EncodedDataset {
        column_kinds: vec![ColumnKindTag::Continuous; headers.len()],
        headers,
        values: nodes.node_data.clone(),
        schema: DataSchema { columns },
    })
}

/// Resolve a formula right-hand side such as `x + s(time)` into the term
/// collection that every mark's log-intensity uses.
pub fn covariate_spec_from_formula(
    right_hand_side: &str,
    nodes: &CohortNodes,
    covariate_names: &[String],
) -> Result<TermCollectionSpec, EventHistoryError> {
    let rhs = right_hand_side.trim().trim_start_matches('~').trim();
    let formula = format!("events ~ {}", if rhs.is_empty() { "1" } else { rhs });
    let parsed = parse_formula(&formula).map_err(|error| EventHistoryError::InvalidInput {
        reason: format!("event-history formula {right_hand_side:?}: {error}"),
    })?;
    let dataset = node_dataset(nodes, covariate_names)?;
    let col_map = dataset.column_map();
    let mut notes = Vec::new();
    build_termspec(
        &parsed.terms,
        &dataset,
        &col_map,
        &mut notes,
        &gam_runtime::resource::ResourcePolicy::default_library(),
    )
    .map_err(|error| EventHistoryError::InvalidInput {
        reason: format!("event-history formula {right_hand_side:?}: {}", String::from(error)),
    })
}
