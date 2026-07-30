# `tests/src_modules/` — fixtures sourced into crate test trees, not a test target

This directory is **not** a Cargo test target. It has no `main.rs`, so
`autotests` discovery never builds it, and nothing `mod`s it into a binary.

Every file here is instead pulled directly into some crate's own `#[cfg(test)]`
tree by `include!` or `#[path]`, because the assertions need crate-internal
items that an integration test (which sees only the public API) cannot reach:

| fixture | pulled in by |
| --- | --- |
| `misc/cli_tests.rs` | `crates/gam-cli/src/main.rs` via `#[path]` |
| `misc/families_bms_identifiability_rigid_tests.rs` | `crates/gam-models/src/bms/mod.rs` via `include!` |
| `optimization/families_bms_joint_hessian_hvp_correction_tests.rs` | `crates/gam-models/src/bms/mod.rs` via `include!` |
| `smooths/basis_duchon_matern_jet_derivative_tests.rs` | `crates/gam-terms/src/basis/tests.rs` via `include!` |
| `smooths/basis_radial_periodic_thinplate_tests.rs` | `crates/gam-terms/src/basis/tests.rs` via `include!` |

So a fixture's coverage is only as live as its consumer: deleting the
`include!`/`#[path]` line silently retires the guards in the file it names,
with nothing here to signal that. Grep for the filename before assuming a
fixture still runs.

## Why there are no `mod.rs` files

Each subdirectory used to carry a `mod.rs` listing its fixtures. Those files
were the residue of a real defect rather than working wiring: because
`tests/src_modules/` is `mod`'d into no binary, the `mod` lines resolved to
nothing and the guards they listed ran nowhere. Under #1601 the three
`smooth_*` fixtures were re-homed into the `gam-models` drivers test tree
(`design_assembly_constraint_tests.rs`, `adaptive_bounded_duchon_tests.rs`,
`matern_nfree_rekey_topology_tests.rs`), where their cross-crate dependencies
resolve, and the dead copies were deleted; the survivors were rewired to the
`include!`/`#[path]` sites tabulated above.

The `mod.rs` stubs outlived that repair and were removed, since a `mod` list
that binds nothing reads like wiring and is the reason the original breakage
went unnoticed. Do not reintroduce one — adding a fixture here means adding
its `include!`/`#[path]` at the consumer and a row in the table above.
