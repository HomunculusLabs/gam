//! Integration-test harness for gam-row-macros: every module here was a
//! standalone tests/*.rs crate and therefore its own link of gam-row-macros and
//! its dependency tree. One binary, same tests, same names.

mod cause_specific_codegen_perf;
mod gaussian_codegen_perf;
mod rigid_bms_codegen_perf;
mod sls_codegen_perf;
