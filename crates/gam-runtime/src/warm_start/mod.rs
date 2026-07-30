//! On-disk warm-start store.
//!
//! Persists periodic checkpoints of in-progress fits so a subsequent run
//! (possibly after a SIGKILL or weeks later) auto-resumes from the
//! best-known iterate. Keyed on a SHA-256 fingerprint of the data + fit
//! spec, so re-fitting the same model on the same data reuses the matching
//! persisted warm-start entry.
//!
//! The store does not choose its own root — [`WarmStartStore::open`] takes one.
//! The persistent checkpoint root gam-solve passes is
//! `std::env::temp_dir()/gam/warm/v1` (see `gam_solve::persistent_warm_start`),
//! NOT a user cache directory: `dirs::cache_dir()` reads `XDG_CACHE_HOME`/`HOME`
//! through `env::var`, which is banned in that crate. `temp_dir()` is therefore
//! MACHINE-LOCAL and shared by every process on the host, which is the property
//! to keep in mind when reasoning about why two runs of the same fit differ
//! (#2486) — searching a user cache directory for these entries finds nothing.
//!
//! Layout under that root:
//!
//! ```text
//! <keyhex>/
//!   <runid>.json    metadata (objective, iter, checksum, kind)
//!   <runid>.bin     opaque payload bytes
//! ```
//!
//! All writes are tmp-file + fsync + rename, so a hard crash leaves either
//! the pre-write state or a fully-written entry on disk — never half-written.
//! Per-entry SHA-256 checksums catch any residual corruption.
//!
//! Multiple entries can coexist for one key (concurrent fits, prior aborted
//! runs). `lookup` prefers a completed fit's [`EntryKind::Final`] write over
//! any [`EntryKind::Checkpoint`], takes the LATEST of several terminal writes
//! (which completed fit a resume carries is a provenance question, not a
//! quality one), and orders checkpoints by lowest objective.
//!
//! Disk is bounded by [`StoreOptions::size_budget_bytes`] (default ~1 GiB);
//! oldest entries are evicted to fit. Entries older than
//! [`StoreOptions::ttl`] (default 30 days) are dropped on every save.

pub mod key;
pub mod session;
pub mod store;

pub use key::{Fingerprint, Fingerprinter};
pub use session::{LoadSource, LoadedEntry, Session};
pub use store::{EntryKind, StoreError, StoreOptions, WarmStartEntry, WarmStartStore};
