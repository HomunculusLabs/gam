//! Cheap reproducer for the sparse-dictionary DECODER REFRESH wall (#2283, #2441, #1017).
//!
//! # What this measures and why it exists
//!
//! The `#2283` authoritative hybrid row is blocked behind one cost: the per-epoch
//! decoder refresh of `fit_sparse_dictionary`. At the production shape
//! (`N=96000, P=2048, K=32672, s=28`) that refresh is hundreds of seconds per epoch
//! while the device routing stage is flat at ~15-19 s, and — the thing `#2441` is
//! named for — it **inflates over the epochs of a single fit** rather than being a
//! fixed per-epoch price. Every prior measurement of that curve cost a multi-hour
//! (at one point 24-hour) GPU allocation on a 1.4 GB real activation harvest, so a
//! refresh hypothesis was never disposable.
//!
//! This bench makes it disposable. It runs the SAME production entry point
//! (`fit_sparse_dictionary` → `run_linear_reml_schedule` → the recycled block-CG
//! decoder refresh) on planted-dictionary synthetic activations at shapes small
//! enough that a whole sweep finishes in minutes, and reports the per-epoch trace
//! the production heartbeat already emits:
//!
//! ```text
//! refresh_s route_s cg_columns cg_iterations recycled_rank max_component
//! max_component_nnz operator_build_s cg_kappa_bound births
//! ```
//!
//! Two things are deliberately NOT abstracted away, because both are what a
//! synthetic decoder-refresh microbenchmark gets wrong:
//!
//!   * **The fit is real, not a static normal-equation fixture.** `#2441`'s own
//!     record is that a fixture "too short for any atom to be born" measures the
//!     epoch-1 cost and reports a speedup the second half of a real fit never
//!     sees. The refresh operator here is assembled from codes that the router
//!     actually produced, epoch after epoch, so the co-firing graph and the
//!     operator's spectrum evolve exactly as they do at scale.
//!   * **The whole epoch curve is reported, never an aggregate.** A single mean
//!     over epochs hides the growth this exists to detect. The `growth` column of
//!     the summary is `last_refresh_s / first_refresh_s`, which is the quantity
//!     `#2441` measured as `5.5x` at production scale.
//!
//! # Reading it
//!
//! Every configuration prints one `[refresh-epoch]` CSV row per epoch and one
//! `[refresh-config]` summary row. `growth > 1` reproduces the `#2441` shape; a
//! `growth` near 1 with `cg_iterations` flat says the fit's Krylov work is not
//! inflating at that shape and the sweep must be pushed further before a fix is
//! attributed to anything.
//!
//! # Running
//!
//! ```sh
//! cargo bench --bench sae_decoder_refresh_scaling
//! # or an explicit ladder: n,p,k,s,epochs;...
//! cargo bench --bench sae_decoder_refresh_scaling -- \
//!   --configs 4096,64,2048,8,12 --seed 20283
//! ```
//!
//! The default ladder sweeps `K`, `N` and `P` one axis at a time from a shared
//! centre point, so the scaling law in each axis is readable rather than a single
//! point that any cost model can be fitted to.

use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use gam::terms::sae::sparse_dict::{SparseDictConfig, fit_sparse_dictionary};
use ndarray::Array2;

/// One measured epoch of a fit, parsed from the production `[SAE epoch …]`
/// heartbeat. Parsing the heartbeat rather than re-deriving the numbers keeps the
/// bench honest: it reports exactly the fields a production run reports, so a
/// number here is comparable to a number from the real creditscope/Qwen job.
#[derive(Clone, Debug, Default)]
struct EpochRow {
    epoch: usize,
    refresh_s: f64,
    route_s: f64,
    births: usize,
    cg_columns: usize,
    cg_iterations: usize,
    recycled_rank: usize,
    tile_columns: usize,
    accumulate_s: f64,
    sigma_s: f64,
    graph_build_s: f64,
    precond_s: f64,
    cg_solve_s: f64,
    block_sweeps: usize,
    max_component: usize,
    max_component_nnz: usize,
    operator_build_s: f64,
    kappa_bound: f64,
    ev: f64,
}

/// Captured heartbeat lines. The fit logs on `log::warn!`, so a bench-local
/// logger is the only way to observe the per-epoch trace without changing the
/// production code path.
fn capture() -> &'static Mutex<Vec<String>> {
    static CAPTURE: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
    CAPTURE.get_or_init(|| Mutex::new(Vec::new()))
}

struct HeartbeatLogger;

/// `log` is built here without its `std` feature, so the logger must be a
/// `&'static` value rather than a boxed one.
static HEARTBEAT_LOGGER: HeartbeatLogger = HeartbeatLogger;

impl log::Log for HeartbeatLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Warn
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let line = format!("{}", record.args());
        if line.starts_with("[SAE epoch") {
            capture()
                .lock()
                .expect("heartbeat capture mutex")
                .push(line);
        }
    }

    fn flush(&self) {}
}

/// Pull `key=<value>` out of a heartbeat line. The heartbeat is a flat
/// space-separated `key=value` record, so this is total: a missing key is a
/// contract change in the production trace and must fail loudly rather than
/// silently reporting zero.
fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    line.split_whitespace()
        .find_map(|token| token.strip_prefix(key).and_then(|r| r.strip_prefix('=')))
}

fn number(line: &str, key: &str) -> f64 {
    let raw = field(line, key)
        .unwrap_or_else(|| panic!("heartbeat lost the `{key}` field; line was: {line}"));
    // `cg_kappa_bound` is an `Option<f64>` rendered by `Debug`: `Some(1.5e13)` /
    // `None`. Everything else is a bare number.
    let raw = raw.trim_start_matches("Some(").trim_end_matches(')');
    if raw == "None" {
        return f64::NAN;
    }
    raw.parse::<f64>()
        .unwrap_or_else(|error| panic!("heartbeat field `{key}` was not a number ({error}): {raw}"))
}

fn count(line: &str, key: &str) -> usize {
    // Fields like `live=1234/2048` carry a denominator; take the numerator.
    let raw = field(line, key)
        .unwrap_or_else(|| panic!("heartbeat lost the `{key}` field; line was: {line}"));
    let head = raw.split('/').next().unwrap_or(raw);
    head.parse::<usize>()
        .unwrap_or_else(|error| {
            panic!("heartbeat field `{key}` was not an integer ({error}): {raw}")
        })
}

fn parse_epochs(lines: &[String]) -> Vec<EpochRow> {
    lines
        .iter()
        .enumerate()
        .map(|(index, line)| EpochRow {
            // The fit may run several inner epoch sweeps across the outer REML
            // schedule, so the heartbeat's own `epoch i/N` restarts. The bench
            // index is the monotone refresh counter, which is what a growth
            // curve needs.
            epoch: index + 1,
            refresh_s: number(line, "refresh_s"),
            route_s: number(line, "route_s"),
            births: count(line, "births"),
            cg_columns: count(line, "cg_columns"),
            cg_iterations: count(line, "cg_iterations"),
            recycled_rank: count(line, "recycled_rank"),
            tile_columns: count(line, "tile_columns"),
            accumulate_s: number(line, "accumulate_s"),
            sigma_s: number(line, "sigma_s"),
            graph_build_s: number(line, "graph_build_s"),
            precond_s: number(line, "precond_s"),
            cg_solve_s: number(line, "cg_solve_s"),
            block_sweeps: count(line, "block_sweeps"),
            max_component: count(line, "max_component"),
            max_component_nnz: count(line, "max_component_nnz"),
            operator_build_s: number(line, "operator_build_s"),
            kappa_bound: number(line, "cg_kappa_bound"),
            ev: number(line, "ev"),
        })
        .collect()
}

/// Deterministic xorshift64*, so the bench needs no RNG dependency and the same
/// ladder is bit-reproducible on every host.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform on `[-0.5, 0.5)`.
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// Planted-dictionary activations: `n` rows in `p` dimensions, each a sparse
/// combination of `s` of `k_true` unit atoms plus small noise.
///
/// The planted width is the FIT's width, so the dictionary is exactly critically
/// complete and the router's co-firing graph percolates the way an overcomplete
/// LLM-activation fit's does — which is the regime whose giant component drives
/// the refresh. Uniform atom selection makes the expected co-firing degree
/// `≈ s(s-1)n/k`, so the sweep's degree is a known function of the shape rather
/// than an artifact of the generator.
fn planted_activations(n: usize, p: usize, k_true: usize, s: usize, seed: u64) -> Array2<f32> {
    let mut rng = Rng::new(seed);
    let mut atoms = Array2::<f64>::zeros((k_true, p));
    for a in 0..k_true {
        let mut norm = 0.0f64;
        for c in 0..p {
            let v = rng.uniform();
            atoms[[a, c]] = v;
            norm += v * v;
        }
        let norm = norm.sqrt().max(1.0e-12);
        for c in 0..p {
            atoms[[a, c]] /= norm;
        }
    }

    let mut x = Array2::<f32>::zeros((n, p));
    for i in 0..n {
        for _ in 0..s {
            let a = rng.below(k_true);
            let coefficient = 0.5 + rng.uniform().abs();
            for c in 0..p {
                x[[i, c]] += (coefficient * atoms[[a, c]]) as f32;
            }
        }
        for c in 0..p {
            x[[i, c]] += (0.01 * rng.uniform()) as f32;
        }
    }
    x
}

/// Achievable memory bandwidth on THIS host, measured with the SAME rayon pool
/// the refresh runs on.
///
/// Every seconds-number this bench prints is uninterpretable without it. A
/// refresh sitting at 90% of the host's streaming ceiling and one sitting at 9%
/// look identical from the wall and call for completely opposite work: the
/// first can only be improved by moving fewer bytes (or moving them on a device
/// with more bandwidth), while the second has headroom that a better kernel,
/// a better preconditioner or more threads can still take.
///
/// Two numbers, because the block-CG matvec does two different things:
///
/// * **triad** — `a = b + c` over an array far past any cache. The sequential
///   ceiling, and an upper bound on anything the refresh could achieve.
/// * **gather** — the CSR access pattern itself: for each of `m` rows, read
///   `degree` neighbour rows of `t` contiguous `f64`s and fold them. This is
///   what `solve_block_cg`'s operator closure actually does, and it is the
///   honest denominator, because the matvec never streams — it re-reads
///   neighbour rows in graph order with no reuse guarantee.
///
/// The gather is reported at two block sizes on purpose. The resident block is
/// `m·t·8` bytes, so a small-`K` fixture can sit entirely in cache while the
/// production shape does not — which would make a small-`K` witness reproduce
/// the production ITERATION regime while missing its MEMORY regime entirely.
/// Reporting both sizes is what stops that being assumed either way.
fn host_bandwidth() -> Vec<(String, f64)> {
    use rayon::prelude::*;

    let mut out = Vec::new();

    // --- triad --------------------------------------------------------------
    let n = 128usize << 20 / 8; // ~1 GiB per array, 3 arrays
    let b = vec![1.0f64; n];
    let c = vec![2.0f64; n];
    let mut a = vec![0.0f64; n];
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let started = Instant::now();
        a.par_chunks_mut(1 << 16)
            .zip(b.par_chunks(1 << 16))
            .zip(c.par_chunks(1 << 16))
            .for_each(|((achunk, bchunk), cchunk)| {
                for i in 0..achunk.len() {
                    achunk[i] = bchunk[i] + cchunk[i];
                }
            });
        best = best.min(started.elapsed().as_secs_f64());
    }
    // Two reads and one write per element.
    out.push(("triad".to_string(), 3.0 * n as f64 * 8.0 / best));
    drop((a, b, c));

    // --- gather, at the bench scale and at the production scale --------------
    for (m, t, degree) in [(1024usize, 409usize, 380usize), (32768, 409, 380)] {
        let block = vec![1.0f64; m * t];
        let mut state = 0x2283u64;
        let indices: Vec<u32> = (0..m * degree)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % m as u64) as u32
            })
            .collect();
        let mut sink = vec![0.0f64; m * t];
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let started = Instant::now();
            sink.par_chunks_mut(t).enumerate().for_each(|(i, row)| {
                for slot in row.iter_mut() {
                    *slot = 0.0;
                }
                for e in 0..degree {
                    let j = indices[i * degree + e] as usize;
                    let neighbour = &block[j * t..(j + 1) * t];
                    for (slot, value) in row.iter_mut().zip(neighbour.iter()) {
                        *slot += value;
                    }
                }
            });
            best = best.min(started.elapsed().as_secs_f64());
        }
        out.push((
            format!("gather_m{m}_t{t}_deg{degree}_block{}MiB", (m * t * 8) >> 20),
            m as f64 * degree as f64 * t as f64 * 8.0 / best,
        ));
    }
    out
}

/// Bytes the CPU block-CG moves per operator application, derived from the
/// source rather than assumed (`solve_block_cg`'s closure plus
/// `CpuPcgBlockBackend`'s five `m × tile` state blocks).
///
/// Per iteration: the CSR matvec gathers `8·t` bytes for each of `nnz` stored
/// couplings and walks `12·nnz` bytes of structure; three block inner products,
/// the `x`/`r` update, the preconditioner apply and the `p` update together
/// touch the dense state about eighteen times (`18·8·m·t`); the low-rank
/// correction adds `16·m·rank`. The gather term dominates once the mean
/// co-firing degree exceeds ~36, which every shape here is far past.
fn bytes_per_sweep(m: usize, nnz: usize, tile: usize, rank: usize) -> f64 {
    8.0 * tile as f64 * (nnz as f64 + 18.0 * m as f64)
        + 12.0 * nnz as f64
        + 16.0 * m as f64 * rank as f64
}

#[derive(Clone, Copy, Debug)]
struct Shape {
    n: usize,
    p: usize,
    k: usize,
    s: usize,
    epochs: usize,
}

/// One axis at a time from a shared centre, so each column of the summary is a
/// scaling law rather than one point.
fn default_ladder() -> Vec<Shape> {
    const CENTRE: Shape = Shape {
        n: 4096,
        p: 64,
        k: 1024,
        s: 8,
        epochs: 12,
    };
    let mut ladder = Vec::new();
    for k in [256usize, 512, 1024, 2048] {
        ladder.push(Shape { k, ..CENTRE });
    }
    for n in [1024usize, 2048, 8192] {
        ladder.push(Shape { n, ..CENTRE });
    }
    for p in [32usize, 128, 256] {
        ladder.push(Shape { p, ..CENTRE });
    }
    ladder
}

fn parse_ladder(spec: &str) -> Vec<Shape> {
    spec.split(';')
        .filter(|entry| !entry.trim().is_empty())
        .map(|entry| {
            let parts: Vec<usize> = entry
                .split(',')
                .map(|f| {
                    f.trim().parse::<usize>().unwrap_or_else(|error| {
                        panic!("bad config field `{f}` in `{entry}`: {error}")
                    })
                })
                .collect();
            assert_eq!(
                parts.len(),
                5,
                "each config is `n,p,k,s,epochs`; got `{entry}`"
            );
            Shape {
                n: parts[0],
                p: parts[1],
                k: parts[2],
                s: parts[3],
                epochs: parts[4],
            }
        })
        .collect()
}

/// `--configs n,p,k,s,epochs;…` and `--seed <u64>`, both optional.
///
/// The ladder is a command-line argument rather than an environment variable on
/// purpose: this repo bans environment reads outright, and a bench whose shape
/// is on the command line is a shape that appears verbatim in the run log next
/// to the numbers it produced.
fn command_line() -> (Vec<Shape>, u64) {
    let mut ladder: Option<Vec<Shape>> = None;
    let mut seed = 20_283u64;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--configs" => {
                let spec = args
                    .next()
                    .expect("--configs takes `n,p,k,s,epochs;…` as its value");
                ladder = Some(parse_ladder(&spec));
            }
            "--seed" => {
                let raw = args.next().expect("--seed takes a u64 value");
                seed = raw
                    .parse::<u64>()
                    .unwrap_or_else(|error| panic!("--seed `{raw}` is not a u64: {error}"));
            }
            // `cargo bench` forwards harness flags (`--bench`, `--nocapture`)
            // to every bench binary; a harness=false bench must tolerate them
            // rather than refuse to run under its own documented invocation.
            "--bench" | "--nocapture" | "--test" => {}
            other => panic!("unknown argument `{other}`; expected --configs / --seed"),
        }
    }
    (ladder.unwrap_or_else(default_ladder), seed)
}

fn main() {
    log::set_logger(&HEARTBEAT_LOGGER).expect("bench installs the only logger");
    log::set_max_level(log::LevelFilter::Warn);

    let (ladder, seed) = command_line();

    for (name, bytes_per_second) in host_bandwidth() {
        println!("[refresh-host] {name} GBs {:.1}", bytes_per_second / 1.0e9);
    }
    println!(
        "[refresh-epoch] n,p,k,s,epoch,refresh_s,route_s,refresh_over_route,births,\
         cg_columns,cg_iterations,recycled_rank,tile_columns,max_component,max_component_nnz,\
         operator_build_s,accumulate_s,sigma_s,graph_build_s,precond_s,cg_solve_s,\
         block_sweeps,kappa_bound,ev"
    );
    println!(
        "[refresh-config] n,p,k,s,epochs_run,fit_s,refresh_total_s,route_total_s,\
         refresh_frac,first_refresh_s,last_refresh_s,growth,\
         first_cg_iterations,last_cg_iterations,cg_growth,\
         first_tile,last_tile,first_nnz,last_nnz,first_kappa,last_kappa,\
         acc_total_s,graph_total_s,opbuild_total_s,precond_total_s,cgsolve_total_s,\
         total_block_sweeps,last_iters_per_column,\
         cg_bytes_moved_TB,cg_achieved_GBs"
    );

    for shape in ladder {
        let Shape { n, p, k, s, epochs } = shape;
        let x = planted_activations(n, p, k, s, seed);
        let config = SparseDictConfig {
            n_atoms: k,
            active: s,
            minibatch: 1024.min(n),
            max_epochs: epochs,
            // A plateau stop would truncate the growth curve this bench exists
            // to show, so the tolerance is set below the fit's own rounding
            // floor: every configured epoch runs.
            tolerance: 0.0,
            ..SparseDictConfig::new(k)
        };

        capture().lock().expect("heartbeat capture mutex").clear();
        let started = Instant::now();
        let fit = fit_sparse_dictionary(x.view(), &config);
        let fit_s = started.elapsed().as_secs_f64();
        let lines = capture()
            .lock()
            .expect("heartbeat capture mutex")
            .clone();
        // A fit that errors still leaves a usable refresh curve behind, and the
        // curve is the measurement. Report the error rather than aborting the
        // sweep: a shape that refuses is itself a datum for #2396/#2441.
        let outcome = match &fit {
            Ok(f) => format!("ok ev={:.6}", f.explained_variance),
            Err(e) => format!("err {e}"),
        };
        let rows = parse_epochs(&lines);
        eprintln!(
            "[refresh-run] n={n} p={p} k={k} s={s} epochs_observed={} fit_s={fit_s:.2} {outcome}",
            rows.len()
        );
        if rows.is_empty() {
            continue;
        }

        for row in &rows {
            println!(
                "[refresh-epoch] {n},{p},{k},{s},{},{:.4},{:.4},{:.3},{},{},{},{},{},{},{},{:.4},\
                 {:.4},{:.4},{:.4},{:.4},{:.4},{},{:.6e},{:.6}",
                row.epoch,
                row.refresh_s,
                row.route_s,
                if row.route_s > 0.0 {
                    row.refresh_s / row.route_s
                } else {
                    f64::NAN
                },
                row.births,
                row.cg_columns,
                row.cg_iterations,
                row.recycled_rank,
                row.tile_columns,
                row.max_component,
                row.max_component_nnz,
                row.operator_build_s,
                row.accumulate_s,
                row.sigma_s,
                row.graph_build_s,
                row.precond_s,
                row.cg_solve_s,
                row.block_sweeps,
                row.kappa_bound,
                row.ev,
            );
        }

        let first = rows.first().expect("non-empty");
        let last = rows.last().expect("non-empty");
        // Traffic is per-epoch: `nnz`, the tile and the rank all move over a
        // fit, so the model has to be evaluated on each epoch's own numbers and
        // summed, never on the last epoch's numbers times the epoch count.
        let cg_bytes: f64 = rows
            .iter()
            .map(|r| {
                bytes_per_sweep(
                    r.max_component,
                    r.max_component_nnz,
                    r.tile_columns,
                    r.recycled_rank,
                ) * r.block_sweeps as f64
            })
            .sum();
        let cg_seconds: f64 = rows.iter().map(|r| r.cg_solve_s).sum();
        let refresh_total: f64 = rows.iter().map(|r| r.refresh_s).sum();
        let route_total: f64 = rows.iter().map(|r| r.route_s).sum();
        println!(
            "[refresh-config] {n},{p},{k},{s},{},{fit_s:.2},{refresh_total:.3},{route_total:.3},\
             {:.4},{:.4},{:.4},{:.3},{},{},{:.3},{},{},{},{},{:.4e},{:.4e},\
             {:.3},{:.3},{:.3},{:.3},{:.3},{},{:.2},{:.4},{:.1}",
            rows.len(),
            refresh_total / (refresh_total + route_total).max(f64::MIN_POSITIVE),
            first.refresh_s,
            last.refresh_s,
            last.refresh_s / first.refresh_s.max(f64::MIN_POSITIVE),
            first.cg_iterations,
            last.cg_iterations,
            last.cg_iterations as f64 / (first.cg_iterations.max(1)) as f64,
            first.tile_columns,
            last.tile_columns,
            first.max_component_nnz,
            last.max_component_nnz,
            first.kappa_bound,
            last.kappa_bound,
            rows.iter().map(|r| r.accumulate_s).sum::<f64>(),
            rows.iter().map(|r| r.graph_build_s).sum::<f64>(),
            rows.iter().map(|r| r.operator_build_s).sum::<f64>(),
            rows.iter().map(|r| r.precond_s).sum::<f64>(),
            rows.iter().map(|r| r.cg_solve_s).sum::<f64>(),
            rows.iter().map(|r| r.block_sweeps).sum::<usize>(),
            // The block recurrence advances all columns of a tile together, so
            // per-column iterations is the honest convergence measure; the raw
            // `cg_iterations` sum scales with the column count.
            last.cg_iterations as f64 / (last.cg_columns.max(1)) as f64,
            // Bytes the block recurrence moved across the whole fit, and the
            // rate it moved them at. Compare `cg_achieved_GBs` to the
            // `[refresh-host] gather` line above: close to it means the solve is
            // memory-bound and only fewer bytes (or a device with more
            // bandwidth) can help; far below it means there is headroom a
            // better kernel or preconditioner could still take.
            cg_bytes / 1.0e12,
            if cg_seconds > 0.0 { cg_bytes / cg_seconds } else { f64::NAN } / 1.0e9,
        );
    }
}
