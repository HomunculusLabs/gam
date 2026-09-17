//! One Riemannian trust-region run on a Grassmann frame at LLM width holds no
//! `d × d` matrix, let alone the `(dk)²` metric tensor (#2946).
//!
//! `Gr(16, 2560)` stored as a `d·k = 40960`-vector has a dense metric of
//! `40960²` doubles, 13.4 GB, and the trust region's adapter used to build it on
//! every metric product, so the #2946 frame gradient could not reach the
//! optimizer at real width. The adapter now applies the manifold's own
//! matrix-free `metric_product`.
//!
//! The measurement is the process-wide peak of live heap bytes above what was
//! live when a run started, taken by a counting global allocator. That is why
//! this is its own test binary with a single test: nothing else allocates while
//! it measures.
//!
//! The process's first matrix product initializes the linear-algebra runtime,
//! which keeps a process-lifetime allocation that depends on neither the manifold
//! nor its width: on MSI job 1172753 the first `fast_atb` call left one
//! `2³⁰`-byte block live and later calls allocated nothing like it. So the run
//! asserted on is the second one, after a first run has paid that
//! initialization. The dense metric this test rules out was formed on every
//! metric product of every run, so no warm-up can hide it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use gam_geometry::{
    GeometryResult, GrassmannManifold, RiemannianManifold, RiemannianObjective,
    RiemannianTrustRegion,
};
use ndarray::{Array1, ArrayView1};

struct PeakCountingAllocator;

static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

fn note_allocated(bytes: usize) {
    let live = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
}

fn note_released(bytes: usize) {
    LIVE_BYTES.fetch_sub(bytes, Ordering::Relaxed);
}

// SAFETY: `PeakCountingAllocator` delegates every allocation operation to
// `System` with the original pointer/layout contract unchanged. Its only side
// effect is updating atomic byte counters, which allocates nothing and cannot
// affect ownership.
unsafe impl GlobalAlloc for PeakCountingAllocator {
    // SAFETY: callers supply the `GlobalAlloc`-required valid layout;
    // forwarding it unchanged to `System` preserves that contract.
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is valid by this method's `GlobalAlloc` contract.
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            note_allocated(layout.size());
        }
        ptr
    }

    // SAFETY: callers supply the `GlobalAlloc`-required valid layout;
    // forwarding it unchanged to `System` preserves that contract.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is valid by this method's `GlobalAlloc` contract.
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            note_allocated(layout.size());
        }
        ptr
    }

    // SAFETY: `ptr` and `layout` must denote a live `System` allocation by
    // this allocator's contract, and both are forwarded unchanged.
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller guarantees the matching live allocation contract.
        unsafe { System.dealloc(ptr, layout) };
        note_released(layout.size());
    }

    // SAFETY: `ptr` and `layout` must denote a live `System` allocation and
    // `new_size` is forwarded unchanged, exactly as required by `GlobalAlloc`.
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller guarantees the matching live allocation contract.
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            note_released(layout.size());
            note_allocated(new_size);
        }
        new_ptr
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: PeakCountingAllocator = PeakCountingAllocator;

/// Run `work`, returning its output and the peak live heap bytes it added above
/// what was live when it started.
fn peak_added_bytes<T>(work: impl FnOnce() -> T) -> (T, usize) {
    let baseline = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_BYTES.store(baseline, Ordering::Relaxed);
    let output = work();
    let peak = PEAK_BYTES.load(Ordering::Relaxed);
    (output, peak.saturating_sub(baseline))
}

const fn dense_f64_bytes(rows: usize, cols: usize) -> usize {
    rows * cols * std::mem::size_of::<f64>()
}

/// An orthonormal `d × k` frame, flattened row-major: the DCT-II basis vectors
/// of frequencies `1..=k`, `Y_ij = √(2/d)·cos(π(i + ½)(j + 1)/d)`. They are dense,
/// so no diagonal objective is stationary at this frame.
fn dct_frame(d: usize, k: usize) -> Array1<f64> {
    let scale = (2.0 / d as f64).sqrt();
    Array1::from_shape_fn(d * k, |index| {
        let (row, col) = (index / k, index % k);
        scale * (std::f64::consts::PI * (row as f64 + 0.5) * (col as f64 + 1.0) / d as f64).cos()
    })
}

/// `f(Y) = −½·tr(YᵀAY)` with `A = diag(1, 2, …, d)/d`, whose ambient
/// differential `−AY` costs `O(dk)`.
struct DiagonalRayleigh {
    d: usize,
    k: usize,
}

impl RiemannianObjective for DiagonalRayleigh {
    fn value_gradient(&mut self, point: ArrayView1<'_, f64>) -> GeometryResult<(f64, Array1<f64>)> {
        let mut value = 0.0;
        let mut gradient = Array1::<f64>::zeros(point.len());
        for index in 0..point.len() {
            let weight = (index / self.k + 1) as f64 / self.d as f64;
            value -= 0.5 * weight * point[index] * point[index];
            gradient[index] = -weight * point[index];
        }
        Ok((value, gradient))
    }
}

#[test]
fn grassmann_trust_region_at_llm_width_holds_no_d_by_d_matrix() {
    // Positive control, on the allocation this test exists to rule out: the
    // dense metric product the adapter used to take, `metric_tensor(Y)·v`, at a
    // width where it is cheap. The counter must see its `(dk)²` tensor, and the
    // d×d bound asserted below must reject it.
    let (control_d, control_k) = (64usize, 4usize);
    let control_ambient = control_d * control_k;
    let control = GrassmannManifold::new(control_k, control_d).expect("Gr(4, 64)");
    let control_point = dct_frame(control_d, control_k);
    let control_vector = Array1::<f64>::ones(control_ambient);
    let (control_product, control_bytes) = peak_added_bytes(|| {
        control
            .metric_tensor(control_point.view())
            .expect("dense Grassmann metric")
            .dot(&control_vector)
    });
    assert_eq!(control_product.len(), control_ambient);
    assert!(
        control_bytes >= dense_f64_bytes(control_ambient, control_ambient),
        "the counter missed the dense metric: {control_bytes} bytes added, the tensor alone is {} bytes",
        dense_f64_bytes(control_ambient, control_ambient)
    );
    assert!(
        control_bytes >= dense_f64_bytes(control_d, control_d),
        "the d×d bound would accept the dense metric product ({control_bytes} bytes)"
    );

    // `Gr(16, 2560)`: the width the #2946 frame gradient runs at.
    let (d, k) = (2560usize, 16usize);
    let ambient = d * k;
    let manifold = GrassmannManifold::new(k, d).expect("Gr(16, 2560)");
    let start = dct_frame(d, k);
    let mut objective = DiagonalRayleigh { d, k };
    let start_value = objective
        .value_gradient(start.view())
        .expect("start value")
        .0;
    let solver = RiemannianTrustRegion {
        max_iter: 4,
        grad_tol: 0.0,
        ..RiemannianTrustRegion::default()
    };
    let (warm_up, warm_up_bytes) = peak_added_bytes(|| {
        solver.minimize_reporting_termination(&manifold, &mut objective, start.view())
    });
    let warm_up = warm_up.expect("the first trust-region run completes at d·k = 40960");
    assert_eq!(
        warm_up.iterations, solver.max_iter,
        "the first run must spend its budget"
    );
    let (termination, added_bytes) = peak_added_bytes(|| {
        solver.minimize_reporting_termination(&manifold, &mut objective, start.view())
    });
    let termination = termination.expect("the trust region runs at d·k = 40960");
    let end_value = objective
        .value_gradient(termination.point.view())
        .expect("terminal value")
        .0;
    eprintln!(
        "Gr({k}, {d}) trust region: {} iterations, f {start_value:.6e} -> {end_value:.6e}; first run \
         added {warm_up_bytes} bytes of heap (runtime initialization included), measured run \
         {added_bytes} bytes = {:.2} ambient vectors; the dense metric is {} bytes",
        termination.iterations,
        added_bytes as f64 / dense_f64_bytes(ambient, 1) as f64,
        dense_f64_bytes(ambient, ambient)
    );
    assert_eq!(
        termination.iterations, solver.max_iter,
        "the budget must be spent on real iterations"
    );
    assert!(
        end_value < start_value,
        "the run must make progress: f {start_value} -> {end_value}"
    );
    assert!(
        added_bytes < dense_f64_bytes(d, d),
        "one trust-region run on Gr({k}, {d}) added {added_bytes} bytes of heap, at least a d×d \
         matrix ({} bytes); the dense metric would be {} bytes",
        dense_f64_bytes(d, d),
        dense_f64_bytes(ambient, ambient)
    );
}
