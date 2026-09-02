//! The radial-shell chart shared by the in-frame curved lane and the block
//! chart lane: every whitened row is projected onto the shell of the training
//! rows' mean radius, and its held-out fit is scored per row against the linear
//! reconstruction with the same squared error.

use ndarray::Array2;

/// Radial-shell chart: project each whitened row to the train mean radius shell.
pub(crate) fn radial_predict(train: &Array2<f64>, eval: &Array2<f64>) -> Array2<f64> {
    let d = train.ncols();
    let mut radius = 0.0;
    for i in 0..train.nrows() {
        let mut ss = 0.0;
        for j in 0..d {
            ss += train[[i, j]] * train[[i, j]];
        }
        radius += ss.sqrt();
    }
    radius /= train.nrows().max(1) as f64;
    let mut out = Array2::<f64>::zeros(eval.dim());
    for i in 0..eval.nrows() {
        let mut norm = 0.0;
        for j in 0..d {
            norm += eval[[i, j]] * eval[[i, j]];
        }
        norm = norm.sqrt().max(1.0e-12);
        for j in 0..d {
            out[[i, j]] = radius * eval[[i, j]] / norm;
        }
    }
    out
}

/// Squared error between row `row` of `a` and of `b`.
pub(crate) fn row_sse(a: &Array2<f64>, b: &Array2<f64>, row: usize) -> f64 {
    let mut s = 0.0;
    for j in 0..a.ncols() {
        let d = a[[row, j]] - b[[row, j]];
        s += d * d;
    }
    s
}
