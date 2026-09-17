//! Laws the survival marginal-slope tests anchor on: a Gauss–Hermite law, on
//! which the anchor is the closed form to quadrature tolerance, and a skewed
//! two-component law nothing Gaussian describes.

use super::*;

/// Gauss–Hermite nodes and weights for the standard normal law, `m` points:
/// the finite law on which the anchor IS the Gaussian closed form to
/// quadrature tolerance. Nodes ascend, weights sum to one.
///
/// The physicists' rule (weight `e^{−x²}`) is built by Newton on the
/// orthonormal three-term recurrence from the standard asymptotic seeds, then
/// mapped to the probabilists' scale `u = √2·x`, `w ↦ w/√π`.
pub(crate) fn gauss_hermite_probabilists(m: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    if m < 2 {
        return Err("a Gauss–Hermite law needs at least two nodes".to_string());
    }
    const PI_QUARTER_INV: f64 = 0.751_125_544_464_942_5;
    let mut x = vec![0.0_f64; m];
    let mut w = vec![0.0_f64; m];
    let n = m as f64;
    let mut z = 0.0_f64;
    for i in 0..m.div_ceil(2) {
        z = match i {
            0 => (2.0 * n + 1.0).sqrt() - 1.85575 * (2.0 * n + 1.0).powf(-1.0 / 6.0),
            1 => z - 1.14 * n.powf(0.426) / z,
            2 => 1.86 * z - 0.86 * x[0],
            3 => 1.91 * z - 0.91 * x[1],
            _ => 2.0 * z - x[i - 2],
        };
        let mut pp = 0.0;
        for _ in 0..200 {
            let mut p1 = PI_QUARTER_INV;
            let mut p2 = 0.0;
            for j in 1..=m {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = z * (2.0 / jf).sqrt() * p2 - ((jf - 1.0) / jf).sqrt() * p3;
            }
            pp = (2.0 * n).sqrt() * p2;
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() <= 3e-16 * (1.0 + z.abs()) {
                break;
            }
        }
        x[i] = z;
        x[m - 1 - i] = -z;
        w[i] = 2.0 / (pp * pp);
        w[m - 1 - i] = w[i];
    }
    // Physicists' → probabilists': u = √2·x; ascending order; unit total mass.
    let mut nodes: Vec<f64> = x.iter().map(|value| value * std::f64::consts::SQRT_2).collect();
    let mut weights = w;
    nodes.reverse();
    weights.reverse();
    let total: f64 = weights.iter().sum();
    for weight in weights.iter_mut() {
        *weight /= total;
    }
    for k in 1..m {
        if !(nodes[k] > nodes[k - 1]) {
            return Err(format!(
                "Gauss–Hermite construction failed at m={m}: nodes are not ascending"
            ));
        }
    }
    Ok((nodes, weights))
}

/// A deliberately skewed two-component law on 41 nodes.
pub(crate) fn skewed_grid() -> AnchorGridOwned {
    let nodes: Vec<f64> = (0..41).map(|k| -2.5 + 0.15 * k as f64).collect();
    let raw: Vec<f64> = nodes
        .iter()
        .map(|&u| {
            (-0.5 * ((u + 0.9) / 0.5).powi(2)).exp()
                + 0.35 * (-0.5 * ((u - 1.4) / 0.9).powi(2)).exp()
        })
        .collect();
    let total: f64 = raw.iter().sum();
    AnchorGridOwned::new(nodes, raw.into_iter().map(|w| w / total).collect())
}
