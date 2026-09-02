//! gam#2695 — the measurement behind the composed-warp degree floor.
//!
//! For every degree from 3 to 6 and every knot-crossing reading, print the gap
//! at h=1e-3 and h=1e-5 and their ratio (≈100 is a continuous quantity, ≈1 is
//! a step). Printed always, so the floor is set from this table rather than
//! from a derivation about which basis derivative a lowering consumes.

use super::*;

#[test]
fn link_warp_knot_crossing_gap_ladder_by_degree_2695() {
    for degree in 3..=6usize {
        for read in [
            LinkWarpKnotReading::ObservedInformation,
            LinkWarpKnotReading::ObjectiveJeffreysTerm,
            LinkWarpKnotReading::ObjectiveJeffreysGradient,
        ] {
            for amplitude in [1.0e-6_f64, 3.0e-2] {
                let (coarse, fine) =
                    link_warp_knot_crossing_gap_2695(degree, true, amplitude, read);
                println!(
                    "2695-ladder degree={degree} read={read:?} amp={amplitude:.1e} \
                     coarse={coarse:.6e} fine={fine:.6e} ratio={:.3e}",
                    coarse / fine.max(f64::MIN_POSITIVE)
                );
                assert!(coarse.is_finite() && fine.is_finite());
            }
        }
    }
}
