use gam::terms::sae::atom_codes::BitVec;

#[test]
fn bug_atom_codes_collision_free_in_small_supported_space() {
    let mut a = BitVec::zeros(130);
    let mut b = BitVec::zeros(130);
    a.set(1, true);
    a.set(64, true);
    b.set(1, true);
    b.set(65, true);
    assert_ne!(
        a, b,
        "Two distinct active masks in the supported index range should never collide."
    );
}
