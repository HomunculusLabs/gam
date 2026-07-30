//! Shared `f16_to_f32` for the gam-sae examples.
//!
//! Lifted verbatim from the 5 byte-identical copies that used to
//! live one per example binary. Not an example target itself: cargo only
//! auto-discovers `examples/*.rs` and `examples/*/main.rs`, so a helper in
//! `examples/support/` is compiled only where it is `#[path]`-included.

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 0x1;
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    let bits = if exp == 0 {
        if mant == 0 {
            (sign as u32) << 31
        } else {
            let mut m = mant as u32;
            let mut e: i32 = -1;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let exp32 = (127 - 15 + 1 + e) as u32;
            ((sign as u32) << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        ((sign as u32) << 31) | (0xff << 23) | ((mant as u32) << 13)
    } else {
        let exp32 = (exp as i32 - 15 + 127) as u32;
        ((sign as u32) << 31) | (exp32 << 23) | ((mant as u32) << 13)
    };
    f32::from_bits(bits)
}
