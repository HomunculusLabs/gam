//! Shared `.npy` header parsers for the gam-sae examples.
//!
//! Lifted verbatim from the 4 byte-identical copies that used to
//! live one per example binary. Not an example target itself: cargo only
//! auto-discovers `examples/*.rs` and `examples/*/main.rs`, so a helper in
//! `examples/support/` is compiled only where it is `#[path]`-included.
//!
//! [`parse_npy_float_header`] reads a little-endian, C-order `<f2`, `<f4` or `<f8`
//! header of one or two axes. [`parse_npy_header`] keeps the original contract: a
//! 2-D `<f4` or `<f2` array.

use std::path::Path;

/// The element type of a little-endian float `.npy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NpyFloat {
    F2,
    F4,
    F8,
}

impl NpyFloat {
    /// Bytes per element.
    pub fn bytes(self) -> usize {
        match self {
            Self::F2 => 2,
            Self::F4 => 4,
            Self::F8 => 8,
        }
    }
}

/// A little-endian, C-order float `.npy` header: the element type, the shape and the
/// byte offset of the data.
pub struct NpyFloatHeader {
    pub float: NpyFloat,
    pub shape: Vec<usize>,
    pub data_off: usize,
}

pub fn parse_npy_float_header(head: &[u8], path: &Path) -> Result<NpyFloatHeader, String> {
    if head.len() <= 12 || &head[0..6] != b"\x93NUMPY" {
        return Err(format!("{}: not a .npy file", path.display()));
    }
    let major = head[6];
    let (header_len, data_off) = if major >= 2 {
        let header_len = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
        (header_len, 12 + header_len)
    } else {
        let header_len = u16::from_le_bytes([head[8], head[9]]) as usize;
        (header_len, 10 + header_len)
    };
    if data_off > head.len() {
        return Err(format!(
            "{}: header exceeds initial read buffer",
            path.display()
        ));
    }
    let header = std::str::from_utf8(&head[data_off - header_len..data_off])
        .map_err(|err| format!("{}: header is not utf8: {err}", path.display()))?;
    let float = if header.contains("'<f4'") || header.contains("\"<f4\"") {
        NpyFloat::F4
    } else if header.contains("'<f2'") || header.contains("\"<f2\"") {
        NpyFloat::F2
    } else if header.contains("'<f8'") || header.contains("\"<f8\"") {
        NpyFloat::F8
    } else {
        return Err(format!(
            "{}: expected little-endian <f8, <f4 or <f2; header: {header}",
            path.display()
        ));
    };
    if !(header.contains("'fortran_order': False") || header.contains("\"fortran_order\": false")) {
        return Err(format!(
            "{}: expected C-order; header: {header}",
            path.display()
        ));
    }
    let shape_start = header
        .find("'shape':")
        .ok_or_else(|| format!("{}: missing shape key", path.display()))?
        + "'shape':".len();
    let paren_open = header[shape_start..]
        .find('(')
        .ok_or_else(|| format!("{}: missing shape open paren", path.display()))?
        + shape_start
        + 1;
    let paren_close = header[paren_open..]
        .find(')')
        .ok_or_else(|| format!("{}: missing shape close paren", path.display()))?
        + paren_open;
    let dims: Vec<usize> = header[paren_open..paren_close]
        .split(',')
        .filter_map(|token| token.trim().parse::<usize>().ok())
        .collect();
    if dims.is_empty() || dims.len() > 2 {
        return Err(format!(
            "{}: expected a 1-D or 2-D array, got {dims:?}",
            path.display()
        ));
    }
    Ok(NpyFloatHeader {
        float,
        shape: dims,
        data_off,
    })
}

pub fn parse_npy_header(
    head: &[u8],
    path: &Path,
) -> Result<(usize, usize, usize, bool, usize), String> {
    let header = parse_npy_float_header(head, path)?;
    if header.float == NpyFloat::F8 {
        return Err(format!(
            "{}: expected little-endian <f4 or <f2, got <f8",
            path.display()
        ));
    }
    let [rows, cols] = header.shape[..] else {
        return Err(format!(
            "{}: expected a 2-D array, got {:?}",
            path.display(),
            header.shape
        ));
    };
    Ok((
        rows,
        cols,
        header.float.bytes(),
        header.float == NpyFloat::F4,
        header.data_off,
    ))
}
