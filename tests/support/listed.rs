//! A member cursor backed by literal encoded values.

use sra::backend::Members;

/// Lends the listed values in their original order.
pub struct Listed<'a, T>(pub &'a [T]);

impl<T: AsRef<[u64]>> Members for Listed<'_, T> {
    fn next(&mut self) -> Option<&[u64]> {
        Some(self.0.split_off_first()?.as_ref())
    }
}
