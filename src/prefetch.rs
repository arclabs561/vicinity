//! Architecture-specific software prefetch hints.

/// Prefetch a memory address into L1 cache for reading.
///
/// This is a performance hint only. Unsupported platforms compile to a no-op,
/// and callers must not rely on it for correctness.
#[inline(always)]
#[allow(unsafe_code, unused_variables)]
pub(crate) fn prefetch_read_data<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: `_mm_prefetch` is a hint and does not dereference `ptr`.
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `prfm` is a hint and does not affect program semantics.
        // Inline asm is used because the stable intrinsic is not available.
        unsafe {
            std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
        }
    }
}
