use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

struct Meter;
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static REALLOCS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: Each operation forwards the original arguments to System.
unsafe impl GlobalAlloc for Meter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        REALLOCS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, size) }
    }
}

#[global_allocator]
static ALLOCATOR: Meter = Meter;

fn main() {
    let mut packed: Vec<_> = (0..513u64).rev().flat_map(|i| [i, i + 1]).collect();
    ALLOCS.store(0, Ordering::Relaxed);
    REALLOCS.store(0, Ordering::Relaxed);
    sra::scalar::sort_records(&mut packed, 2);
    let allocs = ALLOCS.load(Ordering::Relaxed);
    let reallocs = REALLOCS.load(Ordering::Relaxed);
    assert_eq!(packed.len(), 1026);
    assert!(packed.chunks_exact(2).enumerate().all(|(i, rec)| rec == [i as u64, i as u64 + 1]));
    println!("records=513 stride=2 allocs={allocs} reallocs={reallocs} capacity={}", packed.capacity());
}
