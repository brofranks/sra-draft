use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use sra::backend::{Backend, Function, MapSpec, transfer};
use sra::bdd::BddStore;
use sra::scalar::ScalarSort;

struct Trace;
static ACTIVE: AtomicBool = AtomicBool::new(false);
static WIDE: AtomicUsize = AtomicUsize::new(0);
fn record(size: usize) {
    if size >= 8192 && ACTIVE.load(Ordering::Relaxed) {
        WIDE.fetch_add(1, Ordering::Relaxed);
    }
}
unsafe impl GlobalAlloc for Trace {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        record(l.size());
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        record(l.size());
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, size: usize) -> *mut u8 {
        record(size);
        unsafe { System.realloc(p,l,size) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) { unsafe { System.dealloc(p,l) } }
}
#[global_allocator]
static TRACE: Trace = Trace;
fn main() {
    let sort=ScalarSort::bit_vec(65536);
    let c=Arc::new(BddStore::with_capacity(2,0,1)).column(sort).unwrap();
    let map=MapSpec::Rust(Arc::new(|_,_| panic!("map received an empty input")));
    ACTIVE.store(true,Ordering::Relaxed);
    let result=transfer(sort,&map,&mut *black_box(c.members(Function::FALSE)),|_| true,true);
    ACTIVE.store(false,Ordering::Relaxed);
    assert!(result.is_empty());
    let count=WIDE.load(Ordering::Relaxed);
    println!("allocations of at least 8192 bytes: {count}");
    assert_eq!(count, std::env::args().nth(1).unwrap().parse::<usize>().unwrap());
}
