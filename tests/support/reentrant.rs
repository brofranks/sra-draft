//! A scalar backend that uses an independent index during partition operations.
//!
//! The fixture records the nonconstant references it hands to its caller.
//! Internal references belong to the wrapped algebra.

use std::cell::Cell;
use std::ops::ControlFlow;
use std::sync::{Arc, Mutex};

use sra::backend::{Algebra, Align, Backend, BoolOp, Cursor, Function, MapSpec, Refine, Values};
use sra::index::Index;
use sra::scalar::ScalarSort;
use sra::util::count::Count;

use super::BV2;

thread_local! {
    /// One operation to run when the calling thread builds its next scalar cell.
    pub static ON_CELL: Cell<Option<Box<dyn FnOnce()>>> = const { Cell::new(None) };
}

/// Scalar operations observed by the caller of this backend.
#[derive(Clone, Copy, Default)]
pub struct Calls {
    pub align: usize,
    pub refine: usize,
    pub dec_refs: usize,
}

#[derive(Clone)]
pub struct Reentrant {
    algebra: Arc<dyn Algebra>,
    index: Index,
    refs: Arc<Mutex<Vec<Function>>>,
    calls: Arc<Mutex<Calls>>,
}

impl Reentrant {
    pub fn new(index: Index, algebra: Box<dyn Algebra>) -> Self {
        Self {
            algebra: Arc::from(algebra),
            index,
            refs: Arc::default(),
            calls: Arc::default(),
        }
    }

    /// The outstanding external references in token order.
    pub fn refs(&self) -> Vec<Function> {
        let mut refs = self.refs.lock().unwrap().clone();
        refs.sort_unstable();
        refs
    }

    pub fn calls(&self) -> Calls {
        *self.calls.lock().unwrap()
    }

    fn own(&self, f: Function) -> Function {
        if !f.is_constant() {
            self.refs.lock().unwrap().push(f);
        }
        f
    }
}

impl Backend for Reentrant {
    fn bind(self: Arc<Self>, sort: ScalarSort) -> Option<Box<dyn Algebra>> {
        (sort == BV2).then(|| Box::new((*self).clone()) as Box<dyn Algebra>)
    }
}

impl Algebra for Reentrant {
    fn sort(&self) -> ScalarSort {
        self.algebra.sort()
    }

    fn inc_ref(&self, f: Function) {
        self.algebra.inc_ref(f);
        self.own(f);
    }

    fn dec_ref(&self, f: Function) {
        self.algebra.dec_ref(f);
        if !f.is_constant() {
            let mut refs = self.refs.lock().unwrap();
            let i = refs.iter().position(|&root| root == f).unwrap();
            refs.swap_remove(i);
        }
    }

    fn dec_refs(&self, roots: &[Function]) {
        // An index can release partitions during exclusive maintenance.
        let nested = self.index.values(0, &[3], &[self.index.full()]);
        assert!(nested.contains(&[3]));
        for &f in roots {
            self.dec_ref(f);
        }
        self.calls.lock().unwrap().dec_refs += 1;
    }

    fn evaluate(&self, f: Function, value: &[u64]) -> u32 {
        self.algebra.evaluate(f, value)
    }

    fn count(&self, p: Function) -> Count {
        self.algebra.count(p)
    }

    fn set(&self, values: &[u64]) -> Function {
        self.own(self.algebra.set(values))
    }

    fn lt(&self, bound: &[u64]) -> Function {
        self.own(self.algebra.lt(bound))
    }

    fn apply(&self, op: BoolOp, left: Function, right: Function) -> Function {
        self.own(self.algebra.apply(op, left, right))
    }

    fn image(&self, map: &MapSpec, p: Function, q: Function) -> Function {
        self.own(self.algebra.image(map, p, q))
    }

    fn preimage(&self, map: &MapSpec, p: Function, q: Function) -> Function {
        self.own(self.algebra.preimage(map, p, q))
    }

    fn ite(&self, condition: Function, yes: Function, no: Function) -> Function {
        self.own(self.algebra.ite(condition, yes, no))
    }

    fn align(
        &self,
        left: Function,
        right: Function,
        labels: &mut Vec<(u32, u32)>,
        scratch: &mut Align,
    ) -> Function {
        self.calls.lock().unwrap().align += 1;
        let a = self.index.values(0, &[1], &[self.index.full()]);
        let b = self.index.values(0, &[2], &[self.index.full()]);
        let union = a.apply(BoolOp::Or, &b);
        assert_eq!(
            [0, 1, 2, 3].map(|x| union.contains(&[x])),
            [false, true, true, false]
        );
        self.own(self.algebra.align(left, right, labels, scratch))
    }

    fn quotient(&self, f: Function, classes: &[u32], scratch: &mut Align) -> Function {
        self.own(self.algebra.quotient(f, classes, scratch))
    }

    fn cases(&self, default: u32, cases: &[(Function, u32)], labels: &mut Vec<u32>) -> Function {
        self.own(self.algebra.cases(default, cases, labels))
    }

    fn values(
        &self,
        default: u32,
        keys: &[u64],
        case_labels: &[u32],
        labels: &mut Vec<u32>,
        scratch: &mut Values,
    ) -> Function {
        let root = self.index.values(0, &[1], &[self.index.full()]);
        assert!(root.contains(&[1]));
        self.own(
            self.algebra
                .values(default, keys, case_labels, labels, scratch),
        )
    }

    fn refine(
        &self,
        f: Function,
        p: Function,
        scratch: &mut Refine,
        select: &mut dyn FnMut(u32) -> bool,
        then: &mut dyn FnMut(u32, &mut dyn FnMut() -> Function) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        self.calls.lock().unwrap().refine += 1;
        self.algebra.refine(f, p, scratch, select, &mut |i, build| {
            then(i, &mut || {
                if let Some(run) = ON_CELL.take() {
                    run();
                }
                self.own(build())
            })
        })
    }

    fn members(&self, p: Function) -> Box<dyn Cursor + '_> {
        self.algebra.members(p)
    }
}
