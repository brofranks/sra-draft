//! A scalar backend that uses an independent index during partition operations.

use std::ops::ControlFlow;
use std::sync::Arc;

use sra::backend::{Algebra, Align, Backend, BoolOp, Cursor, Function, MapSpec, Refine, Values};
use sra::index::Index;
use sra::scalar::ScalarSort;
use sra::sparse::SparseStore;
use sra::util::count::Count;

use super::BV2;

#[derive(Clone)]
pub struct Reentrant {
    algebra: Arc<dyn Algebra>,
    index: Index,
}

impl Reentrant {
    pub fn new(index: Index) -> Self {
        let store = Arc::new(SparseStore::with_capacity(8, 16, 1));
        Self {
            algebra: Arc::from(store.bind(BV2).unwrap()),
            index,
        }
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
    }

    fn dec_ref(&self, f: Function) {
        self.algebra.dec_ref(f);
    }

    fn evaluate(&self, f: Function, value: &[u64]) -> u32 {
        self.algebra.evaluate(f, value)
    }

    fn count(&self, p: Function) -> Count {
        self.algebra.count(p)
    }

    fn set(&self, values: &[u64]) -> Function {
        self.algebra.set(values)
    }

    fn lt(&self, bound: &[u64]) -> Function {
        self.algebra.lt(bound)
    }

    fn apply(&self, op: BoolOp, left: Function, right: Function) -> Function {
        self.algebra.apply(op, left, right)
    }

    fn image(&self, map: &MapSpec, p: Function, q: Function) -> Function {
        self.algebra.image(map, p, q)
    }

    fn preimage(&self, map: &MapSpec, p: Function, q: Function) -> Function {
        self.algebra.preimage(map, p, q)
    }

    fn ite(&self, condition: Function, yes: Function, no: Function) -> Function {
        self.algebra.ite(condition, yes, no)
    }

    fn align(
        &self,
        left: Function,
        right: Function,
        labels: &mut Vec<(u32, u32)>,
        scratch: &mut Align,
    ) -> Function {
        let a = self.index.values(0, &[1], &[self.index.full()]);
        let b = self.index.values(0, &[2], &[self.index.full()]);
        let union = a.apply_bool(BoolOp::Or, &b);
        assert_eq!(
            [0, 1, 2, 3].map(|x| union.contains(&[x])),
            [false, true, true, false]
        );
        self.algebra.align(left, right, labels, scratch)
    }

    fn quotient(&self, f: Function, classes: &[u32], scratch: &mut Align) -> Function {
        self.algebra.quotient(f, classes, scratch)
    }

    fn cases(&self, default: u32, cases: &[(Function, u32)], labels: &mut Vec<u32>) -> Function {
        self.algebra.cases(default, cases, labels)
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
        self.algebra
            .values(default, keys, case_labels, labels, scratch)
    }

    fn refine(
        &self,
        f: Function,
        p: Function,
        scratch: &mut Refine,
        select: &mut dyn FnMut(u32) -> bool,
        then: &mut dyn FnMut(u32, &mut dyn FnMut() -> Function) -> ControlFlow<()>,
    ) -> ControlFlow<()> {
        self.algebra.refine(f, p, scratch, select, then)
    }

    fn members(&self, p: Function) -> Box<dyn Cursor + '_> {
        self.algebra.members(p)
    }
}
