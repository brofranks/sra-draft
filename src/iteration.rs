//! Semi-naive fixed points over stored relations.
//!
//! This is the recursive layer and it is deliberately thin: three relations
//! of one index and one barrier. How a round derives its candidates is an
//! ordinary [`Query`](crate::Query), so recursion adds no execution
//! machinery of its own. It only decides what each round is allowed to read.

use crate::backend::BoolOp;
use crate::rel::Rel;

/// One recursively defined relation, midway through its fixed point.
///
/// ```text
/// stable  facts accepted before the current frontier
/// recent  the accepted current frontier
/// pending candidates submitted during the current round
/// ```
///
/// Candidates stay invisible until [`Self::changed`]. For positive linear
/// recursion, a round reads `recent` with fixed operands and submits
/// collected candidates through [`Self::insert`]. Every accepted fact enters
/// `recent` once, including when one symbolic row holds many facts. Callers
/// must evaluate all their rules before advancing again; the barrier does
/// not run rules.
///
/// Maintain the protocol through `insert` and `changed`; the public fields
/// expose low-level state. Nonlinear rule variants, mutual or chaotic
/// iteration, rule differentiation, scheduling, and deletion maintenance
/// remain later layers. A finite fact universe bounds monotone growth, but
/// neither the number of rounds nor symbolic representation size is
/// necessarily small.
pub struct Variable {
    /// Everything accepted before the current frontier.
    pub stable: Rel,
    /// The current frontier: what the next round is entitled to derive from.
    pub recent: Rel,
    /// Candidates submitted during the current round.
    pending: Rel,
}

impl Variable {
    /// Starts with candidates that become visible at the first call to
    /// [`Self::changed`].
    ///
    /// The seed relation supplies the context, endpoint sorts, universe, and
    /// physical index every later round must share.
    #[inline(always)]
    pub fn new(initial: Rel) -> Self {
        Self {
            stable: initial.empty_like(),
            recent: initial.empty_like(),
            pending: initial,
        }
    }

    /// Everything accepted so far, the union of [`Self::stable`] and
    /// [`Self::recent`], as an owning snapshot.
    #[inline(always)]
    pub fn current(&self) -> Rel {
        self.stable.apply_bool(BoolOp::Or, &self.recent)
    }

    /// Adds `rel` to this round's candidates using native stored union.
    ///
    /// # Panics
    ///
    /// Panics if the candidate, even an empty one, belongs to another context
    /// or differs in logical endpoint sorts, declared universe, or physical
    /// index.
    #[inline(always)]
    pub fn insert(&mut self, rel: &Rel) {
        self.pending = self.pending.apply_bool(BoolOp::Or, rel);
    }

    /// Closes the round: promotes the frontier into [`Self::stable`], accepts the
    /// candidates that are new as the next frontier, and reports whether any
    /// were.
    ///
    /// Novelty compares complete facts with native set difference rather than
    /// keyed query `minus`, preserving symbolic correlations.
    #[inline(always)]
    pub fn changed(&mut self) -> bool {
        let known = self.current();
        self.recent = self.pending.apply_bool(BoolOp::AndNot, &known);
        self.pending = self.pending.empty_like();
        self.stable = known;
        !self.recent.is_empty()
    }

    /// Consumes drained iteration state.
    ///
    /// First call [`Self::changed`] until it reports no new facts. This checks
    /// for outstanding work, and the caller remains responsible for evaluating
    /// its rules.
    ///
    /// # Panics
    ///
    /// Panics if any recent or pending facts remain.
    #[inline(always)]
    pub fn complete(self) -> Rel {
        assert!(
            self.recent.is_empty() && self.pending.is_empty(),
            "cannot complete a variable with recent or pending facts",
        );
        self.stable
    }
}

#[cfg(test)]
#[path = "../tests/unit/iteration.rs"]
mod tests;
