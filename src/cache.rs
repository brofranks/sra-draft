//! Whole-call partition caches shared with the backend interface.

use crate::backend::Function;
use crate::util::dag::{Map, NeedGrowth};

/// Entries above which a cache is cleared, retaining its allocations.
const BOUND: usize = 1 << 20;

/// Whole-call alignment and quotient results cached on one store.
///
/// Entries borrow tokens, so the cache clears when its store or maintenance
/// epoch changes.
#[derive(Debug)]
pub struct Cache {
    /// The store whose results are cached.
    store: u64,
    /// The store's maintenance epoch when the results were computed.
    epoch: u64,
    /// The entry-count threshold, `1 << 20` by default.
    ///
    /// An alignment or quotient attempt clears the cache at its start if the
    /// existing entry count exceeds this bound. Payload bytes are unbounded.
    pub bound: usize,
    /// `(left, right)` to `(result, labels start, labels length)` in `labels`.
    aligned: Map<(Function, Function), (Function, usize, usize)>,
    /// Pairs of input class numbers for each cached alignment, back to back.
    labels: Vec<(u32, u32)>,
    /// `(root, class map hash)` to `(result, start, length)` in `classes`.
    quotients: Map<(Function, u64), (Function, usize, usize)>,
    /// Class maps appended since the last clear, including those displaced by
    /// hash collisions.
    classes: Vec<u32>,
}

impl Cache {
    /// An empty cache with the given [entry-count threshold](Self::bound).
    #[inline(always)]
    pub fn with_bound(bound: usize) -> Self {
        Self {
            store: 0,
            epoch: 0,
            bound,
            aligned: Map::default(),
            labels: Vec::new(),
            quotients: Map::default(),
            classes: Vec::new(),
        }
    }

    /// Clears all cached results, retaining the allocations for reuse.
    pub fn clear(&mut self) {
        self.aligned.clear();
        self.labels.clear();
        self.quotients.clear();
        self.classes.clear();
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self::with_bound(BOUND)
    }
}

/// Binds `cache` to `store` at `epoch`, clearing it when either changes or
/// it exceeds its bound.
#[inline(always)]
pub fn restart(cache: &mut Cache, store: u64, epoch: u64) {
    if cache.store != store
        || cache.epoch != epoch
        || cache.aligned.len() + cache.quotients.len() > cache.bound
    {
        cache.clear();
        cache.store = store;
        cache.epoch = epoch;
    }
}

/// Appends the cached or newly built alignment's labels and returns its root.
///
/// The caller holds store admission and has restarted the cache for this
/// attempt. A failed build rolls back its suffix before maintenance and retry.
pub fn align(
    cache: &mut Cache,
    key: (Function, Function),
    labels: &mut Vec<(u32, u32)>,
    build: impl FnOnce(&mut Vec<(u32, u32)>, usize) -> Result<Function, NeedGrowth>,
) -> Result<Function, NeedGrowth> {
    if let Some(&(result, at, len)) = cache.aligned.get(&key) {
        labels.extend_from_slice(&cache.labels[at..at + len]);
        return Ok(result);
    }
    let start = labels.len();
    let result = build(labels, start).inspect_err(|_| labels.truncate(start))?;
    let at = cache.labels.len();
    cache.labels.extend_from_slice(&labels[start..]);
    cache
        .aligned
        .insert(key, (result, at, labels.len() - start));
    Ok(result)
}

/// The quotient for this root and class map, built on a miss.
///
/// The caller holds admission in the cache's current store and epoch. A hash
/// match still compares the full map. A collision replaces the old entry.
pub fn quotient(
    cache: &mut Cache,
    key: (Function, u64),
    classes: &[u32],
    build: impl FnOnce() -> Result<Function, NeedGrowth>,
) -> Result<Function, NeedGrowth> {
    if let Some(&(result, at, len)) = cache.quotients.get(&key)
        && cache.classes[at..at + len] == *classes
    {
        return Ok(result);
    }
    let result = build()?;
    let at = cache.classes.len();
    cache.classes.extend_from_slice(classes);
    cache.quotients.insert(key, (result, at, classes.len()));
    Ok(result)
}

#[cfg(test)]
#[path = "../tests/unit/cache.rs"]
mod tests;
