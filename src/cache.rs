//! Whole-call partition caches shared with the backend interface.

use crate::backend::Function;
use crate::util::dag::{Map, NeedGrowth};

/// Entries above which a store cache is cleared, retaining its allocations.
const CACHE_BOUND: usize = 1 << 20;

/// Whole-call alignment and quotient results cached on one store.
///
/// Entries borrow tokens without retaining them, so the next alignment or
/// quotient clears the cache whenever the store or its maintenance epoch
/// changes.
#[derive(Debug)]
pub struct StoreCache {
    /// The store whose results are cached.
    store: u64,
    /// The store's maintenance epoch when the cached results were computed.
    epoch: u64,
    /// Entries above which the next alignment or quotient clears the cache,
    /// `1 << 20` by default.
    ///
    /// This bounds entries rather than retained bytes.
    pub bound: usize,
    /// `(left, right)` to `(result, labels start, labels length)` in `labels`.
    aligned: Map<(Function, Function), (Function, usize, usize)>,
    /// Pairs of input class numbers for each cached alignment, back to back.
    labels: Vec<(u32, u32)>,
    /// Results keyed by input root and the hash of its class map.
    ///
    /// Each entry holds the result root and the start and length of the input
    /// map in `classes`. A hit compares the full map.
    quotients: Map<(Function, u64), (Function, usize, usize)>,
    /// The class maps of every cached quotient, back to back.
    classes: Vec<u32>,
}

impl StoreCache {
    /// An empty cache that the next alignment or quotient clears once it holds
    /// more than `bound` entries.
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

impl Default for StoreCache {
    fn default() -> Self {
        Self::with_bound(CACHE_BOUND)
    }
}

/// Binds `cache` to `store` at maintenance epoch `epoch`, clearing every entry
/// when either differs from the previous call or the entry count exceeds the
/// bound.
///
/// Allocations remain available for reuse.
#[inline(always)]
pub fn restart(cache: &mut StoreCache, store: u64, epoch: u64) {
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
    cache: &mut StoreCache,
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
    cache: &mut StoreCache,
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
