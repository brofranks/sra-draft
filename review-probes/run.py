import shutil
import subprocess
import sys
from pathlib import Path

probes = Path("review-probes")
cache_test = "a_later_alignment_cache_hit_returns_its_own_labels"
wide_test = "wide_keys_take_one_label_per_record"
filter_test = "a_transfer_image_outside_the_sort_is_rejected"

def replace(path, old, new):
    path = Path(path)
    source = path.read_text()
    assert source.count(old) == 1, (path, old, source.count(old))
    path.write_text(source.replace(old, new))

def run(args, diagnostics=()):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, flush=True)
    if diagnostics:
        assert result.returncode and all(marker in result.stdout for marker in diagnostics)
    else:
        result.check_returncode()
    return result.stdout

filter_before = '''        assert!(
            domain.accepts(&y),
            "map image does not encode the scalar sort"
        );
        if holds(&y) {'''
filter_after = '''        if holds(&y) {
            assert!(
                domain.accepts(&y),
                "map image does not encode the scalar sort"
            );'''

mutations = {
    "cache": ("src/cache.rs", "&cache.labels[at..at + len]", "&cache.labels[..len]",
              "bdd", cache_test, "left: [(9, 9), (0, 0), (1, 0), (0, 1)]"),
    "word-count": ("src/bdd.rs", "            keys.len() / self.sort().words(),", "            keys.len(),",
                   "bdd", wide_test, "there must be one case label per key"),
    "label-offset": ("src/bdd.rs", "                offset + start,", "                offset + start * words,",
                     "bdd", wide_test, "index out of bounds"),
    "filter": ("src/backend.rs", filter_before, filter_after,
               "backend", filter_test, "filter received an invalid image"),
}

probe = sys.argv[1]
if probe in mutations:
    path, old, new, target, name, diagnostic = mutations[probe]
    for faulty in [False, True]:
        if faulty:
            replace(path, old, new)
        for profile in [[], ["--release"]]:
            cmd = ["cargo", "test", "--locked", *profile, "--test", target, name, "--", "--exact"]
            if faulty:
                run(cmd, (name, diagnostic, "test result: FAILED. 0 passed; 1 failed"))
            else:
                assert "test result: ok. 1 passed;" in run(cmd)
    print(f"VERIFIED {probe}: permanent regression passed on candidate and caught fault in debug and release")

elif probe == "cursor-set":
    counters = "const LEVEL_BITS: u32 = 6;"
    for name in ["REVIEW_EDGES", "REVIEW_OWNED"]:
        counters += f"\nstatic {name}: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);"
    replace("src/bdd.rs", "const LEVEL_BITS: u32 = 6;", counters)
    for signature, name in [
        ("fn covering_edge(edges: &[Edge], bit: u64) -> &Edge {", "REVIEW_EDGES"),
        ("fn owned(&self, mut build: impl FnMut(&State) -> Result<Function, NeedGrowth>) -> Function {", "REVIEW_OWNED"),
    ]:
        replace("src/bdd.rs", signature,
                signature + "\n        " + name + ".fetch_add(1, std::sync::atomic::Ordering::Relaxed);")
    with Path("tests/unit/bdd.rs").open("a") as suite:
        suite.write((probes / "counters.rs.in").read_text())
    output = run(["cargo", "test", "--locked", "--release", "--lib", "manual_operation_counts", "--", "--nocapture"])
    assert "OBS edge lookups=0 empty-set admissions=0" in output

elif probe == "allocations":
    Path("examples").mkdir(exist_ok=True)
    shutil.copy(probes / "alloc.rs.in", "examples/review_alloc.rs")
    lines = run(["cargo", "run", "--locked", "--release", "--example", "review_alloc"]).splitlines()
    for result in ["OBS singleton sort allocations=0", "OBS count limbs=1 allocations=0", "OBS count limbs=64 allocations=0"]:
        assert result in lines

elif probe == "oracle":
    shutil.copy(probes / "oracle.rs.in", "tests/review_oracle.rs")
    output = run(["cargo", "test", "--locked", "--release", "--test", "review_oracle", "--", "--nocapture"])
    for width in [4, 7, 9]:
        assert f"OBS finite oracle width={width} domain={1 << width}" in output

else:
    raise ValueError(probe)
