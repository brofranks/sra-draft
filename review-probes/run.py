import shutil
import subprocess
import sys
from pathlib import Path

root = Path("review-probes")

def replace(path, old, new):
    path = Path(path)
    source = path.read_text()
    assert source.count(old) == 1, (path, old, source.count(old))
    path.write_text(source.replace(old, new))

def run(args, expect_fail=None):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in result.stdout.splitlines():
        if any(s in line for s in ("OBS ", "test result:", "error", "FAILED", "wrong cached suffix", "filter received", "did not panic")):
            print(line, flush=True)
    if expect_fail:
        assert result.returncode and expect_fail in result.stdout and "test result: FAILED" in result.stdout, result.stdout[-5000:]
    else:
        assert result.returncode == 0, result.stdout[-5000:]
    return result.stdout

def suite():
    for mode in [[], ["--release"]]:
        run(["cargo", "test", "--locked", "--all-targets", *mode])

mode = sys.argv[1]
if mode in ("cache", "filter"):
    if mode == "cache":
        path = "src/cache.rs"
        old = "&cache.labels[at..at + len]"
        new = "&cache.labels[..len]"
        def install_test():
            shutil.copy(root / "cache.rs.in", "tests/manual_cache.rs")
        cmd = ["cargo", "test", "--locked", "--test", "manual_cache"]
        diagnostic = "wrong cached suffix"
    else:
        path = "src/backend.rs"
        old = '''        assert!(
            domain.accepts(&y),
            "map image does not encode the scalar sort"
        );
        if holds(&y) {'''
        new = '''        if holds(&y) {
            assert!(
                domain.accepts(&y),
                "map image does not encode the scalar sort"
            );'''
        def install_test():
            replace("tests/backend.rs",
                    "let mut members = Listed(&[[1]]);\n    transfer(BV4, &map, &mut members, |_| true, true);",
                    'let mut members = Listed(&[[1]]);\n    transfer(BV4, &map, &mut members, |_| panic!("filter received an invalid image"), true);')
        cmd = ["cargo", "test", "--locked", "--test", "backend", "a_transfer_image_outside_the_sort_is_rejected", "--", "--exact"]
        diagnostic = "filter received an invalid image"
    replace(path, old, new)
    suite()
    print("OBS mutant survived all original native debug and release tests", flush=True)
    replace(path, new, old)
    install_test()
    run(cmd)
    replace(path, old, new)
    run(cmd, diagnostic)
    print("OBS stronger test passed on original and detected mutant", flush=True)

elif mode == "paths":
    replace("src/bdd.rs", "const LEVEL_BITS: u32 = 6;", """const LEVEL_BITS: u32 = 6;
static REVIEW_EDGES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static REVIEW_OWNED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);""")
    for signature, counter in [
        ("fn covering_edge(edges: &[Edge], bit: u64) -> &Edge {", "REVIEW_EDGES"),
        ("fn owned(&self, mut build: impl FnMut(&State) -> Result<Function, NeedGrowth>) -> Function {", "REVIEW_OWNED"),
    ]:
        replace("src/bdd.rs", signature, signature + "\n        " + counter + ".fetch_add(1, std::sync::atomic::Ordering::Relaxed);")
    with Path("tests/unit/bdd.rs").open("a") as f:
        f.write((root / "counters.rs.in").read_text())
    cmd = ["cargo", "test", "--locked", "--release", "--lib", "manual_operation_counts", "--", "--nocapture"]
    assert "OBS edge lookups=32 empty-set admissions=1" in run(cmd)
    child = """            let child = if node.level() == *level {
                covering_edge(node.edges, 1 << assignment).child
            } else {
                *f
            };
"""
    replace("src/bdd.rs", child, "")
    next_level = "            let next_level = *level + 1;"
    replace("src/bdd.rs", next_level, child + next_level)
    replace("src/bdd.rs", """        if keys.is_empty() {
            return Ok(default);
        }
""", "")
    anchor = "        scalar::check_packed(self.sort(), values);\n"
    replace("src/bdd.rs", anchor, anchor + """        if values.is_empty() {
            return Function::FALSE;
        }
""")
    assert "OBS edge lookups=0 empty-set admissions=0" in run(cmd)
    suite()

elif mode == "alloc":
    Path("examples").mkdir(exist_ok=True)
    shutil.copy(root / "alloc.rs.in", "examples/manual_alloc.rs")
    cmd = ["cargo", "run", "--locked", "--release", "--example", "manual_alloc"]
    baseline = run(cmd)
    assert all(x in baseline for x in ["singleton sort allocations=2", "count limbs=1 allocations=1", "count limbs=64 allocations=1"])
    anchor = "    let record = |i: usize| &packed[i * words..][..words];"
    replace("src/scalar.rs", anchor, """    if packed.len() <= words {
        return;
    }
""" + anchor)
    anchor = "    fn add_assign(&mut self, rhs: Self) {"
    replace("src/util/count.rs", anchor, anchor + """
        if self.0.is_empty() {
            *self = rhs;
            return;
        }""")
    changed = run(cmd)
    assert all(x in changed for x in ["singleton sort allocations=0", "count limbs=1 allocations=0", "count limbs=64 allocations=0"])
    suite()

elif mode == "oracle":
    shutil.copy(root / "oracle.rs.in", "tests/manual_oracle.rs")
    run(["cargo", "test", "--locked", "--release", "--test", "manual_oracle", "--", "--nocapture"])
    replace("tests/bdd.rs", """        let mut reached = HashSet::new();
        for x in 0..1 << 18 {
            let pair = (c.evaluate(f, &[x]), c.evaluate(parity, &[x]));
            assert_eq!(labels[c.evaluate(h, &[x]) as usize], pair, "at {x}");
            reached.insert(pair);
        }
        assert_eq!(labels.iter().copied().collect::<HashSet<_>>(), reached);
        assert_eq!(labels.len(), reached.len());""",
    """        let mut first = Vec::new();
        for x in 0..1 << 18 {
            let pair = (left(x), (x & 1) as u32);
            assert_eq!(labels[c.evaluate(h, &[x]) as usize], pair, "at {x}");
            if !first.contains(&pair) {
                first.push(pair);
            }
        }
        assert_eq!(labels, first);""")
    run(["cargo", "test", "--locked", "--test", "bdd", "alignment_shares_pairs_and_splits_below_a_terminal_side", "--", "--exact"])
    print("OBS independent full-domain alignment fixture passed", flush=True)
else:
    raise ValueError(mode)
