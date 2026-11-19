"""Focused checks for the disposable BDD review branch."""
import subprocess
import sys
from pathlib import Path

src = Path("src/bdd.rs")
fixture = Path("tests/unit/bdd.rs")
sort = Path("src/scalar.rs")
mode = sys.argv[1]


def replace(path, old, new, count=1):
    source = path.read_text()
    assert source.count(old) == count, (path, old)
    path.write_text(source.replace(old, new))


def run(*args, failure=None):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if failure is None and result.returncode != 0:
        print(result.stdout)
        result.check_returncode()
    if failure is not None:
        print(result.stdout)
        assert result.returncode != 0 and failure in result.stdout
    else:
        print("\n".join(line for line in result.stdout.splitlines() if "test result:" in line or "records=" in line))
    return result.stdout


def test(name, failure=None):
    output = run("cargo", "test", "--locked", "--lib", "bdd::tests::" + name, "--", "--exact", failure=failure)
    assert "running 1 test\n" in output


if mode == "allocations":
    # Standalone processes leave the allocator counters independent.
    print("current", flush=True)
    run("cargo", "run", "--locked", "--release", "--example", "review_allocations")
    replace(sort, "    *packed = order.iter().flat_map(|&i| record(i)).copied().collect();",
            "    let mut out = Vec::with_capacity(order.len() * words);\n"
            "    for i in order { out.extend_from_slice(record(i)); }\n    *packed = out;")
    print("exact capacity", flush=True)
    run("cargo", "run", "--locked", "--release", "--example", "review_allocations")
else:
    original = src.read_text()
    if mode == "hits":
        old = "            self.store.restart_cache(cache);"
        changed = old + "\n            cache.clear();"
        count = 2
        cases = [
            ("alignment_numbers_reachable_pairs_in_discovery_order", "warm alignment rebuilt"),
            ("quotients_merge_classes_and_check_the_cached_map", "warm quotient rebuilt"),
        ]
    else:
        old = "            self.store.restart_cache(cache);\n            cache::quotient"
        changed = "            cache::quotient"
        count = 1
        cases = [("review_quotient_restarts_after_collection", "cannot reference a reclaimed DAG node")]

    replace(src, old, changed, count)
    print("Current suite on mutant", flush=True)
    run("cargo", "test", "--locked", "--all-targets")
    src.write_text(original)

    if mode == "hits":
        for field, binding, call, marker in (
            ("memo", "again", "c.align(f, g, &mut labels, &mut scratch)", cases[0][1]),
            ("map", "hit", "c.quotient(f, &class_map, &mut scratch)", cases[1][1]),
        ):
            statement = f"    let {binding} = {call};"
            replace(fixture, statement, f"    scratch.{field}.clear();\n" + statement
                    + f'\n    assert!(scratch.{field}.is_empty(), "{marker}");')
    else:
        with fixture.open("a") as out:
            out.write("""
#[test]
fn review_quotient_restarts_after_collection() {
    let store = roomy();
    let c = col(&store, BV4);
    let f = residues(&*c, 6);
    let mut scratch = Alignment::default();
    let map = [0, 1, 1, 0, 1, 1];
    let q = c.quotient(f, &map, &mut scratch);
    c.dec_ref(q);
    collect(&store);
    assert!(store.read().get(q).is_none());
    let q = c.quotient(f, &map, &mut scratch);
    for x in 0..16 {
        assert_eq!(c.evaluate(q, &[x]), u32::from(x % 3 != 0));
    }
    c.dec_ref(q);
    c.dec_ref(f);
}
""")
    print("Stronger checks on original", flush=True)
    for name, _ in cases:
        test(name)
    replace(src, old, changed, count)
    print("Stronger checks on mutant", flush=True)
    for name, marker in cases:
        test(name, failure=marker)
