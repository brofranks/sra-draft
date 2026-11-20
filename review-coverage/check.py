import collections
import io
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path

profile = os.environ["PROFILE"]
revision = os.environ["SOURCE_REV"]
source = Path(os.environ["RUNNER_TEMP"]) / "candidate"
output = Path.cwd() / "report"
source.mkdir()
output.mkdir()
assert subprocess.check_output(["git", "rev-parse", "HEAD^"], text=True).strip() == revision
tree = subprocess.check_output(["git", "rev-parse", revision + "^{tree}"], text=True).strip()
(output / "tree.txt").write_text(tree + "\n")
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(["git", "archive", revision]))) as archive:
    archive.extractall(source, filter="data")

env = dict(os.environ, CARGO_TARGET_DIR=str(source.parent / ("target-" + profile)))
rows = {}
def run(name, args):
    result = subprocess.run(args, cwd=source, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (output / (name+".log")).write_text(result.stdout)
    row = {"args":args, "exit":result.returncode,
           "passed":sum(map(int, re.findall(r"test result: ok\. (\d+) passed", result.stdout)))}
    rows[name] = row
    (output / "commands.json").write_text(json.dumps(rows, indent=2))
    print(name, row, flush=True)
    return result
def check(name, args):
    result = run(name, args)
    assert result.returncode == 0, result.stdout[-4500:]
    return result.stdout

check("rustc", ["rustc","-Vv"])
if profile == "native":
    check("rustfmt", ["cargo","fmt","--all","--","--check"])
    check("clippy", ["cargo","clippy","--locked","--all-targets","--","-D","warnings"])
    check("debug", ["cargo","test","--locked","--all-targets"])
    check("release", ["cargo","test","--locked","--release","--all-targets"])
    check("doctests", ["cargo","test","--locked","--doc"])
elif profile == "coverage":
    check("cov-version", ["cargo","llvm-cov","--version"])
    keep = ["--ignore-filename-regex","/tests/|/examples/|/macros/"]
    check("tests", ["cargo","llvm-cov","--locked","--all-targets","--branch","--show-missing-lines",*keep])
    path = output / "coverage.json"
    check("export", ["cargo","llvm-cov","report","--json",*keep,"--output-path",str(path)])
    f = next(f for f in json.loads(path.read_text())["data"][0]["files"] if f["filename"].endswith("/src/bdd.rs"))
    outcomes = collections.defaultdict(lambda: [False,False])
    for sl,sc,el,ec,yes,no,*_ in f["branches"]:
        pair = outcomes[sl,sc,el,ec]
        pair[0] |= yes > 0
        pair[1] |= no > 0
    (output / "bdd.json").write_text(json.dumps({
        "llvm":f["summary"]["branches"], "covered":sum(map(sum,outcomes.values())),
        "count":2 * len(outcomes), "missing":[list(k) for k,v in sorted(outcomes.items()) if not all(v)]
    }, indent=2))
else:
    bdd = source / "src/bdd.rs"
    data = bdd.read_text()
    if profile in ("ignore-level","ignore-edges"):
        before = "|node, stored| node.data == data && stored == edges"
        after = "|_node, stored| stored == edges" if profile == "ignore-level" else "|node, _stored| node.data == data"
        assert data.count(before) == 1
        data = data.replace(before,after)
        selector = ["--lib"]
        test = "bdd::tests::colliding_nodes_are_distinguished_by_level_and_edges"
        reason = "merged different levels" if profile == "ignore-level" else "merged different edges"
    else:
        assert profile in ("image","preimage","ite")
        start = data.index("    fn "+profile+"(&self")
        end = data.index("\n    }", start)
        body = data[start:end]
        guard = re.search(r'            debug_assert!\(\n                state\.has_refs\(.*?\n                "cannot use an unreferenced BDD node"\n            \);\n', body)
        assert guard is not None
        data = data[:start] + body[:guard.start()] + body[guard.end():] + data[end:]
        selector = ["--test","bdd"]
        test = "conditionals_check_all_three_input_roots" if profile == "ite" else "bit_maps_check_both_input_roots"
        reason = "the operation accepted an unreferenced BDD node"
    bdd.write_text(data)
    probe = run("probe",["cargo","test","--locked",*selector,test,"--","--exact"])
    detected = (probe.returncode == 101 and ("test "+test+" ... FAILED") in probe.stdout
                and "test result: FAILED. 0 passed; 1 failed" in probe.stdout and reason in probe.stdout)
    assert detected, probe.stdout[-4000:]
    (output / "fault.json").write_text(json.dumps({"fault":profile,"test":test,"assertion":reason},indent=2))
