import io
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path

checkout = Path.cwd()
job = os.environ["PROFILE"]
out = checkout / "report"
out.mkdir()
source = Path(os.environ["RUNNER_TEMP"]) / "audit-source"
source.mkdir()
rev = "3af3babb73584b0c169d74ce8ea46ed5ef791958"
assert subprocess.check_output(["git", "rev-parse", "HEAD^"], text=True).strip() == rev
(out / "source-tree").write_bytes(subprocess.check_output(["git", "rev-parse", rev+"^{tree}"]))
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(["git", "archive", rev]))) as archive:
    archive.extractall(source, filter="data")
env = dict(os.environ, CARGO_TARGET_DIR=str(source.parent / ("target-"+job)))
result = {}
def run(name, args):
    process = subprocess.run(args, cwd=source, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    (out / (name+".out")).write_text(process.stdout)
    (out / (name+".err")).write_text(process.stderr)
    result[name] = dict(args=args, code=process.returncode,
                        passed=sum(int(n) for n in re.findall(r"test result: ok\. (\d+) passed", process.stdout+process.stderr)))
    (out / "commands.json").write_text(json.dumps(result, indent=2))
    print(name,process.returncode,result[name]["passed"],flush=True)
    return process
def once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)
bdd = source / "src/bdd.rs"
text = bdd.read_text()
if job.startswith("count-"):
    edits = [
        ("        let mut stack = vec![(p, 0)];\n        'nodes: while let Some((f, next)) = stack.last_mut() {\n            let f = *f;",
         "        let mut stack = Vec::new();\n        let (mut f, mut next) = (p, 0);\n        'nodes: loop {"),
        ("            while let Some(edge) = node.edges.get(*next) {\n                *next += 1;",
         "            while let Some(edge) = node.edges.get(next) {\n                next += 1;"),
        ("                    stack.push((edge.child, 0));\n                    continue 'nodes;",
         "                    stack.push((f, next));\n                    (f, next) = (edge.child, 0);\n                    continue 'nodes;"),
        ("            if f == p {\n                break;\n            }\n            memo.insert(f, (pool.len(), sum.len()));\n            pool.extend_from_slice(&sum);\n            stack.pop();",
         "            let Some(parent) = stack.pop() else { break };\n            memo.insert(f, (pool.len(), sum.len()));\n            pool.extend_from_slice(&sum);\n            (f, next) = parent;"),
    ]
    for before,after in edits:
        text = once(text, before, after)
elif job in ("image", "preimage", "ite"):
    start = text.index("    fn "+job+"(&self")
    end = text.index("\n    }",start)
    body = text[start:end]
    guard = re.search(r'            debug_assert!\(\n                state\.has_refs\(.*?\n                "cannot use an unreferenced BDD node"\n            \);\n',body)
    assert guard
    text = text[:start]+body[:guard.start()]+body[guard.end():]+text[end:]
elif job in ("ignore-level","ignore-edges"):
    replacement = "|_node, stored| stored == edges" if job=="ignore-level" else "|node, _stored| node.data == data"
    text = once(text, "|node, stored| node.data == data && stored == edges", replacement)
bdd.write_text(text)

if job.startswith("count-"):
    from difflib import unified_diff
    original=subprocess.check_output(["git","show",rev+":src/bdd.rs"],text=True)
    (out/"count.diff").write_text("".join(unified_diff(original.splitlines(True),text.splitlines(True),fromfile="a/src/bdd.rs",tofile="b/src/bdd.rs")))

mutant = job in ("image","preimage","ite","ignore-level","ignore-edges")
if mutant:
    suite = run("existing", ["cargo","test","--locked","--all-targets"])
    if suite.returncode:
        print((suite.stdout+suite.stderr)[-4000:],flush=True)
    assert suite.returncode == 0 and result["existing"]["passed"] == 203

if job != "count-only":
    for target, snippet in (("tests/bdd.rs","released"),("tests/unit/bdd.rs","collision")):
        path = source / target
        with path.open("a") as f:
            f.write((checkout / "review-coverage" / (snippet+".rs.in")).read_text())
if mutant:
    if job.startswith("ignore-"):
        args=["--lib","bdd::tests::coverage_distinct_nodes_with_one_hash"]
        reasons=["merged different levels" if job=="ignore-level" else "merged different edges"]
    else:
        args=["--test","bdd","coverage_released_operands"]
        reasons={"image":["image(p)","image(q)"],"preimage":["preimage(p)","preimage(q)"],
                 "ite":["ite(condition)","ite(yes)","ite(no)"]}[job]
    probe=run("probe", ["cargo","test","--locked",*args,"--","--exact"])
    message=probe.stdout + probe.stderr
    assert probe.returncode != 0 and "test result: FAILED. 0 passed; 1 failed" in message, message[-3000:]
    assert all(s in message for s in reasons), message[-3000:]
else:
    ignore=["--ignore-filename-regex","/tests/|/examples/|/macros/"]
    suite = run("coverage",["cargo","llvm-cov","--locked","--all-targets","--branch","--show-missing-lines",*ignore])
    assert suite.returncode == 0 and result["coverage"]["passed"] == (203 if job=="count-only" else 205), (suite.stdout+suite.stderr)[-4500:]
    if job!="count-only":
        assert "coverage_released_operands ... ok" in suite.stderr and "coverage_distinct_nodes_with_one_hash ... ok" in suite.stderr
    assert run("export",["cargo","llvm-cov","report","--json",*ignore,"--output-path",str(out/"coverage.json")]).returncode==0
