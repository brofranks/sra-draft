import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

kind = sys.argv[1]
checkout = Path.cwd()
out = checkout / "lint-results"
out.mkdir()
src = Path(os.environ["RUNNER_TEMP"]) / "audit-source"
src.mkdir()
rev = "36987f938edde1fb652dacf5196f991a378d3547" if kind == "parent" else "HEAD"
tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "-z", rev]).decode().split("\0")
workflows = {".github/workflows/bdd-lints.yml", ".github/workflows/bdd-final-ci.yml"}
paths = [p for p in tracked if p and not p.startswith("review-lints/") and p not in workflows]
data = subprocess.check_output(["git", "archive", rev, "--", *paths])
with tarfile.open(fileobj=io.BytesIO(data)) as archive:
    archive.extractall(src, filter="data")
manifest = {p: hashlib.sha256((src / p).read_bytes()).hexdigest() for p in paths}
(out/"manifest.json").write_text(json.dumps(manifest, indent=2))
env = os.environ.copy()
env["CARGO_TARGET_DIR"] = str(src.parent / ("audit-target-" + kind))
log = {}

def command(name, args, extra=None):
    complete = subprocess.run(args, cwd=src, env=env | (extra or {}), capture_output=True)
    log[name] = {"args": args, "env": extra, "exit": complete.returncode}
    for suffix, raw in ((".out", complete.stdout), (".err", complete.stderr)):
        (out / (name + suffix)).write_bytes(raw)
    print(name, complete.returncode, flush=True)
    if complete.returncode:
        print((complete.stdout[-3500:] + complete.stderr[-3500:]).decode(errors="replace"), flush=True)
    return complete

command("rustc-version", ["rustc", "-Vv"])
command("cargo-version", ["cargo", "-V"])

if kind in ("parent", "layer"):
    command("clippy-version", ["cargo", "clippy", "-V"])
    command("rustfmt-version", ["rustfmt", "-V"])
    command("fmt", ["cargo", "fmt", "--all", "--", "--check"])
    profiles = json.loads((checkout / "review-lints/profiles.json").read_text())
    profiles["forced-complexity"] = ["clippy", "--locked", "--all-targets", "--message-format=json",
                                    "--", "--force-warn", "clippy::type_complexity"]
    profiles["doc-deny"] = ["doc", "--locked", "--no-deps", "--message-format=json"]
    for name, flags in profiles.items():
        extra = None
        if name in ("public", "private"):
            extra = {"RUSTDOCFLAGS": "-W rustdoc::all -W rustdoc::unescaped_backticks"}
        elif name == "doc-deny":
            extra = {"RUSTDOCFLAGS": "-D warnings"}
        command(name, ["cargo", *flags], extra)
    if kind == "layer":
        for name, flags in [("debug-tests", ["--all-targets"]), ("release-tests", ["--release", "--all-targets"]), ("doc-tests", ["--doc"])]:
            command(name, ["cargo", "test", "--locked", *flags])

elif kind in ("stable-lines", "nightly-branches", "nightly-unit"):
    command("llvm-cov-version", ["cargo", "llvm-cov", "--version"])
    ignore = ["--ignore-filename-regex", "/tests/|/examples/|/macros/"]
    flags = ["--lib"] if kind == "nightly-unit" else ["--all-targets"]
    if kind.startswith("nightly"):
        flags.extend(["--branch", "--show-missing-lines"])
    command("coverage", ["cargo", "llvm-cov", "--locked", *flags, *ignore])
    command("coverage-export", ["cargo", "llvm-cov", "report", "--json", *ignore, "--output-path", str(out / "coverage.json")])

elif kind == "focused-miri":
    command("miri-version", ["cargo", "miri", "--version"])
    for name, flags in [("count", ["--test", "count"]), ("level-buffer", ["--lib", "level_buffer"]), ("soundness", ["--test", "soundness"])]:
        result = command(name, ["cargo", "miri", "test", "--locked", *flags])
        if result.returncode == 0 and "test result: ok. 0 passed;" in result.stdout.decode(errors="replace"):
            raise AssertionError(f"The filter for {name} selected no tests")
elif kind == "regressions":
    shutil.copy2(checkout / "review-lints/oracle.rs.in", src / "tests/lint_oracle.rs")
    oracle = command("oracle", ["cargo", "test", "--release", "--locked", "--test", "lint_oracle", "--", "--nocapture"])
    for width in (4, 7, 9):
        assert f"OBS finite oracle width={width} domain={1 << width}" in oracle.stdout.decode()
    buffer_test = "bdd::tests::level_buffer_preserves_its_initialized_prefix_when_full"
    push = "    fn push(&mut self, item: T) {"
    experiments = [
        ("lost-last-push", push, push + "\n        if self.len == LEVEL_VALUES - 1 { return; }", ["--lib"], buffer_test, "prefix after pushing 64"),
        ("ignored-overflow", push, push + "\n        if self.len == LEVEL_VALUES { return; }", ["--lib"], buffer_test, "result.is_err()"),
        ("alignment-prefix", "let class = (labels.len() - start) as u32;", "let class = labels.len() as u32;", ["--lib"],
         "bdd::tests::alignment_numbers_reachable_pairs_in_discovery_order", "assertion"),
        ("one-based-class", "let class = first.len() as u32;", "let class = first.len() as u32 + 1;", ["--test","bdd"],
         "wide_keys_take_one_label_per_record", "at [0, 0]"),
    ]
    for name, before, after, target, test, reason in experiments:
        fault = src.parent / ("fault-" + name)
        shutil.copytree(src, fault)
        path = fault / "src/bdd.rs"
        data = path.read_text()
        assert data.count(before) == 1
        path.write_text(data.replace(before, after))
        args = ["cargo","test","--locked",*target,test,"--","--exact"]
        result = subprocess.run(args,cwd=fault,env=env | {"CARGO_TARGET_DIR":str(fault.parent/("target-"+name))},
                                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (out / (name+".out")).write_text(result.stdout)
        detected = (result.returncode != 0 and test in result.stdout and reason in result.stdout
                    and "test result: FAILED. 0 passed; 1 failed" in result.stdout)
        log[name] = {"args":args, "fault_exit":result.returncode, "exit":0 if detected else 1}
        print(name, "detected" if detected else result.stdout[-3000:], flush=True)
else:
    raise ValueError(kind)

(out / "commands.json").write_text(json.dumps(log, indent=2))
if any(x["exit"] for x in log.values()):
    sys.exit(1)
