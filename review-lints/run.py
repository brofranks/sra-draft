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
paths = [p for p in tracked if p and not p.startswith("review-lints/") and p != ".github/workflows/bdd-lints.yml"]
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
else:
    raise ValueError(kind)

(out / "commands.json").write_text(json.dumps(log, indent=2))
if any(x["exit"] for x in log.values()):
    sys.exit(1)
