from pathlib import Path
import subprocess

cmd = [
    "cargo", "miri", "test", "--locked", "--release", "--target",
    "i686-unknown-linux-gnu", "--lib", "--",
    "backend::tests::an_unrepresentable_class_leaves_grouping_reusable", "--exact",
]
subprocess.run(cmd, check=True)
src = Path("src/backend.rs")
original = src.read_text()
checked = """            let len = idx
                .checked_add(1)
                .expect("class lookup length exceeds usize");
            self.classes.resize(len, UNREACHED);"""
assert original.count(checked) == 1
try:
    src.write_text(original.replace(checked, "            self.classes.resize(idx + 1, UNREACHED);", 1))
    run = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(run.stdout, end="")
    assert run.returncode == 101 and "index out of bounds: the len is 0 but the index is 1" in run.stdout
finally:
    src.write_text(original)
