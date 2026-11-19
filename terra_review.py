from pathlib import Path
import subprocess
import sys

variant = sys.argv[1]
src = Path("src/bdd.rs")
code = src.read_text()
signature = "    fn ite(&self, condition: Function, yes: Function, no: Function) -> Function {"
shortcut = {
    "true-condition": "if condition.is_t() { return yes; }",
    "false-condition": "if condition.is_f() { return no; }",
    "equal-branches": "if yes == no { return yes; }",
    "identity": "if yes.is_t() && no.is_f() { return condition; }",
}[variant]
assert code.count(signature) == 1
mutation = code.replace(signature, signature + "\n        " + shortcut)


def run(args, expect_failure=False):
    result = subprocess.run(["cargo", "test", "--locked", *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if expect_failure:
        print(result.stdout)
        assert result.returncode and "ite result lost its owner" in result.stdout
    elif result.returncode:
        print(result.stdout)
        result.check_returncode()
    else:
        print("\n".join(line for line in result.stdout.splitlines() if "test result:" in line))
        assert "test result: ok. 1 passed" in result.stdout if "--exact" in args else True


print("Original tests on modified implementation", flush=True)
src.write_text(mutation)
run(["--all-targets"])
src.write_text(code)
with Path("tests/unit/bdd.rs").open("a") as fixture:
    fixture.write("""
#[test]
fn shortcut_result_owns_its_operand() {
    for case in 0..4 {
        let store = roomy();
        let c = col(&store, BV4);
        let p = c.set(&[1, 3]);
        let other = c.set(&[2]);
        let result = match case {
            0 => c.ite(Function::TRUE, p, other),
            1 => c.ite(Function::FALSE, other, p),
            2 => c.ite(other, p, p),
            _ => c.ite(p, Function::TRUE, Function::FALSE),
        };
        assert_eq!(result, p);
        c.dec_ref(p);
        c.dec_ref(other);
        collect(&store);
        assert!(store.read().get(result).is_some(), "ite result lost its owner in case {case}");
        for x in 0..16 {
            assert_eq!(c.evaluate(result, &[x]), u32::from(x == 1 || x == 3));
        }
        c.dec_ref(result);
        collect(&store);
        assert!(store.read().get(result).is_none());
    }
}
""")
test = ["--lib", "bdd::tests::shortcut_result_owns_its_operand", "--", "--exact"]
print("Ownership test on original implementation", flush=True)
run(test)
print("Ownership test on modified implementation", flush=True)
src.write_text(mutation)
run(test, expect_failure=True)
