import subprocess
from pathlib import Path

def cargo(test_target, name, expected=None):
    result = subprocess.run(
        ['cargo', 'test', '--locked', *test_target, name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(result.stdout)
    if expected is None:
        assert result.returncode == 0, name
    else:
        assert result.returncode != 0 and expected in result.stdout, name
        assert any(line.startswith(f'test {name} ') and line.endswith(' ... FAILED')
                   for line in result.stdout.splitlines()), name

probes = ['--test', 'review_sparse']
for name in ['empty_refine_rejects_released_partition', 'partial_map_rejects_released_filter']:
    cargo(probes, name, 'test did not panic as expected')
cargo(probes, 'public_scratch_reuse_after_discarding_lookup_table', 'index out of bounds')
cargo(probes, 'alignment_reports_the_exact_suffix')
cargo(probes, 'sparse_values_rejects_missing_label')
cargo(['--release', *probes], 'sparse_values_rejects_missing_label')

path = Path('src/sparse.rs')
base = path.read_text()
def mutant(old, new, test, expected):
    assert base.count(old) == 1
    try:
        path.write_text(base.replace(old, new))
        cargo(['--lib'], 'sparse::tests::')
        cargo(probes, test, expected)
    finally:
        path.write_text(base)

align = """            merge_entries(l, a, b, |key, x, y| emit(key, (x, y)));
            drop(emit);
            state.intern_referenced(l, 0, entries)"""
mutant(align, align.replace('            state.intern_referenced', '            labels.push((0, 0));\n            state.intern_referenced'),
       'alignment_reports_the_exact_suffix', 'assertion `left == right` failed')
check = """        assert_eq!(
            case_labels.len(),
            keys.len() / self.sort.words(),
            "there must be one case label per key"
        );
"""
mutant(check, '', 'sparse_values_rejects_missing_label', 'test did not panic as expected')
