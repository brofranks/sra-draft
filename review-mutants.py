import subprocess
from pathlib import Path

def test(name, *, original=False, fails=False):
    args = ['cargo', 'test', '--locked', '--lib', name]
    if original:
        args += ['--', '--skip', 'review_']
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, flush=True)
    if fails:
        assert result.returncode != 0
        assert any(line.startswith(f'test sparse::tests::{name} ') and line.endswith(' ... FAILED')
                   for line in result.stdout.splitlines()), name
    else:
        assert result.returncode == 0, name

path = Path('src/sparse.rs')
base = path.read_text()
cases = [
    (
        '            labels.truncate(start);\n            debug_assert!(',
        '            let _ = start;\n            debug_assert!(',
        'review_alignment_retry_appends_one_suffix',
    ),
    (
        '        self.column.dec_ref(self.root);',
        '        if self.value.is_none() { self.column.dec_ref(self.root); }',
        'review_suspended_cursor_releases_its_root',
    ),
    (
        'let mut default = if values[0] != 0 { values[1] } else { values[2] };',
        'let mut default = 0;',
        'review_conditional_preserves_a_true_initial_run',
    ),
    (
        'add(&mut sum, &key_words, next == 1);',
        'add(&mut sum, &key_words[..1], next == 1);',
        'review_counts_a_run_crossing_the_low_word_boundary',
    ),
    (
        'slot_labels.extend([default, label]);\n                state.inc_ref(p);',
        'slot_labels.extend([default, label]);',
        'review_single_case_shortcut_acquires_a_reference',
    ),
    (
        'let initial = value(&active);',
        'let initial = active.keys().next().copied().unwrap_or(default);',
        'review_cases_detect_overlap_starting_at_zero',
    ),
]
test('sparse::tests::review_')
for old, new, name in cases:
    assert base.count(old) == 1, name
    try:
        path.write_text(base.replace(old, new))
        test('sparse::tests::', original=True)
        test(name, fails=True)
    finally:
        path.write_text(base)
