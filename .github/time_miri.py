from pathlib import Path
import re
import subprocess
import sys
import time

selectors = [
    (['--lib'], 'tests/unit/bdd.rs', 'bdd::tests::'),
    (['--lib'], 'tests/unit/cache.rs', 'cache::tests::'),
    *[(['--test', name], f'tests/{name}.rs', '') for name in ['bdd','backend','scalar','count']]
]
cases = []
for target,path,prefix in selectors:
    if not Path(path).exists():
        continue
    for segment in Path(path).read_text().split('#[test]')[1:]:
        name = re.search(r'^fn (\w+)\(',segment,re.M)[1]
        cases.append((target,prefix+name))
cargo=['cargo','miri','test','--locked','--target','x86_64-unknown-linux-gnu']
subprocess.run(cargo+['--no-run'],check=True)
for target,name in cases[int(sys.argv[1])::8]:
    started=time.monotonic()
    run=subprocess.run(['timeout','-k','2s','70s',*cargo,*target,'--','--exact',name,'--include-ignored'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    print('TIMING',name,round(time.monotonic()-started,1),run.returncode,flush=True)
    if run.returncode not in (0,124,137) or run.returncode==0 and '1 passed' not in run.stdout:
        print(run.stdout)
        raise SystemExit(1)
