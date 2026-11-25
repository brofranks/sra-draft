import concurrent.futures as cf
import os, pathlib, re, resource, subprocess, threading, time, json
P=pathlib.Path
suites=['context','dag','index','sparse','soundness','count','scalar']
def cpu():
    r=resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime+r.ru_stime
def avail():
    return int(re.search(r'MemAvailable:\s+(\d+)',P('/proc/meminfo').read_text())[1])/1024
def run(suite):
    t=time.monotonic()
    output=subprocess.check_output(['cargo','miri','test','--locked','--target','x86_64-unknown-linux-gnu','--test',suite],stderr=subprocess.STDOUT,text=True)
    print(suite, round(time.monotonic()-t,1), re.findall(r'test result: ok[^\n]+',output)[-1],flush=True)
def sample(stop,stats):
    sid=os.getsid(0)
    while not stop.is_set():
        total=0
        for f in P('/proc').glob('[0-9]*/stat'):
            try:
                fields=f.read_text().rsplit(')',1)[1].split()
                if int(fields[3])==sid:
                    total+=int(re.search(r'^Pss:\s+(\d+)',(f.parent/'smaps_rollup').read_text(),re.M)[1])
            except (OSError,TypeError,ValueError):pass
        stats[0]=max(stats[0],total/1024)
        stats[1]=min(stats[1],avail())
        stop.wait(1)
print('CPUs',len(os.sched_getaffinity(0)),'mem',P('/proc/meminfo').read_text().splitlines()[0], 'quota',P('/sys/fs/cgroup/cpu.max').read_text().strip())
for n in [1,2,4]:
    stats=[0,avail()]; stop=threading.Event(); thread=threading.Thread(target=sample,args=(stop,stats));thread.start()
    t=time.monotonic(); c=cpu()
    try:
        with cf.ThreadPoolExecutor(n) as pool:list(pool.map(run,suites))
    finally:stop.set();thread.join()
    dt=time.monotonic()-t
    print('RESULT',json.dumps(dict(n=n,seconds=round(dt,1),cores=round((cpu()-c)/dt,2),peak_pss_mib=round(stats[0]),min_available_mib=round(stats[1]))))
