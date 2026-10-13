"""Independent, reversible CUDA GEMM contention; no GPU clock/power changes."""
import argparse
import json
import time
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument('--duty',type=float,required=True)
p.add_argument('--ready',type=Path,required=True)
p.add_argument('--stop',type=Path,required=True)
a=p.parse_args()
assert 0 <= a.duty <= 1
torch.set_num_threads(1)
torch.manual_seed(20261013)
x=torch.randn((2048,2048),device='cuda',dtype=torch.float16)
y=torch.randn_like(x)
z=torch.empty_like(x)
for _ in range(20):
    torch.mm(x,y,out=z)
torch.cuda.synchronize()
a.ready.write_text(json.dumps({'duty':a.duty,'matrix':2048,'dtype':'float16','allocated_bytes':torch.cuda.memory_allocated()}))
cycles=0
active=0.0
started=time.perf_counter()
while not a.stop.exists():
    begin=time.perf_counter()
    until=begin+.04*a.duty
    while time.perf_counter()<until:
        torch.mm(x,y,out=z)
        torch.cuda.synchronize()
    active+=time.perf_counter()-begin
    time.sleep(max(0,.04-(time.perf_counter()-begin)))
    cycles+=1
print(json.dumps({'cycles':cycles,'active_wall_fraction':active/(time.perf_counter()-started),'requested_duty':a.duty}),flush=True)
