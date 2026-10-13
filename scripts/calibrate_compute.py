"""Standalone competition-worker calibration; not TTFT or a GPU-equivalence test."""
import time
from benchmark import Competitor,Telemetry,ROOT

(ROOT/'results'/'private').mkdir(exist_ok=True)
for duty in [0,.25,.5,.75]:
    with Competitor(duty,f'calibration-{duty}'),Telemetry('calibration',f'gemm-{duty}'):
        time.sleep(5)
    print('calibrated duty',duty,flush=True)
