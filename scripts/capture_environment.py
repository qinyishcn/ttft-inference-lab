import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
query='name,driver_version,memory.total,power.limit,power.default_limit,power.min_limit,power.max_limit'
gpu=subprocess.check_output(['nvidia-smi','--query-gpu='+query,'--format=csv,noheader,nounits'],text=True).strip().split(', ')
cpu=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command','Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress'],text=True))
ram=int(subprocess.check_output(['powershell','-NoProfile','-Command','(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory'],text=True).strip())
packages={}
for name in ['requests','numpy','pillow','matplotlib','torch']:
    try:
        packages[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name]=None
data={'os':platform.system(),'os_release':platform.release(),'os_version':platform.version(),'python':sys.version.split()[0],'gpu':dict(zip(query.split(','),gpu)),'cpu':cpu,'ram_bytes':ram,'packages':packages,'power_settings_modified':False,'user_timezone':'Asia/Shanghai'}
(ROOT/'data'/'environment.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
print(json.dumps(data,indent=2))
