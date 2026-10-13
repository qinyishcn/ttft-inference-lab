"""Save hashes and line anchors for pinned primary-source timer implementations."""
import hashlib
import json
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1]
audit=[]
for tag in ['b11516','b11577']:
    for name in ['server-context.cpp','server-common.h','server-common.cpp']:
        url=f'https://raw.githubusercontent.com/ggml-org/llama.cpp/{tag}/tools/server/{name}'
        response=requests.get(url,timeout=30)
        response.raise_for_status()
        content=response.content
        (ROOT/'results'/'private'/f'{tag}-{name}').write_bytes(content)
        lines=content.decode().splitlines()
        anchors=[{'line':i+1,'text':line.strip()} for i,line in enumerate(lines)
                 if any(s in line for s in ['update_prompt_start','update_prompt_last','t_prompt_ms()','"prompt_ms",','encode_mtmd_chunk(slot','process_mtmd_chunk(slot'])]
        audit.append({'tag':tag,'url':url,'sha256':hashlib.sha256(content).hexdigest(),'anchors':anchors})
(ROOT/'data'/'timing-source-audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
