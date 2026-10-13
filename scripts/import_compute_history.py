"""Import numeric computation records only, with pinned public source provenance."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--grace',type=Path,required=True)
    a=p.parse_args()
    out=ROOT/'results'/'history'
    out.mkdir(exist_ok=True)
    manifest={'policy':'Numeric computation observations; no transport or split Testbed timings. Historical warmup is not assumed.', 'sources':[]}
    rows=[]
    commit='7896d6c0540ba411147364a54cf190900ff0e336'
    for mode in ['single','four']:
        relative=f'results/q4_k_m_{mode}_native/predictions.jsonl'
        path=a.assets/relative
        manifest['sources'].append({'kind':'MiniCPM native','url':f'https://github.com/qinyishcn/minicpm-o-2_6-streamingbench-eval/blob/{commit}/{relative}', 'sha256':sha(path)})
        for line in path.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            t=r['timings']
            rows.append({'study':'minicpm_native_vulkan','condition':mode,'sample_id':r['question_id'],
                         'video_id':r['video_id'],'compute_ttft_ms':t['prompt_ms'],
                         'compute_answer_ms':t['prompt_ms']+t['predicted_ms'],
                         'input_tokens':t['prompt_n'],'cache_n':t['cache_n'],
                         'generated_tokens':t['predicted_n'],'correct':r['correct'],
                         'warmup_status':'not explicitly controlled in historical run'})
    (out/'minicpm-native.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    # Retain the cached language computation evidence, but never relabel it fresh full-model TTFT.
    study=a.assets/'publication'/'nextqa-vlm-fault-tolerance'
    commit='dd8841f419f47c903773d5e4df12064b522409ab'
    llava=[]
    for path in sorted((study/'results'/'offline_20261012'/'predictions').glob('*.jsonl')):
        relative=path.relative_to(study).as_posix()
        manifest['sources'].append({'kind':'LLaVA cached-language answer computation, NOT full TTFT',
            'url':f'https://github.com/qinyishcn/nextqa-vlm-fault-tolerance/blob/{commit}/{relative}', 'sha256':sha(path)})
        for line in path.read_text(encoding='utf-8').splitlines():
            r=json.loads(line)
            llava.append({'study':'llava_video_7b_nf4_cached','condition':r['condition_id'],
                          'sample_id':r['question_id'],'status':r['status'],
                          'cached_answer_ms':1000*r['suffix_decode_seconds'],
                          'generated_tokens':len(r.get('generated_token_ids',[])),
                          'eligible_full_model_ttft':False})
    (out/'llava-cached-answer.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in llava),encoding='utf-8')
    path=a.grace/'results'/'codec_benchmark.json'
    (out/'grace-codec.json').write_bytes(path.read_bytes())
    manifest['sources'].append({'kind':'GRACE codec computation; not LLM TTFT',
        'url':'https://github.com/qinyishcn/grace-rtp-psnr-testbed/blob/62c3ae031d4d410ea98b0ff06f3df140d82135ee/results/codec_benchmark.json','sha256':sha(path)})
    manifest['excluded_or_missing']=[
        {'model':'Liquid_V1_7B','reason':'No comparable per-question native compute timing found; RTC session duration is ineligible. Different model from LiquidAI LFM2.'},
        {'model':'LLaVA-Video-7B-Qwen2','reason':'Saved answer timers reuse visual features and prefix KV. Vision preparation and prefix audit timers include IO/hash/preparation; do not sum into fresh full inference TTFT.'},
        {'model':'DCVC','reason':'No completed comparable inference timing found in inspected project evidence.'}]
    (out/'provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    prov=json.loads((a.assets/'results'/'provenance.json').read_text())
    artifacts={'repo':'openbmb/MiniCPM-o-2_6-gguf','revision':prov['model_revision'],'runtime':'llama.cpp b11577 CUDA12.4',
               'dataset_revision':prov['dataset_revision'],'artifacts':[]}
    for entry in prov['model_files']:
        path=a.assets/'models'/entry['rfilename']
        digest=sha(path)
        assert digest==entry['lfs']['sha256'],path.name
        artifacts['artifacts'].append({'file':path.name,'bytes':path.stat().st_size,'sha256':digest,
            'url':f'https://huggingface.co/{artifacts["repo"]}/resolve/{artifacts["revision"]}/{path.name}'})
    for precision,fallback in [('q8_0',27),('q4_k_m',163)]:
        path=ROOT/'models'/f'minicpm-o26-mmproj-{precision}.gguf'
        artifacts['artifacts'].append({'file':path.name,'bytes':path.stat().st_size,'sha256':sha(path),
            'derived_from':'mmproj-model-f16.gguf','requested_precision':precision.upper(),
            'fallback_tensors':fallback,'total_tensors':455,
            'command':f'llama-quantize mmproj-model-f16.gguf {path.name} {precision.upper()} 8',
            'note':'Mixed tensor types; not uniformly 8-bit or 4-bit. Language weights remain original Q4_K_M.'})
    (ROOT/'data'/'minicpm-artifacts.json').write_text(json.dumps(artifacts,indent=2),encoding='utf-8')
    print('Imported',len(rows),'native MiniCPM and',len(llava),'conditional cached-language records; model hashes verified.')

if __name__=='__main__':
    main()
