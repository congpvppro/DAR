"""Re-decode all inputs and verify tensor hashes and real HF temporal grids on CPU."""
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
OUT=Path(__file__).resolve().parent
DATA=OUT/'timing-corrected'
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'DAR'))


def main():
    import torch
    from transformers import AutoProcessor
    import test as official
    from study import intervene,build_cases
    torch.set_num_threads(2)
    manifest=json.loads((DATA/'manifest.json').read_text())
    records={}
    for path in DATA.glob('*-shard*.jsonl'):
        for line in path.read_text().splitlines():
            r=json.loads(line)
            records[(r['model'],r['case_id'])]=r
    assert len(records)==736,len(records)
    processor=AutoProcessor.from_pretrained(str(ROOT/'model/Qwen2.5-VL-3B-Instruct'))
    result=[]
    for row in manifest['videos']:
        messages=official.make_messages(row['video_path'],row['video_duration'])
        messages[0]['content'][0].update(nframes=16,min_pixels=100352,max_pixels=100352)
        base=official.build_input(processor,messages)
        tensor,metadata=base['multi_modal_data']['video'][0]
        frames=tensor.numpy()
        fps=len(frames)*float(metadata['fps'])/int(metadata['total_num_frames'])
        conditions=sorted({c['condition'] for c in build_cases([row])})
        inputs=[]
        for condition in conditions:
            transformed=intervene(frames,condition)
            digest=hashlib.sha256(transformed.tobytes()).hexdigest()
            if condition=='static_middle':
                assert np.all(transformed==frames[8:9])
            elif condition=='gray':
                assert np.all(transformed==127.5)
            elif condition=='gray_to_white':
                assert np.all(transformed[:8]==127.5) and np.all(transformed[8:]==255.)
            related=[records[(model,c['case_id'])] for model in ('dar','qwen') for c in build_cases([row]) if c['condition']==condition]
            for rec in related:
                assert rec['input_sha256']==digest,(rec['case_id'],'tensor')
                assert rec['input_shape']==list(transformed.shape)
                assert rec['mm_processor_kwargs']['fps']==[fps]
                assert abs(rec['expected_second_per_grid_ts']-2/fps)<1e-12
            prompt=related[0]['prompt']
            hf=processor(text=[prompt],videos=[torch.from_numpy(transformed)],do_sample_frames=False,fps=[fps],return_tensors='pt')
            actual=hf['second_per_grid_ts']
            assert np.allclose(np.asarray(actual),[2/fps],rtol=1e-6,atol=1e-6)
            effective=dict(input_ids=hashlib.sha256(hf['input_ids'].numpy().tobytes()).hexdigest(),
                           pixels=hashlib.sha256(hf['pixel_values_videos'].numpy().tobytes()).hexdigest(),
                           timegrid_float32=np.asarray(actual).tolist(),
                           timegrid_bfloat16=torch.as_tensor(actual).to(torch.bfloat16).float().tolist())
            inputs.append(dict(condition=condition,shape=list(transformed.shape),raw_sha256=digest,
                               matching_records=len(related),effective=effective))
        result.append(dict(video_id=row['video_id'],kind=row['kind'],sampled_fps=fps,inputs=inputs))
        print(json.dumps(dict(verified=len(result),total=len(manifest['videos']))),flush=True)
    out=dict(status='passed',videos=len(result),records_verified=sum(x['matching_records'] for r in result for x in r['inputs']),
             scope='CPU re-decode plus actual HF processing, cross-model recorded metadata; separate exact vLLM forwarding proof in parent artifact',results=result)
    (OUT/'input-verification.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    main()
