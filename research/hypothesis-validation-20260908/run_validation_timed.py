"""Paired frozen-input inference; no selective retries or replacement outputs."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from study import SEED,CONSERVATIVE,FACTUAL,intervene,build_cases

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'timing-corrected'
sys.path.insert(0,str(ROOT/'DAR'))
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=['dar','qwen'],required=True)
    ap.add_argument('--shard',type=int,choices=[0,1],required=True)
    args=ap.parse_args()
    result_path=OUT/f'{args.model}-shard{args.shard}.jsonl'
    if result_path.exists():
        raise FileExistsError(f'Refusing to overwrite or rerun {result_path}')
    import torch
    import transformers
    import vllm
    import test as official
    torch.set_num_threads(2)
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert hashlib.sha256((OUT/'protocol.md').read_bytes()).hexdigest()==manifest['protocol_sha256']
    videos=manifest['videos'][args.shard::2]
    model=ROOT/'model'/('DAR-R1' if args.model=='dar' else 'Qwen2.5-VL-3B-Instruct')
    llm=vllm.LLM(model=str(model),trust_remote_code=False,gpu_memory_utilization=.7,
                  tensor_parallel_size=1,max_model_len=12288,seed=SEED,
                  enforce_eager=True,max_num_seqs=8)
    processor=transformers.AutoProcessor.from_pretrained(str(model))
    params={p:vllm.SamplingParams(temperature=0,max_tokens=256 if p=='factual' else 4096,seed=SEED)
            for p in ('released','conservative','factual')}
    meta=dict(model=str(model),shard=args.shard,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
              started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              torch=torch.__version__,transformers=transformers.__version__,vllm=vllm.__version__,
              seed=SEED,temperature=0,nframes=16,min_pixels=100352,max_pixels=100352,
              max_tokens=4096,factual_max_tokens=256,max_model_len=12288,
              manifest_sha256=hashlib.sha256((OUT/'manifest.json').read_bytes()).hexdigest(),
              source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__),OUT/'study.py',ROOT/'DAR/test.py']})
    (OUT/f'{args.model}-shard{args.shard}-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    total=len(build_cases(videos))
    completed=0
    with result_path.open('x') as stream:
        for start in range(0,len(videos),2):
            inputs=[]
            details=[]
            for row in videos[start:start+2]:
                messages=official.make_messages(row['video_path'],row['video_duration'])
                messages[0]['content'][0].update(nframes=16,min_pixels=100352,max_pixels=100352)
                base=official.build_input(processor,messages)
                tensor,video_metadata=base['multi_modal_data']['video'][0]
                frames=tensor.numpy()
                sampled_fps=len(frames)*float(video_metadata['fps'])/int(video_metadata['total_num_frames'])
                assert sampled_fps>0
                base['mm_processor_kwargs']['fps']=[sampled_fps]
                assert frames.shape[0]==16
                if row['kind']=='natural' and args.model=='dar':
                    (OUT/'inputs').mkdir(exist_ok=True)
                    from PIL import Image
                    Image.fromarray(frames[8].transpose(1,2,0).clip(0,255).astype(np.uint8)).save(OUT/'inputs'/f"{row['video_id']}-middle.png")
                for case in build_cases([row]):
                    transformed=intervene(frames,case['condition'])
                    msg=copy.deepcopy(messages)
                    if case['prompt_kind']=='conservative':
                        msg[0]['content'][1]['text']+='\n\n'+CONSERVATIVE
                    elif case['prompt_kind']=='factual':
                        msg[0]['content'][1]['text']=FACTUAL
                    prompt=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
                    inp=dict(prompt=prompt,multi_modal_data={'video':[(torch.from_numpy(transformed),copy.deepcopy(video_metadata))]},
                             mm_processor_kwargs=copy.deepcopy(base['mm_processor_kwargs']))
                    inputs.append(inp)
                    details.append(dict(**case,kind=row['kind'],video_duration=row['video_duration'],
                                        sampled_fps=sampled_fps,expected_second_per_grid_ts=2.0/sampled_fps,input_shape=list(transformed.shape),input_dtype=str(transformed.dtype),
                                        input_sha256=hashlib.sha256(transformed.tobytes()).hexdigest(),
                                        identical_frames=bool(np.all(transformed==transformed[:1])),
                                        input_min=float(transformed.min()),input_max=float(transformed.max()),
                                        video_metadata=video_metadata,mm_processor_kwargs=base['mm_processor_kwargs'],
                                        prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
            for offset in range(0,len(inputs),8):
                batch=inputs[offset:offset+8]
                ds=details[offset:offset+8]
                started=time.monotonic()
                outputs=llm.generate(batch,sampling_params=[params[d['prompt_kind']] for d in ds])
                elapsed=time.monotonic()-started
                assert len(outputs)==len(ds)
                for detail,output in zip(ds,outputs):
                    gen=output.outputs[0]
                    parsed=official.try_parse_json(gen.text)
                    uncorrected=parsed.get('segments',[]) if isinstance(parsed,dict) else[]
                    error=None
                    try:
                        segments=official.validate_and_fix_segments(copy.deepcopy(uncorrected),detail['video_duration'])
                    except Exception as exc:
                        segments=[]
                        error=repr(exc)
                    result=dict(**detail,model=args.model,raw_output=gen.text,parsed=parsed,
                                uncorrected_segments=uncorrected,segments=segments,
                                finish_reason=gen.finish_reason,generated_tokens=len(gen.token_ids),
                                prompt_tokens=len(output.prompt_token_ids),batch_elapsed_seconds=elapsed,error=error)
                    stream.write(json.dumps(result,ensure_ascii=False,default=str)+'\n')
                    stream.flush()
                completed+=len(ds)
                print(json.dumps(dict(model=args.model,shard=args.shard,completed=completed,total=total)),flush=True)
    assert completed==total


if __name__=='__main__':
    main()
