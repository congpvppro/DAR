"""Run paired interventions with unchanged released prompt and evaluator cleanup."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'DAR'))
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['dar','qwen'], required=True)
    ap.add_argument('--shard', type=int, required=True)
    args = ap.parse_args()
    import torch
    import transformers
    import vllm
    import test as official
    from metrics import transform_frames
    torch.set_num_threads(2)
    manifest = json.loads((OUT/'manifest.json').read_text())
    videos = manifest['videos'][args.shard::2]
    model = ROOT/'model'/('DAR-R1' if args.model == 'dar' else 'Qwen2.5-VL-3B-Instruct')
    result_path = OUT/f'{args.model}-shard{args.shard}.jsonl'
    done = set()
    if result_path.exists():
        done = {(r['video_id'],r['condition']) for r in map(json.loads,result_path.read_text().splitlines())}
    llm = vllm.LLM(model=str(model), trust_remote_code=False, gpu_memory_utilization=.7,
                   tensor_parallel_size=1, max_model_len=12288, seed=20260907,
                   enforce_eager=True, max_num_seqs=8)
    processor = transformers.AutoProcessor.from_pretrained(str(model))
    sampling = vllm.SamplingParams(temperature=0, max_tokens=4096, seed=20260907)
    metadata = dict(model=str(model), shard=args.shard, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    torch=torch.__version__, transformers=transformers.__version__,vllm=vllm.__version__,
                    nframes=16, min_pixels=100352, max_pixels=100352, max_tokens=4096,
                    temperature=0, seed=20260907, max_model_len=12288)
    (OUT/f'{args.model}-shard{args.shard}-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with result_path.open('a') as stream:
        for start in range(0,len(videos),2):
            inputs, metas = [], []
            for row in videos[start:start+2]:
                todo = [c for c in manifest['conditions'] if (row['video_id'],c) not in done]
                if not todo:
                    continue
                messages = official.make_messages(row['video_path'],row['video_duration'])
                messages[0]['content'][0].update(nframes=16,min_pixels=100352,max_pixels=100352)
                base = official.build_input(processor,messages)
                tensor, video_metadata = base['multi_modal_data']['video'][0]
                frames = tensor.numpy()
                for condition in todo:
                    transformed = transform_frames(frames, condition)
                    item = dict(prompt=base['prompt'],multi_modal_data={'video':[(torch.from_numpy(transformed),copy.deepcopy(video_metadata))]},
                                mm_processor_kwargs=copy.deepcopy(base['mm_processor_kwargs']))
                    inputs.append(item)
                    metas.append(dict(video_id=row['video_id'],condition=condition,video_duration=row['video_duration'],
                                      stratum=row['stratum'],input_shape=list(transformed.shape),
                                      input_sha256=hashlib.sha256(transformed.tobytes()).hexdigest()))
            if not inputs:
                continue
            started = time.monotonic()
            outputs = llm.generate(inputs,sampling_params=sampling)
            elapsed = time.monotonic()-started
            for meta, output in zip(metas,outputs):
                generated = output.outputs[0]
                raw = generated.text
                parsed = official.try_parse_json(raw)
                uncorrected = parsed.get('segments',[]) if isinstance(parsed,dict) else []
                error = None
                try:
                    normalized = official.validate_and_fix_segments(copy.deepcopy(uncorrected),meta['video_duration'])
                except Exception as exc:
                    normalized = []
                    error = repr(exc)
                result = dict(**meta,model=args.model,raw_output=raw,uncorrected_segments=uncorrected,
                              segments=normalized,finish_reason=generated.finish_reason,generated_tokens=len(generated.token_ids),
                              prompt_tokens=len(output.prompt_token_ids),batch_elapsed_seconds=elapsed,error=error)
                stream.write(json.dumps(result,ensure_ascii=False)+'\n')
                stream.flush()
            print(json.dumps(dict(model=args.model,shard=args.shard,completed_videos=min(start+2,len(videos)),total=len(videos))),flush=True)


if __name__ == '__main__':
    main()
