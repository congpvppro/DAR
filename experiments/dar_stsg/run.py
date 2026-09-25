"""Generate train-only pseudo graphs/captions, or evaluate a fresh SFT adapter."""
import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from core import ROOT, dar_prompt, digest, evidence_prompt, parse_json, read_jsonl, validate_graph


class Backend:
    def __init__(self, model, adapter=None, frames=16, pixels=100352):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        if not torch.cuda.is_available():
            raise RuntimeError('This pilot needs a CUDA GPU; use the Kaggle notebook')
        self.torch, self.vision = torch, process_vision_info
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, torch_dtype=torch.float16, device_map='auto', attn_implementation='sdpa')
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.frames, self.pixels = frames, pixels

    def generate(self, row, prompt, max_tokens):
        if not Path(row['video_path']).is_file():
            raise FileNotFoundError(row['video_path'])
        messages = [{'role': 'user', 'content': [
            {'type': 'video', 'video': row['video_path'], 'nframes': self.frames,
             'min_pixels': self.pixels, 'max_pixels': self.pixels},
            {'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Qwen2.5 needs sampled fps, not source fps. The local utility supplies it
        # when return_video_metadata=False. Do not drop these video_kwargs.
        # False is the utility's default; omitting the keyword also supports
        # pre-Qwen3 versions which already return the required sampled fps.
        images, videos, kwargs = self.vision(messages, return_video_kwargs=True)
        if not kwargs.get('fps'):
            raise RuntimeError('Qwen2.5 video input is missing sampled fps')
        frame_hash = digest([self.frames, self.pixels])
        if videos:
            import hashlib
            frame_hash = hashlib.sha256(videos[0].numpy().tobytes()).hexdigest()
        inputs = self.processor(text=[text], images=images, videos=videos, padding=True,
                                return_tensors='pt', **kwargs).to(self.model.device)
        self.torch.cuda.synchronize()
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        self.torch.cuda.synchronize()
        tokens = output[:, inputs.input_ids.shape[1]:]
        result = dict(raw=self.processor.batch_decode(tokens, skip_special_tokens=True)[0],
                      generated_tokens=tokens.shape[1], input_tokens=inputs.input_ids.shape[1],
                      generation_seconds=time.monotonic() - started, frame_sha256=frame_hash,
                      sampled_fps=kwargs.get('fps'), prompt_sha256=digest(prompt),
                      hit_token_limit=tokens.shape[1] >= max_tokens)
        del inputs, output, tokens
        return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['teacher', 'predict'])
    for flag in ('model', 'input', 'output'):
        p.add_argument('--' + flag, required=True)
    p.add_argument('--adapter')
    p.add_argument('--backend', choices=['qwen', 'videollama3'], default='qwen')
    p.add_argument('--kinds', nargs='+', choices=['caption', 'stsg'], default=['caption', 'stsg'])
    p.add_argument('--load-in-4bit', action='store_true')
    p.add_argument('--frames', type=int, default=16)
    p.add_argument('--max-tokens', type=int, default=2048)
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() or out.with_suffix('.meta.json').exists():
        raise FileExistsError('Choose a new output; no silent retries or overwrites')
    rows = read_jsonl(args.input)
    if not rows or len({r['video_id'] for r in rows}) != len(rows):
        raise ValueError('Empty manifest or duplicate video IDs')
    if args.mode == 'teacher' and any(set(r) != {'video_id', 'video_path', 'video_duration'} for r in rows):
        raise ValueError('Teacher input must be label-free; use teacher_inputs.jsonl')
    from transformers import set_seed
    set_seed(args.seed)
    if args.backend == 'videollama3':
        if args.mode != 'teacher' or args.adapter:
            raise ValueError('VideoLLaMA3 is a frozen teacher only')
        from teacher_videollama3 import VideoLLaMA3Backend
        backend = VideoLLaMA3Backend(args.model, args.frames, args.load_in_4bit)
    else:
        if args.load_in_4bit:
            raise ValueError('4-bit is currently implemented for VideoLLaMA3 teacher only')
        backend = Backend(args.model, args.adapter, args.frames)
    out.parent.mkdir(parents=True, exist_ok=True)
    versions = {x: importlib.metadata.version(x) for x in ('torch', 'transformers', 'accelerate')}
    meta = dict(config=vars(args), input_sha256=digest(rows), packages=versions,
                code_sha256={p.name: digest(p.read_text(encoding='utf-8')) for p in [Path(__file__), Path(__file__).with_name('core.py'), ROOT / 'test.py']})
    if args.backend == 'videollama3':
        meta['teacher'] = backend.metadata
        backend_path = Path(__file__).with_name('teacher_videollama3.py')
        meta['code_sha256'][backend_path.name] = digest(backend_path.read_text(encoding='utf-8'))
    out.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    with out.open('x', encoding='utf-8') as stream:
        for i, row in enumerate(rows):
            identity = {k: row[k] for k in ('video_id', 'video_path', 'video_duration')}
            result = dict(video_id=row['video_id'], input_sha256=digest(identity), calls={})
            for kind in (args.kinds if args.mode == 'teacher' else ('dar',)):
                prompt = dar_prompt(row['video_duration']) if kind == 'dar' else evidence_prompt(row['video_duration'], kind)
                try:
                    call = backend.generate(identity, prompt, args.max_tokens)
                    result['calls'][kind] = call
                    obj = parse_json(call['raw'])
                    if kind == 'stsg':
                        validate_graph(obj, row['video_duration'])
                    result[kind] = obj
                except (ValueError, KeyError, TypeError) as exc:
                    result[kind + '_error'] = str(exc)
                # OOM, corrupt/missing video and model errors abort: do not silently
                # turn a broken runtime into an apparent poor benchmark result.
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            print(f'{i + 1}/{len(rows)} {row["video_id"]}', flush=True)


if __name__ == '__main__':
    main()
