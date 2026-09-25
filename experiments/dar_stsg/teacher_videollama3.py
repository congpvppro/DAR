"""Frozen VideoLLaMA3 teacher; run in a separate Transformers 4.57.1 process.

API: https://github.com/DAMO-NLP-SG/VideoLLaMA3/blob/main/inference/example_videollama3.py
Use a full local snapshot including custom Python files, not the Image checkpoint.
"""
import hashlib
import importlib.metadata
import time
from pathlib import Path

from core import digest


class VideoLLaMA3Backend:
    def __init__(self, model, frames=16, load_in_4bit=False):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
        from transformers import image_utils
        if transformers.__version__ != '4.57.1':
            raise RuntimeError(
                'Use the separate teacher runtime: transformers==4.57.1; '
                f'loaded {transformers.__version__} from {transformers.__file__}. '
                'Rerun the offline teacher bootstrap and use its teacher_python/teacher_env.')
        if not torch.cuda.is_available():
            raise RuntimeError('VideoLLaMA3 teacher requires CUDA')
        # The snapshot uses VideoInput only as a type annotation. Transformers
        # 4.57.1 no longer exports it, while its ImageInput covers the same
        # accepted image/frame values used by this processor.
        if not hasattr(image_utils, 'VideoInput'):
            image_utils.VideoInput = image_utils.ImageInput
        root = Path(model)
        if not root.is_dir() or not list(root.glob('modeling*.py')):
            raise ValueError('Attach full local VideoLLaMA3 snapshot, including custom Python code')
        self.torch, self.frames = torch, frames
        options = dict(trust_remote_code=True, local_files_only=True,
                       torch_dtype=torch.float16, device_map='auto', attn_implementation='sdpa')
        if load_in_4bit:
            options['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_skip_modules=['vision_encoder', 'mm_projector', 'lm_head'])
        else:
            # Leave activation/KV-cache headroom on every visible GPU.
            available = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
            if sum(available) < 22 * 1024**3:
                raise RuntimeError('FP16 teacher needs more headroom: use two GPUs or --load-in-4bit')
            options['max_memory'] = {i: int(n * .8) for i, n in enumerate(available)}
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model, **options).eval()
        self.metadata = dict(model_config=self.model.config.to_dict(),
            snapshot_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(root.iterdir()) if p.suffix in ('.py', '.json')},
            device_map={k: str(v) for k, v in getattr(self.model, 'hf_device_map', {}).items()},
            precision='nf4-fp16' if load_in_4bit else 'fp16',
            sampling='uniform up to max_frames; native VideoLLaMA3 preprocessing')
        if load_in_4bit:
            self.metadata['bitsandbytes'] = importlib.metadata.version('bitsandbytes')

    def generate(self, row, prompt, max_tokens):
        import numpy as np
        from decord import VideoReader, cpu
        if not Path(row['video_path']).is_file():
            raise FileNotFoundError(row['video_path'])
        reader = VideoReader(row['video_path'], ctx=cpu(0), num_threads=2)
        fps = float(reader.get_avg_fps())
        if len(reader) < 1 or not np.isfinite(fps) or fps <= 0:
            raise RuntimeError('Video has no frames or invalid FPS')
        indices = np.linspace(0, len(reader) - 1, min(self.frames, len(reader)), dtype=int)
        frames = reader.get_batch(indices).asnumpy().transpose(0, 3, 1, 2)
        timestamps = (indices / fps).tolist()
        conversation = [{'role': 'user', 'content': [
            {'type': 'video', 'video': list(frames), 'num_frames': len(frames),
             'timestamps': timestamps},
            {'type': 'text', 'text': prompt}]}]
        inputs = self.processor(conversation=conversation, add_system_prompt=True,
                                add_generation_prompt=True, return_tensors='pt')
        pixel_hash = hashlib.sha256(inputs['pixel_values'].contiguous().numpy().tobytes()).hexdigest()
        device = self.model.get_input_embeddings().weight.device
        inputs = {k: v.to(device) if isinstance(v, self.torch.Tensor) else v for k, v in inputs.items()}
        inputs['pixel_values'] = inputs['pixel_values'].to(self.torch.float16)
        self.torch.cuda.synchronize()
        started = time.monotonic()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        self.torch.cuda.synchronize()
        # Custom generate uses inputs_embeds: output contains generated tokens only.
        return dict(raw=self.processor.batch_decode(output, skip_special_tokens=True)[0].strip(),
                    generated_tokens=output.shape[1], input_tokens=inputs['input_ids'].shape[1],
                    generation_seconds=time.monotonic() - started,
                    visual_tensor_sha256=pixel_hash, prompt_sha256=digest(prompt),
                    max_frames=self.frames, sampling='uniform', frame_indices=indices.tolist(),
                    timestamps=timestamps, source_fps=fps,
                    hit_token_limit=output.shape[1] >= max_tokens)
