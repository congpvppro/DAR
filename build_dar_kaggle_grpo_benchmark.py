"""Build the standalone, offline Kaggle GRPO benchmark notebook."""
import json
import hashlib
from pathlib import Path


def md(source):
    return {"cell_type": "markdown", "id": hashlib.sha1(source.encode()).hexdigest()[:8], "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "id": hashlib.sha1(source.encode()).hexdigest()[:8], "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


cells = [
    md("""# Benchmark checkpoint GRPO trên DAR (Kaggle, Internet off)

Chạy **Save & Run All** sau khi attach các input dưới đây và bật GPU. Notebook dùng đúng các dataset của notebook SFT:

- `vucongaaa/dar-r1`: mã DAR và wheelhouse offline.
- `vucongaaa/dar-annotation`: `train.jsonl` và **`test.jsonl`**. Nếu phiên bản dataset hiện tại chỉ có `train.jsonl`, cần publish thêm test split thật; không dùng train làm benchmark.
- `vucongaaa/vce-original-videos`: video gốc.
- Kaggle Model/Dataset chứa **full checkpoint GRPO** (config + safetensors). Sửa `GRPO_CHECKPOINT` ở cell kế tiếp. Nếu checkpoint thiếu processor/tokenizer, attach thêm checkpoint SFT hoặc base Qwen2.5-VL-3B để bổ sung các file đó.

Mặc định đánh giá toàn bộ test split. Prompt lấy từ `test.py`; 16 frame/clip, `100352` pixels/frame, greedy decoding, tối đa 4096 tokens. Báo cả chỉ số gốc của repo (có sửa output) và joint F1 nghiêm ngặt với video thiếu/lỗi tính vào mẫu số. Không chạy vLLM hoặc dịch vụ mạng. Không import Torch/Transformers trong kernel trước bước cài wheelhouse.
"""),
    code("""from pathlib import Path
import json

# SỬA đường dẫn này thành thư mục checkpoint GRPO full đã attach.
GRPO_CHECKPOINT = Path('/kaggle/input/models/OWNER/DAR-GRPO/transformers/default/1')
# Chỉ dùng để bổ sung tokenizer/processor nếu GRPO checkpoint thiếu chúng.
SFT_FALLBACK = Path('/kaggle/input/models/OWNER/DAR-SFT/transformers/default/1')
BASE_FALLBACK = Path('/kaggle/input/models/qwen-lm/qwen2.5-vl/transformers/3b-instruct/2')

CONFIG = {
    'grpo_checkpoint': str(GRPO_CHECKPOINT),
    'fallbacks': [str(SFT_FALLBACK), str(BASE_FALLBACK)],
    'repo': '/kaggle/input/datasets/vucongaaa/dar-r1/DAR',
    'wheelhouse': '/kaggle/input/datasets/vucongaaa/dar-r1/offline-wheelhouse-kaggle-py312-cu128',
    'train_jsonl': '/kaggle/input/datasets/vucongaaa/dar-annotation/train.jsonl',
    'test_jsonl': '/kaggle/input/datasets/vucongaaa/dar-annotation/test.jsonl',
    'video_dataset': '/kaggle/input/datasets/vucongaaa/vce-original-videos',
    'work': '/kaggle/working/dar_grpo_benchmark',
    'max_videos': 0,  # 0 = toàn bộ test; đặt số dương để smoke test.
    'frames': 16,
    'pixels': 100352,
    'max_new_tokens': 4096,
    'seed': 1234,
}
work = Path(CONFIG['work'])
work.mkdir(parents=True, exist_ok=True)
config_file = work / 'config.json'
if config_file.exists():
    previous = json.loads(config_file.read_text(encoding='utf-8'))
    if previous != CONFIG and (work / 'predictions.jsonl').exists():
        raise RuntimeError('Cấu hình đã đổi nhưng predictions.jsonl còn tồn tại. Dùng thư mục work mới để tránh trộn kết quả.')
config_file.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding='utf-8')
print('Config:', config_file)
"""),
    code("""%%bash
set -euo pipefail
WHEELHOUSE=/kaggle/input/datasets/vucongaaa/dar-r1/offline-wheelhouse-kaggle-py312-cu128
RUNTIME=/kaggle/working/dar_grpo_benchmark/python
test -d "$WHEELHOUSE"
test -f "$WHEELHOUSE/ms_swift-3.12.5-py3-none-any.whl"
python -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$RUNTIME"
mapfile -d '' WHEELS < <(find "$WHEELHOUSE" -maxdepth 1 -type f -name '*.whl' ! -name '*cp313*' ! -name 'packaging-26.3*' -print0 | sort -z)
test "${#WHEELS[@]}" -gt 0
python -m pip install --no-index --no-deps --upgrade --target "$RUNTIME" "${WHEELS[@]}"
export PYTHONPATH="$RUNTIME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python - <<'PY'
import torch, transformers, qwen_vl_utils, scipy
print('torch', torch.__version__, 'transformers', transformers.__version__, 'scipy', scipy.__version__)
assert torch.cuda.is_available(), 'Bật GPU trong Kaggle settings'
assert transformers.__version__ == '4.57.1'
PY
"""),
    code("""%%bash
set -euo pipefail
cat > /kaggle/working/dar_grpo_benchmark/benchmark_helpers.py <<'PY'
import ast
import copy
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment

TEST_PY = Path('/kaggle/input/datasets/vucongaaa/dar-r1/DAR/test.py')

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def official_functions():
    names = {'EMOTION_CANDIDATES', 'build_eval_prompt', 'calculate_iou',
             'evaluate_single_video', 'compute_overall_metrics',
             'merge_adjacent_same_emotion_segments', 'validate_and_fix_segments'}
    selected = []
    for node in ast.parse(TEST_PY.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            selected.append(node)
    scope = {'List': List, 'Dict': Dict}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(TEST_PY), 'exec'), scope)
    if not names <= scope.keys():
        raise RuntimeError(f'test.py thiếu hàm benchmark: {names - scope.keys()}')
    return scope

def annotation(row, video_root):
    basename = str(row.get('video') or row.get('video_path') or '').replace('\\\\', '/').split('/')[-1]
    duration = row.get('video_duration')
    if not basename or isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Annotation thiếu video hoặc duration hợp lệ')
    answers = [x['value'] for x in row.get('conversations', []) if x.get('from') == 'gpt']
    target = json.loads(answers[-1]) if answers else {'segments': row.get('gt_segments')}
    if not isinstance(target.get('segments'), list) or not target['segments']:
        raise ValueError(f'Annotation thiếu GT segments: {basename}')
    return {'video_id': Path(basename).stem, 'video_path': str(Path(video_root) / basename),
            'video_duration': float(duration), 'target': target}

def parse_json(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\\n', 1)[1].rsplit('```', 1)[0].strip()
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError('Output không phải JSON object')
    return obj

def valid_segments(obj, duration, emotions):
    if not isinstance(obj, dict) or set(obj) != {'segments'} or not isinstance(obj['segments'], list) or not obj['segments']:
        return []
    end, previous = 0., None
    try:
        for seg in obj['segments']:
            start, stop = seg['start_time'], seg['end_time']
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (start, stop)):
                return []
            if abs(start-end) > .051 or stop <= start or stop > duration + .051:
                return []
            if any(abs(x-round(x, 1)) > 1e-6 for x in (start, stop)):
                return []
            if seg['emotion'] not in emotions or seg['emotion'] == previous:
                return []
            if not isinstance(seg['reason'], str) or not seg['reason'].strip():
                return []
            end, previous = stop, seg['emotion']
        return obj['segments'] if abs(end-duration) <= .051 else []
    except (KeyError, TypeError):
        return []

def evaluate(manifest, predictions):
    off = official_functions()
    ids = [p['video_id'] for p in predictions]
    if len(ids) != len(set(ids)) or set(ids) - {r['video_id'] for r in manifest}:
        raise ValueError('Prediction có ID trùng/ngoài manifest')
    by_id = {p['video_id']: p for p in predictions}
    counts, repaired_results = [], {}
    for row in manifest:
        obj = by_id.get(row['video_id'], {}).get('dar')
        gt = row['target']['segments']
        pred = valid_segments(obj, row['video_duration'], off['EMOTION_CANDIDATES'])
        eligible = np.array([[off['calculate_iou'](a, b) >= .5 and a['emotion'] == b['emotion'] for b in gt]
                             for a in pred], dtype=int)
        if len(pred):
            i, j = linear_sum_assignment(eligible, maximize=True)
            tp = int(eligible[i, j].sum())
        else:
            tp = 0
        counts.append((tp, len(pred), len(gt), int(bool(pred)), int(len(pred) == len(gt))))
        try:
            repaired = off['validate_and_fix_segments'](copy.deepcopy(obj['segments']), row['video_duration']) if obj else []
            result = off['evaluate_single_video'](repaired, gt)
            if not all(np.isfinite(v) for v in result['ious']):
                raise ValueError('Non-finite repaired times')
        except (ValueError, TypeError, KeyError, AttributeError):
            result = off['evaluate_single_video']([], gt)
        repaired_results[row['video_id']] = result
    tp, p, g, valid, count = np.asarray(counts).sum(axis=0)
    return {'strict': {'joint_f1_at_05': float(2*tp/(p+g)) if p+g else 0.,
                       'coverage': float(valid/len(manifest)),
                       'count_accuracy': float(count/len(manifest)),
                       'predicted_segments': int(p), 'gt_segments': int(g)},
            'official_repaired': off['compute_overall_metrics'](repaired_results),
            'missing_videos': len(manifest)-len(predictions)}
PY
python -m py_compile /kaggle/working/dar_grpo_benchmark/benchmark_helpers.py
"""),
    code("""%%bash
set -euo pipefail
export PYTHONPATH=/kaggle/working/dar_grpo_benchmark/python:/kaggle/working/dar_grpo_benchmark
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
python - <<'PY'
import json
from pathlib import Path
from benchmark_helpers import annotation, read_jsonl

c = json.loads(Path('/kaggle/working/dar_grpo_benchmark/config.json').read_text(encoding='utf-8'))
repo, work = Path(c['repo']), Path(c['work'])
for p in (repo / 'test.py', Path(c['train_jsonl']),
          Path(c['test_jsonl']), Path(c['video_dataset'])):
    if not p.exists():
        raise FileNotFoundError(f'Thiếu input: {p}')
checkpoint = Path(c['grpo_checkpoint'])
if not (checkpoint / 'config.json').is_file() or not list(checkpoint.glob('*.safetensors')):
    raise FileNotFoundError(f'Cần full GRPO checkpoint có config.json và *.safetensors: {checkpoint}')
if (checkpoint / 'adapter_config.json').exists():
    raise ValueError('Đây là LoRA adapter; hãy export/merge thành full checkpoint trước khi benchmark.')
model_view = work / 'model'
model_view.mkdir(exist_ok=True)
for source in (checkpoint, *(Path(x) for x in c['fallbacks'] if Path(x).is_dir())):
    for p in source.iterdir():
        if source != checkpoint and (p.name.endswith('.safetensors') or p.name in {'config.json', 'model.safetensors.index.json'}):
            continue
        dest = model_view / p.name
        if not dest.exists() and not dest.is_symlink():
            dest.symlink_to(p, target_is_directory=p.is_dir())
for name in ('config.json', 'preprocessor_config.json'):
    if not (model_view / name).is_file():
        raise FileNotFoundError(f'Thiếu {name}; attach SFT/base model đầy đủ ở cell cấu hình')
if not any((model_view / n).exists() for n in ('tokenizer.json', 'tokenizer.model')):
    raise FileNotFoundError('Thiếu tokenizer; attach SFT/base model đầy đủ')

# Notebook SFT tìm 06720.mp4 rồi dùng thư mục chứa video làm VIDEO_ROOT.
matches = list(Path(c['video_dataset']).rglob('06720.mp4'))
if len(matches) != 1:
    raise RuntimeError(f'Cần đúng một 06720.mp4 trong video dataset; tìm thấy {len(matches)}')
video_root = matches[0].parent
train_ids = {Path(str(r['video'])).stem for r in read_jsonl(c['train_jsonl'])}
rows = [annotation(r, video_root) for r in read_jsonl(c['test_jsonl'])]
if not rows or len({r['video_id'] for r in rows}) != len(rows):
    raise ValueError('Test split rỗng hoặc có video ID trùng')
if train_ids & {r['video_id'] for r in rows}:
    raise ValueError('Train/test trùng video ID; không benchmark trên dữ liệu huấn luyện')
missing = [r['video_path'] for r in rows if not Path(r['video_path']).is_file()]
if missing:
    raise FileNotFoundError(f'Thiếu {len(missing)} video test; ví dụ: {missing[:3]}')
if c['max_videos']:
    rows = rows[:c['max_videos']]
manifest = work / 'manifest.jsonl'
if manifest.exists():
    old = read_jsonl(manifest)
    if old != rows and (work / 'predictions.jsonl').exists():
        raise RuntimeError('Manifest đổi; dùng thư mục work mới để tránh trộn dự đoán')
manifest.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\\n' for r in rows), encoding='utf-8')
print('GRPO checkpoint:', checkpoint)
print('Video root:', video_root)
print('Test videos:', len(rows), '(full split nếu max_videos=0)')
print('Manifest:', manifest)
PY
"""),
    code("""%%bash
set -euo pipefail
export PYTHONPATH=/kaggle/working/dar_grpo_benchmark/python:/kaggle/working/dar_grpo_benchmark
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
export FPS_MIN_FRAMES=16 FPS_MAX_FRAMES=16 VIDEO_MAX_PIXELS=100352
python - <<'PY'
import json, time
from pathlib import Path
from benchmark_helpers import official_functions, parse_json, read_jsonl
import hashlib

c = json.loads(Path('/kaggle/working/dar_grpo_benchmark/config.json').read_text(encoding='utf-8'))
work = Path(c['work'])
rows = read_jsonl(work / 'manifest.jsonl')
output = work / 'predictions.jsonl'
done = read_jsonl(output) if output.exists() else []
done_ids = [r['video_id'] for r in done]
if len(done_ids) != len(set(done_ids)) or set(done_ids) - {r['video_id'] for r in rows}:
    raise ValueError('predictions.jsonl chứa ID trùng/ngoài manifest; dùng work mới')
todo = [r for r in rows if r['video_id'] not in set(done_ids)]
print(f'Đã có {len(done)} dự đoán; còn {len(todo)} video', flush=True)
if todo:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed
    from qwen_vl_utils import process_vision_info
    assert torch.cuda.is_available()
    set_seed(c['seed'])
    model_path = str(work / 'model')
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map='cuda:0', local_files_only=True).eval()
    prompt_fn = official_functions()['build_eval_prompt']
    with output.open('a', encoding='utf-8') as stream:
        for i, row in enumerate(todo, 1):
            messages = [{'role': 'user', 'content': [
                {'type': 'video', 'video': row['video_path'], 'nframes': c['frames'],
                 'min_pixels': c['pixels'], 'max_pixels': c['pixels']},
                {'type': 'text', 'text': prompt_fn(row['video_duration'])}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            if not video_kwargs.get('fps'):
                raise RuntimeError(f'Thiếu sampled fps cho {row["video_id"]}')
            inputs = processor(text=[prompt], images=images, videos=videos,
                               padding=True, return_tensors='pt', **video_kwargs).to(model.device)
            start = time.monotonic()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=c['max_new_tokens'], do_sample=False)
            tokens = generated[:, inputs.input_ids.shape[1]:]
            raw = processor.batch_decode(tokens, skip_special_tokens=True)[0]
            identity = {k: row[k] for k in ('video_id', 'video_path', 'video_duration')}
            result = {'video_id': row['video_id'], 'input_sha256': hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                      'raw': raw, 'generated_tokens': int(tokens.shape[1]),
                      'seconds': round(time.monotonic() - start, 2)}
            try:
                result['dar'] = parse_json(raw)
            except (ValueError, IndexError) as exc:
                result['parse_error'] = str(exc)
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\\n')
            stream.flush()
            print(f'{len(done)+i}/{len(rows)} {row["video_id"]} {result["generated_tokens"]} tokens', flush=True)
            del inputs, generated, tokens, images, videos
print('Predictions:', output)
PY
"""),
    code("""%%bash
set -euo pipefail
export PYTHONPATH=/kaggle/working/dar_grpo_benchmark/python:/kaggle/working/dar_grpo_benchmark
python - <<'PY'
import json
from pathlib import Path
from benchmark_helpers import evaluate, read_jsonl

work = Path('/kaggle/working/dar_grpo_benchmark')
manifest = read_jsonl(work / 'manifest.jsonl')
predictions = read_jsonl(work / 'predictions.jsonl')
array, scores = evaluate(manifest, predictions)
scores['test_videos'] = len(manifest)
scores['predictions_written'] = len(predictions)
scores['parse_errors'] = sum('parse_error' in r for r in predictions)
scores['complete'] = len(predictions) == len(manifest)
scores['protocol'] = 'DAR test.py prompt and repaired metrics; strict one-to-one joint F1@IoU>=0.5 on all manifest videos'
result = work / 'metrics.json'
result.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(scores, ensure_ascii=False, indent=2))
print('Files:', work / 'predictions.jsonl', result)
if not scores['complete']:
    print('Benchmark chưa hoàn tất; chạy lại cell inference để tiếp tục.')
PY
"""),
]

notebook = {
    'cells': cells,
    'metadata': {
        'kaggle': {'accelerator': 'gpu', 'isGpuEnabled': True, 'isInternetEnabled': False,
                   'language': 'python', 'sourceType': 'notebook'},
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}
out = Path(__file__).with_name('dar_kaggle_grpo_benchmark.ipynb')
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
print(out)
