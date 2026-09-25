"""Rebuild the portable Kaggle notebook from reviewed source files."""
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md('''# DAR + STSG: train lại SFT trên Kaggle

**Adapt vào dữ liệu và nhiệm vụ phụ SFT.** Mỗi video có target DAR gốc và
target STSG/caption. Mọi nhánh bắt đầu độc lập từ **Qwen2.5-VL-3B-Instruct**.
Không cần checkpoint DAR-SFT/GRPO cũ. Inference dùng một lượt video → DAR JSON;
graph không được đưa vào input benchmark. GRPO không nằm trong pilot này.

Mặc định chỉ chạy `ARM='stsg'`: teacher VideoLLaMA3-7B sinh graph rồi student học SFT.
Đổi `ARM='caption'` hoặc `'stsg_no_links'` để chạy riêng; không chạy lại baseline.
So sánh chính: STSG − caption. Đây là prototype lấy cảm hứng STEP/VoT, không phải
MotionEpic hay toàn bộ VoT. Graph text không có bbox/tracker và chưa được xác minh
thực tế chỉ vì pass schema.

Notebook chứa code cần thiết. Chọn GPU Kaggle, attach base model, train/test JSONL,
video và offline wheelhouse như notebook GRPO hiện tại. `smoke` đã xác minh kỹ thuật;
`full` chạy toàn bộ train.jsonl trừ 64 video dev. Chưa có kết luận cải thiện.''')

code('''from pathlib import Path
import os, sys, subprocess, json

# Sửa các đường dẫn theo Kaggle Inputs. BASE_MODEL phải là backbone Instruct gốc.
TEACHER_MODEL = Path('/kaggle/input/datasets/vucongaaa/videollama3-7b')
TEACHER_4BIT = False  # The RTX PRO 6000 has sufficient VRAM for the FP16 teacher.
TEACHER_GPUS = '0'  # Set '0,1' with TEACHER_4BIT=False for two GPUs.
TEACHER_WHEELHOUSE = None  # Auto-detect transformers 4.57.1 under /kaggle/input; set a Path to override.
TEACHER_VENV = Path('/tmp/dar_videollama3_venv')
BASE_MODEL = Path('/kaggle/input/models/qwen-lm/qwen2.5-vl/transformers/3b-instruct/2')
TRAIN_JSONL = Path('/kaggle/input/datasets/vucongaaa/dar-annotation/train.jsonl')
TEST_JSONL = Path('/kaggle/input/datasets/vucongaaa/dar-annotation/test.jsonl')
VIDEO_ROOT = Path('/kaggle/input/datasets/vucongaaa/vce-original-videos/videos')
WHEELHOUSE = Path('/kaggle/input/datasets/vucongaaa/dar-r1/offline-wheelhouse-kaggle-py312-cu128')
RUNTIME = Path('/tmp/dar_stsg_python')
PROJECT = Path('/kaggle/working/dar_stsg_project')
MODE = 'full'  # Entire train.jsonl except the frozen 64-video dev split.
assert MODE in ('smoke', 'pilot', 'full')
TRAIN_SIZE, DEV_SIZE = ((8, 4) if MODE == 'smoke' else
                        (256, 64) if MODE == 'pilot' else (-1, 64))
NUM_TRAIN_EPOCHS = 0.5
ARM = 'stsg'  # Choose stsg, caption, or stsg_no_links; never runs baseline automatically.
assert ARM in ('stsg', 'caption', 'stsg_no_links')
ARMS = [ARM]
WORK = Path('/kaggle/working/dar_videollama3_full_epoch05_' + MODE + '_' + ARM)
MODEL_OUTPUT_ROOT = Path('/kaggle/temp/dar_stsg_models_' + MODE + '_' + ARM)
LEARNING_RATE = 1e-5
GRADIENT_ACCUMULATION_STEPS = 32  # Matches dar_kaggle_sft_clean.ipynb.
SEED = 1234
for path in (TEACHER_MODEL, BASE_MODEL, TRAIN_JSONL, TEST_JSONL, VIDEO_ROOT, WHEELHOUSE):
    assert path.exists(), f'Sửa đường dẫn chưa tồn tại: {path}'
assert (TEACHER_MODEL / 'config.json').exists()
assert list(TEACHER_MODEL.glob('modeling*.py')), 'Attach full teacher snapshot with custom code'
assert (BASE_MODEL / 'config.json').exists()
assert list(BASE_MODEL.glob('*.safetensors')), 'Cần full base weights, không phải LoRA adapter'

# The Kaggle model mount may omit preprocessor_config.json. Keep the read-only
# mount intact and make a lightweight local view of its files for ms-swift.
BASE_MODEL_SOURCE = BASE_MODEL
BASE_MODEL = Path('/tmp/dar_stsg_base_model')
BASE_MODEL.mkdir(parents=True, exist_ok=True)
for source in BASE_MODEL_SOURCE.iterdir():
    if source.name != 'preprocessor_config.json':
        target = BASE_MODEL / source.name
        if not target.exists():
            target.symlink_to(source, target_is_directory=source.is_dir())
preprocessor_source = BASE_MODEL_SOURCE / 'preprocessor_config.json'
preprocessor = (json.loads(preprocessor_source.read_text()) if preprocessor_source.exists()
                else dict(min_pixels=3136, max_pixels=12845056, patch_size=14,
                          temporal_patch_size=2, merge_size=2,
                          image_mean=[0.48145466, 0.4578275, 0.40821073],
                          image_std=[0.26862954, 0.26130258, 0.27577711]))
preprocessor['image_processor_type'] = 'Qwen2VLImageProcessor'
preprocessor.setdefault('processor_class', 'Qwen2_5_VLProcessor')
(BASE_MODEL / 'preprocessor_config.json').write_text(json.dumps(preprocessor))
print('Student model view:', BASE_MODEL, 'preprocessor:', preprocessor)
''')

md('''Runtime riêng, dùng wheelhouse có sẵn; không thay NumPy/Torch đang load trong
kernel. Tất cả tác vụ ML chạy ở subprocess. Không thêm `ms-swift/` checkout 4.0
vào PYTHONPATH của runtime 3.12.5.''')
code((HERE / 'notebook_student_runtime.py').read_text(encoding='utf-8'))
code('''run(sys.executable, '-c',
    "import sys; from transformers import AutoProcessor; "
    "p = AutoProcessor.from_pretrained(sys.argv[1], local_files_only=True); "
    "print('Student processor preflight:', type(p).__name__, type(p.image_processor).__name__)",
    BASE_MODEL)
''')

md('''Teacher riêng: VideoLLaMA3-7B, Transformers 4.57.1; student dùng 4.57.1.
Notebook chạy **Internet Off**. `TEACHER_WHEELHOUSE=None` tự tìm wheel 4.57.1
trong Kaggle Inputs; dataset runtime riêng cung cấp ffmpeg-python/future.
Nếu thiếu wheel, pip dừng tại bước kiểm tra offline, không thử kết nối PyPI.
Giữ CUDA Torch có sẵn của Kaggle; không cài lại Torch CPU từ wheelhouse.
Mỗi lần chạy bootstrap tạo thư mục runtime mới, giữ nguyên các runtime cũ.
Teacher tạo venv bằng `--without-pip`, dùng pip kernel quản lý qua `--python`
([pip docs](https://pip.pypa.io/en/stable/topics/python-option/), cần pip >=22.3).
Đường dẫn package teacher được ưu tiên trước package Kaggle; không xoá đường dẫn
hỗ trợ của Kaggle. Kiểm tra phiên bản và đường dẫn import trước khi chạy model.
Full teacher snapshot phải có weights, tokenizer, configs và custom Python files.
Teacher FP16 chạy trên GPU đã chọn; NF4 chỉ là tùy chọn khi có wheel tương thích.''')
code((HERE / 'notebook_teacher_runtime.py').read_text(encoding='utf-8'))

files = {f'experiments/dar_stsg/{name}': (HERE / name).read_text(encoding='utf-8')
         for name in ('core.py', 'prepare.py', 'run.py', 'evaluate.py', 'train.sh', 'test_pipeline.py', 'teacher_videollama3.py')}
files['test.py'] = (ROOT / 'test.py').read_text(encoding='utf-8')
md('''Cell sau giải nén code đã nhúng vào notebook; không phụ thuộc bản DAR cũ trong
Kaggle Inputs. Các file hiện có khác nội dung sẽ làm cell dừng để tránh trộn version.''')
code('FILES = ' + repr(files) + '''
for name, content in FILES.items():
    dest = PROJECT / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        assert dest.read_text(encoding='utf-8') == content, f'Code khác version: {dest}'
    else:
        dest.write_text(content, encoding='utf-8')
SCRIPTS = PROJECT / 'experiments/dar_stsg'
WORK.mkdir(parents=True, exist_ok=True)
run(sys.executable, '-m', 'unittest', 'discover', '-s', SCRIPTS, '-p', 'test_pipeline.py', '-v')
''')

md('''Tách dev từ **train**, kiểm tra overlap với test. Không dùng official test để
chọn hyperparameters. File teacher chỉ chứa video ID/path/duration, không có target.
Lỗi file/decoder/GPU sẽ dừng; không lặng lẽ thay video.''')
code('''run(sys.executable, SCRIPTS / 'prepare.py', 'split', '--train', TRAIN_JSONL,
    '--test', TEST_JSONL, '--video-root', VIDEO_ROOT, '--train-size', TRAIN_SIZE,
    '--dev-size', DEV_SIZE, '--output', WORK / 'split')
for name in ('teacher_inputs.jsonl', 'dev.jsonl'):
    rows = [json.loads(x) for x in (WORK / 'split' / name).read_text().splitlines()]
    missing = [r['video_path'] for r in rows if not Path(r['video_path']).is_file()]
    assert not missing, f'VIDEO_ROOT phải chứa trực tiếp các basename: {missing[:5]}'
''')

code('''kind = 'caption' if ARM == 'caption' else 'stsg'
teacher_command = [teacher_python, SCRIPTS / 'run.py', 'teacher', '--backend', 'videollama3',
    '--model', TEACHER_MODEL, '--kinds', kind, '--frames', '16',
    '--input', WORK / 'split/teacher_inputs.jsonl', '--output', WORK / 'evidence.jsonl']
if TEACHER_4BIT:
    teacher_command.append('--load-in-4bit')
subprocess.run(list(map(str, teacher_command)), check=True, env=teacher_env)
evidence = [json.loads(x) for x in (WORK / 'evidence.jsonl').read_text().splitlines()]
for item in evidence:
    call = item.get('calls', {}).get(kind, {})
    print('teacher evidence', item['video_id'],
          'error=', item.get(kind + '_error'),
          'parsed=', kind in item,
          'tokens=', call.get('generated_tokens'),
          'hit_limit=', call.get('hit_token_limit'),
          'raw_prefix=', repr(call.get('raw', '')[:600]), flush=True)
run(sys.executable, SCRIPTS / 'prepare.py', 'build', '--split', WORK / 'split',
    '--evidence', WORK / 'evidence.jsonl', '--output', WORK / 'arms', '--arms', ARM)
print((WORK / 'arms/build.json').read_text())
print(json.dumps(evidence[0], ensure_ascii=False, indent=2))
''')
md('''Trước pilot lớn, đối chiếu graph/caption của một mẫu train được chọn trước với
video. Schema pass chỉ kiểm tra cấu trúc. Chạy từng arm riêng có thể loại các video khác nhau;
đối chiếu accepted IDs trong build.json và cấu hình baseline trước khi so sánh. Nếu đa số
graph lỗi hoặc bịa sự kiện, cải thiện teacher/data trước khi tăng compute.

Train **full SFT** cho LLM + aligner, đóng băng vision encoder như baseline DAR.
BF16, LR=1e-5, AdamW, weight_decay=0, max_length=8192, gradient accumulation=32.
Train 0.5 epoch trên dataset DAR + auxiliary đã build, không giới hạn max_steps.
Pilot vẫn dùng tập con; mỗi video có 2 mẫu nên cùng số epoch chưa bảo đảm cùng ngân sách baseline.
Cần GPU hỗ trợ BF16 và đủ VRAM cho full SFT (baseline notebook ghi RTX 6000 96 GB).
Teacher 4-bit không làm student full SFT tốn ít VRAM hơn. Nếu OOM,
ghi lại cấu hình mới và chạy lại tất cả arm nhất quán; chưa xác nhận VRAM trên Kaggle.''')
code('''run(sys.executable, '-c', "import torch; assert torch.cuda.is_bf16_supported(), 'Full SFT baseline requires BF16-capable GPU'")
import shutil
MODEL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
for location in (Path('/kaggle/working'), MODEL_OUTPUT_ROOT):
    usage = shutil.disk_usage(location)
    print('Disk', location, 'free GiB:', round(usage.free / 1024**3, 2), flush=True)
for arm in ARMS:
    run('bash', SCRIPTS / 'train.sh', extra_env={
        'BASE_MODEL': str(BASE_MODEL), 'DATA_DIR': str(WORK / 'arms'),
        'OUTPUT_ROOT': str(MODEL_OUTPUT_ROOT), 'ARM': arm,
        'NUM_TRAIN_EPOCHS': str(NUM_TRAIN_EPOCHS), 'SEED': str(SEED),
        'LEARNING_RATE': str(LEARNING_RATE),
        'GRADIENT_ACCUMULATION_STEPS': str(GRADIENT_ACCUMULATION_STEPS)})
''')

md('''Inference một lượt, không dùng graph teacher; sampled fps được truyền cho
Qwen2.5. Lưu raw text, frame hash và token/time metadata. Không retry chọn output đẹp.''')
code('''predictions = []
model_checkpoints = {}
for arm in ARMS:
    checkpoints = list((MODEL_OUTPUT_ROOT / f'{arm}-seed{SEED}').rglob('config.json'))
    checkpoints = [p.parent for p in checkpoints
                   if p.parent.name.startswith('checkpoint-') and p.parent.name.split('-')[-1].isdigit()]
    assert checkpoints, f'No full checkpoint saved for {arm}'
    checkpoints = [max(checkpoints, key=lambda p: int(p.name.split('-')[-1]))]
    assert list(checkpoints[0].glob('*.safetensors')), 'Missing full checkpoint weights'
    assert not (checkpoints[0] / 'adapter_config.json').exists(), 'Expected full SFT checkpoint'
    model_checkpoints[arm] = checkpoints[0]
    output = WORK / f'{arm}-predictions.jsonl'
    run(sys.executable, SCRIPTS / 'run.py', 'predict', '--model', checkpoints[0],
        '--input', WORK / 'split/dev.jsonl', '--output', output)
    predictions.append(f'{arm}={output}')
run(sys.executable, SCRIPTS / 'evaluate.py', '--manifest', WORK / 'split/dev.jsonl',
    '--predictions', *predictions, '--reference', ARM, '--output', WORK / 'comparison.json')
print((WORK / 'comparison.json').read_text())
if MODE == 'full':
    source = model_checkpoints[ARM]
    destination = WORK / 'final_checkpoint'
    size = sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
    free = shutil.disk_usage(WORK).free
    print('Final checkpoint GiB:', round(size / 1024**3, 2),
          'working free GiB:', round(free / 1024**3, 2), flush=True)
    assert free > size + 512 * 1024**2, 'Not enough persistent space for final checkpoint'
    shutil.copytree(source, destination)
    assert list(destination.glob('*.safetensors'))
    print('Persistent full checkpoint:', destination, flush=True)
''')
md('''Kết quả `smoke` chỉ kiểm tra kỹ thuật. Với `pilot`, xem joint F1@0.5 strict,
coverage. Một arm chỉ có điểm tuyệt đối, chưa có paired CI hoặc kết luận cải thiện.
Ghép predictions trên cùng dev manifest sau để so sánh. Đơn vị là fraction (0.02 = 2 điểm phần trăm).
`official_repaired` có policy sửa output của repo, cần báo riêng. CI bootstrap theo
video không bao phủ training-seed variance. Sau pilot: lặp 3 seeds, khóa cấu hình,
đánh giá official DAR test rồi mới cân nhắc GRPO lại. Đừng chọn metric thắng sau khi xem.

Nguồn: [STEP §3–4](https://arxiv.org/html/2412.00161v2),
[Video-of-Thought §3–5](https://arxiv.org/html/2501.03230v1).
Không có graph encoder, STEP iterative QRA self-training hoặc VoT verification loop
trong v0 này. Không có kết quả benchmark thật cho tới khi các cell GPU được chạy.''')

nb.cells = cells
nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
               'language_info': {'name': 'python'}}
nbf.validate(nb)
nbf.write(nb, ROOT / 'dar_kaggle_stsg_pilot.ipynb')
print(f'Wrote {len(cells)} cells')
