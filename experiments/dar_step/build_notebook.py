"""Copy the existing pilot and replace only graph preparation/runtime cells."""
import copy
import hashlib
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
source = ROOT / 'dar_kaggle_stsg_pilot.ipynb'
before = hashlib.sha256(source.read_bytes()).hexdigest()
nb = copy.deepcopy(nbf.read(source, as_version=4))
assert len(nb.cells) == 17, 'Original notebook structure changed; review cell mappings'

nb.cells[0].source = '''# DAR + STEP-inspired graph pilot — VILA1.5-3B

Bản copy của `dar_kaggle_stsg_pilot.ipynb`. Đơn vị train vẫn là **video gốc**.

`video → PySceneDetect → scene clips → clustering keyframe → frame graphs →
dynamic merging trong clip → cross-clip bridging → graph toàn video → auxiliary SFT`

Mỗi video có 2 mẫu: video → DAR JSON, và keyframes + timestamps → graph JSON.
Hai tác vụ dùng chung Qwen2.5-VL-3B; chưa có graph encoder/head riêng.
Inference DAR vẫn dùng video, không cần teacher hoặc graph.

**VILA1.5-3B** được chọn cho pilot vì hỗ trợ multi-image và phù hợp nhiều lần gọi
parse/đối chiếu ảnh. STEP dùng VILA3B và VideoChat2-Mistral7B, nhưng không cung cấp
so sánh trực tiếp chất lượng graph teacher. Không suy ra VILA tốt hơn từ kích thước.
Đây là adaptation cách dựng graph; chưa tái tạo QRA/CoT self-training của STEP.
Các prompt trong notebook do dự án viết, không phải prompt nguyên văn từ paper.

Mặc định `smoke`: 8 train / 4 dev. Chưa chạy GPU với checkpoint thật cho bản này.
Teacher chỉ nhận ID/path/duration train. Scene cut không được coi là ranh giới cảm xúc.'''

config = nb.cells[1].source
start, end = config.index('TEACHER_MODEL ='), config.index('BASE_MODEL =')
config = config[:start] + '''TEACHER_MODEL = Path('/kaggle/input/vila15-3b/VILA1.5-3b')
TEACHER_BUNDLE = Path('/kaggle/input/dar-step-vila-runtime')
TEACHER_GPUS = '0'  # Native SigLIP teacher uses one GPU, FP16.
SCENE_THRESHOLD = 27.0
MIN_SCENE_SECONDS = 0.5
# Hecate: all frames; global K=min(N//2, scene count); one frame/subshot.
MAX_KEYFRAMES = 24  # Video-dependent count; exceeding this is a recorded rejection.
MAX_LENGTH = 16384  # Preflight encodes ALL SFT rows before training.
''' + config[end:]
config = config.replace("RUNTIME = Path('/tmp/dar_stsg_python')", "RUNTIME = Path('/tmp/dar_step_student_python')")
config = config.replace("PROJECT = Path('/kaggle/working/dar_stsg_project')", "PROJECT = Path('/kaggle/working/dar_step_hecate_project')")
config = config.replace("MODE = 'full'", "MODE = 'smoke'")
config = config.replace("ARM = 'stsg'  # Choose stsg, caption, or stsg_no_links; never runs baseline automatically.",
                        "ARM = 'stsg'  # Or stsg_no_links: same observations, temporal/reference edges removed.")
config = config.replace("('stsg', 'caption', 'stsg_no_links')", "('stsg', 'stsg_no_links')")
config = config.replace('dar_videollama3_full_epoch05_', 'dar_step_hecate_vila_epoch05_')
config = config.replace('dar_stsg_models_', 'dar_step_hecate_models_')
config = config.replace('for path in (TEACHER_MODEL, BASE_MODEL,', 'for path in (TEACHER_MODEL, TEACHER_BUNDLE, BASE_MODEL,')
config = config.replace("assert list(TEACHER_MODEL.glob('modeling*.py')), 'Attach full teacher snapshot with custom code'", '''for component in ('llm', 'vision_tower', 'mm_projector'):
    assert (TEACHER_MODEL / component / 'config.json').exists(), f'Missing VILA component: {component}'
    assert list((TEACHER_MODEL / component).glob('*.safetensors')), f'Missing weights: {component}'
assert (TEACHER_BUNDLE / 'manifest.json').exists(), 'Run prepare_runtime.py on an online machine first' ''')
config = config.replace("Path('/tmp/dar_stsg_base_model')", "Path('/tmp/dar_step_base_model')")
config = config.replace('# Entire train.jsonl except the frozen 64-video dev split.', '# First validate smoke; then pilot/full.')
nb.cells[1].source = config
nb.cells[5].source = '''## Runtime teacher riêng và dữ liệu cần attach

Giữ student runtime của pilot: Transformers 4.57.1 / ms-swift 3.12.5.
Teacher dùng native VILA revision `6b941da19e31ddfdfaa60160908ccf0978d96615`,
Transformers 4.36.2, FP16, eager attention. Không cài patch training FlashAttention.
Imports của các vision tower không dùng được chuyển sang lazy import; nguồn và diff
được lưu trong `vila/dar-runtime.json`. CUDA Torch của Kaggle được giữ nguyên.
Stack Torch mới của Kaggle cần được kiểm tra bằng smoke, chưa có kiểm chứng GPU thật.

Chuẩn bị trên máy có Internet:
```
python experiments/dar_step/prepare_runtime.py --output experiments/dar_step/runtime-bundle
```
Attach thư mục output thành dataset `dar-step-vila-runtime`. Script chỉ tải code/wheels,
không tải weights. Attach riêng snapshot đầy đủ
[`Efficient-Large-Model/VILA1.5-3b`](https://huggingface.co/Efficient-Large-Model/VILA1.5-3b)
với `llm/`, `vision_tower/`, `mm_projector/` và tokenizer trong `llm/`.
Sửa 2 đường dẫn teacher trong config. Notebook chạy Internet Off, bundle Python 3.12/Linux.

Không dùng runtime VideoLLaMA3 cũ cho VILA. Cell dưới kiểm tra hash bundle và native imports.'''
nb.cells[6].source = (HERE / 'notebook_teacher_runtime.py').read_text(encoding='utf-8')

legacy = ROOT / 'experiments/dar_stsg'
files = {f'experiments/dar_stsg/{name}': (legacy / name).read_text(encoding='utf-8')
         for name in ('core.py', 'prepare.py', 'run.py', 'evaluate.py', 'test_pipeline.py', 'teacher_videollama3.py')}
files['test.py'] = (ROOT / 'test.py').read_text(encoding='utf-8')
for name in ('common.py', 'hecate_keyframes.py', 'HECATE_NOTICE.md', 'HECATE_LICENSE', 'frames.py', 'graph.py', 'prompts.py', 'teacher_vila.py', 'pipeline.py',
             'prepare.py', 'preflight_sft.py', 'train.sh', 'test_step.py'):
    files[f'experiments/dar_step/{name}'] = (HERE / name).read_text(encoding='utf-8')
nb.cells[8].source = 'FILES = ' + repr(files) + '''
for name, content in FILES.items():
    destination = PROJECT / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        assert destination.read_text(encoding='utf-8') == content, f'Code version mismatch: {destination}'
    else:
        destination.write_text(content, encoding='utf-8')
SCRIPTS = PROJECT / 'experiments/dar_stsg'
STEP_SCRIPTS = PROJECT / 'experiments/dar_step'
WORK.mkdir(parents=True, exist_ok=True)
subprocess.run([teacher_python, '-m', 'unittest', 'discover', '-s', str(STEP_SCRIPTS),
                '-p', 'test_step.py', '-v'], check=True, env=teacher_env)
'''
nb.cells[11].source = '''teacher_command = [teacher_python, STEP_SCRIPTS / 'pipeline.py',
    '--model', TEACHER_MODEL, '--input', WORK / 'split/teacher_inputs.jsonl',
    '--output', WORK / 'teacher', '--threshold', SCENE_THRESHOLD,
    '--min-scene-seconds', MIN_SCENE_SECONDS, '--max-frames', MAX_KEYFRAMES, '--seed', SEED]
subprocess.run(list(map(str, teacher_command)), check=True, env=teacher_env)
evidence_path = WORK / 'teacher/evidence.jsonl'
evidence = [json.loads(x) for x in evidence_path.read_text().splitlines()]
for item in evidence:
    print(item['video_id'], 'error=', item.get('error'), 'calls=', item['calls'],
          'keyframes=', len(item.get('sampling', {}).get('frames', [])))
run(sys.executable, STEP_SCRIPTS / 'prepare.py', '--split', WORK / 'split',
    '--evidence', evidence_path, '--output', WORK / 'arms', '--arm', ARM)
print((WORK / 'arms/build.json').read_text())
valid = [e for e in evidence if 'graph' in e]
assert valid, 'No graph passed validation'
print(json.dumps(valid[0]['graph'], ensure_ascii=False, indent=2))
# Review the first frozen train sample, including a failure if it failed (no cherry-picking).
first = evidence[0]
print('Review video:', first['video_id'], 'status:', first.get('error', 'accepted'))
if 'sampling' in first:
    from IPython.display import display, Image
    for frame in first['sampling']['frames']:
        print(frame['id'], frame['scene'], frame['time'])
        display(Image(filename=frame['path'], width=320))
'''
nb.cells[12].source = '''## Kiểm tra dữ liệu và train

PySceneDetect ContentDetector xác định **khoảng scene**, không re-encode video.
Hecate source port: xét mọi frame (ảnh phân tích cạnh dài tối đa 160px), histogram
HSV/cạnh 2000 chiều, global k-means++ với K=min(N//2, số scene), tối thiểu 1.
Tách subshot theo các đoạn nhãn cụm liên tiếp trong từng scene; lấy frame ít thay đổi
nhất/subshot. Không thêm frame đầu/cuối. Scene duy nhất có thể chỉ cho một keyframe.
Nguồn, giấy phép và các sửa lỗi biên/uint8 được nhúng trong HECATE_NOTICE.md/HECATE_LICENSE.
Đây là port bước keyframe, không chạy toàn bộ hệ thống thumbnail/quality filter của Hecate.
Video không có cut vẫn là một scene. Số frame không còn cố định 16 cho teacher.
Decoder timestamps được kiểm tra với CFR; dữ liệu VFR cần chuẩn hóa riêng.
Vượt budget, JSON sai hoặc xung đột danh tính được ghi rõ và loại khỏi tập train.

Parse object/attribute/relation → một lượt visual verification → merge các keyframe
kề nhau trong scene → scene event → xác minh event → đối chiếu đại diện entity trên
mọi cặp scene. Motion chỉ là thay đổi nhìn thấy giữa hai keyframe, không phải trajectory.
Frame instances và attributes theo thời gian được giữ lại. Cross-clip `same_as` không
gộp mất ID cục bộ. Không có self-consistency n lần hoặc human factual verification.
Mô tả scene dùng tối đa 4 keyframe cách đều; mọi keyframe vẫn được parse/merge.
Số lần gọi tăng theo số frame/entity/scene; xem trường `calls` trước khi chạy full.

Nhánh DAR vẫn lấy 16 frame/video. Nhánh graph dùng **đúng ảnh PNG teacher đã chọn**
và timestamp dạng text; không gán fps giả cho chuỗi ảnh không đều.
Graph là target, không xuất hiện trong user prompt. Schema hợp lệ chưa đảm bảo đúng hình ảnh.

Train full SFT LLM + aligner, freeze vision, BF16, LR 1e-5, 0.5 epoch, accumulation 32.
Context tăng lên 16384. Trước SFT, script encode toàn bộ tập bằng ms-swift thực tế;
mẫu quá dài làm dừng để điều chỉnh, không train một graph bị cắt. Chi phí VRAM cao hơn pilot cũ.
Kết quả mới còn bị ảnh hưởng bởi sampling/multi-image/context; muốn quy cải thiện cho graph
cần đối chứng cùng selected frames, accepted IDs và ngân sách train.'''
nb.cells[13].source = nb.cells[13].source.replace("SCRIPTS / 'train.sh'", "STEP_SCRIPTS / 'train.sh'")
nb.cells[13].source = nb.cells[13].source.replace("'BASE_MODEL': str(BASE_MODEL),", "'MAX_LENGTH': str(MAX_LENGTH), 'BASE_MODEL': str(BASE_MODEL),")
nb.cells[16].source = '''## Đọc kết quả

Inference/evaluation giữ quy trình DAR của bản gốc. Một arm chưa đủ kết luận cải thiện.
So sánh trên cùng dev video và accepted train IDs; không dùng test để chọn cấu hình.
`stsg_no_links` giữ entities/attributes/scene intervals, bỏ motion/reference/event links:
đây là ablation về cạnh, không loại mọi thông tin thời gian vì timestamps vẫn còn.

Artifacts: `teacher/run.json`, từng `sampling.json`, `calls/*.json`, `intermediate.json`
(frame graphs + clip graphs), `graph.json`, `arms/build.json`, `arms/*-lengths.json`,
checkpoint và DAR dev predictions. Re-run dùng WORK/OUTPUT mới để tránh trộn kết quả.
Các cell unit test dùng video tổng hợp và fake teacher để kiểm tra wiring. Cần chạy
smoke với checkpoint thật và kiểm tra graph bằng mắt trước khi tăng quy mô.

Nguồn: [STEP §3–4](https://arxiv.org/html/2412.00161v2),
[VILA 1.5 native inference](https://github.com/NVlabs/VILA/tree/6b941da19e31ddfdfaa60160908ccf0978d96615),
[VILA3B model card](https://huggingface.co/Efficient-Large-Model/VILA1.5-3b).
Đây là graph-supervised auxiliary SFT, chưa phải graph encoder hoặc STEP QRA self-training.'''
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None
        compile(cell.source, '<notebook cell>', 'exec')
nb.metadata['dar_step'] = dict(source_notebook=source.name, source_sha256=before,
                              gpu_validated=False, teacher='VILA1.5-3b', selector='hecate_histogram_subshot_stillness')
nbf.validate(nb)
target = ROOT / 'dar_kaggle_step_graph_pilot.ipynb'
nbf.write(nb, target)
assert hashlib.sha256(source.read_bytes()).hexdigest() == before, 'Original notebook changed'
print('Wrote', target, 'Original SHA256:', before)
