"""Offline teacher bootstrap, embedded in the notebook; never import ML in kernel."""
import hashlib
import tempfile
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

bundle_meta = json.loads((TEACHER_BUNDLE / 'manifest.json').read_text())
assert bundle_meta['revision'] == '6b941da19e31ddfdfaa60160908ccf0978d96615'
for name, expected in bundle_meta['files'].items():
    path = (TEACHER_BUNDLE / name).resolve()
    assert path.is_relative_to(TEACHER_BUNDLE.resolve()), name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f'Bundle changed: {path}'
teacher_site = Path(tempfile.mkdtemp(prefix='dar-vila-site-'))
tags = set(sys_tags())
teacher_wheels = []
for path in sorted((TEACHER_BUNDLE / 'wheels').glob('*.whl')):
    name, _, _, wheel_tags = parse_wheel_filename(path.name)
    assert name not in ('torch', 'torchvision', 'torchaudio', 'triton') and not name.startswith('nvidia-')
    if tags & wheel_tags:
        teacher_wheels.append(path)
assert teacher_wheels, 'Prepare the bundle for Kaggle Python 3.12 / Linux'
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                '--target', str(teacher_site), *map(str, teacher_wheels)], check=True, env=base_env)
teacher_python = sys.executable
teacher_env = dict(base_env, PYTHONPATH=os.pathsep.join(
    [str(TEACHER_BUNDLE / 'vila'), str(teacher_site), *base_paths]), CUDA_VISIBLE_DEVICES=TEACHER_GPUS)
subprocess.run([teacher_python, '-c',
    "import torch, transformers, scenedetect, cv2; from llava.model.builder import load_pretrained_model; "
    "assert transformers.__version__ == '4.36.2'; assert scenedetect.__version__ == '0.6.5.2'; "
    "assert torch.cuda.is_available(); print('VILA native imports OK:', transformers.__file__, torch.__version__)"],
    check=True, env=teacher_env)
