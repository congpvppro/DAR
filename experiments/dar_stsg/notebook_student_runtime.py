"""Student bootstrap cell; embedded verbatim by build_notebook.py."""
import tempfile
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

# Preserve Kaggle's package directories (including sitecustomize dependencies).
# Do not inherit an old student/teacher overlay from a previous bootstrap.
base_paths = [p for p in sys.path if p and Path(p).is_dir()
              and ('site-packages' in p or 'dist-packages' in p)]
inherited_paths = os.environ.get('PYTHONPATH', '').split(os.pathsep)
base_paths = list(dict.fromkeys(
    p for p in [*inherited_paths, *base_paths] if p
    and not Path(p).resolve().is_relative_to(Path('/kaggle/working'))))
base_env = dict(os.environ, PYTHONPATH=os.pathsep.join(base_paths),
                PIP_NO_INDEX='1', PIP_DISABLE_PIP_VERSION_CHECK='1',
                HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')

# Check the actual base interpreter, not torch already imported in the kernel.
subprocess.run([sys.executable, '-c',
    "import torch; print('Base torch:', torch.__version__, torch.__file__); "
    "assert torch.version.cuda is not None, 'Kaggle base Torch is CPU-only; use a GPU image with CUDA Torch'; "
    "assert torch.cuda.is_available(), 'Enable a Kaggle GPU accelerator before continuing'"],
    check=True, env={**base_env, 'CUDA_VISIBLE_DEVICES': TEACHER_GPUS})

supported_tags = set(sys_tags())
wheels = []
student_pins = {'transformers': '4.57.1', 'ms-swift': '3.12.5'}
for wheel in sorted(WHEELHOUSE.rglob('*.whl')):
    name, version, _, tags = parse_wheel_filename(wheel.name)
    # Reuse Kaggle's matched CUDA stack; never overlay it with a CPU wheel.
    if name in {'torch', 'torchvision', 'torchaudio', 'triton', 'torchao'} or name.startswith('nvidia-'):
        continue
    if not (tags & supported_tags) or (name == 'packaging' and str(version) == '26.3'):
        continue
    # Do not pass two versions of one package to pip if additional wheels are attached.
    if name in student_pins and str(version) != student_pins[name]:
        continue
    wheels.append(wheel)
assert wheels, f'No compatible student wheels in {WHEELHOUSE}'
assert any(parse_wheel_filename(p.name)[0] == 'transformers'
           and str(parse_wheel_filename(p.name)[1]) == student_pins['transformers']
           for p in wheels), 'Student wheelhouse needs transformers 4.57.1'

# A fresh overlay prevents CPU Torch left by an earlier cell from being imported.
RUNTIME.mkdir(parents=True, exist_ok=True)
student_site = Path(tempfile.mkdtemp(prefix='student-', dir=RUNTIME))
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                '--only-binary=:all:', '--target', str(student_site), *map(str, wheels)],
               check=True, env=base_env)
env = dict(base_env, PYTHONPATH=os.pathsep.join([str(student_site), *base_paths]),
           CUDA_VISIBLE_DEVICES='0')
def run(*args, extra_env=None):
    subprocess.run(list(map(str, args)), check=True, env={**env, **(extra_env or {})})
run(sys.executable, '-c',
    "import torch, transformers, swift, peft, qwen_vl_utils; "
    "print('Student:', torch.__version__, transformers.__version__, swift.__version__); "
    "print('Torch path:', torch.__file__); print('Transformers path:', transformers.__file__); "
    "assert torch.version.cuda is not None and torch.cuda.is_available(), 'Student requires CUDA Torch'; "
    "assert transformers.__version__ == '4.57.1', transformers.__file__; "
    "assert swift.__version__ == '3.12.5'; print(torch.cuda.get_device_name(0))")
