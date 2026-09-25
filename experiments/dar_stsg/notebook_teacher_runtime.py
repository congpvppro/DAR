"""Offline teacher bootstrap cell; embedded verbatim by build_notebook.py."""
import tempfile

requirements = ['transformers==4.57.1', 'accelerate==1.11.0',
                'decord==0.6.0', 'numpy>=1.26', 'einops==0.8.1', 'ffmpeg-python==0.2.0']
if TEACHER_WHEELHOUSE is None:
    teacher_candidates = sorted(Path('/kaggle/input').rglob('transformers-4.57.1-*.whl'))
    assert teacher_candidates, (
        'Internet Off: no transformers-4.57.1 wheel was found under /kaggle/input. '
        'Attach a Kaggle Dataset containing the teacher wheelhouse, then rerun this cell.')
    candidate_roots = sorted({p.parent for p in teacher_candidates})
    assert len(candidate_roots) == 1, (
        f'Found transformers 4.57.1 in multiple directories: {candidate_roots}. '
        'Set TEACHER_WHEELHOUSE explicitly in the configuration cell.')
    teacher_wheelhouse = candidate_roots[0]
    print('Auto-detected teacher wheelhouse:', teacher_wheelhouse)
else:
    teacher_wheelhouse = Path(TEACHER_WHEELHOUSE)
    assert teacher_wheelhouse.is_dir(), f'Missing teacher wheelhouse: {teacher_wheelhouse}'
teacher_wheels = sorted(teacher_wheelhouse.rglob('*.whl'))
assert any(p.name.startswith('transformers-4.57.1-') for p in teacher_wheels), (
    f'Missing transformers-4.57.1 wheel in {teacher_wheelhouse}; attach teacher wheels before running')
install_flags = ['--no-index', '--only-binary=:all:']
for directory in sorted({p.parent for p in Path('/kaggle/input').rglob('*.whl')}):
    install_flags.extend(['--find-links', str(directory)])

# Create the interpreter first. No ensurepip, no deletion of earlier environments.
TEACHER_VENV.mkdir(parents=True, exist_ok=True)
teacher_venv = Path(tempfile.mkdtemp(prefix='teacher-', dir=TEACHER_VENV))
subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages',
                '--without-pip', str(teacher_venv)], check=True, env=base_env)
teacher_python = teacher_venv / 'bin/python'
assert teacher_python.is_file(), f'Failed to create teacher interpreter: {teacher_python}'
teacher_site = teacher_venv / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
teacher_env = dict(base_env,
    PYTHONPATH=os.pathsep.join([str(teacher_site), *base_paths]),
    CUDA_VISIBLE_DEVICES=TEACHER_GPUS)

# Pin the existing CUDA stack so dependency resolution cannot replace it.
stack_probe = subprocess.run([sys.executable, '-c',
    "import importlib.metadata as m, json; "
    "print(json.dumps({d.metadata['Name']: d.version for d in m.distributions() "
    "if d.metadata['Name'] and (d.metadata['Name'].lower() in "
    "('torch', 'torchvision', 'torchaudio', 'triton') or d.metadata['Name'].lower().startswith('nvidia-'))}))"],
    check=True, env=base_env, text=True, capture_output=True)
cuda_stack = json.loads(stack_probe.stdout)
constraints = teacher_venv / 'cuda-constraints.txt'
constraints.write_text(''.join(f'{name}=={version}\n' for name, version in cuda_stack.items()))

# Kernel pip >=22.3 can manage a venv that has no pip of its own.
# Resolve offline first; missing wheels fail without partially installing packages.
pip_command = [sys.executable, '-m', 'pip', '--python', str(teacher_python),
               'install', *install_flags, '--constraint', str(constraints), *requirements]
subprocess.run([*pip_command, '--dry-run'], check=True, env=teacher_env)
subprocess.run(pip_command, check=True, env=teacher_env)

subprocess.run([str(teacher_python), '-c',
    "import sys, torch, transformers, decord, cv2; "
    "print('Teacher Python:', sys.executable); "
    "print('Teacher Transformers:', transformers.__version__, transformers.__file__); "
    "print('Teacher Torch:', torch.__version__, torch.__file__); "
    "assert transformers.__version__ == '4.57.1', 'Wrong teacher Transformers: ' + transformers.__file__; "
    "assert torch.version.cuda is not None and torch.cuda.is_available(), 'Teacher requires CUDA Torch'; "
    "print('Teacher GPUs:', torch.cuda.device_count())"], check=True, env=teacher_env)
