"""Prepare a portable VILA source + Linux cp312 wheel bundle, on an ONLINE machine.

Does not download model weights or upload anything. The Kaggle notebook uses
this bundle offline, separately from the existing student's wheelhouse.
"""
import argparse
import io
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile

from common import dump, file_hash, require
from teacher_vila import VILA_REVISION


def prepare_source(data, destination):
    """Keep native inference, make unused model-family/S2 imports lazy."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for item in archive.infolist():
            relative = Path(*Path(item.filename).parts[1:])
            if item.is_dir() or not relative.parts:
                continue
            target = (destination / relative).resolve()
            require(target.is_relative_to(destination.resolve()), 'Unsafe archive path')
            # Retain native package plus license/readme. No external build scripts run.
            if relative.parts[0] != 'llava' and str(relative) not in ('LICENSE', 'README.md'):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(item))
    patches = []

    def replace(name, old, new):
        path = destination / name
        content = path.read_text(encoding='utf-8')
        require(content.count(old) == 1, f'Pinned VILA source mismatch: {name}')
        before = file_hash(path)
        path.write_text(content.replace(old, new), encoding='utf-8')
        patches.append(dict(path=name, before_sha256=before, after_sha256=file_hash(path), old=old, new=new))

    # SigLIP inference needs none of the dependencies of unused Intern/RADIO towers.
    builder = 'llava/model/multimodal_encoder/builder.py'
    for module, classes, branch in (
        ('clip_encoder', 'CLIPVisionTower, CLIPVisionTowerS2', 'elif "clip" in vision_tower_name:'),
        ('intern_encoder', 'InternVisionTower', 'if "intern" in vision_tower_name.lower():'),
        ('radio_encoder', 'RADIOVisionTower', 'elif "radio" in vision_tower_name:')):
        line = f'from .{module} import {classes}'
        replace(builder, line + '\n', '')
        replace(builder, '    ' + branch + '\n', '    ' + branch + '\n        ' + line + '\n')
    replace('llava/model/multimodal_encoder/vision_encoder.py',
            'from s2wrapper import forward as multiscale_forward',
            'def multiscale_forward(*args, **kwargs):\n'
            '    from s2wrapper import forward\n'
            '    return forward(*args, **kwargs)')
    # No Transformers replacement/FlashAttention training patch: native generate
    # delegates directly to stock LlamaForCausalLM.generate with eager attention.
    dump(destination / 'dar-runtime.json', dict(revision=VILA_REVISION, patches=patches,
         files={str(p.relative_to(destination)).replace('\\', '/'): file_hash(p)
                for p in destination.rglob('*.py')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-archive')
    parser.add_argument('--source-only', action='store_true')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    data = (Path(args.source_archive).read_bytes() if args.source_archive else
            urllib.request.urlopen(f'https://api.github.com/repos/NVlabs/VILA/zipball/{VILA_REVISION}').read())
    prepare_source(data, out / 'vila')
    requirements = Path(__file__).with_name('requirements-teacher.txt')
    (out / requirements.name).write_bytes(requirements.read_bytes())
    if not args.source_only:
        subprocess.run([sys.executable, '-m', 'pip', 'download', '--only-binary=:all:', '--no-deps',
                        '--platform', 'manylinux2014_x86_64', '--python-version', '312',
                        '--implementation', 'cp', '--abi', 'cp312', '-r', str(requirements),
                        '--dest', str(out / 'wheels')], check=True)
    dump(out / 'manifest.json', dict(revision=VILA_REVISION,
         files={str(p.relative_to(out)).replace('\\', '/'): file_hash(p)
                for p in out.rglob('*') if p.is_file()}))
    print('Attach this directory to Kaggle as one dataset:', out.resolve())


if __name__ == '__main__':
    main()
