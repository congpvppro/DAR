"""Exercise bootstrap orchestration without pip installs, CUDA or network."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.wheels = self.root / 'wheels'
        self.wheels.mkdir()
        self.calls = []

    def run_command(self, command, **kwargs):
        command = list(map(str, command))
        self.calls.append((command, kwargs))
        if 'venv' in command:
            executable = Path(command[-1]) / 'bin/python'
            executable.parent.mkdir()
            executable.touch()
        if '--python' in command:
            self.assertTrue(Path(command[command.index('--python') + 1]).is_file())
        return subprocess.CompletedProcess(command, 0, stdout='{"torch": "2.6.0+cu124"}\n')

    def namespace(self):
        return dict(Path=Path, os=os, sys=sys, subprocess=subprocess, json=json,
                    TEACHER_WHEELHOUSE=self.wheels, TEACHER_VENV=self.root / 'venv',
                    TEACHER_GPUS='0', WHEELHOUSE=self.wheels, RUNTIME=self.root / 'runtime',
                    base_env={'PYTHONPATH': '/kaggle/support', 'PIP_NO_INDEX': '1'},
                    base_paths=['/kaggle/support'])

    def execute(self, name, namespace):
        source = (HERE / name).read_text(encoding='utf-8')
        with patch.object(subprocess, 'run', side_effect=self.run_command):
            exec(compile(source, name, 'exec'), namespace)

    def test_missing_teacher_wheels_stops_before_venv_or_pip(self):
        with self.assertRaisesRegex(AssertionError, 'Missing transformers'):
            self.execute('notebook_teacher_runtime.py', self.namespace())
        self.assertEqual(self.calls, [])

    def test_placeholder_is_not_required_for_auto_detection(self):
        ns = self.namespace()
        ns['TEACHER_WHEELHOUSE'] = None
        with patch.object(Path, 'rglob', return_value=[]):
            with self.assertRaisesRegex(AssertionError, 'no transformers-4.57.1 wheel'):
                self.execute('notebook_teacher_runtime.py', ns)
        self.assertEqual(self.calls, [])

    def test_teacher_creates_interpreter_offline_and_prioritizes_local_packages(self):
        (self.wheels / 'transformers-4.57.1-py3-none-any.whl').touch()
        ns = self.namespace()
        self.execute('notebook_teacher_runtime.py', ns)
        self.assertIn('--without-pip', self.calls[0][0])
        installs = [c for c, _ in self.calls if 'install' in c]
        self.assertEqual(len(installs), 2)
        self.assertIn('--dry-run', installs[0])
        for command in installs:
            self.assertIn('--no-index', command)
            self.assertIn('--constraint', command)
            self.assertNotIn('--force-reinstall', command)
        self.assertEqual(ns['teacher_env']['PYTHONPATH'],
                         os.pathsep.join([str(ns['teacher_site']), '/kaggle/support']))
        self.assertIn('torch==2.6.0+cu124', ns['constraints'].read_text())
        previous = ns['teacher_python']
        self.execute('notebook_teacher_runtime.py', ns)
        self.assertTrue(previous.exists())
        self.assertNotEqual(previous, ns['teacher_python'])

    def test_student_filters_cpu_stack_and_ignores_old_overlay(self):
        for name in ['torch-2.10.0+cpu-py3-none-any.whl',
                     'torchvision-0.25.0-py3-none-any.whl',
                     'nvidia_cublas_cu12-12.0-py3-none-any.whl',
                     'transformers-4.46.3-py3-none-any.whl',
                     'transformers-4.57.1-py3-none-any.whl']:
            (self.wheels / name).touch()
        ns = self.namespace()
        ns['RUNTIME'].mkdir()
        stale = ns['RUNTIME'] / 'torch'
        stale.mkdir()
        self.execute('notebook_student_runtime.py', ns)
        install = next(c for c, _ in self.calls if 'install' in c)
        installed_wheels = [Path(p).name for p in install if p.endswith('.whl')]
        self.assertEqual(installed_wheels, ['transformers-4.57.1-py3-none-any.whl'])
        self.assertIn('--no-index', install)
        self.assertTrue(stale.is_dir())
        self.assertNotEqual(ns['student_site'], ns['RUNTIME'])
        self.assertTrue(ns['env']['PYTHONPATH'].startswith(str(ns['student_site'])))


if __name__ == '__main__':
    unittest.main()
