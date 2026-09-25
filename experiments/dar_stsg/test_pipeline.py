import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core import annotation, digest, evidence_prompt, official_functions, sft_rows, validate_graph, write_jsonl, read_jsonl
from evaluate import aggregate, evaluate, stats, valid_segments
from prepare import build, split


def graph():
    return {'entities': [{'id': 'o1', 'label': 'ball'}, {'id': 'o2', 'label': 'table'}],
            'events': [{'id': 'e1', 'start_time': 0., 'end_time': 1., 'observation': 'Ball on table', 'relations': [['o1', 'on', 'o2']]},
                       {'id': 'e2', 'start_time': 1., 'end_time': 2., 'observation': 'Ball rolls', 'relations': []}],
            'temporal_links': [['e1', 'before', 'e2']]}


def segment(a=0., b=2., emotion='Interest'):
    return dict(start_time=a, end_time=b, emotion=emotion, reason='Visible object attracts attention.')


def row(i='v'):
    return dict(video_id=i, video_path=i + '.mp4', video_duration=2., target={'segments': [segment()]})


class PipelineTests(unittest.TestCase):
    def test_graph_prompt_has_no_copyable_placeholder_json(self):
        prompt = evidence_prompt(2, 'stsg')
        self.assertNotIn('"o1"', prompt)
        self.assertNotIn('"e1"', prompt)
        self.assertNotIn('visible object or scene', prompt)
        self.assertIn('Both IDs MUST occur', prompt)

    def test_invalid_references_and_time(self):
        self.assertEqual(validate_graph(graph(), 2), graph())
        for change in ('id', 'time', 'nan', 'reverse'):
            g = graph()
            if change == 'id': g['events'][0]['relations'][0][2] = 'missing'
            if change == 'time': g['events'][0]['end_time'] = 3
            if change == 'nan': g['events'][0]['start_time'] = float('nan')
            if change == 'reverse': g['temporal_links'] = [['e2', 'before', 'e1']]
            with self.assertRaises(ValueError): validate_graph(g, 2)

    def test_auxiliary_does_not_receive_answer(self):
        r = row()
        r['target']['segments'][0]['reason'] = 'SECRET_REFERENCE_TEXT'
        e = dict(stsg=graph(), caption={'caption': 'A ball rolls.'})
        for arm in ('caption', 'stsg', 'stsg_no_links'):
            samples = sft_rows(r, e, arm)
            self.assertIn('SECRET_REFERENCE_TEXT', samples[0]['messages'][1]['content'])
            self.assertNotIn('SECRET_REFERENCE_TEXT', json.dumps(samples[1]))
            self.assertEqual(len(samples), 2)
        self.assertEqual(graph(), e['stsg'])
        no_links = json.loads(sft_rows(r, e, 'stsg_no_links')[1]['messages'][1]['content'])
        self.assertEqual(no_links['temporal_links'], [])
        self.assertEqual(no_links['events'], e['stsg']['events'])

    def test_no_gt_duration_fallback(self):
        with self.assertRaises(ValueError): annotation({'video': 'v.mp4', 'conversations': []})

    def test_full_split_uses_every_non_dev_train_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = lambda i: dict(video=f'{i}.mp4', video_duration=2.,
                                    conversations=[{'from': 'gpt', 'value': json.dumps(row()['target'])}])
            write_jsonl(root / 'train.jsonl', [source(i) for i in range(6)])
            write_jsonl(root / 'test.jsonl', [source(99)])
            split(SimpleNamespace(train=root / 'train.jsonl', test=root / 'test.jsonl',
                                  video_root=tmp, output=root / 'split', train_size=-1,
                                  dev_size=2, seed=1))
            train = read_jsonl(root / 'split/train.jsonl')
            dev = read_jsonl(root / 'split/dev.jsonl')
            self.assertEqual((len(train), len(dev)), (4, 2))
            self.assertEqual(len({r['video_id'] for r in train + dev}), 6)

    def test_matching_penalizes_extra_segments(self):
        gt = [segment()]
        self.assertEqual(stats(gt * 2, gt)[0], 1)
        self.assertAlmostEqual(aggregate(np.asarray([stats(gt * 2, gt)]))['joint_f1_at_05'], 2/3)
        self.assertEqual(stats([segment(emotion='Fear')], gt)[0], 0)

    def test_invalid_outputs_and_missing_denominator(self):
        self.assertEqual(valid_segments({'segments': [segment(b=float('nan'))]}, 2), [])
        self.assertEqual(valid_segments({'segments': [segment(a=1.)]}, 2), [])
        a, report = evaluate([row('a'), row('b')], [{'video_id': 'a', 'dar': row()['target']}])
        self.assertEqual(report['strict']['coverage'], .5)
        self.assertEqual(report['missing_videos'], 1)
        self.assertEqual(report['strict']['gt_segments'], 2)
        self.assertAlmostEqual(report['strict']['joint_f1_at_05'], 2/3)

    def test_duplicate_predictions_fail(self):
        p = {'video_id': 'v', 'dar': row()['target']}
        with self.assertRaises(ValueError): evaluate([row()], [p, p])

    def test_official_functions_preserved(self):
        f = official_functions()
        self.assertIn('SILENT', f['build_eval_prompt'](2.))
        self.assertEqual(f['evaluate_single_video']([segment()], [segment()])['avg_iou'], 1.)

    def test_split_and_common_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def source(i):
                return dict(video=f'{i}.mp4', video_duration=2., conversations=[{'from': 'gpt', 'value': json.dumps(row()['target'])}])
            write_jsonl(root/'train.jsonl', [source(i) for i in range(6)])
            write_jsonl(root/'test.jsonl', [source(99)])
            args = SimpleNamespace(train=root/'train.jsonl', test=root/'test.jsonl', video_root=tmp,
                                   output=root/'split', train_size=3, dev_size=2, seed=1)
            split(args)
            inputs = read_jsonl(root/'split/teacher_inputs.jsonl')
            self.assertTrue(all(set(r) == {'video_id', 'video_path', 'video_duration'} for r in inputs))
            evidence = [dict(video_id=r['video_id'], input_sha256=digest(r), stsg=graph(), caption={'caption': 'Ball'}) for r in inputs]
            evidence[0].pop('stsg')
            write_jsonl(root/'evidence.jsonl', evidence)
            build(SimpleNamespace(split=root/'split', evidence=root/'evidence.jsonl', output=root/'arms'))
            for arm in ('baseline', 'caption', 'stsg', 'stsg_no_links'):
                self.assertEqual(len(read_jsonl(root/'arms'/f'{arm}.jsonl')), 4)
            write_jsonl(root/'overlap.jsonl', [source(1)])
            args.test = root/'overlap.jsonl'
            args.output = root/'bad'
            with self.assertRaises(ValueError): split(args)

    def test_single_arm_only_requires_its_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r = row()
            write_jsonl(root / 'train.jsonl', [r])
            identity = {k: r[k] for k in ('video_id', 'video_path', 'video_duration')}
            for arm, payload in [('stsg', {'stsg': graph()}),
                                 ('caption', {'caption': {'caption': 'A ball rolls.'}})]:
                evidence_path = root / (arm + '-evidence.jsonl')
                write_jsonl(evidence_path, [dict(video_id=r['video_id'], input_sha256=digest(identity), **payload)])
                output = root / arm
                build(SimpleNamespace(split=root, evidence=evidence_path, output=output, arms=[arm]))
                self.assertEqual(len(read_jsonl(output / (arm + '.jsonl'))), 2)
                self.assertEqual([p.name for p in output.glob('*.jsonl')], [arm + '.jsonl'])
                other = 'caption' if arm == 'stsg' else 'stsg'
                with self.assertRaises(ValueError):
                    build(SimpleNamespace(split=root, evidence=evidence_path, output=root/'invalid', arms=[other]))


if __name__ == '__main__':
    unittest.main()
