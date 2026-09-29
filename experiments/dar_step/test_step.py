"""CPU contract/integration tests; fake responses never certify teacher quality."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from common import digest, file_hash
from frames import select_frames
from hecate_keyframes import histogram_features, stillness_costs, cluster_labels, select_subshots
from graph import assemble, fsg, pair_result, scoped, validate, verified_graph, IdentityGroups
from pipeline import create_graph
# Avoid colliding with the original prepare.py if both test suites share a process.
import importlib.util
spec = importlib.util.spec_from_file_location('step_prepare', Path(__file__).with_name('prepare.py'))
step_prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step_prepare)


def frame_graph():
    return dict(objects=[dict(id='o0', label='person', kind='dynamic', attributes=['red shirt']),
                         dict(id='o1', label='wall', kind='static', attributes=['white'])],
                relations=[['o0', 'in front of', 'o1']], observation='A person stands by a wall.')


def fixture():
    frames = [dict(id=f'f{i}', scene=f'c{i // 2}', time=i * .5, path=f'f{i}.png', sha256='fake')
              for i in range(4)]
    sampling = dict(scenes=[dict(id='c0', start=0., end=1.), dict(id='c1', start=1., end=2.)], frames=frames)
    parsed = {f['id']: scoped(frame_graph(), f['id']) for f in frames}
    pairs = [dict(matches=[[f'f{a}_o0', f'f{b}_o0'], [f'f{a}_o1', f'f{b}_o1']],
                  motions=[[f'f{a}_o0', 'moves right', f'f{b}_o0']]) for a, b in ((0, 1), (2, 3))]
    bridges = [dict(matches=[['f0_o0', 'f2_o0'], ['f0_o1', 'f2_o1']], motions=[])]
    descriptions = {'c0': 'Person moves right.', 'c1': 'Person beside wall.'}
    return sampling, parsed, pairs, bridges, descriptions


class GraphTests(unittest.TestCase):
    def test_merge_preserves_temporal_attributes_and_clip_identity(self):
        args = fixture()
        args[1]['f1']['objects'][0]['attributes'] = ['arm raised']
        graph = validate(assemble(*args), 2.)
        self.assertEqual(len(graph['entities']), 4)
        self.assertEqual(len(graph['frames']), 4)
        self.assertEqual(len(graph['motion_links']), 2)
        self.assertEqual(len(graph['reference_links']), 2)
        self.assertEqual(graph['frames'][1]['objects'][0]['attributes'], ['arm raised'])
        self.assertEqual(graph['event_links'], [['c0', 'before', 'c1']])

    def test_verification_removes_attributes_and_dangling_edges(self):
        result = verified_graph(fsg(frame_graph()), dict(objects=['o0'], attributes=[],
                                                          relations=[0], observation=False))
        self.assertEqual(result['relations'], [])
        self.assertEqual(result['objects'][0]['attributes'], [])
        self.assertEqual(result['observation'], '')

    def test_no_matching_by_label_without_visual_match(self):
        args = list(fixture())
        args[2], args[3] = [], []
        self.assertEqual(len(assemble(*args)['entities']), 8)

    def test_collision_rejected_transitively(self):
        union = IdentityGroups({'a': 'f0', 'b': 'f1', 'c': 'f0'})
        union.join('a', 'b')
        with self.assertRaisesRegex(ValueError, 'collision'):
            union.join('b', 'c')

    def test_invalid_motion_static_and_cross_cut(self):
        left, right = scoped(frame_graph(), 'f0'), scoped(frame_graph(), 'f1')
        for result, bridge in ((dict(matches=[['f0_o1', 'f1_o1']],
                                    motions=[['f0_o1', 'moves', 'f1_o1']]), False),
                               (dict(matches=[['f0_o0', 'f1_o0']],
                                     motions=[['f0_o0', 'moves', 'f1_o0']]), True)):
            with self.assertRaises(ValueError):
                pair_result(result, left, right, bridge=bridge)

    def test_schema_catches_invalid_times_and_references(self):
        for mutate in (lambda g: g['frames'][1].update(time=-1),
                       lambda g: g['motion_links'].append(['f0_o0', 'moves', 'f2_o0']),
                       lambda g: g['reference_links'].append(['e0', 'same_as', 'e1'])):
            graph = assemble(*fixture())
            mutate(graph)
            with self.assertRaises(ValueError):
                validate(graph, 2.)

    def test_aux_uses_teacher_images_and_no_target_leak(self):
        sampling, *rest = fixture()
        evidence = dict(graph=assemble(sampling, *rest), sampling=sampling)
        row = dict(video_id='v', video_path='v.mp4', video_duration=2.,
                   target={'segments': [{'emotion': 'UNIQUE_GT_MARKER', 'start': .1, 'end': 1.1}]})
        dar, aux = step_prepare.sft_rows(row, evidence)
        self.assertEqual(aux['images'], [f['path'] for f in sampling['frames']])
        self.assertEqual(aux['messages'][0]['content'].count('<image>'), 4)
        self.assertNotIn('UNIQUE_GT_MARKER', json.dumps(aux))
        self.assertNotIn('videos', aux)
        self.assertEqual(dar['videos'], ['v.mp4'])
        self.assertEqual(json.loads(aux['messages'][1]['content']), evidence['graph'])
        _, no_links = step_prepare.sft_rows(row, evidence, 'stsg_no_links')
        self.assertEqual(json.loads(no_links['messages'][1]['content'])['motion_links'], [])
        self.assertTrue(evidence['graph']['motion_links'])

    def test_pipeline_stage_wiring(self):
        sampling = fixture()[0]
        calls = []
        def ask(frames, prompt, stage, max_tokens=900):
            calls.append(stage)
            if stage == 'parse':
                return frame_graph()
            if stage == 'verify':
                return dict(objects=['o0', 'o1'], attributes=[['o0', 0], ['o1', 0]], relations=[0], observation=True)
            if stage == 'event':
                self.assertLessEqual(len(frames), 4)
                return {'description': 'Person beside wall.'}
            if stage == 'verify-event':
                self.assertLessEqual(len(frames), 4)
                return {'supported': True}
            a, b = [f['id'] for f in frames]
            return dict(matches=[[a + '_o0', b + '_o0'], [a + '_o1', b + '_o1']],
                        motions=[] if stage == 'bridge' else [[a + '_o0', 'moves right', b + '_o0']])
        graph, intermediate = create_graph(sampling, ask)
        validate(graph, 2.)
        self.assertEqual(len(intermediate['clip_graphs']), 2)
        self.assertEqual(calls.count('parse'), 4)
        self.assertEqual(calls.count('merge'), 2)
        self.assertEqual(calls.count('bridge'), 1)
        calls.clear()
        sampling = dict(scenes=[dict(id='c0', start=0., end=3.)],
                        frames=[dict(id=f'f{i}', scene='c0', time=i*.5, path=f'f{i}.png', sha256='fake')
                                for i in range(6)])
        graph, intermediate = create_graph(sampling, ask)
        validate(graph, 3.)
        self.assertEqual(calls.count('parse'), 6)
        self.assertEqual(calls.count('merge'), 5)
        self.assertEqual(intermediate['event_frame_ids']['c0'], ['f0', 'f1', 'f3', 'f5'])


    def test_build_accepts_bound_evidence_and_rejects_changed_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sampling, *rest = fixture()
            video = root / 'v.mp4'
            video.write_bytes(b'video content for hashing')
            sampling['video_sha256'] = file_hash(video)
            for frame in sampling['frames']:
                path = root / frame['path']
                path.write_bytes(frame['id'].encode())
                frame.update(path=str(path), sha256=file_hash(path))
            identity = dict(video_id='v', video_path=str(video), video_duration=2.)
            row = dict(identity, target={'segments': [{'emotion': 'UNIQUE_GT_MARKER'}]})
            (root / 'train.jsonl').write_text(json.dumps(row) + '\n')
            e = dict(video_id='v', input_sha256=digest(identity), config_sha256='one-run',
                     graph=assemble(sampling, *rest), sampling=sampling)
            evidence_path = root / 'evidence.jsonl'
            evidence_path.write_text(json.dumps(e) + '\n')
            step_prepare.build(root, evidence_path, root / 'accepted', 'stsg')
            dataset = [json.loads(s) for s in (root / 'accepted/stsg.jsonl').read_text().splitlines()]
            self.assertEqual(len(dataset), 2)
            self.assertNotIn('UNIQUE_GT_MARKER', json.dumps(dataset[1]))
            Path(sampling['frames'][0]['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'No accepted'):
                step_prepare.build(root, evidence_path, root / 'rejected', 'stsg')
            report = json.loads((root / 'rejected/build.json').read_text())
            self.assertIn('keyframe changed', report['rejected'][0]['error'])


class SamplingTests(unittest.TestCase):
    def test_histogram_source_layout(self):
        x = histogram_features(np.zeros((32, 32, 3), np.uint8))
        self.assertEqual(x.shape, (2000,))
        expected = np.zeros(2000, np.float32)
        expected[np.arange(15) * 128] = 1
        expected[1920 + np.arange(10) * 8] = 1
        np.testing.assert_array_equal(x, expected)

    def test_clustering_determinism_and_source_cluster_count(self):
        x = np.array([[1, 0], [1, .01], [0, 1], [.01, 1]], np.float32)
        labels, k = cluster_labels(x, 2)
        self.assertEqual(k, 2)
        np.testing.assert_array_equal(labels, cluster_labels(x, 2)[0])
        self.assertEqual(labels[0], labels[1])
        self.assertNotEqual(labels[0], labels[2])
        self.assertEqual(cluster_labels(x, 1)[1], 1)
        self.assertEqual(cluster_labels(np.ones((10, 3)), 4)[1], 1)
        self.assertEqual(cluster_labels(np.ones((1, 3)), 1)[1], 1)

    def test_temporal_runs_keep_recurrence_and_new_label_boundary(self):
        runs = select_subshots([0, 0, 1, 1, 0, 0], [4, 2, 0, 3, 8, 1], [(0, 6)])
        self.assertEqual([r['index'] for r in runs], [1, 2, 5])
        self.assertEqual([(r['start'], r['end']) for r in runs], [(0, 2), (2, 4), (4, 6)])
        self.assertEqual(len(select_subshots([0, 0], [1, 1], [(0, 1), (1, 2)])), 2)

    def test_stillness_uses_float_neighbors_and_respects_cuts(self):
        pictures = [np.full((4, 4, 3), v, np.uint8) for v in (0, 100, 100, 100, 255)]
        cost = stillness_costs(pictures, [(0, 4), (4, 5)])
        self.assertGreater(cost[0], 0)
        self.assertAlmostEqual(cost[1], cost[0] / 2)
        np.testing.assert_array_equal(cost[2:], [0, 0, 0])
        self.assertEqual(select_subshots([0]*4, cost[:4], [(0, 4)])[0]['index'], 2)

    def test_real_scene_detection_and_frame_extraction(self):
        import cv2
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for cut in (True, False):
                path = root / f'video-{cut}.avi'
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (64, 64))
                self.assertTrue(writer.isOpened())
                for i in range(20):
                    image = np.zeros((64, 64, 3), np.uint8)
                    image[:] = (255, 0, 0) if i < 10 or not cut else (0, 0, 255)
                    writer.write(image)
                writer.release()
                row = dict(video_id='v', video_path=str(path), video_duration=2.)
                sampling = select_frames(row, root / f'frames-{cut}')
                self.assertEqual(len(sampling['scenes']), 2 if cut else 1)
                self.assertEqual(sampling['frames'][0]['index'], 0)
                self.assertEqual(len(sampling['frames']), 2 if cut else 1)
                self.assertEqual(sampling['selection']['candidate_count'], 20)
                self.assertEqual(sampling['selection']['feature_dim'], 2000)
                self.assertEqual(sampling['frames'][-1]['index'], 10 if cut else 0)
                self.assertEqual(sampling['video_sha256'], file_hash(path))
                for frame in sampling['frames']:
                    self.assertEqual(file_hash(frame['path']), frame['sha256'])
                if cut:
                    with self.assertRaisesRegex(ValueError, 'budget'):
                        select_frames(row, root / f'budget-{cut}', max_frames=1)


if __name__ == '__main__':
    unittest.main()
