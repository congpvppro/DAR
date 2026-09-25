import unittest
import numpy as np
from metrics import video_stats, aggregate, transform_frames


def seg(a, b, e):
    return dict(start_time=a, end_time=b, emotion=e)


class MetricTests(unittest.TestCase):
    def test_missing_prediction_is_counted(self):
        result = aggregate([video_stats([], [seg(0, 1, 'Joy')])])
        self.assertEqual(result['coverage'], 0)
        self.assertEqual(result['gt_segments'], 1)
        self.assertEqual(result['matched_emotion_recall'], 0)

    def test_truncation_cannot_hide_unmatched_ground_truth(self):
        gt = [seg(0, 1, 'Joy'), seg(1, 2, 'Fear')]
        result = aggregate([video_stats(gt[:1], gt)])
        self.assertEqual(result['conditional_emotion_accuracy'], 1)
        self.assertEqual(result['fixed_gt_index_joint_recall'], .5)
        self.assertEqual(result['matched_emotion_recall'], .5)
        self.assertAlmostEqual(result['matched_emotion_f1'], 2/3)

    def test_matching_recovers_shifted_index(self):
        gt = [seg(0, 1, 'Joy'), seg(1, 2, 'Fear')]
        result = aggregate([video_stats(gt[1:], gt)])
        self.assertEqual(result['fixed_gt_index_joint_recall'], 0)
        self.assertEqual(result['matched_emotion_recall'], .5)

    def test_false_positive_lowers_precision(self):
        gt = [seg(0, 1, 'Joy')]
        result = aggregate([video_stats(gt + [seg(2, 3, 'Fear')], gt)])
        self.assertEqual(result['matched_emotion_precision'], .5)
        self.assertEqual(result['matched_emotion_recall'], 1)

    def test_control_transforms_preserve_shape_and_original(self):
        x = np.arange(16*3*2*2).reshape(16, 3, 2, 2).astype(np.float32)
        original = x.copy()
        np.testing.assert_array_equal(transform_frames(x, 'reverse'), x[::-1])
        static = transform_frames(x, 'static_middle')
        self.assertEqual(static.shape, x.shape)
        for frame in static:
            np.testing.assert_array_equal(frame, x[8])
        self.assertTrue((transform_frames(x, 'gray') == 127.5).all())
        np.testing.assert_array_equal(x, original)


if __name__ == '__main__':
    unittest.main()
