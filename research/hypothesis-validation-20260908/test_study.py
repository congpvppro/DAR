import unittest
import numpy as np
try:
    import study
except ModuleNotFoundError:
    study = None


class StudyTests(unittest.TestCase):
    def test_static_intervention_preserves_source_and_exact_frame(self):
        self.assertIsNotNone(study, 'study implementation is missing')
        frames = np.arange(16*3*4*4,dtype=np.float32).reshape(16,3,4,4)
        before = frames.copy()
        out = study.intervene(frames,'static_middle')
        np.testing.assert_array_equal(frames,before)
        np.testing.assert_array_equal(out,np.repeat(frames[8:9],16,axis=0))
        self.assertFalse(np.shares_memory(frames,out))

    def test_gray_is_uniform_and_unknown_rejected(self):
        self.assertIsNotNone(study, 'study implementation is missing')
        frames = np.zeros((16,3,4,4),dtype=np.float32)
        out = study.intervene(frames,'gray')
        self.assertTrue(np.all(out == 127.5))
        step = study.intervene(frames,'gray_to_white')
        self.assertTrue(np.all(step[:8] == 127.5))
        self.assertTrue(np.all(step[8:] == 255.0))
        with self.assertRaises(ValueError):
            study.intervene(frames,'unknown')

    def test_matrix_is_complete_unique_and_factual_static_only(self):
        self.assertIsNotNone(study, 'study implementation is missing')
        natural=[{'video_id':f'n{i}','kind':'natural','gt_segments':['secret']} for i in range(64)]
        gray=[{'video_id':f'g{i}','kind':'gray'} for i in range(12)]
        step=[{'video_id':f's{i}','kind':'step'} for i in range(12)]
        cases=study.build_cases(natural+gray+step)
        self.assertEqual(len(cases),368)
        self.assertEqual(len({c['case_id'] for c in cases}),368)
        self.assertEqual(sum(c['prompt_kind']=='factual' for c in cases),88)
        self.assertTrue(all(c['condition']!='original' for c in cases if c['prompt_kind']=='factual'))
        self.assertTrue(all('gt_segments' not in c for c in cases))


if __name__=='__main__':
    unittest.main()
