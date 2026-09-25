import unittest
try:
    import statistics_study as stats
except ModuleNotFoundError:
    stats=None


class StatisticsTests(unittest.TestCase):
    def test_missing_and_ambiguous_remain_in_denominator(self):
        self.assertIsNotNone(stats,'statistics implementation missing')
        out=stats.event_summary(['confirmed','none','ambiguous','unassessable'])
        self.assertEqual(out['n'],4)
        self.assertEqual(out['confirmed'],1)
        self.assertEqual(out['rate_all'],.25)
        self.assertEqual(out['rate_decidable'],.5)

    def test_paired_direction_and_discordant_counts(self):
        self.assertIsNotNone(stats,'statistics implementation missing')
        out=stats.paired_binary([1,1,1,0],[0,0,1,1])
        self.assertEqual(out['a_only'],2)
        self.assertEqual(out['b_only'],1)
        self.assertEqual(out['difference'],.25)
        self.assertEqual(out['mcnemar_exact_p'],1.)
        with self.assertRaises(ValueError):
            stats.paired_binary([1],[1,0])

    def test_constant_paired_bootstrap_and_holm(self):
        self.assertIsNotNone(stats,'statistics implementation missing')
        out=stats.paired_binary([1,1],[0,0])
        self.assertEqual(out['difference_ci95'],[1.,1.])
        self.assertEqual(stats.holm([.03,.01,.2]),[.06,.03,.2])


if __name__=='__main__':
    unittest.main()
