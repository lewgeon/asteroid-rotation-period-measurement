"""Regression controls for fit quality in cross-feature candidate selection."""
import unittest
import numpy as np
from inversion.src.period_consensus import _cross_feature_consensus
from inversion.src.period_uncertainty import candidate_summary
from inversion.src.inversion import multi_harmonic_lomb_scargle


def feature(*pairs):
    return {'candidates': [dict(period_s=p, score=s, source='measured') for p, s in pairs]}


class ConsensusTests(unittest.TestCase):
    def test_single_feature_preserves_strongest_search_peak(self):
        t = np.linspace(0, 180, 120)
        y = np.cos(2*np.pi*t/20)
        for refine in (False, True):
            result = multi_harmonic_lomb_scargle(t, y, 18, 22, 120, refine_period=refine)
            self.assertGreater(result.candidates[0].score-result.candidates[-1].score, .9)
            self.assertEqual(_cross_feature_consensus({'x': candidate_summary(result)})[0], result.best_period_s)

    def test_equal_support_uses_quality_before_artificially_tight_boundary(self):
        data = {'a': feature((20., .95), (18., .01)),
                'b': feature((20.1, .9), (18., .02))}
        self.assertAlmostEqual(_cross_feature_consensus(data)[0], 20.05)
        # Reversing which cluster explains the data must reverse the decision.
        reversed_quality = {'a': feature((18., .95), (20., .01)),
                            'b': feature((18., .9), (20.1, .02))}
        self.assertEqual(_cross_feature_consensus(reversed_quality)[0], 18.)

    def test_aliases_and_multiple_peaks_do_not_add_votes(self):
        data = {'a': feature((20., .9), (20.2, .8), (10., .1)),
                'b': feature((10.1, .1))}
        data['a']['candidates'].append(dict(period_s=10., score=1., source='measured:half'))
        period, support, _, evidence = _cross_feature_consensus(data)
        self.assertEqual(support, 2)
        self.assertEqual(period, 10.05)  # Support remains the primary rule.
        self.assertEqual(len({v['feature'] for v in evidence}), support)

    def test_equal_quality_retains_spread_and_deterministic_period_tiebreak(self):
        data = {'a': feature((20., .8), (10., .8)),
                'b': feature((20.1, .8), (10., .8))}
        self.assertEqual(_cross_feature_consensus(data)[0], 10.)
        with self.assertRaises(ValueError):
            _cross_feature_consensus({})
