import unittest
import numpy as np
from inversion.src.bounded_features import range_centroid_budget


class BoundedFeatureTests(unittest.TestCase):
    def test_known_perturbations_respect_bound_and_budget_changes_matter(self):
        rng=np.random.default_rng(332)
        axis=np.linspace(-1e-6,2e-6,31)
        samples=rng.normal(size=31)+1j*rng.normal(size=31)
        report=range_centroid_budget(samples,axis,{'receiver':.01,'model':.02})
        center=lambda z:float(np.sum(abs(z)**2*axis)/np.sum(abs(z)**2))
        worst=0.
        for _ in range(100):
            change=rng.normal(size=31)+1j*rng.normal(size=31);change*=.03/np.linalg.norm(change)
            delta=abs(center(samples+change)-center(samples));worst=max(worst,delta)
            self.assertLessEqual(delta,report['centroid_error_bound_s'])
        self.assertGreater(worst,0)
        larger=range_centroid_budget(samples,axis,{'total':.06})
        self.assertGreater(larger['centroid_error_bound_s'],report['centroid_error_bound_s'])

    def test_no_signal_is_not_reported_as_zero_uncertainty(self):
        result=range_centroid_budget([1+0j,0j],[0.,1.],{'total':1.})
        self.assertEqual(result['status'],'unavailable_zero_energy_possible')
        self.assertIsNone(result['centroid_error_bound_s'])
        with self.assertRaises(ValueError):range_centroid_budget([1],[0],{})

    def test_coordinate_shift_is_a_separate_error_not_absorbed_in_sample_budget(self):
        a=range_centroid_budget([1,2],[0.,1.],{'total':.01})
        b=range_centroid_budget([1,2],[100.,101.],{'total':.01})
        self.assertEqual(a['centroid_error_bound_s'],b['centroid_error_bound_s'])
        self.assertIn('Coordinate errors',a['assumptions'])

    def test_large_coordinate_bounds_are_json_safe_or_explicitly_unrepresentable(self):
        import json
        largest=np.finfo(float).max
        a=range_centroid_budget([1,1],[0.,largest],{'total':.2})
        json.dumps(a,allow_nan=False)
        b=range_centroid_budget([1,1],[-largest,largest],{'total':.8})
        self.assertEqual(b['status'],'bound_exceeds_float64')
        self.assertIsNone(b['centroid_error_bound_s'])
        self.assertIn('centroid_error_bound_decimal_s',b)
        json.dumps(b,allow_nan=False)
