import unittest
from fractions import Fraction
from decimal import Decimal
import numpy as np
from inversion.src.bounded_period import Interval,trig_enclosure,exclusion_margin,outer_period_set


class BoundedPeriodTests(unittest.TestCase):
    def test_interval_trig_matches_independent_high_precision_and_exact_special_values(self):
        import mpmath as mp
        with mp.workdps(90):
            for q in (Fraction(0),Fraction(1,4),Fraction(1,2),Fraction(-1,4),Fraction(10**25+1,7),Fraction(-125,37)):
                sine,cosine=trig_enclosure(q)
                for interval,value in ((sine,mp.sin(2*mp.pi*mp.mpf(q.numerator)/q.denominator)),
                                       (cosine,mp.cos(2*mp.pi*mp.mpf(q.numerator)/q.denominator))):
                    self.assertLessEqual(mp.mpf(str(interval.low)),value)
                    self.assertGreaterEqual(mp.mpf(str(interval.high)),value)
                    self.assertLess(float(interval.high-interval.low),1e-35)
        self.assertLess(Interval.rational(Fraction(1,3)).low,Decimal(1)/3+Decimal('1e-27'))

    def fixture(self):
        t=np.linspace(0,90,24);period=7.3
        y=.2+np.cos(2*np.pi*t/period)+.3*np.sin(2*np.pi*t/period)
        return t,y,period

    def test_full_domain_keeps_truth_and_rejects_other_periods_without_truth_selection(self):
        t,y,truth=self.fixture()
        result=outer_period_set(t,y,5.,10.,[1.,2.,2.],{'declared_noise':.002},resolution_s=.02,max_boxes=3000,estimate_s=7.31)
        intervals=result['outer_intervals_s']
        self.assertTrue(any(a<=truth<=b for a,b in intervals))
        self.assertFalse(any(a<=6<=b for a,b in intervals))
        self.assertFalse(any(a<=9<=b for a,b in intervals))
        # Uniform sampling admits an exact frequency alias near 8.43 s.
        # A solver that keeps only the true/local peak would be incorrect.
        self.assertGreater(len(intervals),1)
        alias=1/(23/90-1/truth)
        self.assertTrue(any(a<=alias<=b for a,b in intervals))
        self.assertGreater(result['conditional_error_bound_s'],1.)
        self.assertGreater(result['excluded_boxes'],0)
        self.assertFalse(result['physical_model_error_validated'])

    def test_irregular_sampling_resolves_the_uniform_sampling_alias(self):
        t=np.r_[0.,np.sort(np.random.default_rng(17).uniform(0,90,23))]
        y=.2+np.cos(2*np.pi*t/7.3)+.3*np.sin(2*np.pi*t/7.3)
        result=outer_period_set(t,y,5.,10.,[1.,2.,2.],{'noise':.002},resolution_s=.02,max_boxes=3000,estimate_s=7.31)
        self.assertEqual(len(result['outer_intervals_s']),1)
        self.assertLess(result['conditional_error_bound_s'],.1)

    def test_resource_limit_and_zero_signal_never_discard_unresolved_periods(self):
        t,y,truth=self.fixture()
        result=outer_period_set(t,y,5.,10.,[1.,2.,2.],{'noise':.002},max_boxes=1,estimate_s=7.3)
        self.assertEqual(result['status'],'budget_exhausted_outer_cover')
        self.assertLessEqual(result['outer_intervals_s'][0][0],5.)
        self.assertGreaterEqual(result['outer_intervals_s'][-1][1],10.)
        result=outer_period_set(t,np.zeros_like(t),5.,10.,[1.,2.,2.],{'noise':0.},resolution_s=.2,max_boxes=300)
        self.assertEqual(result['excluded_boxes'],0)
        self.assertTrue(result['all_domain_feasible_proved'])
        self.assertEqual(result['status'],'all_domain_feasible')

    def test_dual_witness_nonexclusion_for_known_feasible_coefficients(self):
        t,y,truth=self.fixture();rng=np.random.default_rng(80)
        for width in (0.,.001,.8):
            margin=exclusion_margin([Fraction(float(v)) for v in t],[Fraction(float(v)) for v in y],
                [Fraction(1,1000)]*len(t),[Fraction(1),Fraction(2),Fraction(2)],
                (Fraction(truth)-Fraction(width),Fraction(truth)+Fraction(width)),
                [Fraction(float(v)) for v in rng.normal(size=len(t))])
            self.assertLessEqual(margin,0)

    def test_error_budget_monotonic_and_bad_inputs_rejected(self):
        t,y,truth=self.fixture()
        small=outer_period_set(t,y,5.,10.,[1.,2.,2.],{'noise':.002},resolution_s=.05,max_boxes=1000)
        large=outer_period_set(t,y,5.,10.,[1.,2.,2.],{'noise':.002,'model':.1},resolution_s=.05,max_boxes=1000)
        def width(r):return sum(b-a for a,b in r['outer_intervals_s'])
        self.assertGreater(width(large),width(small))
        for bounds in ({},{'bad':-.1},{'bad':float('nan')}):
            with self.assertRaises(ValueError):outer_period_set(t,y,5.,10.,[1.,2.,2.],bounds)

    def test_extreme_finite_domain_has_finite_json_artifact(self):
        import json
        upper=np.finfo(float).max
        result=outer_period_set([0.,0.],[0.,0.],1.,upper,[0.,0.,0.],
                                {'zero':0.},max_boxes=1,estimate_s=1.)
        self.assertGreaterEqual(result['outer_intervals_s'][-1][1],upper)
        self.assertGreaterEqual(Fraction(result['conditional_error_bound_s']),Fraction(upper)-1)
        json.dumps(result,allow_nan=False)

    def test_constant_feasibility_exact_touch_and_one_ulp_failure(self):
        # Intersections touch at cap0=1 exactly; both declared components matter.
        def solve(values):
            return outer_period_set([0.,1.],values,3.,4.,[1.,0.,0.],
                                    {'first':.25,'second':.25},max_boxes=1,estimate_s=3.25)
        result=solve([.5,1.5])
        self.assertEqual(result['status'],'all_domain_feasible')
        self.assertEqual(result['constant_feasible_witness_exact'],['1','1'])
        self.assertEqual(result['outer_intervals_s'],[[3.,4.]])
        self.assertEqual(result['conditional_error_bound_s'],.75)
        self.assertEqual(result['visited_boxes'],0)
        # Rounding the lower endpoint would wrongly claim all periods feasible.
        result=solve([.5,np.nextafter(1.5,np.inf)])
        self.assertFalse(result['all_domain_feasible_proved'])
        self.assertIsNone(result['constant_feasible_witness_exact'])
        self.assertNotEqual(result['status'],'all_domain_feasible')

    def test_h2_anisotropic_caps_pointwise_errors_and_epoch_reference(self):
        import mpmath as mp
        epoch=2.**45
        times=epoch+np.array([0.,.125,.5,1.375,3.25,7.125])
        relative=[Fraction(float(z))-Fraction(epoch) for z in times]
        truth=Fraction(13,4)
        coeff=[.125,0.,.25,.5,0.]
        with mp.workdps(90):
            values=np.array([float(mp.mpf('.125')+mp.mpf('.25')*mp.sin(2*mp.pi*mp.mpf(t.numerator)/t.denominator/(mp.mpf(13)/4))+
                                   mp.mpf('.5')*mp.cos(4*mp.pi*mp.mpf(t.numerator)/t.denominator/(mp.mpf(13)/4))) for t in relative])
        errors=np.arange(1,len(times)+1)*1e-12
        witness=[Fraction(float(z)) for z in np.random.default_rng(71).normal(size=len(times))]
        for box in ((truth,truth),(Fraction(3),truth),(truth,Fraction(4))):
            margin=exclusion_margin(relative,[Fraction(float(z)) for z in values],
                                    [Fraction(float(z)) for z in errors],[Fraction(z) for z in coeff],box,witness)
            self.assertLessEqual(margin,0)
        # The epoch affects the declared origin only: caps refer to t_i-t_0.
        result=outer_period_set(times,values,3.,4.,coeff,{'pointwise':errors},max_boxes=1)
        self.assertTrue(any(a<=float(truth)<=b for a,b in result['outer_intervals_s']))

    def test_trig_encloses_random_exact_binary_inputs_after_large_range_reduction(self):
        import mpmath as mp
        rng=np.random.default_rng(141)
        with mp.workdps(90):
            for q in [Fraction(float(z)) for z in rng.uniform(-10.,10.,20)]+[Fraction(2**1000+7,13)]:
                reduced=q-(q+Fraction(1,2))//1
                angle=2*mp.pi*mp.mpf(reduced.numerator)/reduced.denominator
                sine,cosine=trig_enclosure(q)
                for interval,value in ((sine,mp.sin(angle)),(cosine,mp.cos(angle))):
                    self.assertLessEqual(mp.mpf(str(interval.low)),value)
                    self.assertGreaterEqual(mp.mpf(str(interval.high)),value)
