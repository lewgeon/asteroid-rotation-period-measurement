"""Conditional continuous-period exclusion for bounded harmonic observations.

This research API certifies exclusions for the declared scalar-feature model,
not that the declarations describe a physical radar. Numpy chooses witnesses;
only directed Decimal interval arithmetic may exclude a period box.
"""
from decimal import Context,Decimal,ROUND_FLOOR,ROUND_CEILING
from fractions import Fraction
import math
import numpy as np

_LOW=Context(prec=50,rounding=ROUND_FLOOR)
_HIGH=Context(prec=50,rounding=ROUND_CEILING)


class Interval:
    def __init__(self,low,high=None):
        self.low=Decimal(low);self.high=self.low if high is None else Decimal(high)
    @classmethod
    def rational(cls,value):
        f=Fraction(value)
        return cls(_LOW.divide(Decimal(f.numerator),Decimal(f.denominator)),
                   _HIGH.divide(Decimal(f.numerator),Decimal(f.denominator)))
    def __add__(self,other):
        if not isinstance(other,Interval):other=Interval.rational(other)
        return Interval(_LOW.add(self.low,other.low),_HIGH.add(self.high,other.high))
    __radd__=__add__
    def __neg__(self):return Interval(self.high.copy_negate(),self.low.copy_negate())
    def __sub__(self,other):return self+-other
    def __mul__(self,other):
        if not isinstance(other,Interval):other=Interval.rational(other)
        pairs=[(a,b) for a in (self.low,self.high) for b in (other.low,other.high)]
        return Interval(min(_LOW.multiply(a,b) for a,b in pairs),max(_HIGH.multiply(a,b) for a,b in pairs))
    __rmul__=__mul__
    def magnitude(self):return max(self.low.copy_abs(),self.high.copy_abs())


def _atan_bounds(denominator,terms=70):
    q=Fraction(1,denominator)
    value=sum(((-1)**k*q**(2*k+1)/ (2*k+1) for k in range(terms)),Fraction(0))
    next_term=(-1)**terms*q**(2*terms+1)/(2*terms+1)
    return min(value,value+next_term),max(value,value+next_term)


_a,_b=_atan_bounds(5);_c,_d=_atan_bounds(239)
# Machin identity plus alternating-series remainder: no floating pi constant.
_PI=Interval(Interval.rational(16*_a-4*_d).low,Interval.rational(16*_b-4*_c).high)


def trig_enclosure(cycles):
    """Return sin/cos(2*pi*cycles), enclosing exact rational input."""
    q=2*Fraction(cycles)
    q=q-2*((q+1)//2)  # [-1,1), exact range reduction before using pi.
    x=_PI*Interval.rational(q);square=x*x
    sine=Interval(0);cosine=Interval(0);sin_term=x;cos_term=Interval(1)
    terms=34
    for k in range(terms):
        sine=sine+sin_term;cosine=cosine+cos_term
        sin_term=sin_term*(-square)*Interval.rational(Fraction(1,(2*k+2)*(2*k+3)))
        cos_term=cos_term*(-square)*Interval.rational(Fraction(1,(2*k+1)*(2*k+2)))
    # Taylor/Lagrange remainder on |x|<4, derivative magnitude <=1.
    sr=Interval.rational(Fraction(4**(2*terms+1),math.factorial(2*terms+1))).high
    cr=Interval.rational(Fraction(4**(2*terms),math.factorial(2*terms))).high
    return sine+Interval(sr.copy_negate(),sr),cosine+Interval(cr.copy_negate(),cr)


def _fraction(value):return Fraction.from_float(float(value))


def _outward_float(value,upper):
    value=Fraction(value);number=float(value);rounded=_fraction(number)
    if (upper and rounded<value) or (not upper and rounded>value):
        number=float(np.nextafter(number,np.inf if upper else -np.inf))
    return number


def exclusion_margin(times,values,errors,caps,period_box,witness):
    """Positive lower endpoint proves no allowed coefficients fit this box.

    Contract: observations, times, bounds and witness are exact binary64 input
    values. Errors are pointwise absolute bounds in the same feature units.
    """
    left,right=map(Fraction,period_box);mid=(left+right)/2
    if left<=0 or right<left:raise ValueError('Period box must be positive and ordered')
    t=[Fraction(v) for v in times];y=[Fraction(v) for v in values]
    v=[Fraction(z) for z in witness];e=[Fraction(z) for z in errors]
    c=[Fraction(z) for z in caps];h=(len(c)-1)//2
    if not len(t)==len(y)==len(v)==len(e) or not t or len(c)<3 or len(c)%2!=1:
        raise ValueError('Witness arrays must have equal nonzero length and valid harmonic caps')
    if any(z<0 for z in e+c):raise ValueError('Error bounds and coefficient caps must be nonnegative')
    columns=[Interval(0) for _ in caps]
    dot=Interval(0);noise=Interval(0);slope=Interval(0)
    inverse_radius=max(abs(1/left-1/mid),abs(1/right-1/mid))
    for ti,yi,vi,ei in zip(t,y,v,e):
        dot=dot+Interval.rational(vi*yi)
        noise=noise+Interval.rational(abs(vi)*ei)
        columns[0]=columns[0]+Interval.rational(vi)
        slope=slope+Interval.rational(abs(vi*ti))*_PI*2*Interval.rational(inverse_radius)
        for harmonic in range(1,h+1):
            sine,cosine=trig_enclosure(harmonic*ti/mid)
            columns[2*harmonic-1]=columns[2*harmonic-1]+cosine*Interval.rational(vi)
            columns[2*harmonic]=columns[2*harmonic]+sine*Interval.rational(vi)
    nuisance=Interval.rational(c[0])*Interval(columns[0].magnitude())
    for harmonic in range(1,h+1):
        for j in (2*harmonic-1,2*harmonic):
            nuisance=nuisance+Interval.rational(c[j])*(Interval(columns[j].magnitude())+harmonic*slope)
    return (dot-nuisance-noise).low


def outer_period_set(times,values,period_min,period_max,coefficient_caps,error_components,*,
                     resolution_s=.1,max_boxes=20000,estimate_s=None,progress=None):
    """Cover every compatible continuous period, retaining all undecided boxes.

    y_i = a0 + sum_h(ah*cos(2*pi*h*(t_i-t_0)/T)+bh*sin(...)) + error_i.
    Absolute bounds on every coefficient and additive error component must be
    supplied independently; no residual-based bound is inferred here.
    """
    t=np.asarray(times,float);y=np.asarray(values,float);caps=np.asarray(coefficient_caps,float)
    if t.ndim!=1 or y.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError('Finite equal-length 1D observations required')
    if caps.ndim!=1 or len(caps)<3 or len(caps)%2!=1 or np.any(caps<0) or not np.all(np.isfinite(caps)):
        raise ValueError('Finite nonnegative caps for offset and harmonic cos/sin coefficients required')
    if not error_components:raise ValueError('Explicit independently justified error components required')
    if not all(np.isfinite(z) for z in (period_min,period_max,resolution_s)) or not 0<period_min<period_max or resolution_s<=0:
        raise ValueError('Positive finite period domain and resolution required')
    if isinstance(max_boxes,bool) or not isinstance(max_boxes,int) or max_boxes<1:raise ValueError('Positive integer max_boxes required')
    errors=[Fraction(0) for _ in t];components={}
    for name,value in error_components.items():
        a=np.broadcast_to(np.asarray(value,float),t.shape)
        if not np.all(np.isfinite(a)) or np.any(a<0):raise ValueError('Nonnegative finite component bounds required')
        components[name]=a.tolist()
        errors=[old+_fraction(new) for old,new in zip(errors,a)]
    # Exact subtraction ensures very large epoch origins do not change the model.
    exact_t=[_fraction(z)-_fraction(t[0]) for z in t]
    try:tf=np.array([float(z) for z in exact_t])
    except OverflowError:raise ValueError('Time span exceeds supported floating witness range')
    if not np.all(np.isfinite(tf)):raise ValueError('Finite relative time span required')
    exact_y=[_fraction(z) for z in y];exact_caps=[_fraction(z) for z in caps]
    pending=[(_fraction(period_min),_fraction(period_max))];retained=[];excluded=[];visited=0
    # A common admissible constant fits every period, not merely grid points.
    constant_low=max([-exact_caps[0]]+[value-error for value,error in zip(exact_y,errors)])
    constant_high=min([exact_caps[0]]+[value+error for value,error in zip(exact_y,errors)])
    all_feasible=constant_low<=constant_high
    if all_feasible:retained=pending;pending=[]
    resolution=_fraction(resolution_s);h=(len(caps)-1)//2
    while pending and visited<max_boxes:
        box=pending.pop();left,right=box;middle=(left+right)/2;visited+=1
        # Cycle reduction also keeps the non-certifying proposal finite.
        cycles=np.array([float((ti/middle)%1) for ti in exact_t])
        phase=2*np.pi*cycles
        x=np.column_stack([np.ones(len(t))]+[f(k*phase) for k in range(1,h+1) for f in (np.cos,np.sin)])
        # Approximate optimization merely proposes a witness; never rejects data.
        try:
            residual=y-x@np.linalg.lstsq(x,y,rcond=None)[0]
            scale=np.max(abs(residual));witness=residual/scale if scale else residual
            if not np.all(np.isfinite(witness)):witness=np.zeros(len(t))
        except np.linalg.LinAlgError:witness=np.zeros(len(t))
        margin=exclusion_margin(exact_t,exact_y,errors,exact_caps,box,[_fraction(z) for z in witness])
        if margin>0:excluded.append((box,str(margin),witness.tolist()))
        elif right-left<=resolution:retained.append(box)
        else:pending.extend(((middle,right),(left,middle)))
        if progress and visited%100==0:progress(visited,len(pending))
    exhausted=bool(pending);retained+=pending
    merged=[]
    for left,right in sorted(retained):
        if merged and left<=merged[-1][1]:merged[-1]=(merged[-1][0],max(right,merged[-1][1]))
        else:merged.append((left,right))
    def outward(box):return [_outward_float(box[0],False),_outward_float(box[1],True)]
    intervals=[outward(box) for box in merged]
    bound=None
    if estimate_s is not None:
        if not np.isfinite(estimate_s) or not period_min<=estimate_s<=period_max:raise ValueError('Estimate must lie in the declared search domain')
        if merged:
            distance=max(abs(_fraction(estimate_s)-z) for box in merged for z in box)
            bound=_outward_float(distance,True)
    return dict(status='all_domain_feasible' if all_feasible else 'budget_exhausted_outer_cover' if exhausted else 'conditional_outer_cover',
        compatible_set_nonempty_proved=all_feasible,all_domain_feasible_proved=all_feasible,
        constant_feasible_witness_exact=([str(constant_low.numerator),str(constant_low.denominator)] if all_feasible else None),
        outer_intervals_s=intervals,conditional_error_bound_s=bound,
        estimate_s=estimate_s,period_domain_s=[period_min,period_max],resolution_s=resolution_s,
        visited_boxes=visited,excluded_boxes=len(excluded),retained_boxes=len(retained),
        error_components=components,coefficient_caps=caps.tolist(),
        numerical_method='binary64 inputs interpreted exactly; directed Decimal50 enclosure; Machin pi bounds; Taylor remainder; dual witness exclusion',
        time_origin_s=float(t[0]),observation_count=len(t),
        exclusions=[dict(box_s_approx=[float(box[0]),float(box[1])],
            box_exact_rational=[[str(z.numerator),str(z.denominator)] for z in box],
            margin_lower=margin,witness=witness) for box,margin,witness in excluded],
        assumptions='Declared harmonic family, coefficient caps and each pointwise total error bound must contain the true process. These are external premises, not established by this solver.',
        physical_model_error_validated=False,
        warning='Outer cover may include impossible periods; finite resolution and resource limit never discard unresolved boxes. Empty cover means incompatible premises/data, not a zero uncertainty result.')
