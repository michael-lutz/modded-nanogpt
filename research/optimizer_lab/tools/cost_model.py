#!/usr/bin/env python3
"""Conditional cost arithmetic, not a substitute for profiling."""
import argparse
import json
import math

def estimate(step_ratio, optimizer_fraction, optimizer_cost_ratio):
    vals=(step_ratio,optimizer_fraction,optimizer_cost_ratio)
    if not all(math.isfinite(x) for x in vals): raise ValueError('Inputs must be finite')
    if step_ratio<=0 or not 0<=optimizer_fraction<=1 or optimizer_cost_ratio<0:
        raise ValueError('Require s>0, 0<=p<=1 and k>=0')
    ratio=1+optimizer_fraction*(optimizer_cost_ratio-1)
    if ratio<=0: raise ValueError('Degenerate zero-cost step is not a meaningful projection')
    return {'step_time_ratio':ratio,'training_time_ratio':step_ratio*ratio,
            'break_even_step_ratio':1/ratio,
            'assumptions':'Identical work/batching except optimizer; no overlap change; timings not measured.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--step-ratio',type=float,required=True)
    p.add_argument('--optimizer-fraction',type=float,required=True)
    p.add_argument('--optimizer-cost-ratio',type=float,required=True)
    a=p.parse_args()
    print(json.dumps(estimate(a.step_ratio,a.optimizer_fraction,a.optimizer_cost_ratio),indent=2))
