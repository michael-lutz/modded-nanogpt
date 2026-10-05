#!/usr/bin/env python3
"""Extra pilot check: captured backward vs uncaptured compiled backward on same inputs.
Run with the same CLI as gh200_training_smoke.py; stores checks beside results.json.
Does not certify mixed-width FA3 or 8-rank mathematical equivalence.
"""
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import gh200_training_smoke as smoke
original=smoke.ModelRunner.capture

def checked(self):
    original(self)
    assert self.graphs
    self.vgrad.zero_()
    self.fgraph.replay();self.bgraph.replay();torch.cuda.synchronize()
    reference=[None if g is None else g.clone() for g in self.grads]
    vr=self.vgrad.clone();self.vgrad.zero_()
    eager=torch.autograd.grad(self.loss(),self.params+[self.sink],allow_unused=True)
    torch.cuda.synchronize()
    names=[n for n,p in self.model.named_parameters()]+['ngram_sink']
    checks=[]
    for name,a,b in zip(names,reference,eager):
        assert (a is None)==(b is None),name
        if a is None:continue
        af,bf=a.float(),b.float()
        rel=float((af-bf).norm()/bf.norm().clamp_min(1e-12))
        maxabs=float((af-bf).abs().max())
        assert bool(torch.isfinite(af).all()) and bool(torch.isfinite(bf).all()),name
        assert rel<.01,(name,rel,maxabs)
        checks.append({'name':name,'relative_l2_error':rel,'max_absolute_error':maxabs,'reference_l2':float(bf.norm())})
    rel=float((vr.float()-self.vgrad.float()).norm()/self.vgrad.float().norm().clamp_min(1e-12))
    assert rel<.01,('value_embed_grad',rel)
    checks.append({'name':'value_embed_grad','relative_l2_error':rel,'reference_l2':float(self.vgrad.float().norm())})
    self.vgrad.zero_()
    out=self.a.out/'graph_gradient_check.json';out.write_text(json.dumps({'passed':True,'checks':checks},indent=2))
    print('BACKWARD CHECK PASSED',max(c['relative_l2_error'] for c in checks),flush=True)
smoke.ModelRunner.capture=checked
if __name__=='__main__':smoke.main()
