#!/usr/bin/env python3
"""Build a factual ledger from the pinned upstream README tables, plus labeled interpretation."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'research/optimizer_lab'
PIN = '4ea6b937337a4889b8cfe3f38a93d120048d8f71'
BASE = f'https://github.com/KellerJordan/modded-nanogpt/blob/{PIN}/'
T3_NOTES = {
1:'Muon plus auxiliary Adam; initial single-run result.',
2:'Adam baseline, explicitly described as likely undertuned.',
3:'Muon learning-rate and decay tuning.',
4:'AdamH: Adam preconditioning with a hyperball constraint.',
5:'MuonH: hyperball constraint, initialization and schedules.',
6:'Muon plus auxiliary Adam tuning, with more seeds.',
7:'Muon squared: adaptive preconditioning before polar direction.',
8:'NorMuonH: row/column adaptivity combined with norm constraint.',
9:'NorMuon plus minimum update-to-weight norm ratio.',
10:'NorMuon learning-rate/decay tuning.',
11:'Contra-Muon added; pairwise improvement over #9 not significant.',
12:'Plain Muon retuned using the NorMuon recipe.',
13:'MuLoCo-style outer Nesterov around NorMuonH.',
14:'SOAP covariance-basis preconditioning before Muon on MLPs.',
15:'Newton-Muon: activation covariance right-preconditioning.',
16:'Attention SOAP with a trust gate; weak pairwise evidence.',
17:'Aurora applied to the Contra-Muon recipe.',
18:'PMuon: bilateral streaming covariance power preconditioning.',
19:'KL-SOAP with hyperball constraint.',
20:'PowerCool schedule and Contra/Soft-Muon blending.',
21:'Shampoo baseline with inverse-fourth-root preconditioning.',
22:'Spectral descent without momentum; expensive in steps.',
23:'Muown: integrated row-norm control and schedule changes.',
24:'Different cooldown lengths for matrix and auxiliary updates.',
25:'KL-SOAP-H plus PowerCool and nonzero LR floors.',
26:'SinkSOAP: Gram-Sinkhorn preconditioning plus NorMuon postconditioner.',
27:'SOAP-H: SOAP plus hyperball and tuned schedule.',
28:'DynMuon: scheduled spectral transformation.',
29:'Radial brake: suppress outward updates before norm floor.',
30:'Aurora on MLP projections; prune/tune geometry and momentum schedule.',
31:'Muown combined with NorMuon and Contra-Muon.',
32:'SODA-style correction toward initialization, faded late.',
33:'PSGD with Kronecker whitening and hyperball constraint.',
34:'Late reduced-rank extrapolation; weak pairwise evidence.',
35:'One-sided Shampoo, pseudoinverse root and Adam grafting.',
36:'Strong tuned simple Muon plus auxiliary AdamW baseline.',
37:'MuonH benefits from retuning auxiliary Adam parameters.',
38:'Late weight readout and tempered-polar/schedule adjustments.',
39:'EMA-Nesterov lookahead around plain Muon.',
40:'EMA-Nesterov lookahead added to the stronger #30 recipe.',
41:'Circuit-Muon coupling of attention value/output weights.',
42:'Late reference extrapolation and initialization changes.',
43:'Several fixed late-trajectory transformations and anchor blending.',
44:'SOAP on all hidden matrices every step; auxiliary tuning; remove geometry modules.',
45:'Late EMA-weight blending at evaluation.',
46:'Per-row update floor plus post-rescale cautious decay.'}
T1_NOTES = {
1:'AdamW baseline.',2:'Learning-rate tuning alongside rotary architecture change.',
3:'Muon enters for hidden matrices; auxiliary Adam-family updates remain.',
4:'Muon refinements.',6:'Distribute optimizer overhead across GPUs.',
9:'Momentum warmup alongside architecture/logit changes.',
19:'Nonzero terminal learning-rate floor alongside FP8-head changes.',
20:'Batched Muon and Adam epsilon, alongside attention/layout changes.',
22:'Faster gradient reduction.',23:'Overlap communication and computation.',24:'Reduce-scatter replaces all-reduce.',
27:'Symmetric-matmul Triton kernel and matrix orientation.',
36:'Shape-aware Muon and combined gradient reductions.',
38:'Polar Express orthogonalization polynomial schedule.',
39:'Auxiliary Adam updated every other step; batch also changed.',
41:'NorMuon adaptive rescaling enters the record.',42:'NorMuon learning rate and step-logic correction.',
43:'Cautious weight decay with a schedule.',44:'Adam backward hooks and synchronization overlap.',
46:'Batch-size schedule changes the optimizer workload.',
48:'NorMuon axis, matrix layout, implementation and LR fixes.',
50:'Cautious decay extended to Adam parameters.',
52:'Auxiliary beta/LR tuning and scalar freezing at transitions.',
53:'Decay/LR changes accompany MTP and embedding untying.',
56:'Compile Adam; higher-precision state; move gates to Adam groups.',
57:'Mixed-precision Muon and interleaved Adam/Muon execution.',
61:'Unified optimizer implementation and transposed head.',
65:'Group value embeddings into one parameter (layout/system effect).',
71:'Sparse bigram gradient communication.',
72:'Higher minimum LR plus sequence-length schedule.',
80:'Orthogonalize attention Q/K by paired heads.',
91:'Canonical-token evaluation masking; not a new optimizer.',
92:'ANVIL and sparse Adam, bundled with substantial model/loss/systems changes.'}

def family(n):
    if n <= 2: return 'AdamW'
    if n <= 40: return 'Muon + auxiliary Adam-family'
    if n <= 91: return 'NorMuon/Muon family + auxiliary Adam-family'
    return 'ANVIL + auxiliary/sparse Adam'

def parse():
    t3=[]
    path='records/track_3_optimization/README.md'
    for line in (ROOT/path).read_text().splitlines():
        if not re.match(r'^\| \d+ \|',line): continue
        c=[x.strip().strip('|').strip() for x in line.split(' | ')]
        n=int(c[0]); prs=re.findall(r'https://github.com/KellerJordan/modded-nanogpt/pull/\d+',line)
        t3.append(dict(id=n,steps=int(re.search(r'\d+',c[1]).group()),world_record='(!)' in c[1],date=c[4],
                       evidence=c[2],notes=T3_NOTES[n],source=prs[-1] if prs else BASE+path))
    t1=[]
    table=(ROOT/'README.md').read_text().split('## World record history',1)[1].split('## Rules',1)[0]
    for line in table.splitlines():
        if not re.match(r'^\d+ \|',line): continue
        c=[x.strip() for x in line.split(' | ')]
        n=int(c[0]); prs=re.findall(r'https://github.com/KellerJordan/modded-nanogpt/pull/\d+',line)
        t1.append(dict(id=n,time=c[1],date=c[3],family=family(n),
                       optimizer_note=T1_NOTES.get(n,'No optimizer milestone singled out here; see original entry for architectural, systems or recipe changes.'),
                       source=prs[-1] if prs else BASE+'README.md#world-record-history'))
    return t3,t1

def main():
    t3,t1=parse()
    assert [x['id'] for x in t3]==list(range(1,47))
    assert set(x['id'] for x in t1)==set(range(1,93))
    out={'upstream_commit':PIN,'as_of':'2026-10-05','track3':t3,'track1':t1,
         'caveats':['Track 3 ! flags follow upstream, including ties/historical non-significant records.',
                    'Dates are table submission/result dates, not necessarily merge dates; ordering is acceptance ID.',
                    'Track 1 family labels are coarse lineage inferred from documented transitions, not an audit of every historical kernel.',
                    'Track 1 times reflect entire recipes and hardware/timing conventions, not isolated optimizer causal effects.',
                    'Record 21 appears multiple times because upstream records retimings.']}
    (OUT/'history.json').write_text(json.dumps(out,indent=2)+'\n')
    lines=['# Optimizer history: what changed, and what survived','',
           'Snapshot: 2026-10-05; upstream commit `'+PIN+'`. This is an evidence ledger, not an assertion that every accepted change has an independently proven causal benefit.','',
           '## Track 3: all upstream-marked world records','',
           'Order follows accepted record ID. Dates below are the upstream result/submission dates, which can be out of order. The first historical records fail the later statistical rule. The `!` label is upstream\'s; #9 ties a preceding 3250-step entry.','',
           '| ID | Listed date | Steps | Change |','|---|---|---:|---|']
    for x in t3:
        if x['world_record']: lines.append(f"| [#{x['id']}]({x['source']}) | {x['date']} | {x['steps']} | {x['notes']} |")
    lines += ['', '### Interpretation (our research hypotheses, not causal proof)','',
              '- Momentum plus a matrix-aware direction is a strong starting point; the no-momentum spectral-descent result is far worse, but this is not a controlled one-knob comparison.',
              '- Covariance-aware preconditioning is the most persistent major branch in the current champion. The important research problem is how cheaply to retain its benefit.',
              '- Norm/radius control, auxiliary Adam tuning and late averaging matter enough to confound an allegedly new update rule. Test those separately.',
              '- Not everything accumulates: #44 removes Circuit-Muon, Contra-Muon and Aurora while improving the whole recipe. Their historical success does not prove they help in the final combination.',
              '- The reported pairwise p-values for #11 vs #9 (0.69), #16 vs #14 (0.34), and #34 vs #30 (0.168) do not establish superiority. Passing the target threshold is a different hypothesis.',
              '- 3600 -> 2690 is about 25.3% fewer steps; the better-tuned simple baseline #36 is 3250, making the champion about 17.2% fewer steps. Neither is an isolated-optimizer wall-time gain.',
              '', '## Track 3: all 46 accepted entries, including non-record alternatives','',
              'This table prevents omission of useful competing branches merely because they did not set a global record. No claim that the optimizer family was exhaustively tuned.','',
              '| ID | Date | Steps | Global record flag | Evidence | Method / lesson |','|---|---|---:|---|---|---|']
    for x in t3:
        lines.append(f"| [#{x['id']}]({x['source']}) | {x['date']} | {x['steps']} | {'!' if x['world_record'] else '-'} | {x['evidence']} | {x['notes']} |")
    lines += ['', '## Track 1: optimizer-relevant milestones','',
              'The speedrun is not an optimizer-only leaderboard. These are whole-recipe record times; architecture, precision, batch schedule, communication and timing conventions change too. The historical 21st record was retimed; do not interpret the full series as a controlled optimizer sweep.','',
              '| ID | Date | Whole-recipe time | Optimizer-related change |','|---|---|---|---|']
    seen=set()
    for x in t1:
        if x['id'] in T1_NOTES and x['id'] not in seen:
            lines.append(f"| [#{x['id']}]({x['source']}) | {x['date']} | {x['time']} | {x['optimizer_note']} |")
            seen.add(x['id'])
    lines += ['', '### Full record-to-family index','',
              'Every Track 1 record is listed below. Family labels are COARSE INFERENCES from documented introductions: AdamW (#1-2), Muon plus auxiliary Adam-family (#3-40), NorMuon/Muon plus auxiliary Adam-family (#41-91), ANVIL with auxiliary/sparse Adam (#92). This is not independent source-level verification of every record. Parameters such as gates/embeddings can move between optimizer groups.','',
              '| ID | Date | Time | Coarse family |','|---|---|---|---|']
    for x in t1:
        lines.append(f"| [#{x['id']}]({x['source']}) | {x['date']} | {x['time']} | {x['family']} |")
    lines += ['', '### What not to infer','',
              '- Most speedrun gains after Muon did not come from replacing it with a wholly new optimizer. Layout, kernels, precision, communication, cadence and schedules mattered.',
              '- NorMuon normalization, cautious decay, momentum scheduling and auxiliary updates suggest inexpensive mechanisms worth testing across tasks, not guaranteed transferable wins.',
              '- #92 is based on #89, not a clean incremental ablation of #91. ANVIL is one component of a bundle that also changes the model and training loss. The 39.9s figure cannot be attributed to ANVIL alone.',
              '- Original certified #92 and current root trainer differ: #373 later adds #91 canonical masking and other changes. Exact replication must pin the certified artifact.',
              '- A broad optimizer breakthrough needs robust gains beyond one tuned architecture, one tiny endpoint margin and one favorable implementation.','']
    (OUT/'HISTORY.md').write_text('\n'.join(lines))
    print(f'Generated {len(t3)} Track 3 entries ({sum(x["world_record"] for x in t3)} marked records), {len(t1)} Track 1 rows / 92 unique records.')

if __name__=='__main__': main()
