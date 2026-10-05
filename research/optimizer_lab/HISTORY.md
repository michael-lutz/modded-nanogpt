# Optimizer history: what changed, and what survived

Snapshot: 2026-10-05; upstream commit `4ea6b937337a4889b8cfe3f38a93d120048d8f71`. This is an evidence ledger, not an assertion that every accepted change has an independently proven causal benefit.

## Track 3: all upstream-marked world records

Order follows accepted record ID. Dates below are the upstream result/submission dates, which can be out of order. The first historical records fail the later statistical rule. The `!` label is upstream's; #9 ties a preceding 3250-step entry.

| ID | Listed date | Steps | Change |
|---|---|---:|---|
| [#1](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_3_optimization/README.md) | 2026/04/26 | 3600 | Muon plus auxiliary Adam; initial single-run result. |
| [#3](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_3_optimization/README.md) | 2026/04/26 | 3500 | Muon learning-rate and decay tuning. |
| [#5](https://github.com/KellerJordan/modded-nanogpt/pull/267) | 2026/04/30 | 3325 | MuonH: hyperball constraint, initialization and schedules. |
| [#9](https://github.com/KellerJordan/modded-nanogpt/pull/274) | 2026/04/29 | 3250 | NorMuon plus minimum update-to-weight norm ratio. |
| [#11](https://github.com/KellerJordan/modded-nanogpt/pull/275) | 2026/05/01 | 3225 | Contra-Muon added; pairwise improvement over #9 not significant. |
| [#13](https://github.com/KellerJordan/modded-nanogpt/pull/277) | 2026/05/04 | 3210 | MuLoCo-style outer Nesterov around NorMuonH. |
| [#14](https://github.com/KellerJordan/modded-nanogpt/pull/278) | 2026/05/04 | 3150 | SOAP covariance-basis preconditioning before Muon on MLPs. |
| [#16](https://github.com/KellerJordan/modded-nanogpt/pull/283) | 2026/05/05 | 3125 | Attention SOAP with a trust gate; weak pairwise evidence. |
| [#20](https://github.com/KellerJordan/modded-nanogpt/pull/291) | 2026/05/09 | 3030 | PowerCool schedule and Contra/Soft-Muon blending. |
| [#29](https://github.com/KellerJordan/modded-nanogpt/pull/294) | 2026/05/11 | 2990 | Radial brake: suppress outward updates before norm floor. |
| [#30](https://github.com/KellerJordan/modded-nanogpt/pull/300) | 2026/05/14 | 2930 | Aurora on MLP projections; prune/tune geometry and momentum schedule. |
| [#34](https://github.com/KellerJordan/modded-nanogpt/pull/305) | 2026/05/20 | 2925 | Late reduced-rank extrapolation; weak pairwise evidence. |
| [#38](https://github.com/KellerJordan/modded-nanogpt/pull/307) | 2026/05/20 | 2900 | Late weight readout and tempered-polar/schedule adjustments. |
| [#40](https://github.com/KellerJordan/modded-nanogpt/pull/309) | 2026/05/22 | 2890 | EMA-Nesterov lookahead added to the stronger #30 recipe. |
| [#41](https://github.com/KellerJordan/modded-nanogpt/pull/311) | 2026/05/23 | 2875 | Circuit-Muon coupling of attention value/output weights. |
| [#42](https://github.com/KellerJordan/modded-nanogpt/pull/312) | 2026/05/25 | 2860 | Late reference extrapolation and initialization changes. |
| [#43](https://github.com/KellerJordan/modded-nanogpt/pull/318) | 2026/05/29 | 2850 | Several fixed late-trajectory transformations and anchor blending. |
| [#44](https://github.com/KellerJordan/modded-nanogpt/pull/321) | 2026/06/10 | 2750 | SOAP on all hidden matrices every step; auxiliary tuning; remove geometry modules. |
| [#45](https://github.com/KellerJordan/modded-nanogpt/pull/325) | 2026/06/12 | 2720 | Late EMA-weight blending at evaluation. |
| [#46](https://github.com/KellerJordan/modded-nanogpt/pull/328) | 2026/06/19 | 2690 | Per-row update floor plus post-rescale cautious decay. |

### Interpretation (our research hypotheses, not causal proof)

- Momentum plus a matrix-aware direction is a strong starting point; the no-momentum spectral-descent result is far worse, but this is not a controlled one-knob comparison.
- Covariance-aware preconditioning is the most persistent major branch in the current champion. The important research problem is how cheaply to retain its benefit.
- Norm/radius control, auxiliary Adam tuning and late averaging matter enough to confound an allegedly new update rule. Test those separately.
- Not everything accumulates: #44 removes Circuit-Muon, Contra-Muon and Aurora while improving the whole recipe. Their historical success does not prove they help in the final combination.
- The reported pairwise p-values for #11 vs #9 (0.69), #16 vs #14 (0.34), and #34 vs #30 (0.168) do not establish superiority. Passing the target threshold is a different hypothesis.
- 3600 -> 2690 is about 25.3% fewer steps; the better-tuned simple baseline #36 is 3250, making the champion about 17.2% fewer steps. Neither is an isolated-optimizer wall-time gain.

## Track 3: all 46 accepted entries, including non-record alternatives

This table prevents omission of useful competing branches merely because they did not set a global record. No claim that the optimizer family was exhaustively tuned.

| ID | Date | Steps | Global record flag | Evidence | Method / lesson |
|---|---|---:|---|---|---|
| [#1](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_3_optimization/README.md) | 2026/04/26 | 3600 | ! | 3.2777 (n=1)Ⓧ | Muon plus auxiliary Adam; initial single-run result. |
| [#2](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_3_optimization/README.md) | 2026/04/26 | 5625 | - | 3.2790 (n=1)Ⓧ | Adam baseline, explicitly described as likely undertuned. |
| [#3](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_3_optimization/README.md) | 2026/04/26 | 3500 | ! | 3.2767 (n=1)Ⓧ | Muon learning-rate and decay tuning. |
| [#4](https://github.com/KellerJordan/modded-nanogpt/pull/272) | 2026/04/30 | 4875 | - | 3.2741 (n=5)✓ | AdamH: Adam preconditioning with a hyperball constraint. |
| [#5](https://github.com/KellerJordan/modded-nanogpt/pull/267) | 2026/04/30 | 3325 | ! | 3.2782 (n=10)✓ | MuonH: hyperball constraint, initialization and schedules. |
| [#6](https://github.com/KellerJordan/modded-nanogpt/pull/271) | 2026/05/01 | 3375 | - | 3.2788 (n=20)✓ | Muon plus auxiliary Adam tuning, with more seeds. |
| [#7](https://github.com/KellerJordan/modded-nanogpt/pull/266) | 2026/04/29 | 3325 | - | 3.2752 (n=1)✓ | Muon squared: adaptive preconditioning before polar direction. |
| [#8](https://github.com/KellerJordan/modded-nanogpt/pull/273) | 2026/04/30 | 3250 | - | 3.2778 (n=10)✓ | NorMuonH: row/column adaptivity combined with norm constraint. |
| [#9](https://github.com/KellerJordan/modded-nanogpt/pull/274) | 2026/04/29 | 3250 | ! | 3.2771 (n=8)✓ | NorMuon plus minimum update-to-weight norm ratio. |
| [#10](https://github.com/KellerJordan/modded-nanogpt/pull/276) | 2026/05/03 | 3250 | - | 3.2789 (n=20)✓ | NorMuon learning-rate/decay tuning. |
| [#11](https://github.com/KellerJordan/modded-nanogpt/pull/275) | 2026/05/01 | 3225 | ! | 3.2785 (n=16)✓ | Contra-Muon added; pairwise improvement over #9 not significant. |
| [#12](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_3_optimization/README.md) | 2026/05/03 | 3325 | - | 3.2790 (n=20)✓ | Plain Muon retuned using the NorMuon recipe. |
| [#13](https://github.com/KellerJordan/modded-nanogpt/pull/277) | 2026/05/04 | 3210 | ! | 3.2785 (n=10)✓ | MuLoCo-style outer Nesterov around NorMuonH. |
| [#14](https://github.com/KellerJordan/modded-nanogpt/pull/278) | 2026/05/04 | 3150 | ! | 3.2776 (n=4)✓ | SOAP covariance-basis preconditioning before Muon on MLPs. |
| [#15](https://github.com/KellerJordan/modded-nanogpt/pull/281) | 2026/05/05 | 3275 | - | 3.2785 (n=15)✓ | Newton-Muon: activation covariance right-preconditioning. |
| [#16](https://github.com/KellerJordan/modded-nanogpt/pull/283) | 2026/05/05 | 3125 | ! | 3.2784 (n=8)✓ | Attention SOAP with a trust gate; weak pairwise evidence. |
| [#17](https://github.com/KellerJordan/modded-nanogpt/pull/284) | 2026/05/06 | 3175 | - | 3.2789 (n=20)✓ | Aurora applied to the Contra-Muon recipe. |
| [#18](https://github.com/KellerJordan/modded-nanogpt/pull/285) | 2026/05/07 | 3225 | - | 3.2776 (n=9)✓ | PMuon: bilateral streaming covariance power preconditioning. |
| [#19](https://github.com/KellerJordan/modded-nanogpt/pull/290) | 2026/05/08 | 3125 | - | 3.2780 (n=6)✓ | KL-SOAP with hyperball constraint. |
| [#20](https://github.com/KellerJordan/modded-nanogpt/pull/291) | 2026/05/09 | 3030 | ! | 3.2790 (n=30)✓ | PowerCool schedule and Contra/Soft-Muon blending. |
| [#21](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_3_optimization/README.md) | 2026/05/13 | 4100 | - | 3.2776 (n=4)✓ | Shampoo baseline with inverse-fourth-root preconditioning. |
| [#22](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_3_optimization/README.md) | 2026/05/17 | 8225 | - | 3.2774 (n=4)✓ | Spectral descent without momentum; expensive in steps. |
| [#23](https://github.com/KellerJordan/modded-nanogpt/pull/288) | 2026/05/10 | 3075 | - | 3.2790 (n=30)✓ | Muown: integrated row-norm control and schedule changes. |
| [#24](https://github.com/KellerJordan/modded-nanogpt/pull/292) | 2026/05/09 | 3175 | - | 3.2782 (n=10)✓ | Different cooldown lengths for matrix and auxiliary updates. |
| [#25](https://github.com/KellerJordan/modded-nanogpt/pull/293) | 2026/05/11 | 3040 | - | 3.2781 (n=5)✓ | KL-SOAP-H plus PowerCool and nonzero LR floors. |
| [#26](https://github.com/KellerJordan/modded-nanogpt/pull/298) | 2026/05/14 | 3090 | - | 3.2785 (n=10)✓ | SinkSOAP: Gram-Sinkhorn preconditioning plus NorMuon postconditioner. |
| [#27](https://github.com/KellerJordan/modded-nanogpt/pull/302) | 2026/05/18 | 3125 | - | 3.2782 (n=6)✓ | SOAP-H: SOAP plus hyperball and tuned schedule. |
| [#28](https://github.com/KellerJordan/modded-nanogpt/pull/304) | 2026/05/19 | 3175 | - | 3.2790 (n=25)✓ | DynMuon: scheduled spectral transformation. |
| [#29](https://github.com/KellerJordan/modded-nanogpt/pull/294) | 2026/05/11 | 2990 | ! | 3.2787 (n=11)✓ | Radial brake: suppress outward updates before norm floor. |
| [#30](https://github.com/KellerJordan/modded-nanogpt/pull/300) | 2026/05/14 | 2930 | ! | 3.2784 (n=16)✓ | Aurora on MLP projections; prune/tune geometry and momentum schedule. |
| [#31](https://github.com/KellerJordan/modded-nanogpt/pull/301) | 2026/05/15 | 2995 | - | 3.2789 (n=20)✓ | Muown combined with NorMuon and Contra-Muon. |
| [#32](https://github.com/KellerJordan/modded-nanogpt/pull/303) | 2026/05/18 | 3000 | - | 3.2778 (n=9)✓ | SODA-style correction toward initialization, faded late. |
| [#33](https://github.com/KellerJordan/modded-nanogpt/pull/316) | 2026/05/28 | 3375 | - | 3.2779 (n=5)✓ | PSGD with Kronecker whitening and hyperball constraint. |
| [#34](https://github.com/KellerJordan/modded-nanogpt/pull/305) | 2026/05/20 | 2925 | ! | 3.2781 (n=8)✓ | Late reduced-rank extrapolation; weak pairwise evidence. |
| [#35](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/records/track_3_optimization/README.md) | 2026/06/10 | 3375 | - | 3.2767 (n=2)✓ | One-sided Shampoo, pseudoinverse root and Adam grafting. |
| [#36](https://github.com/KellerJordan/modded-nanogpt/pull/323) | 2026/06/11 | 3250 | - | 3.2787 (n=10)✓ | Strong tuned simple Muon plus auxiliary AdamW baseline. |
| [#37](https://github.com/KellerJordan/modded-nanogpt/pull/324) | 2026/06/11 | 3250 | - | 3.2786 (n=10)✓ | MuonH benefits from retuning auxiliary Adam parameters. |
| [#38](https://github.com/KellerJordan/modded-nanogpt/pull/307) | 2026/05/20 | 2900 | ! | 3.2786 (n=9)✓ | Late weight readout and tempered-polar/schedule adjustments. |
| [#39](https://github.com/KellerJordan/modded-nanogpt/pull/308) | 2026/05/22 | 3125 | - | 3.2786 (n=20)✓ | EMA-Nesterov lookahead around plain Muon. |
| [#40](https://github.com/KellerJordan/modded-nanogpt/pull/309) | 2026/05/22 | 2890 | ! | 3.2788 (n=16)✓ | EMA-Nesterov lookahead added to the stronger #30 recipe. |
| [#41](https://github.com/KellerJordan/modded-nanogpt/pull/311) | 2026/05/23 | 2875 | ! | 3.2790 (n=20)✓ | Circuit-Muon coupling of attention value/output weights. |
| [#42](https://github.com/KellerJordan/modded-nanogpt/pull/312) | 2026/05/25 | 2860 | ! | 3.2789 (n=16)✓ | Late reference extrapolation and initialization changes. |
| [#43](https://github.com/KellerJordan/modded-nanogpt/pull/318) | 2026/05/29 | 2850 | ! | 3.2786 (n=13)✓ | Several fixed late-trajectory transformations and anchor blending. |
| [#44](https://github.com/KellerJordan/modded-nanogpt/pull/321) | 2026/06/10 | 2750 | ! | 3.2789 (n=20)✓ | SOAP on all hidden matrices every step; auxiliary tuning; remove geometry modules. |
| [#45](https://github.com/KellerJordan/modded-nanogpt/pull/325) | 2026/06/12 | 2720 | ! | 3.2786 (n=10)✓ | Late EMA-weight blending at evaluation. |
| [#46](https://github.com/KellerJordan/modded-nanogpt/pull/328) | 2026/06/19 | 2690 | ! | 3.2783 (n=8)✓ | Per-row update floor plus post-rescale cautious decay. |

## Track 1: optimizer-relevant milestones

The speedrun is not an optimizer-only leaderboard. These are whole-recipe record times; architecture, precision, batch schedule, communication and timing conventions change too. The historical 21st record was retimed; do not interpret the full series as a controlled optimizer sweep.

| ID | Date | Whole-recipe time | Optimizer-related change |
|---|---|---|---|
| [#1](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 05/28/24 | 45 minutes | AdamW baseline. |
| [#2](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 06/06/24 | 31.4 minutes | Learning-rate tuning alongside rotary architecture change. |
| [#3](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 10/04/24 | 24.9 minutes | Muon enters for hidden matrices; auxiliary Adam-family updates remain. |
| [#4](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 10/11/24 | 22.3 minutes | Muon refinements. |
| [#6](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 10/18/24 | 13.1 minutes | Distribute optimizer overhead across GPUs. |
| [#9](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 11/06/24 | 8.2 minutes | Momentum warmup alongside architecture/logit changes. |
| [#19](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 01/13/25 | 3.142 minutes | Nonzero terminal learning-rate floor alongside FP8-head changes. |
| [#20](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 01/16/25 | 2.992 minutes | Batched Muon and Adam epsilon, alongside attention/layout changes. |
| [#22](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 05/24/25 | 2.990 minutes | Faster gradient reduction. |
| [#23](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 05/25/25 | 2.979 minutes | Overlap communication and computation. |
| [#24](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 05/30/25 | 2.966 minutes | Reduce-scatter replaces all-reduce. |
| [#27](https://github.com/KellerJordan/modded-nanogpt/pull/109) | 07/18/25 | 2.817 minutes | Symmetric-matmul Triton kernel and matrix orientation. |
| [#36](https://github.com/KellerJordan/modded-nanogpt/pull/132) | 09/23/25 | 2.495 minutes | Shape-aware Muon and combined gradient reductions. |
| [#38](https://github.com/KellerJordan/modded-nanogpt/pull/134) | 09/29/25 | 2.476 minutes | Polar Express orthogonalization polynomial schedule. |
| [#39](https://github.com/KellerJordan/modded-nanogpt/pull/136) | 09/30/25 | 2.447 minutes | Auxiliary Adam updated every other step; batch also changed. |
| [#41](https://github.com/KellerJordan/modded-nanogpt/pull/144) | 10/24/25 | 2.345 minutes | NorMuon adaptive rescaling enters the record. |
| [#42](https://github.com/KellerJordan/modded-nanogpt/pull/146) | 10/27/25 | 2.313 minutes | NorMuon learning rate and step-logic correction. |
| [#43](https://github.com/KellerJordan/modded-nanogpt/pull/154) | 11/10/25 | 2.284 minutes | Cautious weight decay with a schedule. |
| [#44](https://github.com/KellerJordan/modded-nanogpt/pull/149) | 11/16/25 | 2.269 minutes | Adam backward hooks and synchronization overlap. |
| [#46](https://github.com/KellerJordan/modded-nanogpt/pull/163) | 11/29/25 | 2.203 minutes | Batch-size schedule changes the optimizer workload. |
| [#48](https://github.com/KellerJordan/modded-nanogpt/pull/168) | 12/11/25 | 2.170 minutes | NorMuon axis, matrix layout, implementation and LR fixes. |
| [#50](https://github.com/KellerJordan/modded-nanogpt/pull/172) | 12/18/25 | 2.128 minutes | Cautious decay extended to Adam parameters. |
| [#52](https://github.com/KellerJordan/modded-nanogpt/pull/177) | 12/21/25 | 2.037 minutes | Auxiliary beta/LR tuning and scalar freezing at transitions. |
| [#53](https://github.com/KellerJordan/modded-nanogpt/pull/178) | 12/22/25 | 1.988 minutes | Decay/LR changes accompany MTP and embedding untying. |
| [#56](https://github.com/KellerJordan/modded-nanogpt/pull/187) | 12/31/25 | 1.894 minutes | Compile Adam; higher-precision state; move gates to Adam groups. |
| [#57](https://github.com/KellerJordan/modded-nanogpt/pull/190) | 01/04/26 | 1.878 minutes | Mixed-precision Muon and interleaved Adam/Muon execution. |
| [#61](https://github.com/KellerJordan/modded-nanogpt/pull/200) | 01/18/26 | 1.748 minutes | Unified optimizer implementation and transposed head. |
| [#65](https://github.com/KellerJordan/modded-nanogpt/pull/215) | 01/30/26 | 1.613 minutes | Group value embeddings into one parameter (layout/system effect). |
| [#71](https://github.com/KellerJordan/modded-nanogpt/pull/221) | 02/06/26 | 1.516 minutes | Sparse bigram gradient communication. |
| [#72](https://github.com/KellerJordan/modded-nanogpt/pull/224) | 02/10/26 | 1.496 minutes | Higher minimum LR plus sequence-length schedule. |
| [#80](https://github.com/KellerJordan/modded-nanogpt/pull/253) | 04/08/26 | 1.406 minutes | Orthogonalize attention Q/K by paired heads. |
| [#91](https://github.com/KellerJordan/modded-nanogpt/pull/350) | 08/06/26 | 1.126 minutes | Canonical-token evaluation masking; not a new optimizer. |
| [#92](https://github.com/KellerJordan/modded-nanogpt/pull/360) | 08/30/26 | 0.665 minutes | ANVIL and sparse Adam, bundled with substantial model/loss/systems changes. |

### Full record-to-family index

Every Track 1 record is listed below. Family labels are COARSE INFERENCES from documented introductions: AdamW (#1-2), Muon plus auxiliary Adam-family (#3-40), NorMuon/Muon plus auxiliary Adam-family (#41-91), ANVIL with auxiliary/sparse Adam (#92). This is not independent source-level verification of every record. Parameters such as gates/embeddings can move between optimizer groups.

| ID | Date | Time | Coarse family |
|---|---|---|---|
| [#1](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 05/28/24 | 45 minutes | AdamW |
| [#2](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 06/06/24 | 31.4 minutes | AdamW |
| [#3](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 10/04/24 | 24.9 minutes | Muon + auxiliary Adam-family |
| [#4](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 10/11/24 | 22.3 minutes | Muon + auxiliary Adam-family |
| [#5](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 10/14/24 | 15.2 minutes | Muon + auxiliary Adam-family |
| [#6](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 10/18/24 | 13.1 minutes | Muon + auxiliary Adam-family |
| [#7](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 10/18/24 | 12.0 minutes | Muon + auxiliary Adam-family |
| [#8](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 11/03/24 | 10.8 minutes | Muon + auxiliary Adam-family |
| [#9](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 11/06/24 | 8.2 minutes | Muon + auxiliary Adam-family |
| [#10](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 11/08/24 | 7.8 minutes | Muon + auxiliary Adam-family |
| [#11](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 11/10/24 | 7.2 minutes | Muon + auxiliary Adam-family |
| [#12](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 11/19/24 | 5.03 minutes | Muon + auxiliary Adam-family |
| [#13](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 11/24/24 | 4.66 minutes | Muon + auxiliary Adam-family |
| [#14](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 12/04/24 | 4.41 minutes | Muon + auxiliary Adam-family |
| [#15](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 12/08/24 | 3.95 minutes | Muon + auxiliary Adam-family |
| [#16](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 12/10/24 | 3.80 minutes | Muon + auxiliary Adam-family |
| [#17](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 12/17/24 | 3.57 minutes | Muon + auxiliary Adam-family |
| [#18](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 01/04/25 | 3.4 minutes | Muon + auxiliary Adam-family |
| [#19](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 01/13/25 | 3.142 minutes | Muon + auxiliary Adam-family |
| [#20](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 01/16/25 | 2.992 minutes | Muon + auxiliary Adam-family |
| [#21](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 01/26/25 | 2.933 minutes | Muon + auxiliary Adam-family |
| [#21](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 02/01/25 | 2.997 minutes | Muon + auxiliary Adam-family |
| [#21](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 05/24/25 | 3.014 minutes | Muon + auxiliary Adam-family |
| [#22](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 05/24/25 | 2.990 minutes | Muon + auxiliary Adam-family |
| [#23](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 05/25/25 | 2.979 minutes | Muon + auxiliary Adam-family |
| [#24](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 05/30/25 | 2.966 minutes | Muon + auxiliary Adam-family |
| [#25](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 07/13/25 | 2.896 minutes | Muon + auxiliary Adam-family |
| [#26](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 07/13/25 | 2.863 minutes | Muon + auxiliary Adam-family |
| [#27](https://github.com/KellerJordan/modded-nanogpt/pull/109) | 07/18/25 | 2.817 minutes | Muon + auxiliary Adam-family |
| [#28](https://github.com/KellerJordan/modded-nanogpt/pull/117) | 08/23/25 | 2.812 minutes | Muon + auxiliary Adam-family |
| [#29](https://github.com/KellerJordan/modded-nanogpt/pull/118) | 09/03/25 | 2.731 minutes | Muon + auxiliary Adam-family |
| [#30](https://github.com/KellerJordan/modded-nanogpt/pull/120) | 09/05/25 | 2.717 minutes | Muon + auxiliary Adam-family |
| [#31](https://github.com/KellerJordan/modded-nanogpt/pull/122) | 09/10/25 | 2.656 minutes | Muon + auxiliary Adam-family |
| [#32](https://github.com/KellerJordan/modded-nanogpt/pull/125) | 09/11/25 | 2.625 minutes | Muon + auxiliary Adam-family |
| [#33](https://github.com/KellerJordan/modded-nanogpt/pull/127) | 09/15/25 | 2.565 minutes | Muon + auxiliary Adam-family |
| [#34](https://github.com/KellerJordan/modded-nanogpt/pull/130) | 09/18/25 | 2.547 minutes | Muon + auxiliary Adam-family |
| [#35](https://github.com/KellerJordan/modded-nanogpt/pull/131) | 09/21/25 | 2.527 minutes | Muon + auxiliary Adam-family |
| [#36](https://github.com/KellerJordan/modded-nanogpt/pull/132) | 09/23/25 | 2.495 minutes | Muon + auxiliary Adam-family |
| [#37](https://github.com/KellerJordan/modded-nanogpt/pull/133) | 09/27/25 | 2.483 minutes | Muon + auxiliary Adam-family |
| [#38](https://github.com/KellerJordan/modded-nanogpt/pull/134) | 09/29/25 | 2.476 minutes | Muon + auxiliary Adam-family |
| [#39](https://github.com/KellerJordan/modded-nanogpt/pull/136) | 09/30/25 | 2.447 minutes | Muon + auxiliary Adam-family |
| [#40](https://github.com/KellerJordan/modded-nanogpt/pull/140) | 10/04/25 | 2.358 minutes | Muon + auxiliary Adam-family |
| [#41](https://github.com/KellerJordan/modded-nanogpt/pull/144) | 10/24/25 | 2.345 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#42](https://github.com/KellerJordan/modded-nanogpt/pull/146) | 10/27/25 | 2.313 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#43](https://github.com/KellerJordan/modded-nanogpt/pull/154) | 11/10/25 | 2.284 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#44](https://github.com/KellerJordan/modded-nanogpt/pull/149) | 11/16/25 | 2.269 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#45](https://github.com/KellerJordan/modded-nanogpt/pull/159) | [log](records/track_1_short/2025-11-18_RefineSkip/00f4e1e6-0044-4a08-b88a-3b7ec0624081.txt),[PR](https://github.com/KellerJordan/modded-nanogpt/pull/159) | 2.248 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#46](https://github.com/KellerJordan/modded-nanogpt/pull/163) | 11/29/25 | 2.203 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#47](https://github.com/KellerJordan/modded-nanogpt/pull/166) | 12/10/25 | 2.193 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#48](https://github.com/KellerJordan/modded-nanogpt/pull/168) | 12/11/25 | 2.170 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#49](https://github.com/KellerJordan/modded-nanogpt/pull/169) | 12/14/25 | 2.146 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#50](https://github.com/KellerJordan/modded-nanogpt/pull/172) | 12/18/25 | 2.128 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#51](https://github.com/KellerJordan/modded-nanogpt/pull/175) | 12/19/25 | 2.075 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#52](https://github.com/KellerJordan/modded-nanogpt/pull/177) | 12/21/25 | 2.037 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#53](https://github.com/KellerJordan/modded-nanogpt/pull/178) | 12/22/25 | 1.988 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#54](https://github.com/KellerJordan/modded-nanogpt/pull/181) | 12/26/25 | 1.940 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#55](https://github.com/KellerJordan/modded-nanogpt/pull/186) | 12/29/25 | 1.918 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#56](https://github.com/KellerJordan/modded-nanogpt/pull/187) | 12/31/25 | 1.894 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#57](https://github.com/KellerJordan/modded-nanogpt/pull/190) | 01/04/26 | 1.878 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#58](https://github.com/KellerJordan/modded-nanogpt/pull/191) | 01/07/26 | 1.820 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#59](https://github.com/KellerJordan/modded-nanogpt/pull/197) | 01/10/26 | 1.781 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#60](https://github.com/KellerJordan/modded-nanogpt/pull/199) | 01/16/26 | 1.765 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#61](https://github.com/KellerJordan/modded-nanogpt/pull/200) | 01/18/26 | 1.748 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#62](https://github.com/KellerJordan/modded-nanogpt/pull/201) | 01/19/26 | 1.655 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#63](https://github.com/KellerJordan/modded-nanogpt/pull/209) | 01/26/26 | 1.650 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#64](https://github.com/KellerJordan/modded-nanogpt/pull/214) | 01/30/26 | 1.630 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#65](https://github.com/KellerJordan/modded-nanogpt/pull/215) | 01/30/26 | 1.613 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#66](https://github.com/KellerJordan/modded-nanogpt/blob/4ea6b937337a4889b8cfe3f38a93d120048d8f71/README.md#world-record-history) | 01/31/26 | 1.595 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#67](https://github.com/KellerJordan/modded-nanogpt/pull/207) | 01/31/26 | 1.540 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#68](https://github.com/KellerJordan/modded-nanogpt/pull/216) | 01/31/26 | 1.535 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#69](https://github.com/KellerJordan/modded-nanogpt/pull/217) | 02/02/26 | 1.528 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#70](https://github.com/KellerJordan/modded-nanogpt/pull/218) | 02/03/26 | 1.521 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#71](https://github.com/KellerJordan/modded-nanogpt/pull/221) | 02/06/26 | 1.516 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#72](https://github.com/KellerJordan/modded-nanogpt/pull/224) | 02/10/26 | 1.496 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#73](https://github.com/KellerJordan/modded-nanogpt/pull/230) | 02/12/26 | 1.485 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#74](https://github.com/KellerJordan/modded-nanogpt/pull/233) | 02/16/26 | 1.468 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#75](https://github.com/KellerJordan/modded-nanogpt/pull/235) | 02/23/26 | 1.453 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#76](https://github.com/KellerJordan/modded-nanogpt/pull/240) | 02/28/26 | 1.446 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#77](https://github.com/KellerJordan/modded-nanogpt/pull/241) | 03/06/26 | 1.435 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#78](https://github.com/KellerJordan/modded-nanogpt/pull/246) | 03/22/26 | 1.426 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#79](https://github.com/KellerJordan/modded-nanogpt/pull/251) | 04/04/26 | 1.411 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#80](https://github.com/KellerJordan/modded-nanogpt/pull/253) | 04/08/26 | 1.406 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#81](https://github.com/KellerJordan/modded-nanogpt/pull/259) | 04/22/26 | 1.363 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#82](https://github.com/KellerJordan/modded-nanogpt/pull/264) | 04/29/26 | 1.353 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#83](https://github.com/KellerJordan/modded-nanogpt/pull/299) | 05/20/26 | 1.328 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#84](https://github.com/KellerJordan/modded-nanogpt/pull/306) | 05/21/26 | 1.320 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#85](https://github.com/KellerJordan/modded-nanogpt/pull/315) | 05/27/26 | 1.271 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#86](https://github.com/KellerJordan/modded-nanogpt/pull/317) | 05/27/26 | 1.266 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#87](https://github.com/KellerJordan/modded-nanogpt/pull/322) | 06/11/26 | 1.256 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#88](https://github.com/KellerJordan/modded-nanogpt/pull/337) | 07/13/26 | 1.243 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#89](https://github.com/KellerJordan/modded-nanogpt/pull/342) | 07/17/26 | 1.23 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#90](https://github.com/KellerJordan/modded-nanogpt/pull/344) | 08/03/26 | 1.13 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#91](https://github.com/KellerJordan/modded-nanogpt/pull/350) | 08/06/26 | 1.126 minutes | NorMuon/Muon family + auxiliary Adam-family |
| [#92](https://github.com/KellerJordan/modded-nanogpt/pull/360) | 08/30/26 | 0.665 minutes | ANVIL + auxiliary/sparse Adam |

### What not to infer

- Most speedrun gains after Muon did not come from replacing it with a wholly new optimizer. Layout, kernels, precision, communication, cadence and schedules mattered.
- NorMuon normalization, cautious decay, momentum scheduling and auxiliary updates suggest inexpensive mechanisms worth testing across tasks, not guaranteed transferable wins.
- #92 is based on #89, not a clean incremental ablation of #91. ANVIL is one component of a bundle that also changes the model and training loss. The 39.9s figure cannot be attributed to ANVIL alone.
- Original certified #92 and current root trainer differ: #373 later adds #91 canonical masking and other changes. Exact replication must pin the certified artifact.
- A broad optimizer breakthrough needs robust gains beyond one tuned architecture, one tiny endpoint margin and one favorable implementation.
