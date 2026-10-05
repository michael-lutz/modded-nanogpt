# Research protocol: optimize evidence per GPU-hour

## 1. Correcting the benchmark hierarchy

There is no evidence yet that Track 1 #91 ranks optimizers like Track 3. A 13-minute full #91 run is inexpensive, but cheap does not mean representative. Likewise, #92's sparse table, sampled training loss, mixed-width attention and precision/schedules create another regime; neither transfer nor failure to transfer can be assumed.

Primary development target: the real Track 3 model/data/global batch. Side-by-side transfer target: Track 1 #91 because it runs on one GH200. Track 2 and original #92 are later external tests, not extrapolated promises. Do not use #91 loss to veto a Track 3 candidate until empirical calibration supports that policy. Maintain separate step-efficiency, time-efficiency and iteration-cost frontiers.

Use two complementary Track 3 contexts:

1. **Mechanism context:** upstream tuned simple Muon + auxiliary AdamW (#36) on the SAME Track 3 architecture/data/batch. Remove neither model difficulty nor global batch. This is easier to interpret than the champion's many interacting components. Its runtime has not yet been measured here.
2. **Competitive context:** #46's complete champion recipe. Test surgical replacements against this, then tune the candidate. Beating only an undertuned baseline is insufficient.

Use both *controlled ablation* (same wrapper, LR schedule and readout except intended change) and *equal-budget best recipe* comparisons (small independent tuning budgets for each optimizer). Identical hyperparameters are not automatically fair. Do not mix these two questions in one headline.

## 2. Correctness gates (proposed tests, not yet implemented GPU tests)

All gates require an experiment manifest naming candidate, reference, exact math, versions, source hashes and seed policy.

| Gate | Concrete test | Acceptance / interpretation |
|---|---|---|
| Update equation | Tiny FP64 reference, deterministic pre-generated gradients, compare weights AND all state after 1, 2, 10 and 100 updates | Start with relative error <=1e-10 and absolute <=1e-12 for well-conditioned FP64 tests; specify epsilon conventions. Legitimately ill-conditioned/discontinuous operations need dedicated tests, not silently relaxed tolerances. |
| Compiled implementation | Eager vs compiled FP32 on identical inputs and state, including refresh boundaries | Starting FP32 relative update-error budget 1e-5 with absolute floor 1e-7; calibrate to conditioning and reduction order before screening. Compare DELTA, not just large nearly equal weights. |
| Mixed precision | FP32 reference vs proposed BF16/FP8 storage/arithmetic path; separately compare optimized kernel vs straightforward same-precision implementation | Predeclare tolerances per operation from incumbent error envelopes; no universal BF16 threshold. Record update cosine, relative norm error, state drift and swallowed sub-ULP updates. Trajectory changes must be assessed by training. |
| Shapes / layouts | All real parameter groups plus tall/wide/square/rank-deficient matrices, non-contiguous/transposed/banked views, supported odd dimensions | No wrong-axis normalization, incorrect aliasing, unintended group mixing, silent fallback or invalid shape support. Reject unsupported layouts explicitly. |
| Edge cases | Zero gradients, missing gradients, zero parameters, tiny norms, repeated singular values, large gradients, abrupt rotation, long constant-gradient sequences | Behavior matches the written algorithm. Missing gradients need not equal explicit zeros. Decay need not vanish with zero gradient. No unexplained NaN/Inf. |
| Claimed properties | Gradient rescaling, orthogonal basis changes, row/column permutations, rank reduction | Test only invariances the method actually claims; epsilon, elementwise adaptivity and selective decay can legitimately break them. Orthogonalization tests do not demand identity for rank-deficient inputs. |
| Persistence | Save/load optimizer and RNG state at steps around refresh/schedule boundaries; compare uninterrupted continuation | Exact where deterministic, otherwise within predeclared numerical envelope. No dropped covariance, momentum, counters or tail-average state. |
| Accumulation | One global gradient vs sum of microbatch gradients, with one optimizer step after accumulation | Gradients/counts agree within precision tolerance; state must not accidentally advance once per microbatch. Multi-GPU agreement remains a separate test when hardware exists. |
| Parameter coverage | Audit IDs, groups, tied weights, inactive parameters, sparse touched rows, state allocation | Every intended parameter updated once, none unintentionally updated; sparse semantics match reference. |
| Short training | At least 50 actual updates plus tests spanning each optimizer-specific cadence boundary | Finite loss/state, sane update/weight ratios, no leak. A smoke test is not convergence evidence. |

If model/autograd kernels change, also require forward/backward numerical checks. Do not demand differentiation through a normal optimizer unless that is part of its intended use.

## 3. Cost gates

### 3.1 What to measure

**Parameter census:** name, shape, dtype, layout, number of copies, trainable size, state bytes, scratch bytes, and update cadence for every parameter group. Include embedding/head, auxiliary scalars, attention, MLP and any sparse table. Do not extrapolate from one 768x768 GEMM.

**Isolated optimizer timing:** use real gradient/state snapshots from early/mid/late trajectories once capture tooling is added; until then mark synthetic-gradient tests as synthetic. Include changing gradients so covariance decompositions are not benchmarked only at an artificially easy steady state.

- Compile once; record cold compile time separately.
- Warm 20 updates, then time 10 blocks of at least 100 updates each (proposed starting protocol; lengthen when timer resolution/noise requires it).
- If refresh period is K, each measured block must span at least 5K steps or combine full cadence cycles. Report refresh and non-refresh separately plus weighted average.
- Use CUDA events for device work and synchronized host elapsed time for user-visible launch/CPU overhead. Do not place a synchronization between every constituent kernel in the throughput measurement.
- Report block mean/median, distribution of individual updates if collected, refresh p95/max, memory allocated/reserved/peak, and bytes of persistent state per weight. Account for CPU work, device-host copies and eigendecomposition/QR refresh spikes.
- Use a dedicated idle GPU, fixed software and normal power settings. Log clocks/power but do not change shared-host settings. Alternate baseline and candidate timing blocks.

**In-context timing:** instrument 100+ representative training updates per schedule phase, with the actual forward/backward, optimizer overlap, communication and precision. Include both tied/untied stages and optimizer transition points. Separate tracing/profiler runs from final non-profiled timing. The isolated optimizer percentage is NOT the critical-path percentage when work overlaps.

**End-to-end:** final authority is quality vs elapsed training time on a fixed hardware/software setup. Also report cold/warm startup, validation time and total research GPU occupancy. GH200 timing is a development measurement, not an 8xH100 result.

### 3.2 Conditional break-even model

Plain ASCII notation:

```text
p = baseline fraction of step time spent in optimizer (no overlap approximation)
k = candidate optimizer cost / baseline optimizer cost
s = candidate required steps / baseline required steps

step_time_ratio ~= 1 + p * (k - 1)
training_time_ratio ~= s * (1 + p * (k - 1))

speedup requires training_time_ratio < 1
```

Example: doubling an optimizer that is 10% of step cost adds approximately 10% to total step time, so it needs more than 9.1% fewer steps to break even. This formula does NOT apply unchanged when batch schedules, architectures or overlap differ; directly time those cases.

### 3.3 Decisions

- Hard fail: wrong math, non-finite/unbounded behavior without an intended recovery policy, memory beyond available capacity, illegal benchmark change, or unintended parameter updates.
- **Fast-path preference, not a scientific law:** <=5% measured whole-step slowdown makes a candidate inexpensive to investigate; >5% needs an explicit plausible break-even improvement. Do not discard a potentially large breakthrough solely on this heuristic.
- **Step-efficiency lane:** an expensive candidate may remain useful for Track 3 or as a teacher for a cheaper approximation; label it accordingly.
- Require claimed speed improvements to exceed measured run-to-run timing uncertainty. No fixed 1% gain claim without an uncertainty estimate.
- Never optimize arithmetic fidelity away to pass the timing gate: report changed numerics and re-test learning.

## 4. Experimental funnel and budgets

```text
Written hypothesis + proposed cost model
                  |
           Correctness/cost gates
                  |
     Same-model Track 3 development
          /                    \
 full Track 3 confirmation    #91 transfer panel (not a veto)
          \                    /
        fresh-seed confirmation + ablations
                  |
     Track 2 and exact #92 transfer tests
                  |
     hardware-matched public submission
```

Short Track 3 prefixes preserve the original full-run LR/momentum schedule and are used for stability/cost diagnostics. Compressing cooldown creates a DIFFERENT proxy. It can be investigated, but needs its own calibration. Near-zero early loss gains are not grounds for rejection; EMA/preconditioning methods can improve late.

Checkpoint continuations are valid full-run evidence only if the candidate generated its complete preceding trajectory and all state is faithfully restored. Switching optimizers at a champion checkpoint measures a warm-start intervention, not training from initialization. Shared-prefix reuse is exact only for algorithms identical over that prefix. Traces of fixed baseline gradients test implementations/cost, not final optimizer quality, since changing the optimizer changes future gradients.

Initial tuning budget: 3 LR/update-scale choices for a preselected development seed, with fixed secondary hyperparameters and endpoint; reuse reasonable reference values only as a starting point. Larger allocations must be recorded for both methods. Expensive jobs are allocated per hypothesis rather than blind full Cartesian sweeps. This is a proposed starting budget, not a guarantee of enough tuning.

Promotion: retain candidates for full Track 3 tests based on mechanism plausibility, correctness, affordable cost and whatever calibrated proxy evidence exists. Before calibration, use prefixes only to discard clear failures, not to certify rankings. Maintain a small exploration allocation for candidates the current proxy dislikes.

## 5. Calibrate transfer, do not assume it

A proposed initial panel of eight variants includes the tuned Muon baseline, NorMuon, the current SOAP-Muon stack, a less-frequent SOAP refresh, an inexpensive diagonal/factored approximation, altered momentum, norm control and tail averaging. Full-family comparisons and individual-component ablations must be labeled separately. Each benchmark gets its own equally budgeted LR tuning.

Run the panel on the genuine Track 3 task and #91, initially 2 paired seeds per recipe, adding replication when differences are within noise. Report:

- Within-task effect vs incumbent, normalized only for display, not mixed across incompatible objectives.
- Pairwise sign agreement and Spearman rank correlation WITH bootstrap uncertainty; eight configurations do not establish universal transfer.
- Top-candidate recall and especially false negatives: how often would #91 discard a genuine Track 3 improvement?
- Optimizer overhead, state memory and compile/startup cost, not only loss rankings.
- Repeat/hold out algorithm families when evaluating a fitted proxy; correlation among nearly identical LR choices can be misleading.

No mandatory cross-task rejection gate unless the calibration shows sufficiently reliable winner recall under an explicitly chosen risk budget. An example policy target is 90% recall; this small initial panel cannot tightly certify it. If uncertain, use #91 to prioritize or break ties, not to reject.

Track 2 must appear as soon as its baseline is runnable locally and one or two promising candidates exist. Its higher-quality target probes longer optimization, but is still the same benchmark family and not proof of broad language-model transfer. General-breakthrough claims need at least another model scale/task outside these leaderboards and equal tuning/compute accounting.

## 6. Confirmation and anti-overfitting

Predeclare endpoints, development seeds and final evaluation seeds. Track all attempts and tuning cost, including failures. Use paired seeds/data order when possible, but numerical nondeterminism prevents assuming identical trajectories across hardware. Freeze optimizer code/hyperparameters before fresh-seed confirmation. Fixed 4-seed initial confirmation with a predeclared expansion to 8 (or a justified sequential test) is preferable to repeatedly peeking until significant. Report effect size and uncertainty; significance against the loss threshold is not evidence of superiority over the incumbent.

Track 3's currently published requirement is:

```text
(3.28 - mean_loss) * sqrt(n) >= 0.004
```

A fixed common endpoint across trials is required; never choose each seed's best stopping point. Track 1/2 use their own threshold/significance/timing rules. Re-read current rules before submission. Fresh seeds reduce seed selection but do NOT erase overfitting from repeated tuning to the same validation distribution. Reserve an additional untouched development evaluation split for research where feasible; official reported scores must still use the official validation.

## 7. First implementation milestones

1. Instrument cold vs warm startup and real optimizer critical-path cost on existing baselines. Do not equate the 14m26s residual in #91's first run entirely with compilation.
2. Build reference-update tests and parameter-shape inventory; implement one optimizer interface without changing official model/data logic.
3. Reproduce the simple Track 3 baseline and calibrate selected prefixes against completed same-task runs.
4. Build the paired cross-task panel, keeping #91 a side test rather than proxy ground truth.
5. Investigate cheaper approximations to successful covariance/geometry mechanisms, plus inexpensive momentum/norm/decay improvements.
6. Prepare Track 2 baseline and pin original #92 artifact. A memory-heavy #92 port is not a prerequisite for doing useful core optimizer research.

Sources: [Track 3 rules/history](../../records/track_3_optimization/README.md), [Track 1/2 rules/history](../../README.md), [version correction](https://github.com/KellerJordan/modded-nanogpt/commit/f380c1f009def747c8843a5c2894b28ac72cf50b).
