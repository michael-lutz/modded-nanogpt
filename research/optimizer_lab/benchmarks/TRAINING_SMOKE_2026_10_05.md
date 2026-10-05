# GH200: complete training-step pilot with the full CPU-resident table

Date: 2026-10-05. **Feasibility/timing pilot of an adapted #92 descendant, not a
certified #92 reproduction or a completed training-to-quality run.**

## Answer

Yes: real forward, backward, dense ANVIL/Adam, sparse table Adam and CPU writeback
run successfully on one GH200. The full 129.95 GB table remains in Grace CPU RAM.
No gradient checkpointing, reduced hash table, reduced model, or reduced global
batch was needed for the early-stage workload tested here.

| Complete early-stage global training step | Run A | Run B | Combined mean |
|---|---:|---:|---:|
| GH200, full CPU table | 133.48 ms | 132.71 ms | **133.09 ms** |
| GH200, HBM-resident finite-trace control | 126.23 ms | 126.19 ms | **126.21 ms** |
| Difference | +7.25 ms | +6.52 ms | **+6.88 ms / +5.5%** |
| Published #92 8xH100, early stage | — | — | **16.981 ms** |

The GH200 pilot is about **7.84x** the published eight-H100 early-stage step time.
This is a cross-hardware, cross-implementation contextual comparison, not an
8-GPU test we ran, an exact strong-scaling measurement, or an official score.
The directly controlled storage comparison is the first two rows: roughly **5-6%**
additional complete-step time for Grace backing instead of HBM backing.

## Experiment

- Two independent **20-global-step runs per storage mode**, in host/HBM/HBM/host
  order. Each uses seed 0, the same fresh token trace and the same initial model.
- Global batch **131,072 tokens**: eight serial microbatches of **16,384**, followed
  by one dense optimizer step. The sparse table updates every two global steps.
- First 8 global steps are untimed training warmup; remaining 12 are measured.
  These are short samples, not a long-run stability or confidence-interval study.
- Real FineWeb data, using the upstream eight-logical-rank BOS packing and
  document limits, per-rank sampled candidate sets, MTP and prefix objectives.
- Full FP8 forward/backward with the current upstream model, compiled and captured
  into CUDA graphs; upstream ANVIL/Adam update rules and optimizer graph checks.
- Sparse gradients come from **actual autograd**, not the prior synthetic-gradient
  benchmark. Hundreds of thousands of rows become nonzero at each table event.
- Model and table objectives/state stay finite. Mean composite training objective
  per two-step cycle goes from **17.091** to about **11.973-11.977** over the short
  runs. This includes auxiliary losses and sampled softmax: it is **not validation
  CE**, and cannot be compared with the leaderboard's 3.28 target.
- CPU-only table NUMA placement is asserted after allocation and every cycle.
  Peak PyTorch allocated GPU memory is approximately **8.72 GiB** for the host
  run (not total device/reserved memory). No activation checkpointing is used.

### What the HBM control means

A full 130 GB table does not fit in this GPU. The control allocates only the union
of global table rows visited by this **fixed finite trace**, about **4.32 GB**.
It does **not** shrink the hash space, rehash tokens, introduce extra collisions,
reset rows between cycles, or omit a row that the trace touches. All stored rows
and their optimizer states persist across the run. It is an exact capacity-limited
control for the visited row identities, not a practical replacement for the full
training table and not an eight-GPU sharded implementation.

Both paths gather into the same active-row cache and write back after sparse Adam.
The HBM path additionally maps full global IDs into its compact backing store.
The timed GPU intervals for hash/plan/pull plus sparse update/writeback are about
**9.22 ms/step on host** versus **1.61 ms/step on HBM**, excluding the common real
gradient merge (~0.68 ms/step). These component intervals are instrumented and not
independent estimates of the total slowdown: use the complete-step A/B above.

## Additional correctness check

On identical inputs and state, the captured forward matched the uncaptured
compiled forward exactly. All tested model-parameter and n-gram-sink gradients
matched numerically exactly between captured and uncaptured compiled backward.
The FP16-atomic value-embedding gradient accumulator differed by **0.174% in
relative L2**, within the diagnostic's 1% tolerance. This is a local graph-replay
check, not equivalence to the official eight-rank trainer. The diagnostic
reported a cross-stream AccumulateGrad warning but completed successfully; it is
not included in the timing pool.

## Timing boundaries and departures from the record

**Timed:** token/target/candidate uploads, GPU n-gram hashing/deduplication, row/state
fetch, FP8 refresh, sampled lm-head weight gather, model forward/backward, real
ngram gradient aggregation, dense and sparse optimizer updates, table writeback,
and synchronization at each global step. Host dispatch and GPU event profiling
costs inside the loop are included. No transfer/Transformer overlap is assumed.

**Outside timing:** data reading/BOS packing, CPU candidate-ID construction,
initial allocation/zeroing, compilation/capture, warmup, and correctness checks.
There is no validation run. Thus these are **complete training-step throughput**
measurements, not elapsed-from-launch benchmark scores. Data-loader integration
and the later, larger training stages remain untested by this pilot.

Adaptations, deliberately isolated in the research script rather than upstream:

1. Source is refactored descendant `4ea6b937337a4889b8cfe3f38a93d120048d8f71`, not
   byte-identical certified #360 source. Existing source files were not edited.
2. The pinned mixed-width attention download failed with HTTP 401, and its CUDA13
   build/driver requirements differ from our ARM64/driver570 environment. We use
   the previously working, pinned ARM CUDA12.6 stock FA3 binary instead. Zero-pad
   Q/K or V to equal head width, preserve the explicit softmax scale, then slice
   back to the original V width. This preserves the attention algebra, **not
   bitwise kernel results or the record's kernel efficiency**.
3. Serialize eight logical ranks. Each rank retains its own token boundaries and
   sampled-negative sequence. Dense gradients are averaged explicitly after
   backward: the custom CE backward ignores `grad_output`, so dividing the scalar
   loss would NOT accomplish gradient accumulation correctly.
4. Accumulate real table gradients in FP32, cast the aggregate to BF16, then apply
   the upstream row Adam with division by eight. Value-embedding gradients use the
   upstream FP16 accumulator and dense shard update on the one physical rank.
   This is **not** the official distributed reduction/rounding order.
5. The short run does not exercise later stages, tail averaging, embed untying,
   final validation, or all scheduling transitions. No convergence, optimizer
   ranking preservation, or statistical parity claim follows.

The first compiled pilot reached graph capture completion in 80.2 seconds; later
processes did so in about 28-29 seconds with reused caches. These are partial-stage
startup measurements, and the first pilot already reused some kernels from debug
runs: **not** a fully cold five-stage compile benchmark. Complete pilot processes
lasted 87.2 seconds initially and 32-34 seconds for the subsequent three runs.

## Reference and evidence

The eight-H100 number is calculated from each of the **17 shipped raw logs**:
subtract cumulative training time at step 50 from step 300, divide by 250, then
average across logs. Range 16.968-16.992 ms. This isolates the same early-stage
batch, rather than comparing our early stage to the all-stage average. Reference:
[certified #92 source/log pool](../../../records/track_1_short/2026-08-30_ANVIL2/this_pr/).
The official 39.914-second full-run mean includes different stages and validation;
**do not multiply our early-stage timing by 1,194 to predict the full run.**

- [Curated raw measurements and provenance](training_results_2026_10_05.json)
- [Training-step harness](gh200_training_smoke.py)
- [Extra captured-backward check](gh200_verify_training_graph.py)
- [Earlier synthetic table-only benchmark](RESULTS_2026_10_05.md)

## Reproduce

Same GH200 environment/cache/data prerequisites as the earlier benchmark. The
attention loader reads the existing pinned ARM64 FA3 cache; set `FA3_CACHE_ROOT`
to its local Hub-cache root. Point `CUDA_HOME` to CUDA runtime headers and keep
Inductor/Triton caches on local disk. Use a new output directory for every process.

```bash
python research/optimizer_lab/benchmarks/gh200_training_smoke.py \
  --mode host --cycles 10 --warmup 4 --out /tmp/gh200-host
python research/optimizer_lab/benchmarks/gh200_training_smoke.py \
  --mode hbm --cycles 10 --warmup 4 --out /tmp/gh200-hbm
python research/optimizer_lab/benchmarks/gh200_verify_training_graph.py \
  --mode hbm --cycles 2 --warmup 1 --out /tmp/gh200-graph-check
```

Run sequentially, not concurrently. The host mode requires at least 200 GiB free
system memory. Do not use `--no-compile` together with graph capture: eager loss
code creates CPU-origin scale tensors, which CUDA capture rejects. That failed
intermediate attempt and the earlier single-microbatch debug run are retained
locally and excluded from the results above.
