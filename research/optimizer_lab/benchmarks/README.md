# GH200 lookup-table feasibility benchmarks

These are **table-subsystem measurements, not a #92 training reproduction**.
They do not establish final loss, optimizer ranking preservation, or full-run timing.

See [measured results](RESULTS_2026_10_05.md) and [curated data](results_2026_10_05.json).

## Reproduce

Requires the GH200 host, at least 200 GiB available system memory, CUDA-enabled
PyTorch 2.10 / Triton 3.6, libnuma, and the cached FineWeb shard. The scripts currently
use the project's local data-cache path; adjust it for another checkout. No datasets,
environments or compiler caches are committed. Source benchmark model code is unchanged.

```bash
# Small mapped-host-memory read/write smoke test.
python research/optimizer_lab/benchmarks/probe_mapped_memory.py

# Separate table transfer/update measurements and naive CPU planning.
python research/optimizer_lab/benchmarks/gh200_table_path.py --out /tmp/table-path

# GPU hash/unique/inverse planning and a synthetic FP32 gradient-merge surrogate.
python research/optimizer_lab/benchmarks/gh200_trace_plan.py --out /tmp/table-plan

# Preferred end-to-end TABLE SUBSYSTEM measurement with fresh token windows.
python research/optimizer_lab/benchmarks/gh200_table_cycle.py \
  --traces 36 --out /tmp/table-cycle

# Separate instrumented diagnostic; do not use this as final unprofiled timing.
python research/optimizer_lab/benchmarks/gh200_table_cycle.py \
  --traces 36 --profile --out /tmp/table-profile
```

## Design and boundaries

- Allocate the full 84,602,880 by 768 BF16 table (129.95 GB) and scalar row state in
  explicitly NUMA-bound CPU DRAM. Reapply binding after CUDA host registration;
  verify the full table remains on node 0 after every stage.
- Direct GPU reads/writes to registered host RAM gather active rows into an HBM
  cache. The upstream sparse Adam kernel operates on that cache, then weights and
  moment/timestamp state are written back.
- Read real FineWeb token windows, hash each of eight logical-rank segments using
  the upstream hash semantics, and deduplicate. These are not the official loader's
  BOS-aligned batches. The hash implementation is checked bit-for-bit.
- Correctness checks compare cached/offloaded weights AND moment/timestamp state
  against the upstream all-GPU kernel, including changed row subsets, missed-event
  decay, a beta transition, zero gradients and untouched rows.
- Integrated runs include token upload, hashing, unique/inverse construction,
  workspace allocation, table/state fetch, gradient aggregation, sparse update,
  writeback and final synchronization. Each case has 6 warmup cycles and 30 timed
  cycles. `--traces 36` gives a different token window to every timed cycle.
- Gradients are synthetic. The included gradient merger accumulates in FP32 before
  casting to FP16; this is **not** the certified eight-GPU trainer's reduction
  ordering/rounding. Its cost is measured, but ML parity is not established.
- Disk/CPU input staging, Transformer forward/backward, actual autograd embedding
  scatter, sampled softmax, complete distributed semantics and convergence are not
  measured. There is no assumed overlap with model work.
- All benchmark weights/state are private allocations, released at process exit.
  No shared power/clocks were changed, existing training jobs stopped, or system
  cache dropped. Large pinned allocations can still reclaim host filesystem cache.

## Pitfalls found and corrected

1. The initial ordinary pinned allocation acquired some GPU-NUMA pages. Those
   diagnostic measurements are not the CPU-DRAM proof. Explicit `mbind` after CUDA
   registration keeps the complete table on CPU node 0.
2. A synthetic merger initially used 32-bit flattened offsets and failed at the
   largest stage (>2^31 gradient elements). Offsets are now 64-bit; the complete
   largest-stage benchmark passes.
3. Making unique-row count a Triton constexpr compiled new kernels for fresh
   windows. The final transfer/state kernels use a non-specialized runtime count.
   Earlier 12/36-window timings mixed compilation with execution and are not steady
   throughput measurements. Final runs start a dedicated cache and vary all 36
   windows without per-row-count compilation.

Failed/intermediate diagnostic logs are retained in the local run directory. Public
JSON contains curated successful measurements, method limitations and source hashes,
not process addresses or private runtime paths.
