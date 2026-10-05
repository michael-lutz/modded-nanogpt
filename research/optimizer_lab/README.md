# Optimizer research laboratory

Objective: discover an optimizer with reproducible improvements in learning per update AND end-to-end time-to-quality, with affordable development cycles. Do not equate a leaderboard win with a general optimizer breakthrough.

This is a research protocol and evidence index, not an implemented GPU experiment scheduler or a claim of new optimizer results.

- [Protocol and concrete gates](PROTOCOL.md)
- [Track 3 history and Track 1 optimizer lineage](HISTORY.md)
- [Local reproduction measurements](baselines.json)
- [Machine-readable upstream history](history.json)

## Status (2026-10-05)

- Track 3 #46: one GH200 pilot completed, seed 0, 2690 steps, scored EMA loss 3.27813, training 6246.827 s, end-to-end 6483 s. Not multi-seed parity; not the benchmark's one-run significance margin.
- Track 1 #91: exact merged source at `a3e9f12ba17d83ace5dc6915b838642561898d90`, one GH200 pilot completed, seed 0, 1290 steps, loss 3.2772, training 776.415 s, end-to-end 1642 s. Not official 8xH100 timing or multi-seed parity.
- Track 1 #92: NOT run. Current root code is a later descendant, not byte-identical certified #360. Follow-up #373 added canonical masking, changed init RNG through pruning, and changed sparse gradient storage. Pin a certified source artifact before claiming exact reproduction.
- Track 2: NOT run. Hardware feasibility, local runtime and optimizer transfer are unmeasured.
- No cross-track rank correlation, optimizer-component profile or warm-cache startup measurement exists yet.

## Checkout and publishing

Our fork: https://github.com/michael-lutz/modded-nanogpt

Working branch: `research/optimizer-lab`. Upstream source snapshot: `4ea6b937337a4889b8cfe3f38a93d120048d8f71`. The benchmark source is unchanged by this research commit. The original baseline checkouts and private runtime/cache directories remain separate. Only curated public metrics are included; no environments, datasets, host identifiers, credentials or generated compiler caches are committed.

## Small audit tools

```bash
python3 research/optimizer_lab/tools/build_history.py
python3 -m unittest discover -s research/optimizer_lab/tests -v
python3 research/optimizer_lab/tools/cost_model.py --step-ratio 0.95 --optimizer-fraction 0.10 --optimizer-cost-ratio 1.50
```

The cost model is a conditional back-of-envelope calculation, NOT measured end-to-end speed. It assumes equal work per step apart from optimizer cost, fixed optimization overhead fractions, and no changing overlap. Real timing is required.
