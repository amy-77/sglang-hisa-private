# Router + Global Unique-Head Budget: implementation design

## Contract

For each query row and transformer layer, the router emits two objects:

- `unique_head_ids: [nq, M]`: exactly `M` global Indexer heads;
- `pair_to_slot: [nq, 64]`: each MLA pair chooses a slot in `[0, M)`.

The Indexer scan produces `unique_indices: [nq, M, K]`. Sparse MLA consumes
`(unique_indices, pair_to_slot)` directly. It must not expand the table to
`[nq, 64, K]`, because expansion preserves the old indices-memory cost.

For int32 `K=720`, the compact payload is `M*720*4 + 64` bytes per query and
layer when the map is uint8: 11,584 bytes at `M=4` and 23,104 bytes at `M=8`,
versus 184,320 bytes for 64 independent lists.

## Hierarchical router

Use two predictions with separate losses:

1. A global set scorer predicts pair-by-head utility scores and applies greedy
   facility-location selection or a distilled exact `TopM` head score.
2. A pair assignment scorer is masked to those `M` heads and chooses one slot
   for each two-head MLA pair.

The first SGLang offline model uses normalized MLA-pair queries, a low-rank
projection, layer priors, and the existing 64-way Indexer gate. A second
variant also projects the current Indexer queries as dynamic keys. A hybrid
training target combines soft utility rankings and the hard best-head label.

The 8-query temporal experiment should determine router frequency. If an
`M=8` set selected at the first decode query retains the per-query set oracle,
cache the global set for a request or short query block and run only the cheap
pair assignment every token.

## TP8 execution

The correctness-first path may replicate the selected set on every rank. It
changes per-rank scans from 8 to `M`, but cluster scans from 64 to `8M`, so it
only saves scan work for `M < 8`.

The intended path owner-shards selected heads:

1. all ranks deterministically produce the same `unique_head_ids`;
2. owner rank `head_id % tp_size` scans each selected head once;
3. all-gather `(head_id, Top-K indices)` in a fixed `M`-slot layout;
4. each rank runs sparse MLA for its local 8 pairs through `pair_to_slot`.

The communicated Top-K payload is 11,520 bytes for `M=4` and 23,040 bytes for
`M=8`, before fixed-layout padding. Kernel latency and the collective must be
benchmarked; the oracle scan ratio alone is not an end-to-end speedup claim.

## Online distillation in Megatron-LM

Recommended configuration surface:

- `dsa_unique_head_budget` (`M`);
- `dsa_router_rank`;
- `dsa_router_loss_coeff`;
- `dsa_router_teacher_interval` and `dsa_router_teacher_rows`;
- `dsa_router_set_loss_weight` and `dsa_router_assignment_loss_weight`;
- `dsa_router_mode = train_only | routed`.

Integration point: `DSAttention.forward`, immediately after
`DSAIndexer.forward_before_topk`. The existing model and Indexer stay frozen
for the first stage; router inputs and teacher utilities are stop-gradient.

On sampled query rows, each TP rank computes the full 64-candidate utility
only for its 16 local MLA heads. An exact global facility teacher does not
need to all-gather `[64,128]`: at each of `M` greedy steps, all-reduce the 64
candidate marginal-gain sums and select the same global head on every rank.
Local pair argmax labels train assignment.

Suggested loss:

`L_router = lambda_set * L_set + lambda_assign * L_assign + lambda_regret * L_regret`

- `L_set`: multi-label or ordered marginal-gain distillation for the greedy
  global set;
- `L_assign`: cross entropy over the selected `M` heads for local MLA pairs;
- `L_regret`: soft utility-weighted ranking loss to avoid treating nearly tied
  heads as completely different labels.

Use hard exact Top-M only at inference. Do not initially backpropagate through
the token Top-K operator or sparse MLA output. After the router matches oracle
coverage, enable routed forward and optionally fine-tune the Indexer.

## Staged acceptance criteria

1. Static `M=4` and `M=8` head dictionaries establish scan/collective/kernel
   baselines.
2. Offline prompt-grouped CV must beat layer-static assignment and approach
   predicted-set + oracle-assignment quality.
3. Online training is justified if more independent 128K queries leave a
   material assignment gap, not merely because the first 16-query dataset is
   small.
4. A routed training path is accepted only after checking attention mass,
   downstream LongBench/RULER quality, peak indices memory, per-layer scan
   time, collective time, and end-to-end tokens/s.
