# Router + Global Unique-Head Budget: real 128K oracle

## Executive conclusion

The real 128K decomposition supports a hard global unique-head budget. It does
not support treating 64 independent pair routers as the system sparsity
mechanism: per-pair sparsity alone does not bound the union of scanned Indexer
heads.

The strongest current operating points are:

- `M=4`: a cross-prompt static set plus per-query oracle pair assignment retains
  98.88% of the unrestricted per-pair oracle while reducing the logical set
  from 64 candidate heads to 4.
- `M=8`: the same ratio is 99.19%.
- A dynamic per-query set is even more concentrated: `M=1` retains 98.82% and
  `M=2` retains 99.54% of the unrestricted per-pair oracle.

Thus the recommended abstraction is a hierarchical constrained router:

1. choose at most `M` unique Indexer heads globally for each query and layer;
2. assign each MLA pair only within that selected set;
3. scan each selected Indexer head once and reuse its Top-K indices.

## Experiment

- Model: DeepSeek-V3.2 checkpoint used by the current SGLang-HISA server.
- Data: 16 real RULER 128K prompts from CWE, NIAH variants, QA and variable
  tracking, with 8 consecutive decode queries per prompt (128 queries total);
  observed model sequence lengths were 128,156 to 131,007.
- Layers: 0, 2, 7, 15, 30, 45 and 60.
- Tensor-parallelism: TP8.
- For every query/layer, every TP rank evaluated all 64 Indexer query heads
  against its 16 local MLA heads. Aggregation gives a complete
  `[64 Indexer, 128 MLA]` attention-mass matrix.
- Each Indexer candidate list contains at most 720 indices from the current
  chunk16-quota per-head selector.
- The main router unit is one Indexer head per two-head MLA pair. The set
  objective is facility location:

  `F(S) = mean_p max_{i in S} U[p, i]`, subject to `|S| <= M`.

## Attention-mass baselines

| Method | Mean retained mass |
|---|---:|
| Fixed 1:2 per-head Top-720 | 0.4045 |
| Unrestricted best Indexer per MLA pair | 0.6117 |
| Unrestricted best Indexer per individual MLA head | 0.6160 |
| Official shared DSA Top-2048 | 0.7493 |
| Exact MLA Top-720 ceiling | 0.7616 |

The pair restriction costs only 0.0043 absolute mass (about 0.70%) compared
with routing all 128 MLA heads independently. Pair routing is therefore the
better system unit unless later task-quality evaluation contradicts it.

The reduced per-head Top-720 scheme still trails official shared Top-2048.
Unique-head reduction preserves the ceiling of the current per-head selector;
it does not by itself solve candidate-quality loss relative to official DSA.

## Global unique-head budget oracle

| M | Dynamic set + dynamic assignment | Static set + dynamic assignment (2-fold CV) | Static set + static assignment (2-fold CV) | Static-set fraction of unrestricted |
|---:|---:|---:|---:|---:|
| 1 | 0.6045 | 0.5846 | 0.5846 | 95.56% |
| 2 | 0.6089 | 0.5982 | 0.5864 | 97.78% |
| 4 | 0.6108 | 0.6049 | 0.5871 | 98.88% |
| 8 | 0.6115 | 0.6068 | 0.5873 | 99.19% |
| 16 | 0.6117 | 0.6079 | 0.5873 | 99.37% |
| 32 | 0.6117 | 0.6093 | 0.5873 | 99.59% |
| 64 | 0.6117 | 0.6117 | 0.5873 | 100.00% |

Two points matter:

- Global set concentration is very strong. Most of the unrestricted benefit
  is available at `M=1..4`.
- Dynamic assignment is still useful. With `M=8`, static assignment scores
  0.5873 whereas oracle per-query assignment scores 0.6068. The remaining
  learning problem is primarily assignment, not finding a large candidate
  set.

Layer 7 is the strongest counterexample to a static `M=1`: static `M=1`
retains 0.6275 versus an unrestricted 0.7085, while static `M=4` recovers
0.6989. This is why `M=4` is a safer first implementation point than `M=1`.

## Decode-time decomposition

The discrete oracle head IDs change often, but their utility is redundant. At
`M=8`, the first decode query's selected set reused for all eight queries scores
0.6064 versus 0.6115 for a freshly selected set every query. Reusing the first
query's pair assignment as well scores only 0.5313. At `M=4`, the corresponding
numbers are 0.5946, 0.6108 and 0.5310.

This supports two router time scales: update the global set at request or block
granularity (especially at `M=8`), but compute a cheap assignment inside the
set every query. The mean set Jaccard to the first query is only 0.382 at
`M=8`, so hard head-ID classification is the wrong sole objective; training
must preserve utility and treat near-equivalent sets as equivalent.

## Offline router result

A low-rank router was trained from MLA pair queries, layer/head and
layer/pair/head priors, and the cheap 64-way Indexer gate. Variants used
separate set/assignment scorers, dynamic Indexer-query keys, budget-conditioned
assignment, ranks 8 and 32, and listwise, hard, regression and hybrid losses.
Cross-validation splits entire prompts, not pair rows.
The expanded dataset contains 896 query-layer contexts and 57,344 pair rows,
with all eight queries from one prompt kept in the same CV fold.

The best `M=1` result was 0.5897, compared with 0.5846 for the static baseline
and 0.6045 for the dynamic oracle. At `M=4`, a predicted set plus oracle
assignment reached 0.6076, confirming that global set selection is learnable.
The best joint result was only 0.5877; even a static set plus learned assignment
peaked at 0.5917, versus 0.6049 with oracle assignment. The existing Indexer
gate's argmax scored only 0.3656.

The expanded experiment rules out the original small-data explanation for
most of the gap. Query-only offline routing reliably learns the global set but
not the rapidly changing pair assignment. Online utility distillation is now
justified, initially on sampled rows with the base model and Indexer frozen.

## TP8 system semantics

The current per-head path scans eight local Indexer heads on every TP rank.
There are two implementation choices:

1. Replicated selected set: every rank scans the same `M` heads. Per-GPU scan
   count changes from 8 to `M`, and cluster scan count changes from 64 to
   `8M`. This saves compute only for `M < 8`.
2. Owner-sharded selected set: deterministically assign each selected head to
   one TP rank, scan it once, and all-gather `(head_id, Top-K indices)`. This
   changes the cluster scan count from 64 to `M`, realizing the intended
   `64/M` reduction.

An int32 Top-720 list is 2,880 bytes. The owner-sharded communication payload
is therefore 11,520 bytes at `M=4` and 23,040 bytes at `M=8` per query/layer,
before collective padding. To reduce indices memory rather than only scan
compute, sparse MLA must consume compact `unique_indices[n,M,K]` plus a uint8
`pair_to_slot[n,64]` map, without expanding back to 64 lists. Compact storage
is 11,584 bytes at `M=4` and 23,104 bytes at `M=8`. The design should benchmark
this collective latency rather than claim a 64/M speedup from oracle cardinality alone.

## Recommended router

Use an exact hard budget, not an entropy penalty:

- Global logits `g_i(q, layer)` select `TopM(g)`.
- Pair logits `a_{p,i}(q_p, layer)` are masked to the selected set.
- An Indexer head is scanned once if any pair uses it.
- The training teacher is the attention-mass utility matrix. Distill global
  facility marginal gains and pairwise rankings separately.
- Do not backpropagate through Top-K indices initially. Freeze the model and
  Indexer, train only the router, and use stopped-gradient oracle targets.

The query projection for all 64 Indexer heads is small compared with scanning
128K keys. It is acceptable for the first implementation to compute all query
vectors but run the expensive scan only for the selected `M` heads.

## Megatron online-training feasibility

The local Megatron-LM tree already contains the necessary DSA substrate:

- `DSAIndexer.forward_before_topk` produces per-head Indexer queries, shared
  keys and gates.
- `DSAttention.forward` owns Top-K generation, SparseMLA consumption and
  Indexer loss attachment.
- The TileLang/cuDNN paths already implement fused Indexer Top-K and sparse
  KL targets.

Online adaptation is feasible, but it needs a new per-head candidate tensor
and cannot be implemented by changing only the existing scalar DSA Indexer
loss. The minimal staged integration is:

1. add a router module and configuration, initially training-only;
2. periodically compute full 64-head teacher utilities on sampled layers and
   query rows;
3. distill global-set and assignment losses with an exact `M` constraint;
4. add owner-sharded selected-head Top-K plus index all-gather;
5. only then enable the routed SparseMLA forward path.

A disabled-by-default Megatron scaffolding module now implements batched exact
facility Top-M, pair-to-selected-slot routing, and separate soft-utility,
exact-membership and within-budget assignment losses. Its 8-rank unit test
passes. This is a training primitive, not yet a claim that the per-head scan
or compact sparse-attention kernels are integrated.

Given the 98.88% static-set oracle at `M=4`, first implement and benchmark a
static `M=4`/`M=8` system baseline. The larger offline dataset did not close
the roughly 0.019 assignment-mass gap at `M=8`, so the next implementation
stage is sampled online router distillation in Megatron.

## Artifacts

- `oracle_summary.json`: full aggregate and per-layer metrics, selections and
  system budget accounting.
- `oracle_summary.md`: compact oracle table.
- `router_*.json` and `.md`: offline-router CV configurations and ablations.
- `manifest.json`: exact prompt/task mapping and timings.
- `stability.json/.md`: first-query reuse, per-query set and assignment
  decomposition.
- `results/`: 7,168 JSON shards and 7,168 feature tensors.
