# TP-local 16-head MISA assignment router

The physical attention TP layout remains TP=8 over 128 MLA heads. Rank r owns
heads `[16*r, 16*r+15]` in their existing order. The assignment decision changes
from 64 pairs to 8 TP-local groups, with no head migration or learned regrouping.
The M Indexer candidates are independent of these 8 MLA groups: several groups
may choose the same candidate, and not every candidate must be used.

## Training contract

- Input: concatenate each rank's 16 original absorbed MLA queries, preserving
  order. For D=576, `q_group` is `[B,8,9216]`, replacing `[B,64,1152]`.
- Target: average the original attention mass over all 16 heads for each
  candidate, yielding utility `[B,8,M]`. Aggregate utility before constructing
  soft targets; do not vote over old pair labels or average pair predictions.
- Model/loss: retain the existing per-layer PyTorch projections, candidate
  features, AdamW, utility distillation and expected-regret loss. Only the query
  input width and group axis change. No new group embeddings or cross-rank layer
  is added, and there is no checkpoint warm start.
- Every training call constructs a fresh model. Format-3 checkpoints record
  `mla_heads_per_group=16`, `attention_tp_size=8`, `assignment_unit=tp_local`, and
  `initialization=random`. The config builder rejects pair-router checkpoints
  for the new learned deployment path.

Offline dataset assembly uses one existing rank record per group on CPU. It
requires ranks 0..7, each with its original 16 global head IDs in order. This is
ordinary batching of training examples, not moving deployed MLA heads across
GPUs. The training implementation remains a standard PyTorch module and loop;
Megatron integration is outside this change.

## Runtime contract

Each rank reshapes only its local `[B,16,D]` query to `[B,1,16*D]`, scores M
candidates once, and selects one token list. The existing attention interface
still has 8 local two-head slots; the selected token list is repeated into those
8 slots locally. All 16 heads therefore consume the same candidate. Neither the
MLA tensors nor their placement change, and no new communication is introduced.
The group16 path rejects attention TP other than 8 or local head count other
than 16. Optional runtime traces now record `q_group` and group metadata.

## First training run

Output directory on h20-9-57:
`/DATA/disk0/qyl/data/misa_assignment_router_group16_v2_20260910`.

The input is the original per-head probe from
`misa_assignment_router_pilot_v2/probe`, using `samples.jsonl` and
`--probe-sample-stride 2`. Each replay saved two adjacent decode probes, so raw
probe ID `2*j` maps to manifest sample j. The available data covers 2,464 replay
contexts before the minimum-length filter: 2,178 train and 286 validation,
635 train and 85 validation prompts, across 61 layers. The final 3 manifest
replays have no saved probes. Train/validation source IDs are disjoint.

This raw probe predates the new 256/top8 candidate policy. Its associated
runtime config specifies 512-token pooling with 60% retention; raw shards have
no policy field. The run isolates the new group16 assignment constraint using
existing raw supervision. It must not be described as an accuracy validation
of the new mean top-8 candidate selector. New-policy teacher collection and
full-model evaluation remain separate experiments.

Training command uses the existing `scripts/train_misa_assignment_router.py`:
10,000 steps, batch 32, rank 64, AdamW lr 3e-4, default loss coefficients, seed
20260907, minimum visible length 4096, and fresh initialization. The checkpoint
and report are written to new paths; existing pair checkpoints are untouched.

## Completed results (2026-09-10)

The fresh 10,000-step run completed successfully with 150,304 layer-context
examples (2,464 replay contexts x 61 layers) and 720 prompt groups. The saved
checkpoint has 37,228,910 parameters and q projection shape `[61,64,9216]`.
Validation mean attention mass is 0.7054157, versus 0.7244165 for the group
candidate oracle and 0.7041977 for the static group-assignment baseline. Mean
regret is 0.0190008. The improvement over static assignment is small. There is
no separate test split in this probe manifest and no full-model evaluation.

The matched pair-router run on the same old candidates has validation learned
mass 0.7163590 and pair candidate oracle 0.7315941. Group16 therefore changes
the all-validation oracle by -0.0071775 and learned mass by -0.0109433. On the
RULER validation subset, pair versus Group16 is 0.6486159 versus 0.6430063 for
the oracle (-0.0056096), and 0.6321624 versus 0.6200721 for learned assignment
(-0.0120903). This is the controlled estimate of the grouping constraint itself;
the larger learned delta also includes the Group16 model's assignment regret.

`runtime_probe_policy.json` reproduces the source 512/60% candidate policy.
`runtime_mean_top8.json` enables the previously implemented 256/top8 selector
with the new group16 checkpoint, but its accuracy has not been validated.
Both paths use the same fixed TP-local 16-head input and shared group decision.
Configuration checkpoint paths use the existing `/workspace/qyl` container mount.

47 unique tests passed across the existing CUDA selector and group16 suites,
including the candidate-subset normalization regression. Validation logs and
checkpoint load verification are stored under the experiment's `verification/`.
