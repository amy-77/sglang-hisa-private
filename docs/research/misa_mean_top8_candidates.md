# Mean-ranked 256-token top-8 candidate regions

Implemented on 2026-09-10. This is a candidate-selection policy, not a claim of
lossless retrieval or measured end-to-end accuracy.

Stage A computes all 64 Indexer heads' coarse affinities. For each head, it takes
the eight highest `q @ mean_k256` regions and uses the signed Indexer gate times
the sum of those eight affinities after ReLU as that head's importance. It then
selects M heads and retains the same eight region IDs for Stage B; it does not
return to an all-region sum or choose coarse regions again. Negative affinities
remain ordered, invisible blocks are masked, and short rows pad missing region
IDs with -1. No sink/tail coarse region is forced into this fixed-count policy.

Unpruned selection and the legacy fractional-retention ablation continue to use
the all-coarse-block importance sum, so old experiment configurations keep their
original semantics.

Stage B directly expands these IDs into at most 128 logical chunk16 candidates.
Only these candidates are scored, using `q @ chunk_sum / visible_token_count`.
Their fine mean scores determine quota rank: the first 52 chunks retain their
8 highest exact token scores; the remaining 76 retain 4. With full candidates,
this yields `52*8 + 76*4 = 720` unique token indices. There is no additional
coarse-region selection and no boundary chunk/token score boost. Partial chunks
or fewer visible regions may yield fewer valid picks; padding remains -1.
Visible sequences of length <=720 continue to retain every visible token.

Runtime configuration fields:

```json
{
  "misa_chunk_size": 256,
  "misa_prune_topk": 8,
  "misa_prune_keep_fraction": null
}
```

These are the new config-builder defaults. Existing configurations explicitly
setting `misa_prune_keep_fraction` retain the legacy fraction policy, including
its boundary pinning. The two pruning options are mutually exclusive.

Teacher collection uses `SGLANG_NSA_HEADMAP_PROBE_MISA_BLOCK_SIZE=256` and
`SGLANG_NSA_HEADMAP_PROBE_MISA_TOPK=8`. The pilot script sets both and builds a
matching runtime config. Explicit `SGLANG_NSA_HEADMAP_PROBE_MISA_KEEP` selects
the legacy fraction ablation instead; do not combine it with TOPK.

Mean ranking has no upper-bound guarantee: an excluded coarse region can hide
important fine chunks. Candidate recall/MLA attention mass and downstream
accuracy must be measured separately. Assignment models should be trained on
candidates produced by this same policy.

Validation: all 24 tests in `test/registered/unit/test_misa_head_router.py`
passed in the project's `qyl/sglang-hisa:eval` image on an H20 GPU, including
CPU mean-region references, CUDA exact candidate-set references, mixed lengths,
partial tails, noncontiguous page mappings, and the legacy selector regression.
Python syntax and pilot-shell syntax checks also passed. No full-model accuracy
or latency result is claimed by this change.

## Matched Group16 probe result

After changing Stage A head importance to the signed-gate Top-8 coarse-affinity
sum, the full policy was recollected from the original 720 trajectories and
unchanged train/validation split. The probe contains 150,304 layer-context
examples. Its Group16 candidate oracle is 0.5036115 on all validation examples
and 0.3110046 on RULER validation, versus 0.7244165 and 0.6430063 respectively
for the matched old 512/60%-retention candidates. The deltas are -0.2208050 and
-0.3320017, so the training gate failed and no new Group16 router was trained.

This result tests the complete constrained candidate path, not merely using a
Top-8 coarse prefix to rank which Indexer heads enter M. The earlier selector
ablation reused old candidate token lists and therefore did not establish that
searching only eight 256-token regions retained sufficient token recall.

Artifacts are under
`/DATA/disk0/qyl/data/misa_assignment_router_group16_256_top8sum_v1_20260910`;
`gate_report.json` contains the exact comparison.
