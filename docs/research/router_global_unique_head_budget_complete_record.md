# Router + Global Unique-Head Budget：128K Oracle、离线路由与在线训练完整记录

更新时间：2026-09-03  
实验服务器：`h20-9-57`，8 × NVIDIA H20 144 GB，TP8  
主要代码树：

- SGLang-HISA：`/DATA/disk0/qyl/code/dpskv32/sglang-hisa`
- Megatron-LM：`/DATA/disk0/qyl/code/Megatron-LM`
- 7 层实验目录：`/DATA/disk0/qyl/data/headwise_minmax_e2e/exp5_128k_multidecode_router`
- 全 61 层实验目录：`/DATA/disk0/qyl/data/headwise_minmax_e2e/exp6_128k_all61_multidecode_router`

## 1. 最终结论

这轮实验验证了研究问题的核心：真正需要约束的不是每个 MLA pair 使用几个
Indexer heads，而是一次 query/layer 上所有 pair 选择结果的全局并集。

设第 `p` 个 MLA pair 使用的 Indexer head 集合为 `S_p`。仅约束
`|S_p|` 很小，并不能保证系统只扫描少量 Indexer heads，因为

```text
per-pair sparsity != system compute sparsity
```

真正的系统约束应当是：

```text
|union_p S_p| <= M
```

因此推荐设计不是简单的 64 个独立 pair router，而是：

```text
Router + exact Global Unique-Head Budget
```

并进一步分解为两个时间尺度：

1. 低频 global-set router：为当前 request、layer 或短 query block 选出至多
   `M` 个全局 Indexer heads。
2. 高频 pair-assignment router：每个 decode token 把 64 个 MLA pairs 映射到
   已选集合内的 slot。

全 61 层真实 128K oracle 表明，动态 global set 下 `M=4` 已达到
unrestricted per-pair oracle 的 99.93%，`M=8` 达到 99.99%。因此 global
unique-head budget 是成立的，
而且强度足以支撑后续系统实现。

训练方面也有明确结论：global set 可以用离线数据学习；per-token assignment
用现有 q-only 离线特征学习仍有明显 gap，需要 sampled online utility
distillation。不是训练整个 DeepSeek-V3.2，而是先冻结 base model 和 Indexer，
只训练小型 router。

## 2. 研究问题与目标函数

当前逐 Indexer query head 方案对每个候选 Indexer head 扫描 128K keys，获得
自己的 Top-K indices，再交给 MLA heads。即使每个 MLA pair 最终只选一个
Indexer head，如果 64 个 pair 的选择并集接近 64，GPU 仍要执行几乎全部 scan。

定义 utility：

```text
U[p, i] = MLA pair p 使用 Indexer head i 的 Top-K indices 时保留的真实 attention mass
```

对全局集合 `S` 使用 facility-location 目标：

```text
F(S) = mean_p max_{i in S} U[p, i]
subject to |S| <= M
```

选出 `S` 后，每个 pair 的 oracle assignment 是：

```text
a_p = argmax_{i in S} U[p, i]
```

这个定义直接把两件事情解耦：

- `M` 决定需要扫描多少个 unique Indexer heads；
- assignment 保留 MLA pair 的 specialization。

## 3. 我具体做了什么

### 3.1 环境与代码审计

首先检查了 H20 节点、SGLang-HISA 代码、已有 per-head Top-K 路径、现有 probe、
实验数据和 GPU 状态。SGLang 工作树在实验前已经存在用户修改，因此整个过程
没有执行 reset、checkout 或覆盖无关改动。

随后检查 Megatron-LM 的 DSA 实现，确认：

- `DSAIndexer.forward_before_topk` 能提供每个 Indexer head 的 query、共享 key
  和 gate；
- `DSAttention.forward` 控制 Top-K、SparseMLA 与 Indexer loss；
- 官方 Megatron DSA 当前生成的是 weighted aggregation 后的共享 Top-K list，
  并不是本研究需要的 per-Indexer-head candidate table；
- 因此在线训练可接入现有 DSA loss/forward 生命周期，但 routed forward 需要
  新的 per-head scan 与 compact indirect sparse-attention 数据通路。

### 3.2 改造 128K probe

在 SGLang-HISA 中完成了以下改造：

- `python/sglang/srt/layers/attention/nsa/nsa_indexer.py`
  - per-head paged Top-K 支持可选 head range；
  - probe 模式能够遍历全部 64 个 Indexer query heads；
  - 导出 MLA query、Indexer query 与 gate 特征。
- `python/sglang/srt/layers/attention/nsa/headmap_probe.py`
  - 每个 TP rank 计算全部 64 个 Indexer heads 对本地 16 个 MLA heads 的
    attention mass/recall；
  - 保存 JSON 指标与 PT feature shard；
  - 旧版本保留为 `headmap_probe.py.pre128k`。

新增脚本：

- `scripts/run_128k_unique_head_probe.py`
- `scripts/analyze_128k_unique_head_oracle.py`
- `scripts/analyze_multidecode_head_stability.py`
- `scripts/train_offline_unique_head_router.py`

所有相关 Python 文件均通过 `py_compile`。

### 3.3 采集真实 128K 数据

最终数据集不是合成矩阵，而是模型真实 forward 中采集的 Top-K indices 与 MLA
attention mass：

- 模型：当前 SGLang-HISA 使用的 DeepSeek-V3.2 checkpoint；
- prompt：16 个真实 RULER 128K 样本；
- task 覆盖：CWE、多个 NIAH、QA、variable tracking；
- 每个 prompt：连续采集 8 个 decode queries；
- query 总数：128；
- 代表层：0、2、7、15、30、45、60；
- 并行：TP8；
- 实际序列长度：128,156 至 131,007；
- Indexer candidates：64；
- MLA heads：128，主路由单位为 64 个 two-head MLA pairs；
- candidate Top-K：当前 chunk16-quota per-head selector 的 Top-720；
- 数据量：7,168 JSON shards + 7,168 PT feature shards；
- 聚合 utility tensor：`[128 queries, 7 layers, 64 pairs, 64 indexer heads]`。

每个 TP rank 负责本地 16 个 MLA heads，但都评估全部 64 个 Indexer candidates；
聚合 8 个 rank 后得到完整的 `[64 Indexer, 128 MLA]` attention-mass matrix。

### 3.4 执行 oracle 分解

对每个 query/layer 分别计算：

- 当前固定 1:2 映射；
- unrestricted best Indexer per MLA pair；
- unrestricted best Indexer per individual MLA head；
- exact greedy facility-location budget `M=1,2,4,8,16,32,64`；
- dynamic set + dynamic assignment；
- prompt-grouped 两折交叉验证下的 static set + dynamic assignment；
- static set + static assignment；
- official shared DSA Top-2048 与 exact MLA Top-720 参照。

### 3.5 多 decode-token 稳定性分解

同一 prompt 的 8 个连续 query 被当作一个 temporal group，分别测量：

- 每个 token 都重新选 global set；
- 只复用第一个 token 的 global set、每 token 重新 assignment；
- global set 与 assignment 都复用；
- 整个 prompt 的 oracle set；
- selected-head ID Jaccard 和 8-token union size。

该实验用来回答 router 应该每 token 更新，还是可以 request/block 级缓存。

### 3.6 离线 router 训练与消融

从 PT shard 中恢复：

- MLA pair query：`[64, 1152]`；
- Indexer query：`[64, 128]`；
- 64-way gate；
- layer id；
- oracle utility：`[64 pairs, 64 indexer heads]`。

训练集共有 896 个 query-layer contexts 和 57,344 个 pair rows。交叉验证按
prompt 分组，同一 prompt 的 8 个 decode queries 永远处于同一个 fold，防止
temporal leakage。

实际测试了：

- low-rank 8 与 32；
- listwise、hard-label、regression、hybrid losses；
- 当前 Indexer queries 作为 dynamic keys；
- layer/head prior 与 layer/pair/head prior；
- set scorer 和 assignment scorer 分离；
- 只在 oracle `M=4` 或 `M=8` 集合内训练 assignment；
- 静态 set、预测 set、oracle assignment 和 learned assignment 的交叉组合。

### 3.7 Megatron 在线训练骨架

在 Megatron-LM 中新增：

- `megatron/core/transformer/experimental_attention_variant/dsa_unique_head_router.py`
- `tests/unit_tests/transformer/experimental_attention_variant/test_dsa_unique_head_router.py`

实现内容：

- batched exact greedy facility-location Top-M teacher；
- exact cardinality 与无重复 head 保证；
- MLA pair 到 selected-head slot 的映射；
- soft utility、exact set membership、within-budget assignment 三类蒸馏 loss；
- teacher utility stop-gradient。

验证：

- Python 编译通过；
- isort 与 119-column 检查通过；
- 8-rank distributed pytest 中每个 rank 均为 `5 passed`。

模块当前没有 import 到 `DSAttention.forward`，默认不影响现有训练。这是训练
primitive，不代表 owner-sharded scan 或 compact SparseMLA kernel 已完成。

### 3.8 补齐可保存的离线训练闭环

`train_offline_unique_head_router.py` 新增：

- `--checkpoint-dir`：保存每个 CV fold checkpoint；
- `--fit-all-steps`：在全部 prompt groups 上做最终拟合；
- checkpoint 内保存模型结构、训练参数、layer mapping、训练 sample ids、最终 loss
  和完整 state dict；
- 训练结果可恢复，不再只有汇总 JSON。

两步 smoke run 已验证生成：

- `router_fold0.pt`
- `router_fold1.pt`
- `router_all.pt`

checkpoint 大小约 540 KB，能够正常反序列化，结构为 rank-32、7 layers、pair
query dimension 1152、Indexer query dimension 128。

## 4. 主要实验结果

### 4.1 总体基线

| 方法 | Mean retained attention mass |
|---|---:|
| 当前固定 1:2 per-head Top-720 | 0.4045 |
| Unrestricted best Indexer per MLA pair | 0.6117 |
| Unrestricted best Indexer per individual MLA head | 0.6160 |
| Official shared DSA Top-2048 | 0.7493 |
| Exact MLA Top-720 ceiling | 0.7616 |

pair 路由相对逐 MLA-head oracle 只损失 0.0043，约 0.70%。因此使用 64 个 MLA
pairs 作为消费者单位是合理的，系统复杂度明显低于 128 个独立 head router。

同时必须注意：unique-head routing 只是在保持当前 per-head Top-720 selector
上限的前提下降低 scan cost；它不会自动消除当前 selector 与 official shared
Top-2048 之间的 candidate-quality gap。

### 4.2 Global unique-head budget

| M | Dynamic set + dynamic assignment | Static set + dynamic assignment | Static set + static assignment | Static-set fraction of unrestricted |
|---:|---:|---:|---:|---:|
| 1 | 0.6045 | 0.5846 | 0.5846 | 95.56% |
| 2 | 0.6089 | 0.5982 | 0.5864 | 97.78% |
| 4 | 0.6108 | 0.6049 | 0.5871 | 98.88% |
| 8 | 0.6115 | 0.6068 | 0.5873 | 99.19% |
| 16 | 0.6117 | 0.6079 | 0.5873 | 99.37% |
| 32 | 0.6117 | 0.6093 | 0.5873 | 99.59% |
| 64 | 0.6117 | 0.6117 | 0.5873 | 100.00% |

最重要的解释是：

- global set 高度集中，`M=1..4` 已覆盖绝大部分 utility；
- dynamic assignment 仍有价值；
- `M=4` 是第一个安全系统点；
- `M=8` 是接近无损并适合 temporal set reuse 的点。

Layer 7 是不应直接使用 static `M=1` 的反例：unrestricted 为 0.7085，static
`M=1` 只有 0.6275，而 static `M=4` 恢复到 0.6989。

### 4.3 分层结果

| Layer | Fixed | Unrestricted pair | Official shared | Exact MLA | Dynamic M=1 | Static M=1 | Static M=4 | Static M=8 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0170 | 0.0639 | 0.4107 | 0.4449 | 0.0607 | 0.0387 | 0.0577 | 0.0580 |
| 2 | 0.1928 | 0.3414 | 0.4428 | 0.4892 | 0.3220 | 0.3083 | 0.3325 | 0.3369 |
| 7 | 0.2708 | 0.7085 | 0.7846 | 0.7932 | 0.7025 | 0.6275 | 0.6989 | 0.7011 |
| 15 | 0.6208 | 0.7993 | 0.8505 | 0.8506 | 0.7950 | 0.7849 | 0.7922 | 0.7938 |
| 30 | 0.6802 | 0.8542 | 0.9623 | 0.9609 | 0.8486 | 0.8400 | 0.8477 | 0.8503 |
| 45 | 0.5135 | 0.8476 | 0.9519 | 0.9513 | 0.8406 | 0.8353 | 0.8391 | 0.8415 |
| 60 | 0.5363 | 0.6674 | 0.8419 | 0.8414 | 0.6621 | 0.6574 | 0.6659 | 0.6659 |

### 4.4 Decode-time stability

| M | Dynamic set/assignment | First-query set + dynamic assignment | First-query set + first assignment | Prompt-oracle set + dynamic assignment | Mean Jaccard to first | 8-token union |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6045 | 0.5287 | 0.5287 | 0.5995 | 0.218 | 3.60 |
| 2 | 0.6089 | 0.5830 | 0.5363 | 0.6050 | 0.222 | 6.46 |
| 4 | 0.6108 | 0.5946 | 0.5310 | 0.6080 | 0.272 | 11.46 |
| 8 | 0.6115 | 0.6064 | 0.5313 | 0.6093 | 0.382 | 18.43 |

`M=8` 下复用第一个 query 的 global set 只损失 0.0051，但复用 assignment
损失 0.0802。离散 head ID 的 Jaccard 较低，却仍能保留 utility，说明候选 head
之间存在大量功能冗余。因此训练不能只做 hard selected-head classification，必须
用 utility/regret soft targets 容忍等价集合。

### 4.5 离线 router

| 指标 | M=4 | M=8 |
|---|---:|---:|
| Fully dynamic oracle | 0.6108 | 0.6115 |
| Layer-static set + oracle assignment | 0.6049 | 0.6068 |
| Layer-static set + static assignment | 0.5871 | 0.5873 |
| Best predicted set + oracle assignment | 0.6076 | 0.6097 |
| Best static set + learned assignment | 0.5917 | 0.5902 |
| Best joint learned set + assignment | 0.5877 | 0.5873 |

现有 64-way Indexer gate argmax 只有 0.3656。

离线结果不能简单概括成“router 不行”：

- set selection 是成功的；预测 set + oracle assignment 已优于 cross-prompt
  static-set oracle；
- assignment 是失败的部分；现有 q-only features 无法追上在线 utility；
- 因此最佳策略是保留 offline/slow set router，把在线训练预算集中到
  per-token assignment。

## 5. 系统收益与必须满足的实现语义

### 5.1 Scan compute

当前 TP8 per-head 路径中，每个 rank 扫描 8 个本地 Indexer heads，总计 64 次。

错误但容易实现的方案是让每个 TP rank 都扫描相同的 `M` 个 heads：

```text
per-GPU scans: 8 -> M
cluster scans: 64 -> 8M
```

这在 `M=8` 时没有集群总 scan 收益。

目标实现必须 owner-shard：

```text
owner(head_id) = head_id % tp_size
```

selected head 只由 owner rank 扫描一次，再 all-gather `(head_id, Top-K indices)`。
这样 cluster scans 才真正从 64 下降到 `M`。

### 5.2 Indices 内存

SparseMLA 应直接消费：

```text
unique_indices: [num_query, M, K]
pair_to_slot:   [num_query, 64]
```

不能在进入 attention kernel 前展开回 `[num_query, 64, K]`。

Top-720、int32 时，每 query/layer：

| 表示 | 字节数 |
|---|---:|
| 64 个独立 indices lists | 184,320 |
| Compact M=4，含 uint8 slot map | 11,584 |
| Compact M=8，含 uint8 slot map | 23,104 |

对应约 15.9× 和 8.0× indices-memory reduction。owner-sharded collective 的
纯 indices payload 分别为 11,520 和 23,040 bytes，尚未包含固定布局 padding。

### 5.3 不能过度声称的部分

目前已经证明的是 utility concentration 与内存/scan 上限，不是端到端速度提升。
最终必须实测：

- selected-head scan kernel 时间；
- TP all-gather 时间；
- compact indirect SparseMLA kernel 时间；
- peak memory；
- prefill/decode tokens/s；
- RULER、LongBench 与真实生成质量。

## 6. 是否需要训练，以及现在能训练什么

### 6.1 需要训练，但只训练 router

建议 Stage 1 冻结：

- DeepSeek base model；
- MLA；
- 当前 DSA Indexer。

只更新：

- global-set router；
- within-budget assignment router。

teacher utility 与 greedy Top-M target 全部 stop-gradient；第一阶段不穿过 hard
Top-K indices 反向传播。

建议 loss：

```text
L_router = lambda_set * L_soft_utility
         + lambda_member * L_set_membership
         + lambda_assign * L_within_budget_assignment
         + lambda_regret * L_regret
```

其中 set 和 assignment 必须分开记录与调权，不能只报告总 loss。

### 6.2 已经完成的训练

离线 rank-32 regression router 的正式 CV + all-data final fit 已完成。该任务使用
已有 128K feature shards，只占 CPU，没有争抢当前 GPU 作业。

训练配置：

```text
rank=32
loss=regression
learning_rate=0.003
CV steps=600 per fold
all-data final-fit steps=600
prompt-grouped 2-fold CV
budgets evaluated: 1,2,4,8,16
```

输出目录：

```text
/DATA/disk0/qyl/data/headwise_minmax_e2e/exp5_128k_multidecode_router/
  training/offline_router_regression_r32/
```

实际产物：

- `router.json` 与 `router.md`；
- `checkpoints/router_fold0.pt`；
- `checkpoints/router_fold1.pt`；
- `checkpoints/router_all.pt`；
- `train.log`。

两折最终 train loss 分别为 6.0857 和 6.1598，all-data final-fit loss 为
6.1952。保存的 `router_all.pt` 含 12 个 state tensors，能够正常反序列化。

本次正式复跑的 CV 指标为：

| M | Joint router | Predicted set + oracle assignment | Static set + router assignment | Static set + oracle assignment |
|---:|---:|---:|---:|---:|
| 1 | 0.5834 | 0.5834 | 0.5846 | 0.5846 |
| 2 | 0.5858 | 0.5988 | 0.5881 | 0.5982 |
| 4 | 0.5877 | 0.6056 | 0.5902 | 0.6049 |
| 8 | 0.5873 | 0.6081 | 0.5894 | 0.6068 |
| 16 | 0.5872 | 0.6092 | 0.5886 | 0.6079 |

它再次复现了同一个判断：预测 global set 是有效的，但 learned assignment 没有
接近 oracle assignment。保存 final checkpoint 的价值是提供一个可部署、可比较的
离线 baseline，而不是宣称离线训练已解决问题。

这次 final fit 是可恢复的离线基线和部署 smoke checkpoint，不应当被解释为已经
解决泛化问题，因为训练数据只有 16 个独立 prompt groups。

### 6.3 尚不能直接启动的训练

Megatron sampled online distillation 还不能安全地直接开正式长跑，原因是：

1. Megatron 中目前只有 router teacher/loss primitive，还没有把 trainable router
   注册进 `DSAttention` 和 optimizer；
2. sampled full-64 teacher utility 尚未接入 Megatron forward；
3. routed per-head Top-K 和 compact SparseMLA 尚未实现；
4. 服务器上找到了 HF checkpoint
   `/DATA/disk0/qyl/models/deepseek-v3.2`，但没有发现可直接 resume 的
   Megatron distributed checkpoint；
5. 当前 8 张 H20 都被已有 SGLang/Ray 作业占用约 81 GB，不能抢占或终止未知任务。

因此“马上起一个 Megatron 命令”只会得到无 teacher、无 router 参数接线或无可用
checkpoint 的伪训练。正确做法是先完成下面的最小闭环，再启动 20–100 steps
online smoke run。

## 7. Megatron 在线训练实施顺序

### Stage A：训练接线，不改变 attention 输出

1. 在 TransformerConfig 增加：
   - `dsa_unique_head_budget`；
   - `dsa_router_rank`；
   - `dsa_router_loss_coeff`；
   - `dsa_router_teacher_interval`；
   - `dsa_router_teacher_rows`；
   - set/assignment/regret weights；
   - `dsa_router_mode=train_only|routed`。
2. `DSAttention` 构造 trainable router。
3. 在 `DSAIndexer.forward_before_topk` 后获得 router features。
4. 只在采样 layer/query rows 上计算 full-64 teacher utility。
5. 把 router loss 通过类似 `DSAIndexerLossAutoScaler` 的机制附着到主 loss。
6. 保持原 DSA output 不变，只验证 loss、gradient 与 checkpoint。

### Stage B：分布式 oracle teacher

每个 TP rank 只计算本地 16 个 MLA heads 的 utility。facility greedy 的每一步：

1. 本地计算 64 个 candidate 的 marginal-gain sums；
2. TP all-reduce 64 个标量；
3. 所有 rank 选择相同 head；
4. 重复 `M` 次。

不需要 all-gather 完整 `[64,128]` utility tensor。assignment target 保留在本地
pair 上即可。

### Stage C：在线训练 smoke acceptance

20–100 steps 必须检查：

- router 参数存在非零 gradient；
- base model/Indexer 在 frozen stage 没有梯度或更新；
- set、membership、assignment loss 都有限且下降；
- sampled teacher 的 collective 在 TP8 无 deadlock；
- checkpoint 保存与 reload 后 router 输出一致；
- 额外显存与 step time 可接受。

### Stage D：routed forward

在 train-only router 达到 oracle coverage 后再实现：

- owner-sharded selected-head scan；
- fixed-layout Top-K indices all-gather；
- compact `unique_indices + pair_to_slot`；
- indirect SparseMLA kernel；
- static M=4/M=8 与 learned router 的端到端对比。

## 8. 推荐的首轮训练超参数

建议从 `M=8` 开始在线训练，因为它允许 global set 在多个 decode tokens 间复用，
oracle 损失也最小；系统性能基线同时保留 `M=4`。

建议首轮：

```text
base/indexer: frozen
router rank: 32
M: 8
teacher interval: 8 steps
teacher rows: 每个 sampled layer 1-4 个 query rows
sampled layers: 先使用 0,2,7,15,30,45,60
optimizer: AdamW
router lr: 1e-3 起步，必要时降到 3e-4
weight decay: 1e-4
gradient clip: 1.0
temperature: 0.04
precision: router logits/loss FP32，features 可 BF16
```

首轮目标不是 downstream quality，而是：

```text
predicted-set + oracle-assignment >= 0.605 at M=4
static-set + learned-assignment 明显超过 0.5917
joint learned route 接近 static-set + oracle-assignment
```

## 9. 复现与产物索引

### 实验结果

```text
/DATA/disk0/qyl/data/headwise_minmax_e2e/exp5_128k_multidecode_router/
  CONCLUSIONS.md
  DESIGN.md
  FULL_RESEARCH_RECORD.md
  manifest.json
  oracle_summary.json
  oracle_summary.md
  stability.json
  stability.md
  router_*.json
  router_*.md
  router_ablation_summary.md
  results/
  training_smoke/
  training/
```

### SGLang-HISA 代码

```text
/DATA/disk0/qyl/code/dpskv32/sglang-hisa/
  python/sglang/srt/layers/attention/nsa/nsa_indexer.py
  python/sglang/srt/layers/attention/nsa/headmap_probe.py
  scripts/run_128k_unique_head_probe.py
  scripts/analyze_128k_unique_head_oracle.py
  scripts/analyze_multidecode_head_stability.py
  scripts/train_offline_unique_head_router.py
  docs/research/router_global_unique_head_budget_128k.md
  docs/research/router_global_unique_head_budget_design.md
```

### Megatron-LM 代码

```text
/DATA/disk0/qyl/code/Megatron-LM/
  megatron/core/transformer/experimental_attention_variant/dsa_unique_head_router.py
  tests/unit_tests/transformer/experimental_attention_variant/test_dsa_unique_head_router.py
```

## 10. 2026-09-03 评测前状态

已经完成：真实 128K full 64×128 candidate oracle、unique-budget 曲线、多-token 稳定性、离线路由
消融、TP8 系统语义、Megatron teacher/loss primitive、8-rank unit test，以及可保存
checkpoint 的离线训练闭环。

已经完成：基于 128K probe 数据的 rank-32 regression router 正式离线 CV 与
all-data final fit，三个 checkpoint 均已保存并验证可加载。

这里必须澄清：“full oracle”指每个采样 context 上完整的 64 个 Indexer heads ×
128 个 MLA heads utility matrix；exp5 只采样了 7 个模型层
`0,2,7,15,30,45,60`，不是 61 层全覆盖。因此 exp5 checkpoint 不能直接被描述成
完整模型的 downstream router。

截至这一阶段，尚未得到 LongBench v2、AIME 2025、RULER 上由该 checkpoint
实际生成的 downstream accuracy。已有的 held-out attention-mass 是机制指标，不能
替代任务正确率。

原计划的下一步是：

1. 等当前 GPU 作业释放；
2. 完成 Megatron Stage A train-only 接线；
3. 解决 HF-to-Megatron checkpoint 或选择一个较小 DSA-compatible checkpoint
   做训练机制验证；
4. 跑 20–100 step sampled online smoke；
5. assignment 指标确实提升后，再实现 owner-sharded routed forward。

研究命题当前可概括为：在一个 exact global unique-head budget 下复用少数
Indexer scans，同时允许 MLA pairs 保留细粒度 specialization；global set 的变化
慢且具有 utility 冗余，assignment 的变化快并需要在线蒸馏。这比“给 64 个 pair
各训练一个 sparse router”更直接地对应实际 GPU compute 与 indices memory。

## 11. 下游精度接线、全 61 层训练与正式评测

### 11.1 为什么之前不能声称已经测过精度

exp5 的 `router_all.pt` 已经完成离线训练并可正常加载，但当时 SGLang 仍只执行
固定 1:2 per-head Top-K；checkpoint 没有接进真实 decode forward。直接运行
LongBench/AIME/RULER 只会测到旧方案，而不是 router。

此外，exp5 checkpoint 只有 7 层参数。把它静默复用到其余 54 层会产生没有训练
依据的结果。因此本次把工作拆为两个层次：

1. 7 层 canary：只在 checkpoint 覆盖层启用 global M=8 + learned assignment，
   其余层回退官方 shared DSA；只用于验证接线。
2. 全 61 层正式模型：重新采集所有层的 128K oracle，训练全层 checkpoint 后才跑
   三数据集完整矩阵。

### 11.2 新增的 SGLang 两阶段 runtime

新增：

```text
python/sglang/srt/layers/attention/nsa/offline_unique_head_router.py
scripts/build_offline_router_runtime_config.py
docs/research/offline_router_m8_partial.json
```

并修改：

```text
python/sglang/srt/layers/attention/nsa/per_head_paged.py
python/sglang/srt/layers/attention/nsa/nsa_indexer.py
python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
python/sglang/srt/managers/schedule_batch.py
python/sglang/srt/model_executor/forward_batch_info.py
```

真实 forward 分成两阶段：

```text
Indexer 阶段：只扫描该层配置中的 M 个全局 unique indexer heads
             -> candidates [token, M, TopK]

MLA 阶段：得到 post-RoPE absorbed MLA query 后
          -> learned assignment scorer
          -> 每个本地 MLA pair 选择一个 candidate slot
          -> [token, local_pairs, TopK]
```

这保证逻辑 candidate head 集合由 `M` 约束，而不是由 64 个 pair 各自选择后的
union 偶然决定。当前 accuracy-first 实现会在每个 TP rank 上复制扫描这 M 个 heads；
还没有做 owner-sharded scan + indices all-gather。因此 M=8 相对“每 rank 固定扫描
8 个本地 heads”的 cluster 总 scan 数暂时不下降，不能把本轮 accuracy canary 的
延迟当作最终系统加速。真正做到 cluster 只扫描 M 次仍需 owner-sharded 实现。

M=8 的有效 Top-720 payload 是 `8×720×4=23,040` bytes，而 64 heads 是
184,320 bytes；但当前 SparseMLA 接口把每行 pad 到模型 `index_topk=2048`，所以
实际 candidate tensor 分别是 `8×2048×4=65,536` 与 524,288 bytes，仍为 1/8。
若和当前 TP8 fixed 1:2 的“每 rank 8 heads”比较，M=8 的 per-rank candidate tensor
并未下降；M=4 才下降一半。最终 compact `unique_indices + pair_to_slot` 仍是必要的
系统工作。

当前 accuracy runtime 使用“每层离线 static global set + learned per-query pair
assignment”。checkpoint 中的 dynamic set scorer 暂未接入 scan 前决策；原因是它
依赖 post-RoPE MLA query，而当前 Indexer scan 在该 query 完整可用前发生。这个
canary 对应离线表中的 `static set + router assignment`，必须与动态 set router
区分。下一版可把 Indexer scan 延迟到 MLA query 生成后，再接 dynamic set scorer。

验证结果：

- 六个修改模块通过远端 `py_compile`；
- SGLang 完整模块在 `qyl/sglang-hisa:eval` 中实际 import 通过；
- runtime assignment 与训练脚本 scorer 在 CPU 上逐项数值一致；
- minibatch 训练和 evaluation smoke 通过。

### 11.3 AIME 生成 3K token 后才切 experimental sparse

新增环境变量：

```text
SGLANG_NSA_EXPERIMENTAL_DECODE_START_TOKEN=3000
```

Scheduler 把每个 request 的 `len(output_ids)` 作为 `generated_lens_cpu` 传到
ForwardBatch；这个数不包含 prompt 和 prefix-cache token。因此阈值严格定义在
“已生成 token 数”，而不是总 sequence length。

AIME 的策略是：

```text
generated tokens < 3000 : 官方 shared DSA Top-K
generated tokens >= 3000: fixed per-head 或 Router + global M
```

这里的“阈值前 dense”准确说是“模型原生 official shared DSA”，不是构造完整
128K dense MHA。DeepSeek-V3.2 本身使用 DSA；若要求真正 dense MHA decode，需要
另一条全 KV dense kernel，显存和计算语义都不同。当前实现满足的是“先不启用我们
的实验 sparse policy，3K 后再切换”。同一 batch 中请求位于阈值两侧时，会按行
合并 official 与 experimental indices；常见的全 early/全 late 情况不产生每层
GPU→CPU 同步。

### 11.4 全 61 层 128K 数据与离线训练

新 pipeline：

```text
/DATA/disk0/qyl/code/dpskv32/run_all_layer_offline_router_pipeline.sh
```

目标数据量：

```text
16 real RULER 128K prompts
× 8 decode queries per prompt
× 61 model layers
× 8 TP rank shards
= 62,464 PT feature shards（另有同数 JSON 指标）
```

probe 新增 `SGLANG_NSA_HEADMAP_PROBE_SAMPLE_OFFSET`、`--start-prompt` 与
`--resume`，可以从完整 prompt 边界恢复。第一次运行已完整得到 sample 0–63：
64 queries × 61 layers × 8 ranks = 31,232 PT shards；随后从 sample 64 恢复。

全层训练配置：

```text
router rank: 32
loss: regression
M evaluated: 4, 8
prompt-grouped 2-fold CV
optimizer steps: 1200/fold + 1200 all-data final fit
minibatch: 256 contexts
device: CUDA（base model 不参与，只训练小 router）
learning rate: 0.003
temperature: 0.04
```

训练脚本增加了 CUDA minibatch 支持，同时保留 `batch_size=0, device=cpu` 的旧
full-batch 行为。CV 的 static route 也改为每层预计算，避免对每个 context 重复
facility greedy。

全 61 层训练已经完成，final checkpoint 为：

```text
/DATA/disk0/qyl/data/headwise_minmax_e2e/exp6_128k_all61_multidecode_router/
  training/offline_router_regression_r32_all61/checkpoints/router_all.pt
```

对应的两个 M=8 runtime artifacts 为：

```text
offline_router_m8_all61.json  learned per-query assignment
offline_static_m8_all61.json  no-training static assignment ablation
```

静态 artifact 中显式存储每层 64 个 pair 的 global indexer-head id，而不是依赖
candidate set 排序的隐式 slot id；config builder 会校验所有 assignment 均属于当层
M=8 set。

prompt-grouped CV 的关键结果是：

| M | Joint router | Static set + static assignment | Static set + router assignment | Static set + oracle assignment |
|---:|---:|---:|---:|---:|
| 4 | 0.7444 | 0.7450 | 0.7447 | 0.7587 |
| 8 | 0.7440 | 0.7450 | 0.7445 | 0.7605 |

因此离线训练并非没有评估：它已经做了严格的 prompt-grouped held-out
attention-mass CV；结果显示 learned assignment 没有超过 static assignment，M=8
仍与 set 内 oracle assignment 相差约 0.016 attention mass。尚缺的是本文下一节的
下游任务 accuracy，而二者不能混为一谈。

### 11.5 三数据集正式评测矩阵

统一脚本：

```text
/DATA/disk0/qyl/code/dpskv32/run_router_global_budget_accuracy.sh
/DATA/disk0/qyl/code/dpskv32/run_post_all61_accuracy.sh
```

四臂：

```text
official_dsa       官方 shared Top-2048 DSA
fixed_per_head     当前固定 1:2 的逐 Indexer-head Top-720
global_m8_static   全 61 层 global M=8 + 不训练的静态 pair assignment
router             全 61 层 Router + global unique-head M=8
```

三数据集：

```text
LongBench v2 : 完整 503 题，max_context_tokens=131072，max_new_tokens=128
RULER        : 13 tasks × {32K,128K} 的固定 test index 7，共 26 条
AIME 2025    : 30 题，max_new_tokens=65536，temperature=1，top_p=.95
```

RULER oracle 数据来自各 task 的 offset 0/1，正式任务评测使用固定 test index 7，
不存在相同 example 的直接重叠。LongBench v2 与 AIME 2025 未用于 router 训练。

AIME 三臂使用相同的 per-problem deterministic seed。三个 evaluator 的 resume
summary 改为每个 `source_id` 只取最后一条，修复过去 append 重跑导致重复计分的
问题。每个 arm 先跑三数据集 smoke，全部成功后才进入完整矩阵。

输出目录：

```text
/DATA/disk0/qyl/data/router_global_budget_accuracy_20260903/
  smoke/{router,global_m8_static,official_dsa,fixed_per_head}/
  full/{router,global_m8_static,official_dsa,fixed_per_head}/
```

每个 server phase 都保存 image inspect、git status、git diff、router config、
启动前 nvidia-smi、server log、逐题 JSONL 和去重后的 summary JSON。

LongBench 最初 smoke 沿用了 probe 收集器的 `16 tokens + ignore_eos=true`。
4 条中有 1 条的文本在输出选项字母前被硬截断，因此该设置不适合正式精度。
完整 503 题矩阵已统一为 `max_new_tokens=128`且正常尊重 EOS，在不强制生成
128 token 的前提下给答案格式足够余量。

### 11.6 当前运行状态与决策门槛

全层恢复、61 层离线训练与 checkpoint/config 生成均已完成。第一次
router smoke 确实在 LongBench v2 第一条 131072-token 样本首个 decode
step 暴露过 CUDA illegal-memory-access，但根因已找到并修复：

```text
per-head candidate 实际只有 720 个有效 indices
page table 为了适配 SparseMLA 接口 pad 到 2048，后 1328 个值是 -1
FA3 wrapper 却把每行 cache_seqlen 硬置为 2048
=> FA3 把 -1 当 KV 地址读取，导致 illegal memory access
```

修复后 FA3 和 FlashMLA-KV 两条 per-head 路径都通过
`(page_table >= 0).sum()` 计算每行真实有效长度；同时保留页表/物理页边界、
non-finite query/score 和 CUDA kernel launch 诊断检查。修复后的验证为：

- 7 层 partial-router，同一条 exact 131072-token canary：成功；
- 全 61 层 M=8 router，同一条 exact 131072-token canary：成功，
  LongBench 得分 1.0，单条耗时 77.2 s；
- 正式 router smoke 已完成 LongBench v2 4/4 条，得分 0.5；
- 正式 router smoke 已完成 RULER 2/2 条（CWE 32K/128K），两条的
  partial-credit 得分都是 0.1。

这些 smoke 数字只证明路径能在真实 128K 上稳定生成，样本太小，不能用来
判断 router 是否掉精度。完整配对矩阵已在后台运行：

```text
PID:      2356005（AIME evaluator 修正后的 resume 作业）
watcher:  2500483（主矩阵结束后自动补 global_m8_static 并汇总四臂）
run root: /DATA/disk0/qyl/data/router_global_budget_accuracy_20260903
driver:   /DATA/disk0/qyl/data/router_global_budget_accuracy_20260903/post_all61_driver.log
```

`global_m8_static` 的每层 64-pair 映射从同一批离线 utility 的均值上生成，每个
assignment 都被校验只使用该层 M=8 set 内的 head。CPU runtime unit test 已验证
TP8 下的 global-pair 到 local-pair 切片及 candidate-slot gather，且 static mode 不加载
learned checkpoint。这一臂是判断“是否真的需要训练 router”的必要对照。

AIME 第一次 smoke 在请求入队前被 SGLang HTTP 层拒绝：评测器传了不被当前
`SamplingParams` 支持的 `seed`，正确字段是 `sampling_seed`。两条失败都只耗时
0.06 s，所以可确定它们尚未进入模型 forward，也不是 router 精度或稳定性失败。
字段已修正且 pipeline 从现有 JSONL 断点重启。AIME server 已核对运行时环境：
`EXPERIMENTAL_DECODE_START_TOKEN=3000`，
router config 指向全 61 层 M=8 config。每条记录保存
`completion_tokens`，所以完整评测后会同时报告有多少 AIME 生成真正跨过
3000 token 以及 3K 后实际有多少 routed decode tokens；若某题在 3000 token
前 EOS，它全程不会启用实验 sparse policy。layer 0 还会在
`generated_lens == 3000` 时输出一次 `experimental_sparse_transition` 日志，用来核对
真实运行时切换，而不只是核对环境变量。

当前两条 AIME router smoke 已给出更直接的运行证据：第 0 题在 1,725 token
EOS，答案 70 正确，因此全程是 official DSA；第 1 题到达 3,000 token 时，
8 个 TP rank 都输出了 `generated_lens=[3000]` marker，之后已在 router sparse
路径上继续稳定生成超过 2,000 token。第 1 题仍在生成，所以尚不报两题
pass@1。后续 smoke 最多生成 8,192 token，正式 AIME 仍保留 65,536 token 上限。

在正式 summary 产生前，不报告“router downstream accuracy 已验证”。决策顺序：

1. 若全层 router 在三数据集上接近 official DSA 且优于/不劣于 fixed 1:2，继续做
   owner-sharded scan、compact indices 与 dynamic global-set router。
2. 若 static-set + offline assignment 明显掉点，而 oracle/static-set mass 仍高，
   说明瓶颈是 assignment 泛化，转 Megatron sampled online distillation。
3. 若 global M=8 的 oracle 本身在新增 61 层显著不足，先调整 M、pair 多选或分层
   budget，不急于在线训练 assignment。
