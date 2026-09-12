# Group16-to-Indexer Router Ablation

## 1. 目标

当前系统先使用 Indexer 自身的 coarse importance 从 64 个 Indexer heads
中选择 M=8，再为这 8 个 heads 生成 Top-720 candidates。固定的每组 16 个
MLA heads 只能在这 8 套 candidates 中选择一套。

现有 RULER all-64 probe 表明，主要损失来自 M=8 head selection：

- 实际 M8 Group16 oracle：0.64501；
- 从 64 个 candidates 中为每个 Group16 选择最佳 Indexer：0.69663；
- head-selection regret：0.05162；
- M8 内逐 MLA head 选择与 Group16 共享一套的差：0.00889。

因此本实验不再训练一个只能在既定 M8 内选择的 assignment router，而是
直接预测：

> 对于当前 layer 的固定 Group16，64 个 Indexer 中哪个最可能生成最有用的
> Top-720 indices？

8 个 Group16 各选择一个 Indexer，选择结果去重后最多包含 8 个 Indexer。
只有这些 Indexer 执行后续细粒度检索。

## 2. 固定定义

### 2.1 MLA 分组

保持 attention TP=8 的现有物理布局：

```text
group 0: MLA heads   0--15
group 1: MLA heads  16--31
...
group 7: MLA heads 112--127
```

不迁移 MLA heads，不学习重新分组。`group_id` 就是 TP-local group/rank ID。

### 2.2 候选和监督

令 \(C_i\) 为 Indexer head \(i\) 产生的 Top-720 token indices。Group16 对
该候选的真实 utility 为：

\[
U_{g,i}
=
\frac{1}{16}
\sum_{h\in g}
\sum_{t\in C_i}
\alpha_{h,t}.
\]

离线 probe 才计算 \(C_i\) 和真实 attention mass。训练目标是让
\(R_{g,i}\) 的排序逼近 \(U_{g,i}\)；运行时不可能提前看到真实 utility。

### 2.3 Layer 信息如何进入

所有方案都使用 layer-specific projection：

\[
z_g=P_l(\operatorname{LN}(Q_g^{MLA})),\qquad
c_i=A_l(\operatorname{LN}(Q_i^{Indexer})).
\]

因此 `layer` 不是与 query 拼接的单独向量。当前 layer ID 用于选择
\(P_l\) 和 \(A_l\) 的参数。即使没有显式 layer embedding，模型也已经是
layer-conditioned。

### 2.4 所有方案共有的 Indexer 信息

所有方案都使用：

- \(Q_i^{Indexer}\)：量化前 BF16 Indexer query，不是把 FP8 code 强转
  BF16；
- \(gate_i\)：原始 DSA 的 `weights_proj(x)` 动态标量，不包含
  \(H^{-1/2}\)、FP8 query scale 或 softmax scale；
- \(H_{l,i}\)：每层、每个 Indexer ID 的可学习 embedding；
- \(Q_g^{MLA}\)：组内 16 个 MLA queries 按原顺序拼接。

候选表示的共同部分为：

\[
\tilde c_i
=
A_l(\operatorname{LN}(Q_i^{Indexer}))
+H_{l,i}
+\operatorname{Std}(gate_i)v_l^{gate}.
\]

`Std` 表示在当前 64 个候选之间标准化。将 gate 乘到一个可学习隐向量后，
它可以通过与 group 表示的内积产生 group-dependent 影响，而不只是给所有
groups 增加相同偏置。

基础匹配分数为：

\[
R_{g,i}=\frac{z_g^\top\tilde c_i}{\sqrt d},
\]

其中当前实验 \(d=64\)。

候选生成仍使用 FP8 query 和包含 query scale 的 effective gate，以保持
线上 `512/60%/chunk16-quota` 检索语义。只有 router 输入的 query、gate
以及 `256/Top-8` coarse context 使用未量化表示。用于同批 MISA baseline
的 `candidate_importance` 仍保存实际 effective importance。

## 3. 四个严格对照

### 3.1 A：Query + Gate

\[
z_g=P_l(\operatorname{LN}(Q_g^{MLA})),
\]

\[
R_{g,i}
=
f(Q_g^{MLA},Q_i^{Indexer},gate_i,layer).
\]

它没有显式 `group_id`，也没有历史 key summary。不同 groups 只能通过当前
16 个 MLA queries 的内容区分。

这个版本回答：

> 当前 query 和 DSA gate 是否已经足以在细粒度检索前选择 Indexer？

如果它表现良好，运行时可以在任何 coarse history scan 之前执行，速度潜力
最大。

### 3.2 B：Query + Gate + 跨层共享 Group ID

\[
z_g
=
P_l(\operatorname{LN}(Q_g^{MLA}))
+E_g.
\]

\(E_g\) 是 8 个固定 groups 各自的 embedding，在所有 layers 之间共享。
与方案 A 相比只增加 Group ID 信息，尽量不引入 layer×group 的额外容量。

这个版本回答：

> 在 query 之外，固定 MLA group 是否存在跨层稳定的 Indexer 偏好？

方案 B 相对方案 A 的增益可以较干净地归因于显式 Group ID。

### 3.3 C：Query + Gate + Layer-specific Group ID

\[
z_g
=
P_l(\operatorname{LN}(Q_g^{MLA}))
+E_{l,g}.
\]

\(E_{l,g}\) 为每个 layer 和 group 单独学习先验。它可以表达类似：

```text
layer 30 / group 5 通常偏好 Indexer 12
```

这个版本比方案 B 参数更多。它不是严格的纯 Group ID 消融，而是用于判断
是否存在明显的 layer×group 静态偏好：

- B > A：跨层稳定的 Group ID 有用；
- C > B：还需要 layer-specific group prior；
- C 与 B 接近：没有必要部署 \(E_{l,g}\)。

### 3.4 D：完整 Coarse-context

首先对所有 64 个 Indexer heads 和 256-token coarse key means 计算：

\[
a_{i,b}
=
Q_i^{Indexer}\cdot\operatorname{Mean}(K_b).
\]

对每个 Indexer \(i\) 取分数最高的 8 个 coarse blocks：

\[
Top8Sum_i
=
\sum_{b\in Top8(i)}
\operatorname{ReLU}(a_{i,b}).
\]

`Top8Sum_i` 是一个标量，描述该 Indexer 在当前前缀中最强区域的总体匹配
强度。它与 `gate_i` 分开输入，不提前合并成 `importance`。

Top-8 coarse key vector summary 为：

\[
CoarseSummary_i
=
\sum_{b\in Top8(i)}
\operatorname{softmax}(a_{i,b}/T)
\operatorname{Mean}(K_b).
\]

它是一个 128 维向量，描述 Indexer \(i\) 当前匹配到的历史区域。

完整 candidate 表示为：

\[
\hat c_i
=
\tilde c_i
+\operatorname{Std}(Top8Sum_i)v_l^{top8}
+C_l(\operatorname{LN}(CoarseSummary_i)).
\]

group 表示沿用严格方案 B 的跨层共享 \(E_g\)：

\[
R_{g,i}
=
\frac{
  (P_l(\operatorname{LN}(Q_g^{MLA}))+E_g)^\top
  \hat c_i
}{\sqrt d}.
\]

这个版本回答：

> 在 query 和静态 group prior 之外，当前历史前缀的 coarse key 信息是否
> 能显著改善 Indexer 选择？

如果 D 与 B 接近，应删除 `CoarseSummary`，避免额外的 all-64
gather/reduction。如果 D 明显更好，说明 query-only router 看不到足够的
当前前缀信息。

## 4. 两条 coarse 路径不要混淆

本实验刻意分开“router 特征”和“监督候选生成”。

Router coarse 特征：

```text
全部 256-token coarse blocks
→ 每个 Indexer 取 Top-8
→ Top8Sum + Top-8 CoarseSummary
```

监督候选 \(C_i\)：

```text
全部 512-token coarse blocks
→ 保留最高 60% 区域
→ chunk16 细粒度检索
→ 生成 Top-720
→ 计算 Group16 utility
```

60% 区域不是在 Top-8 区域内选择，也不作为方案 D 的输入。使用旧
`512/60%` 候选是为了保持候选质量，并与已有 all-64 RULER oracle 可比。

此前已经验证“仅在 8×256 token 区域中生成 Top-720”会导致 Group16 oracle
显著下降，因此本实验只把 Top-8 coarse 信息作为选择 Indexer 的特征，不把
它作为最终检索区域的硬限制。

## 5. 训练数据和测试约束

### 5.1 训练

- 复用原来的 trajectories 和 train/validation split；
- 主 attention 的 prefill、decode 都保持原始 DSA；
- probe 只捕获带稳定 `request_id` 的指定 decode query；
- 每个请求记录实际 `seq_len`、query token、prefix hash、layer 和 TP rank；
- 固定采样点之外，短 prompt 会增加“首次跨过 4096”的 decode position；
- 采集前打印每个 `dataset × split × length` 的 trajectory/sample 覆盖；
- 每个训练 context 必须包含 64 个 Indexer IDs 各一次、61 层和 8 个 TP
  ranks，缺失或重复直接报错；
- `source_id` 和完整 prompt 内容指纹不得跨 train/validation/test；
- 每层每卡达到预期请求数后 flush 并写 completion marker，采集入口在停止
  server 后检查 61×8 个 marker；
- 每个 context 的监督为完整 `[8,64]` utility；
- 所有方案随机初始化，不复用旧 pair 或 Group16 checkpoint。

回放请求生成两个 token：第一个 token 是指定 decode query，第二个 token
保证正常调度下实际执行一次 decode。probe 只在 request ID 中编码的目标
`seq_len` 捕获一次，重试不会产生重复记录。canonical pipeline 明确关闭
experimental sparse prefill，因此部分 chunk 的因果性问题不进入本实验。

### 5.2 Held-out 测试

使用固定 RULER test trajectories 的 all-64 probe。对每个 Group16：

```text
64-way router score
→ argmax Indexer ID
→ 读取该候选的真实 teacher utility
```

8 个 group choices 去重后的数量也会记录。该数量决定理论上需要执行多少次
细粒度检索，范围为 1--8。

## 6. 主要评价指标

每个方案报告：

- `learned_utility`：router 选中 Indexer 的真实 Group16 attention mass；
- `oracle_utility`：同一候选集合中的 Group16 oracle；
- `regret = oracle - learned`；
- `top1_accuracy`：8 个 groups 中选中 oracle candidate 的比例；
- `unique_selected_indexers`：每个 context 的 8 个 choices 去重后数量；
- `static_utility`：只用训练 split 得到的每层每组静态最佳 Indexer；
- `misa_top1_utility`：实际 MISA importance 第一名对应候选的 Group16
  utility；
- `misa_m8_group_oracle`：实际 MISA Top-8 内的 Group16 oracle；
- `group16_constraint_loss`、`candidate_generation_loss` 和 router regret
  三段 oracle 分解；
- 32K 和 128K 分项结果；
- all-64 validation 与 held-out all-64 test 分开报告。

必须与以下无训练参考对齐：

- 实际 MISA M8 Group16 oracle：0.64501；
- all-64 Group16 oracle：0.69663；
- all-64 oracle 每个 context 平均只使用 2.31 个不同 Indexer。

## 7. Legacy 对照结果

以下结果来自旧的 M8 partial-label 训练，只用于证明该监督不能可靠外推到
64-way；它不是新 all-64 训练的最终结果。四个方案均使用相同配置随机初始化
训练 10,000 步。Held-out RULER all-64
测试包含 5,063 个 context（83/83 个请求），结果如下：

- A `query_gate`：utility 0.63082，regret 0.06515，top-1 13.24%，
  平均 2.92 个不同 Indexer；
- B `query_gate_group`：utility 0.62989，regret 0.06608，
  top-1 12.53%，平均 2.91 个不同 Indexer；
- C `query_gate_layer_group`：utility 0.62447，regret 0.07150，
  top-1 12.46%，平均 2.94 个不同 Indexer；
- D `coarse_context`：utility 0.62968，regret 0.06629，
  top-1 12.05%，平均 2.96 个不同 Indexer。

同一测试集上的 all-64 Group16 oracle 为 0.69597。最佳方案是最简单的 A，
但仍比实际 MISA M8 Group16 oracle 0.64501 低 0.01419，比 all-64 oracle
低 0.06515。32K/128K 的最佳 A utility 分别为 0.66402/0.59841，对应
regret 0.05951/0.07065。

M8 partial validation 上，A/B/C/D utility 分别为
0.70589/0.70599/0.70542/0.70358，oracle 为 0.72383。四种结构在训练分布
内接近，但从 M8 partial labels 外推到 all-64 candidates 后均明显下降。

这组 legacy 结果仅支持：

- 显式 Group ID 没有改善 A，layer-specific Group ID 反而下降；
- `Top8Sum + CoarseSummary` 没有改善 query-only 版本，不值得为它增加
  all-64 coarse context 开销；
- 约 2.9 个去重 Indexer 说明 router 确实会复用 Indexer，但当前选择质量
  不足，暂不进入 runtime/kernel 速度验证；
- 必须使用完整 all-64 labels 重新训练；在该对照完成前，不能把失败归因于
  Group16 设计本身。

本实验只验证 Indexer head selection，不证明共享 Top-720 的最终输出误差，
也不修改 CUDA attention kernel。

## 8. 精简后的采集与训练入口

主流程只需要检查三个文件：

- `scripts/run_group16_all64_router.sh`：固定 all-64 采集与训练入口；
- `python/sglang/srt/layers/attention/nsa/headmap_probe.py`：带 request ID 的
  teacher 记录与落盘；
- `scripts/train_group16_indexer_router_ablation.py`：自包含的 Group16
  数据加载、64-way loss、四模型训练和评测。

训练集采集：

```bash
STAGE=collect \
EXP_NAME=group16_indexer_router_train_all64 \
TRAJECTORIES=/DATA/disk0/qyl/data/misa_assignment_router_pilot_v2/trajectories.jsonl \
CONTAINER_TRAJECTORIES=/workspace/qyl/data/misa_assignment_router_pilot_v2/trajectories.jsonl \
bash scripts/run_group16_all64_router.sh
```

测试集采集：

```bash
STAGE=collect \
EXP_NAME=group16_indexer_router_test_all64 \
TRAJECTORIES=/DATA/disk0/qyl/data/ruler_router_decomposition_20260909/trajectories.jsonl \
CONTAINER_TRAJECTORIES=/workspace/qyl/data/ruler_router_decomposition_20260909/trajectories.jsonl \
bash scripts/run_group16_all64_router.sh
```

训练四个方案：

```bash
STAGE=train \
TRAIN_EXP=group16_indexer_router_train_all64 \
TEST_EXP=group16_indexer_router_test_all64 \
EXP_NAME=group16_indexer_router_all64 \
bash scripts/run_group16_all64_router.sh
```

训练保持相同的 10,000-step 预算，每 500 步计算一次 validation。每种方案
分别输出：

```text
<out>/<mode>/best.pt  # validation learned utility 最高
<out>/<mode>/last.pt  # 固定预算终点，含 optimizer state 和 step
```

held-out test 只评估 `best.pt`，不参与 checkpoint 选择。

旧 M8/pair-router 脚本位于 `scripts/legacy/`，不再被主流程导入。这里的
“Top-720”始终指 `512 / keep60% / chunk16-quota` 生成的最多 720 个有效
token，不是全历史精确 Top-720。

## 9. Static Group16 decode 路径

如果大样本验证表明每层、每个 TP-local Group16 的最佳 Indexer 足够稳定，
可以完全移除 decode 时的 Assignment Router。离线拟合：

\[
i^{static}_{l,g}
=\arg\max_i\mathbb{E}_{x\in D_{train}}[U_{l,g,i}(x)]
\]

得到一张 `[61, 8]` 固定映射。线上每个 TP rank 只读取本层对应的一个
Indexer ID：

```text
static_assignment_by_layer[layer][tp_rank]
→ 只为该 Indexer 计算 512-token coarse scores
→ 保留 60% coarse blocks
→ chunk16 quota 检索，最多 720 tokens
→ 同一卡 16 个 MLA heads（8 个 pair slots）复用该 indices
```

线上 `budget=1` 表示**每张卡只扫描一个 Indexer**，不是全局只有一个
Indexer。八张卡的固定选择可以重复，因此每层全局不同 Indexer 数量最多为
8。该路径不加载 router checkpoint，也不运行 MISA 的 64-head head
selection。

离线 Static 评估及 runtime config 导出：

```bash
python scripts/evaluate_group16_static.py \
  --calibration-probe-dir <train>/probe \
  --calibration-manifest <train>/samples.jsonl \
  --test-probe-dir <test>/probe \
  --test-manifest <test>/samples.jsonl \
  --out <out>/static_report.json \
  --runtime-weighting context \
  --runtime-config-out <out>/static_group16_runtime.json
```

启动服务时设置：

```bash
SGLANG_NSA_OFFLINE_ROUTER_CONFIG=<out>/static_group16_runtime.json
```

正式部署前必须使用完整 calibration 数据重新导出；小批 smoke 数据生成的
映射只能验证代码路径，不能作为最终固定表。
