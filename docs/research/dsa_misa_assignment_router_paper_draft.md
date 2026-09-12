# 从共享 DSA Top-K 到 MISA 全局 Head Budget 与 MLA Assignment Router

> DeepSeek-V3.2 / SGLang-HISA 研究与实现记录（paper draft）  
> 更新时间：2026-09-07  
> 实验节点：`h20-9-57`，8 × NVIDIA H20 144 GB，TP8  
> 代码树：`/DATA/disk0/qyl/code/dpskv32/sglang-hisa`  
> 基线 commit：`e399d9f33fecfe50b90fa65e3e0da01ea1572f8f`；本文涉及的 MISA、per-head 与 router 文件仍在未提交工作树中。

## 摘要

DeepSeek-V3.2 的 DSA Indexer 使用 64 个 Indexer heads 共同产生一份共享 Top-2048 token 列表。共享列表对 128 个 MLA heads 的真实 attention mass 很高，但它的主要代价来自 64 个 Indexer heads 对长历史的扫描。我们的起点是一个看似直接的想法：让不同 MLA heads 使用不同 Indexer head 的短候选列表，例如 Top-720，并且只计算真正被消费的少数 Indexer heads。

实验把这个想法修正为一个更准确的结论：**逐 MLA head 的精确稀疏预算确实远小于共享 2048，但现有 Indexer heads 不是与 MLA heads 固定一一或 1:2 对齐的独立检索器。** 固定 1:2 映射几乎等于随机映射；最佳映射高度多对一、随层和 query 变化。因此，per-head 稀疏不能直接从普通 MHA/GQA 迁移到一个先跨 Indexer heads 聚合的 DSA Indexer。

本文实现并记录了一个四阶段方案：第一阶段用无训练的 MISA pooled-key router 为每个 query 选择 `M` 个 Indexer heads；第二阶段仅为这 `M` 个 heads 生成各自的 Top-720；第三阶段用离线训练的低秩 assignment scorer，在已选集合内为每个 MLA two-head pair 选择一个 candidate slot；第四阶段 gather 得到 `[B, local_pairs, topk]` 的稀疏 attention 索引。128K oracle 显示，全局动态集合 `M=4/8` 已能保留 unrestricted pair oracle 的 99.93%/99.99%；但当前 q-only assignment scorer 仍落后于集合内 oracle 约 0.016 attention mass。当前 MISA runtime 已接通并通过定向单元测试，但尚无与旧静态集合实验等价的完整端到端精度结果，也尚未实现 owner-sharded scan。

## 1. 研究问题与核心判断

### 1.1 原始 DSA 做了什么

对一个 query，原始 DSA 先计算 64 个 Indexer head 对历史 token 的分数，再用 head gate 加权并跨 head 聚合，最后从聚合分数中取一份共享 Top-2048：

```text
64 Indexer heads × all historical keys
        -> weighted sum over heads
        -> one shared Top-2048 list
        -> all MLA heads consume the same list
```

这种 shared selection 的优点是稳定：即使不同 MLA heads 关注不同 token，跨 Indexer heads 聚合后的候选集合仍能覆盖多数重要位置。缺点是长上下文下的主要开销仍是 `64 × history_length` 个 Indexer QK 分数。

### 1.2 原始假设与实验后的修正

原始假设是：

```text
每个 MLA head 单独找短 Top-K
应当优于所有 MLA heads 共享一份 Top-2048。
```

这个判断只对“真实 MLA QK 的 exact Top-K”成立。128K 数据上：

- official shared DSA Top-2048 的平均真实 MLA attention mass 为 `0.7672`；
- exact MLA Top-512 为 `0.7557`；
- exact MLA Top-720 为 `0.7769`。

但是，“某个 Indexer head 的 Top-720”不等于“某个 MLA head 的 exact Top-720”。现有 Indexer head 与 MLA head 的固定 1:2 映射只有 `0.4168` mass；即使每个 MLA head 从全部 64 个 Indexer heads 中独立选最优，4K 实验中也只有 `0.7020` mass，仍低于 shared DSA 的 `0.9555`。

因此本文主张的不是“把 shared Top-2048 机械地换成 per-Indexer-head Top-720”，而是：

1. exact MLA 结果证明短的 per-consumer budget 有潜力；
2. Indexer-to-MLA 的对应关系必须动态路由，不能使用固定 1:2；
3. 真正决定系统 scan 数的是所有 MLA consumers 使用的 Indexer heads 的并集；
4. shared DSA 的高质量来自跨 head 互补，减少 heads 后必须同时验证 head selection 和 candidate quality。

### 1.3 系统目标必须是全局 unique-head budget

令第 `p` 个 MLA pair 使用的 Indexer head 集合为 `S_p`。只约束每个 pair 选一个 head，并不保证计算量小：64 个 pair 仍可能合计使用全部 64 个 heads。

```text
per-pair sparsity != system scan sparsity
```

系统真正需要约束的是：

```text
| union_p S_p | <= M
```

定义 `U[p,i]` 为 MLA pair `p` 使用 Indexer head `i` 的候选列表时保留的真实 attention mass，则全局集合的 oracle 目标为：

```text
F(S) = mean_p max_{i in S} U[p,i]
subject to |S| <= M
```

选好 `S` 后，pair assignment 为：

```text
a_p = argmax_{i in S} U[p,i]
```

这正是当前实现中“先选 M，再给每个 MLA pair 分配 candidate slot”的数学动机。

## 2. 符号与张量形状

| 符号 | 含义 | 全局/TP8 典型大小 |
|---|---|---:|
| `b` | 一个 query row；decode 时通常对应 batch 中一个请求 | `B` |
| `l` | Transformer layer | 61 层，编号 0–60 |
| `h` | MLA head | 128；TP8 每 rank 16 |
| `p` | 相邻两个 MLA heads 组成的 pair | 64；TP8 每 rank 8 |
| `i` | Indexer head id | 64 |
| `m` | 被 MISA 选中的 candidate slot | `0..M-1` |
| `r` | assignment scorer 的低秩 latent 维度 | 32 |
| `k` | 每个 candidate list 中的 token slot | 有效 720；接口通常 pad 到 2048 |

因此：

```text
assignment_score[b,p,i]
```

中的 `b` 枚举 query rows，`p` 枚举 MLA pairs，`i` 枚举 64 个 Indexer heads。对所有组合做一次 32 维点积，自然得到：

```text
[B, 64 pairs, 64 Indexer heads]
```

TP8 runtime 不在单个 rank 上构造全局 64 pairs，而是：

```text
[B, local_pairs=8, 64]
```

## 3. 当前四阶段方法

### 3.1 总流程

| 阶段 | 是否训练 | 输入 | 输出 |
|---|---|---|---|
| 1. MISA 选 M | 否 | Indexer Q、缩放后的 64-way gate/weight、历史 Indexer K 的 pooled means | `[B,M]` Indexer head ids |
| 2. 产生 candidates | 否 | 被选中的 M 个 Indexer queries、paged Indexer K、chunk sums | `[B,M,topk]` token indices |
| 3. assignment | 是 | post-RoPE absorbed MLA pair Q、layer、learned priors、标准化 gate；以及 `[B,M]` 用于限制候选 head | `[B,local_pairs,M]` scores 与 `[B,local_pairs]` slot |
| 4. gather | 否 | candidates 与 slot | `[B,local_pairs,topk]` |

完整数据流为：

```text
Indexer Q + gate + pooled historical K
        |
        v
MISA Top-M heads ------------------------------ [B,M]
        |
        v
M independent chunk16 Top-720 scans ----------- [B,M,topk]
                                                    |
MLA pair Q + layer + priors + gate                 |
        |                                           |
        v                                           |
score all 64 heads ----------------------------- [B,P_local,64]
        |
gather M selected head scores ------------------ [B,P_local,M]
        |
argmax candidate slot -------------------------- [B,P_local]
        |                                           |
        +---------------- gather candidates <-------+
                            |
                            v
                      [B,P_local,topk]
```

### 3.2 当前实现的精确边界

- 当前 MISA/per-head runtime 只支持 decode/idle；非 decode 路径会抛 `NotImplementedError`。
- `M` 是每个 query 动态选择的，不是每层固定 head ids。
- MISA 不使用 MLA Q/K，也不使用 assignment checkpoint。
- assignment scorer 使用 MLA Q，但不使用 MLA K。
- 当前 checkpoint 的 `dynamic_indexer_keys=false`，所以 assignment scorer 也不使用 Indexer Q；训练函数保留了这个可选分支，但 runtime 明确拒绝该类 checkpoint。
- assignment 函数接收 candidate indices 作为最后 gather 的 payload，但 scoring 过程不读取这些 indices 的值。
- 历史正式 checkpoint 仍包含旧 `set_scorer` 权重，但 runtime 只加载 `assignment_scorer.*`；当前训练代码已删除 `set_scorer`，选 M 完全由 MISA 负责。

## 4. 第一阶段：MISA 如何选择 M 个 Indexer heads

### 4.1 输入和输出

输入：

```text
q_indexer:  [B,64,128]
weights:    [B,64]       # DSA head gate 经 n_heads^-0.5、q_scale、softmax_scale 缩放
chunk_sum:  [physical_chunks,128]
block_table / visible_length
```

历史 K cache 已经维护精确的 16-token chunk sums。MISA 先通过 block table 恢复逻辑 chunk 顺序，再把这些 sums 合并成默认 1024-token pooling blocks，并除以有效 token 数得到 `mean_k`。

输出：

```text
selected_heads: [B,M]
```

### 4.2 当前代码使用的打分

对 query row `b`、Indexer head `i`、1024-token pooling block `j`：

```text
affinity[b,i,j] = q_indexer[b,i]^T mean_k[b,j]

importance[b,i]
  = sum_j ReLU(affinity[b,i,j]) * abs(weights[b,i])

selected_heads[b]
  = TopM_i importance[b,i]
```

代码省略了对 pooling block 数量取平均的公共因子，因为它不改变同一 query 内 64 个 heads 的排序。可见尾部之外的 stale physical chunks 会在 pooling 前被 mask。

核心实现：

```python
# python/sglang/srt/layers/attention/nsa/per_head_paged.py:265-274
pooled_mean = pooled_sum / pool_counts.clamp_min(1).unsqueeze(-1)
affinities = torch.bmm(
    q_bf[start:end], pooled_mean.to(torch.bfloat16).transpose(1, 2)
).float()
importance = (
    torch.relu(affinities) * weights_f32[start:end].unsqueeze(-1)
).sum(dim=-1)
selected[start:end] = torch.topk(
    importance, k=topk_heads, dim=-1, largest=True, sorted=False
).indices
```

### 4.3 这里的 gate 到底是什么

这里的 `weights` 不是另一个训练出的 router gate。它来自原始 DSA Indexer 的 `weights_proj(x)`，随后乘：

```text
n_heads^-0.5 × q_quant_scale × softmax_scale
```

MISA 用其绝对值调节每个 Indexer head 的 pooled-key relevance。assignment scorer 则把同一 64-way weight 在 head 维做 z-normalization，再乘一个可学习的 per-head `gate_scale[i]`。

### 4.4 `selected_heads_by_layer` 在 MISA 模式下不是静态集合

当前配置文件仍要求每层存在 `selected_heads_by_layer`。但在 MISA 模式中，这个字段只作为“哪些 layer 启用 router”的 coverage marker，里面的具体 ids 不参与选择；真正的 `[B,M]` 由 MISA 每个 query 重新产生。

需要特别注意：2026-09-06 之后，旧 learned-assignment config 即使没有写 `head_selection_mode`，也会默认解释成 `misa`；static assignment config 则默认 `static`。为了可复现，后续配置应显式写出：

```json
{
  "head_selection_mode": "misa",
  "assignment_mode": "learned",
  "misa_chunk_size": 1024
}
```

## 5. 第二阶段：M 个 heads 如何产生 Top-720 candidates

每个被选中的 Indexer head 独立执行 chunk16 quota selector：

1. 用 `q_i · chunk_sum` 计算每个 16-token chunk 的 coarse score；
2. 选 Top-128 chunks，sink 和 tail chunk 强制保留；
3. 对选中 chunks 做 exact FP8 QK；
4. 最好的 52 个 chunks 各保留 8 tokens；其余 76 个各保留 4 tokens；
5. 得到 `52×8 + 76×4 = 720` 个有效 token indices；短于等于 720 的可见历史全部保留。

输出为：

```text
candidates: [B,M,index_topk]
```

其中有效 payload 最多 720 个，当前 SparseMLA 接口通常令 `index_topk=2048`，剩余位置用 `-1` padding。

核心常量位于：

```python
# python/sglang/srt/layers/attention/nsa/per_head_paged.py:44-49
SEL = 128
NDENSE = 52
QDENSE = 8
QSPARSE = 4
OUTK = NDENSE * QDENSE + (SEL - NDENSE) * QSPARSE  # 720
```

动态 head ids 既支持 `[M]`，也支持逐 query 的 `[B,M]`，后者正是当前 MISA 路径。

## 6. 第三阶段：assignment scorer

### 6.1 它不是一个多层 MLP

当前 scorer 是“一个线性投影 + 低秩双线性点积 + 三类 bias/prior”，没有隐藏 MLP，也没有 activation stack。rank 为 32。

对 query `b`、MLA pair `p`：

```text
q_pair[b,p] = concat(q_mla[b,2p], q_mla[b,2p+1])   # 2×576 = 1152

z[b,p] = W_q LN(q_pair[b,p]) + E_layer[l]          # [32]
```

当前部署 checkpoint 不使用动态 Indexer query projection，所以：

```text
d[b,p,i] = z[b,p]^T E_indexer[i]
```

最终打分为：

```text
s[b,p,i]
  = d[b,p,i]
  + B_layer[l,i]
  + B_pair[l,p,i]
  + alpha[i] * normalized_gate[b,i]
```

输出：

```text
score: [B,local_pairs,64]
```

如果未来启用 `dynamic_indexer_keys`，一般形式可额外加入：

```text
E_indexer[i] + W_k LN(q_indexer[b,i])
```

但**当前正式 checkpoint 的该开关为 false，当前 runtime 也不支持 true**。因此本文所有已部署结果都不应声称 assignment scorer 输入了 Indexer Q。

### 6.2 `q_proj`、layer embedding、pair prior 分别是什么

`q_proj`：

```text
Linear(1152 -> 32, bias=False)
```

它把两个相邻 MLA heads 的 post-RoPE absorbed query 拼接向量压到 32 维。它不是 MLA 模型原本的 Q projection，而是 router 自己的小投影矩阵。

`layer_embedding`：

```text
E_layer: [61,32]
```

每层一个可训练 32 维向量，加到 pair latent 上，让同一个 MLA query pattern 在不同层有不同表示。

`pair_prior`：

```text
B_pair: [61,64 pairs,64 Indexer heads]
```

`B_pair[l,p,i]` 表示在不看当前 query 内容时，第 `l` 层的 MLA pair `p` 对 Indexer head `i` 的静态偏好。实验观察到强烈的 layer/pair/head specialization，因此这个张量是模型中最大的参数块。

`prior`：

```text
B_layer: [61,64]
```

它表示一层内某个 Indexer head 对所有 MLA pairs 的共同偏好。

### 6.3 模型大小

全 61 层、rank 32 的单个 assignment scorer 参数量：

| 参数 | 形状 | 参数量 |
|---|---:|---:|
| `q_proj.weight` | `[32,1152]` | 36,864 |
| `indexer_embedding` | `[64,32]` | 2,048 |
| `layer_embedding.weight` | `[61,32]` | 1,952 |
| `prior` | `[61,64]` | 3,904 |
| `pair_prior` | `[61,64,64]` | 249,856 |
| `gate_scale` | `[64]` | 64 |
| 合计 |  | **294,688** |

FP32 约 1.12 MiB。历史正式 checkpoint 同时含 `set_scorer` 和 `assignment_scorer` 两套独立参数，合计 589,376；runtime 只加载后者。当前训练代码已删除 `set_scorer`，新 checkpoint 只保存 Assignment Router。

### 6.4 为什么先算 64，再 gather 成 M

runtime 先给 64 个 Indexer head 全部打分：

```python
# offline_unique_head_router.py:368-392
pair_latent = F.linear(q_pair, state["q_proj.weight"])
pair_latent = pair_latent + state["layer_embedding.weight"][layer_id]
dynamic = torch.einsum("npr,ir->npi", pair_latent, state["indexer_embedding"])
score = (
    dynamic
    + state["prior"][layer_id].view(1, 1, 64)
    + state["pair_prior"][layer_id, pair_ids].unsqueeze(0)
    + gate.unsqueeze(1) * state["gate_scale"].view(1, 1, 64)
)
```

再用 MISA 给出的真实 head ids gather：

```python
# offline_unique_head_router.py:393-416
selected_score = score.gather(
    -1, selected_heads.unsqueeze(1).expand(-1, local_pairs, -1)
)
choice = selected_score.argmax(-1)
gather = choice.unsqueeze(-1).expand(-1, -1, candidates.shape[-1])
expanded = candidates.unsqueeze(1).expand(-1, local_pairs, -1, -1)
output = expanded.gather(2, gather.unsqueeze(2)).squeeze(2)
```

于是形状变化为：

```text
[B,local_pairs,64]
  --gather selected head ids-->
[B,local_pairs,M]
  --argmax-->
[B,local_pairs]
```

### 6.5 candidate indices 为什么不作为 scorer 输入

训练 teacher 的确需要 candidate indices；部署 scorer 当前不需要。这是 teacher–student distillation 的基本分工：

- teacher 用候选 indices 和真实 MLA QK 计算“这个候选列表到底保留了多少 attention mass”；
- student 只从推理时便宜可得的 MLA query、layer prior 和 gate 预测这个 utility；
- utility table 是监督标签，不是模型输入，类似训练分类器时 label 不会在 inference 输入。

但是这也是当前 scorer 的主要限制：两个候选列表即使内容完全不同，只要来自同一个 Indexer head `i`，scorer 对它们的区别只能通过 query/gate 间接推测。它不知道当前 Top-720 的 token ids、候选 key summary 或候选上的 MLA QK statistic。held-out 结果显示这个假设仍不够强。

## 7. 第四阶段：从 M 份 candidates gather 到 MLA pairs

`choice[b,p]` 是 `0..M-1` 的 candidate slot，不是全局 Indexer head id。最终：

```text
output[b,p,:] = candidates[b, choice[b,p], :]
```

输出契约：

```text
[B,local_pairs,topk]
```

在 TP8 下每 rank 为 `[B,8,topk]`，8 个 ranks 合起来覆盖全局 64 MLA pairs。

这里没有重新做 Top-K，也没有合并多个 candidate list；每个 pair 只消费一个被选中 Indexer head 的完整列表。

## 8. Assignment Router 训练方案

当前推理流程已经变成：

```text
64 Indexer heads
        ↓
MISA 每个 query 动态选择 M=8 heads
        ↓
8 个 heads 各自产生一套 Top-720 candidates
        ↓
Assignment Router
        ↓
每个 MLA pair 从 8 套 candidates 中选择一套
```

MISA 第一阶段已经是 query-dependent 的动态选择。旧 assignment scorer 虽然保存了 `q_indexer`，但正式 checkpoint 并未使用它，而是主要依赖 MLA pair query、静态 Indexer-head embedding、layer/pair prior 和 DSA gate。

因此下一版训练的核心目标是：

> **训练时直接面对运行时真实的 MISA Top-8 candidate set，并让 Router 利用当前 MLA query、当前 Indexer query 和当前 KV context 的 candidate 信息，预测每个 MLA pair 最适合哪一个 candidate。**

旧实现中的 `set_scorer/set_score[p,i]` 已删除：候选集合完全由 MISA 产生，训练只更新 Assignment Router。

### 8.1 Teacher 数据生成

对于每个真实 decode query、每一层：

1. 使用与推理完全一致的 MISA，动态得到

   $$
   S=\{i_1,\dots,i_8\}.
   $$

2. 只让这 8 个 Indexer heads 执行真实 retrieval，分别产生

   $$
   C_i=\operatorname{Top\text{-}720}_i.
   $$

3. 计算完整 MLA attention distribution \(P^{MLA}_{h,t}\)。

4. 对每个 MLA pair \(p=(2p,2p+1)\) 和 candidate \(i\)，定义真实 utility：

   $$
   \boxed{
   U_{p,i}
   =
   \frac12
   \sum_{h\in\{2p,2p+1\}}
   \sum_{t\in C_i}
   P^{MLA}_{h,t}
   }
   $$

因此每个 query-layer context 得到：

$$
\boxed{
U\in\mathbb{R}^{64\times8}
}
$$

它表示 64 个 MLA pairs 分别使用 8 个 MISA candidates 时能够保留的真实 attention mass。Teacher 使用 retained MLA attention mass，而不是把某个 head ID 当作唯一标签。

### 8.2 Router 输入

对于 MLA pair \(p\) 和候选 Indexer head \(i\)，使用当前 MLA pair query：

$$
q_p\in\mathbb R^{1152},
$$

当前 query 下候选 Indexer head \(i\) 的动态 query：

$$
q_i^{Indexer}\in\mathbb R^{128},
$$

以及少量候选辅助信息：

- MISA importance \(m_i\)；
- 原始 DSA gate \(g_i\)；
- Indexer head ID；
- layer 信息；
- 可选：candidate 的 chunk/Top-K score statistics 或 coarse KV summary。

最重要的变化是：

> **不再只用静态 head identity，而是显式输入当前 \(q_i^{Indexer}\)，进一步可加入当前 KV context 的 candidate summary。**

这正对应旧 scorer 的最大缺口：它看不到 candidate content。

### 8.3 Router 结构

使用一个共享的 candidate-wise low-rank scorer，而不是固定的 8-way slot classifier。

MLA side：

$$
z_p=W_q\operatorname{LN}(q_p),
\qquad
W_q:\;1152\rightarrow r.
$$

Indexer side：

$$
r_i
=
e_i
+
W_k\operatorname{LN}(q_i^{Indexer})
+
W_c\operatorname{LN}(c_i).
$$

其中：

- \(e_i\)：可选的 static head-ID embedding；
- \(c_i\)：可选的当前 KV-context summary；
- \(r\) 第一版取 32。

最终：

$$
\boxed{
s_{p,i}
=
z_p^\top r_i
+
\beta_m m_i
+
\beta_g\hat g_i
+
b_{l,i}
}
$$

得到：

$$
s\in\mathbb R^{64\times8}.
$$

然后：

$$
\boxed{
P_{p,i}^{router}
=
\operatorname{softmax}_{i\in S}(s_{p,i})
}
$$

表示 Router 对 8 个 candidates 的选择分布。第一版去掉旧实现中巨大的 \(b_{l,p,i}\)，避免模型主要记忆静态 layer/pair/head 偏好，而不是学习 query-dependent routing。

### 8.4 Loss

该任务不应被建模为普通“8 分类”。多个 candidate 的 utility 往往非常接近，例如：

$$
[0.91,\;0.90,\;0.89,\;0.70,\dots].
$$

如果把 0.91 定义成唯一正确标签，就会把 0.90 和 0.70 同样视为错误。因此主要使用 soft utility distillation。

Teacher distribution：

$$
P_{p,i}^{teacher}
=
\operatorname{softmax}
\left(
\frac{U_{p,i}}{T}
\right).
$$

第一项：

$$
\boxed{
L_{\mathrm{soft}}
=
KL
\left(
P^{teacher}\|
P^{router}
\right)
}
$$

让 Router 学习完整的 candidate utility ranking。

第二项直接优化实际 utility：

$$
U_p^*
=
\max_{i\in S}U_{p,i},
$$

$$
\boxed{
L_{\mathrm{regret}}
=
U_p^*
-
\sum_{i\in S}
P_{p,i}^{router}U_{p,i}
}
$$

它惩罚 Router 把概率放在低-utility candidate 上。

可以再加入低权重 Top-1 CE：

$$
L_{\mathrm{top1}}
=
CE
\left(
s_p,\arg\max_iU_{p,i}
\right).
$$

第一版最终目标：

$$
\boxed{
L
=
L_{\mathrm{soft}}
+
\lambda_rL_{\mathrm{regret}}
+
\lambda_hL_{\mathrm{top1}},
\qquad
\lambda_h\ll1
}
$$

暂时不加入 output distillation，先验证 assignment 本身是否能接近 oracle。

### 8.5 训练方式

冻结：

```text
Base LLM
MLA
Indexer
MISA
DSA gate
```

只训练 Assignment Router。

训练数据按真实 serving/decode 流程采集：

```text
prompt
→ 多个分散 decode positions
→ 61 layers
→ MISA Top-8
→ teacher U[64,8]
```

旧正式离线训练虽然有大量 pair rows，但实际上只有 16 个独立 RULER prompts，因此独立 context 数量仍然偏少。下一版应扩大到至少数百个独立 prompts，并混合 RULER、LongBench、code、math 和普通长文本，严格按 prompt/dataset 做 held-out split。

### 8.6 评估

核心不再报告“Router 选对了多少次 head ID”，而报告：

$$
\boxed{
U_{\mathrm{oracle}}
=
\frac1{64}
\sum_p
\max_{i\in S}U_{p,i}
}
$$

表示 MISA Top-8 set 的 assignment 上限；

$$
\boxed{
U_{\mathrm{router}}
=
\frac1{64}
\sum_p
U_{p,\arg\max_i s_{p,i}}
}
$$

表示 learned Router 的实际 utility；以及：

$$
\boxed{
\operatorname{Regret}
=
U_{\mathrm{oracle}}
-
U_{\mathrm{router}}
}
$$

同时比较：

$$
U_{\mathrm{router}}-U_{\mathrm{static}}.
$$

只有 learned Router 稳定超过 static assignment，才能证明 query-dependent assignment 真正有价值。旧实验中 learned assignment 尚未超过 static assignment，而 set 内 oracle 仍有明显空间，因此 assignment 仍是主要瓶颈。

最终报告：

```text
MISA set-oracle utility
learned assignment utility
regret to oracle
static-assignment baseline
per-layer / p10 tail
RULER / LongBench / AIME accuracy
```

一句话概括：

> **训练时完全复现部署路径：MISA 先动态选 8 个 Indexer heads，只在这 8 个真实 candidates 上构造 MLA attention-mass teacher；Router 根据当前 MLA pair query、动态 Indexer query 和 candidate/context 特征做 candidate-wise utility scoring，并通过 soft utility distillation + expected-regret loss 学习 pair-to-head assignment。**

## 9. 实验记录

### 9.1 实验一：Indexer 时间占比

128K decode profile 的结论：

- 64 Indexer heads 对全历史的 logits 计算约占 Indexer 时间 `89%`；
- 约占单层 attention 总时间 `72%`；
- decode 中 Top-K 本身可忽略；在“很多 query × 很长 KV”的 prefill 中 Top-K 才明显。

因此减少 merge 开销或删除一个 head-weight 乘法不是主优化点。要得到数量级收益，必须减少实际计算的 `q·k_j`：减少被扫描 heads、减少每 head 扫描的 chunks，或增加更便宜的第一阶段。

### 9.2 实验二：chunk16 粗筛表示

设置：chunk size 16，粗筛 `SEL` 个 chunks；其中 `0.4×SEL` dense chunks 取 8 token，其余 `0.6×SEL` sparse chunks 取 4 token。Coverage 定义为真实 Top-112 tokens 中落入被选 chunks 的比例。

| Layer | SEL=20 | 40 | 60 | 80 | 100 | 120 | 140 |
|---|---:|---:|---:|---:|---:|---:|---:|
| L00 | 0.082 | 0.127 | 0.162 | 0.191 | 0.217 | 0.239 | 0.260 |
| L30 | 0.364 | 0.492 | 0.567 | 0.619 | 0.658 | 0.688 | 0.713 |
| L60 | 0.355 | 0.489 | 0.568 | 0.624 | 0.665 | 0.697 | 0.722 |
| 三层均值 | 0.267 | 0.369 | 0.432 | 0.478 | 0.513 | 0.541 | 0.565 |

L30/L60 的 representation 对照：

| measure | SEL=20 | SEL=40 | SEL=60 | SEL=80 |
|---|---:|---:|---:|---:|
| mean | 0.364 / 0.355 | 0.492 / 0.489 | 0.567 / 0.568 | 0.619 / 0.624 |
| `(min+max)/2` | 0.292 / 0.291 | 0.407 / 0.411 | 0.480 / 0.489 | 0.534 / 0.545 |
| Quest-style extrema | 0.190 / 0.192 | 0.282 / 0.293 | 0.349 / 0.365 | 0.401 / 0.420 |

mean chunk representation 在这组真实 trace 上最好，因此当前 chunk16 selector 用 chunk sum/mean 做 coarse ranking。

### 9.3 实验三：共享 DSA Top-2048 的 ID recall 与真实 MLA mass

4K、15 个有效 queries、7 个层的结果：

| Layer | 覆盖 MLA exact Top-720 的 recall | 保留真实 MLA attention mass |
|---:|---:|---:|
| 0 | 92.84% | 94.84% |
| 2 | 89.76% | 88.01% |
| 7 | 94.82% | 96.07% |
| 15 | 94.74% | 96.00% |
| 30 | 98.25% | 97.95% |
| 45 | 96.70% | 99.02% |
| 60 | 96.43% | 96.95% |
| 平均 | **94.79%** | **95.55%** |

这里 shared DSA 的预算是 2048，而真值集合是 Top-720，不能把该 recall 与同预算 Top-720 方法直接比较。

另一个 RULER QA 32K、Layer 0 实验中，shared DSA Top-2048 对 MLA exact Top-2048 的 ID recall 只有 `41.8%`，但 attention mass 仍有 `74.8%`。这说明 ID overlap 会低估“命中了少数高质量 token”的价值，最终应优先报告 attention mass。

### 9.4 实验四：固定 1:2 是否成立

设置：4K，15 queries，层 `0,2,7,15,30,45,60`；每层比较 64 Indexer heads 与 128 MLA heads。

| 映射 | Recall | Attention mass |
|---|---:|---:|
| 固定相邻 1:2 | 21.32% | 41.68% |
| 随机 Indexer head | 21.47% | 41.28% |
| 最优匹配且每个 Indexer 最多服务两个 MLA heads | 26.81% | 51.69% |
| 每个 MLA head 独立选最佳 Indexer | 44.65% | 70.20% |
| 原始 shared Top-2048 | — | 95.55% |

固定 1:2 基本等于随机，假设不成立。最优映射高度多对一，例如 Layer 45 中 123/128 个 MLA heads 都偏向 Indexer head 54。部分层的主要映射：

- L00：I44→70 个 MLA heads，I37→38；
- L07：I23→50，I31→38；
- L30：I36→36，I58→31，I14→27，I48→10；
- L45：I54→123，I33→5；
- L60：I3→58，I41→24，I32→23，I44→12。

这正是 assignment router 必要的直接证据，同时也解释了为什么全局只需要少数 unique Indexer heads。

### 9.5 实验五：128K 下 shared 与 naive per-head

128K probe 的一个直接对照为：

| 方法 | ID recall | Attention mass |
|---|---:|---:|
| official shared selection | 0.5825 | 0.6505 |
| naive/fixed per-head path | 0.0369 | 0.1251 |

不同实验的数据、candidate budget 和测量目标不同，不能把该表与 4K 的 0.9555 或全 61 层训练 utility 直接横比。但方向一致：**简单拆成独立 per-head candidates 会破坏 DSA 原有的跨 head 互补。**

在另一组 7-layer 128K pair-utility 数据中：

| 方法 | Mean retained attention mass |
|---|---:|
| 固定 1:2 per-head Top-720 | 0.4045 |
| unrestricted best Indexer per MLA pair | 0.6117 |
| unrestricted best Indexer per individual MLA head | 0.6160 |
| official shared DSA Top-2048 | 0.7493 |
| exact MLA Top-720 ceiling | 0.7616 |

pair 路由相对逐 MLA-head oracle 只差 0.0043，说明 64 个 two-head pairs 是合理的 consumer 粒度；更大的 gap 来自 Indexer candidate quality，而不是 pair grouping。

### 9.6 实验六：exact MLA 需要多大的 K 才匹配 shared DSA

128K，16 prompts、128 queries、7 layers、114,688 个 MLA-head observations：

| 方法 | 平均 attention mass |
|---|---:|
| official shared DSA Top-2048 | 0.767164 |
| exact MLA Top-512 | 0.755664 |
| exact MLA Top-720 | 0.776911 |
| exact MLA Top-1024 | 0.799600 |

如果只比较全体平均 mass，512 与 720 之间的线性交点约为 `K≈625`，所以 kernel-friendly 的平均预算可取 640。但逐 observation 计算“第一次达到对应 shared-DSA mass 的最小 K”时：

```text
mean = 757.6
median = 759
p90 = 1263
```

固定预算覆盖比例：

| K | 达到/超过各自 shared-DSA mass 的 observations |
|---:|---:|
| 512 | 29.35% |
| 720 | 46.49% |
| 1024 | 73.13% |
| 1536 | 97.95% |

所以“500–700 足够”只适用于平均 mass 叙述；如果希望 90% 的 head/query observations 不差于 DSA，应取约 1280。论文中必须区分 mean crossover 与 per-observation guarantee。

分层差异也很大：

| Layer | shared mass | 最小 K mean | K p50 | K p90 | exact K=720 mass |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.5007 | 751 | 706 | 1584 | 0.5207 |
| 2 | 0.4404 | 540 | 489 | 1070 | 0.4844 |
| 7 | 0.7994 | 646 | 615 | 1105 | 0.8062 |
| 15 | 0.8827 | 649 | 627 | 1101 | 0.8853 |
| 30 | 0.9648 | 999 | 1027 | 1350 | 0.9628 |
| 45 | 0.9547 | 937 | 963 | 1290 | 0.9534 |
| 60 | 0.8274 | 782 | 786.5 | 1260 | 0.8257 |

### 9.7 实验七：端到端精度

2026-09-03 完成的全 61 层正式矩阵中，`router` arm 的含义是：**每层离线静态 M=8 head set + learned per-query assignment**。它不是 2026-09-06 接入的动态 MISA head selection。

| Arm | LongBench-v2 503 | RULER 26 | AIME 2025 30，pass@1 |
|---|---:|---:|---:|
| official shared DSA | 49.30% | 84.10% | 93.33% |
| global M=8 + static assignment | 50.70% | 52.76% | 73.33% |
| global M=8 + learned assignment | 47.91% | 63.72% | 63.33% |

LongBench-v2 按长度：

| Arm | long | medium | short |
|---|---:|---:|---:|
| official DSA | 47.22% | 49.30% | 50.56% |
| M8 static assignment | 47.22% | 52.56% | 50.56% |
| M8 learned assignment | 41.67% | 52.09% | 46.67% |

飞书早期表格将 DSA 写为 49.11%，而当前远端正式 `longbench_summary.json` 为 49.30%（相差 1/503 题）；本文以可复查的当前 summary JSON 为主，同时保留该差异说明。

AIME 的结果需要额外谨慎：每题只有一个 sample；实验 sparse policy 在生成 3000 tokens 后才启用；router/static arms 分别有 8 个 truncated generations，而 official arm 为 0。该数字不能单独归因于 assignment router。

### 9.8 Global unique-head budget oracle

7-layer 128K oracle：

| M | Dynamic set + dynamic assignment | Static set + dynamic assignment |
|---:|---:|---:|
| 1 | 0.6045 | 0.5846 |
| 2 | 0.6089 | 0.5982 |
| 4 | 0.6108 | 0.6049 |
| 8 | 0.6115 | 0.6068 |
| 16 | 0.6117 | 0.6079 |
| 64 | 0.6117 | 0.6117 |

动态 `M=4` 已达到 unrestricted pair oracle 的 99.93%，`M=8` 达到 99.99%。这验证的是“utility 集中在很少的 unique heads”，而不是当前 MISA 已经找到了 oracle 集合。

连续 8 个 decode queries 的 temporal 结果：

- M=8 每 token 动态 set+assignment：0.6115；
- 复用第一个 query 的 set、每 token 重做 assignment：0.6064；
- set 和 assignment 都复用：0.5313；
- set 与第一个 query 的平均 Jaccard 仅 0.382，8-token union 为 18.43。

结论是 set 可以低频更新或容忍等价集合，但 assignment 必须高频更新。

### 9.9 全 61 层 assignment scorer held-out 结果

| M | Joint learned | Predicted set + oracle assignment | Static set + learned assignment | Static set + static assignment | Static set + oracle assignment |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.7444 | 0.7593 | 0.7447 | 0.7450 | 0.7587 |
| 8 | 0.7440 | 0.7619 | 0.7445 | 0.7450 | 0.7605 |

这张表给出最直接的训练结论：

- set scorer 预测出的集合在 oracle assignment 下很好；
- q-only learned assignment 没有超过 static assignment；
- M=8 的 static-set learned assignment 距集合内 oracle 约 0.016；
- 因此没有一个可信的“classification accuracy”可以宣称 assignment 已经训练好，应该报告 retained attention mass 与 regret。

### 9.10 MISA 现有初步结果

#### 对 production DSA Top-2048 的 token recall

61 layers、13 queries、793 records：

| 方法 | 对 production shared Top-2048 的 recall |
|---|---:|
| HISA mean-128 → 8192 candidates → DSA rerank | 60.75% |
| MISA 8 heads → 8192 candidates → DSA rerank | 72.79% |
| MISA 8 heads direct Top-2048 | 45.73% |

这里的 ground truth 是 production DSA token ids，不是 MLA attention mass。它说明 MISA 选 8 heads 后若直接取 token，仍与全 64-head shared selector 有明显差距；扩大候选并用完整 DSA rerank 能显著恢复。

#### 三层真实 128K dump 的 head-routing 分析

每层抽样 48 query rows：

| Layer | MISA@8 与 greedy head oracle 的集合 overlap | MISA 8-head dense mass | 8-head greedy oracle mass | 64-head official mass |
|---:|---:|---:|---:|---:|
| 0 | 22.14% | 0.2978 | 0.3697 | 0.3821 |
| 30 | 8.59% | 0.5497 | 0.6279 | 0.6468 |
| 60 | 30.99% | 0.5454 | 0.5972 | 0.6137 |

MISA 与 64-token extrema router 的选集 overlap 却为 80.2%–96.4%，说明两个便宜启发式彼此相近，但都不等于针对最终 utility 的 oracle。

#### synthetic prefill microbenchmark

H20、`nq=8192`、M=8：

| `nk` | MISA blockmean router | per-head Top-K H64 | per-head Top-K H8 | DSA dense total | router + H8 / DSA |
|---:|---:|---:|---:|---:|---:|
| 32K | 1.44 ms | 59.97 ms | 7.69 ms | 23.78 ms | 2.61× faster |
| 64K | 1.44 ms | 83.77 ms | 10.61 ms | 47.04 ms | 3.91× faster |
| 128K | 2.75 ms | 130.08 ms | 16.60 ms | 93.92 ms | 4.85× faster |

这是 synthetic standalone microbenchmark，不是 SGLang 集成后的 prefill 吞吐；脚本中的 per-head path 取 Top-2048，也不等于当前 runtime 的 chunk16 Top-720。它只能说明“head routing 开销相对 token scan 较小”和“64→8 的 per-head scan 接近线性缩放”。

## 10. 当前实现与结果之间的时间线

| 日期 | head set | assignment | 有什么结果 |
|---|---|---|---|
| 2026-09-03 | 每层离线静态 M=8 | learned 或 static | LongBench/RULER/AIME 完整矩阵 |
| 2026-09-06 | 每 query MISA M=8 | learned | runtime 与单元测试已接通；尚无同规格完整下游矩阵 |

因此：

```text
47.91% LongBench != MISA + assignment 的准确率
47.91% = static M8 set + learned assignment 的准确率
```

同一个旧 `offline_router_m8_all61.json` 没有 `head_selection_mode` 字段；在新代码中 learned mode 会默认走 MISA。重跑历史实验时必须显式固定 config，否则实验语义会静默变化。

## 11. 系统收益与尚未完成的部分

### 11.1 当前 TP8 scan 仍是复制的

当前每个 TP rank 都扫描同一 query 的 M 个动态 heads：

```text
per-rank scans: M
cluster scans:  8M
```

M=8 时 cluster 总扫描仍为 64，与原本每 rank 8 个本地 heads 相同。要实现真正的全局 unique-head 计算收益，需要：

```text
owner(head_id) = head_id % tp_size
```

每个 selected head 只在 owner rank 扫一次，再 all-gather `(head_id, candidate indices)`。否则“global M=8”是逻辑 budget，不是集群 scan budget。

### 11.2 indices 表示仍应 compact

理想接口：

```text
unique_indices: [B,M,K]
pair_to_slot:   [B,64]
```

Top-720、int32，每 query/layer：

| 表示 | 大小 |
|---|---:|
| 64 lists | 184,320 bytes |
| compact M=4 + uint8 slot map | 11,584 bytes |
| compact M=8 + uint8 slot map | 23,104 bytes |

当前 runtime 在 Python 层最终展开为每 local pair 一份 indices，并且底层接口常 pad 到 2048；逻辑复用已经存在，但最终 kernel 尚未原生消费 compact indirect layout。

### 11.3 当前 assignment 不看 candidate content

这是最值得优先验证的模型缺口。可选增强方向：

1. 为每个 candidate list 构造廉价 summary，例如选中 K 的 Indexer score statistics、位置分布、chunk histogram；
2. 在不读取 MLA K 的前提下，加入 candidate pooled Indexer K 与 MLA pair Q 的小规模交互；
3. teacher 只在 MISA 实际选出的 M 个 heads 内训练 assignment，使 train/test candidate distribution 对齐；
4. sampled online distillation，直接用运行中的当前 query candidate utility 监督；
5. 允许每个 pair 选择 2 个 candidate lists 并做 union，测量少量额外预算能否显著缩小 0.016 gap。

### 11.4 candidate quality 仍是上限

即使 assignment 达到 oracle，7-layer unrestricted Indexer-pair utility 也只有 0.6117，而 exact MLA Top-720 是 0.7616。优化 router 不能弥补 candidate selector 的全部质量差距。论文实验应把误差拆成：

```text
exact MLA Top-K ceiling
  -> shared DSA / per-Indexer candidate gap
  -> M-head set-selection regret
  -> within-set assignment regret
  -> downstream task gap
```

## 12. 推荐的下一轮实验

优先级从高到低：

1. **补当前 MISA+assignment 的完整 paired E2E**：与 official、static M8+static assignment、static M8+learned assignment 同 prompt、同 seed 比较。
2. **测 MISA set oracle regret**：在已有 64×64 utility table 上，用每个 query 的 MISA `[B,M]` 直接计算 oracle assignment mass；这能把 MISA head selection 与 assignment 模型完全解耦。
3. **MISA-aligned retraining**：只在 MISA 选集内优化 assignment，并与 all-64 regression 比较。
4. **candidate-aware scorer**：先加入最便宜的候选 summary，观察是否超过 static pair prior。
5. **分层 K/M**：Layer 30/45 对 K 更敏感，Layer 2/7/15 可以更激进；不要强制所有层使用同一 720。
6. **owner-sharded scan + compact kernel**：只有这一步完成后才能报告集群级 speedup。
7. **prefill integration**：当前 production runtime 只接 decode；standalone prefill microbenchmark 不能替代端到端 prefill 测量。

建议论文主表至少同时报告：

```text
quality: attention mass, ID recall, downstream accuracy
router:  MISA set-oracle mass, learned assignment mass, regret to oracle
system:  actual unique scans/cluster, router ms, scan ms, collective ms,
         sparse-attention ms, end-to-end tokens/s, peak memory
```

## 13. 代码地图

### SGLang-HISA runtime

```text
/DATA/disk0/qyl/code/dpskv32/sglang-hisa/
  python/sglang/srt/layers/attention/nsa/per_head_paged.py
    175-276  MISA pooled-key head selector
    279-394  dynamic head-id Top-720 candidate generation

  python/sglang/srt/layers/attention/nsa/nsa_indexer.py
    495-629  MISA/select heads -> per-head scan -> stash gate/head ids

  python/sglang/srt/layers/attention/nsa/offline_unique_head_router.py
     56-152  config semantics
    204-227  only load assignment_scorer.*
    285-416  score 64 -> gather M -> argmax -> gather candidates

  python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
    341-357  post-RoPE MLA Q 调用 assignment router

  python/sglang/srt/layers/attention/nsa/csrc/chunk16_quota_paged.cu
    fused chunk16 quota kernel
```

### 训练与配置

```text
/DATA/disk0/qyl/code/dpskv32/sglang-hisa/
  scripts/train_offline_unique_head_router.py
     18-61   聚合 TP8 feature shards，构造 q_pair/gate/U
     64-148  RouterScorer 与 OfflineRouter
    156-207  listwise/regression/hybrid loss
    241-308  AdamW training loop

  scripts/build_offline_router_runtime_config.py
     35-66   misa/static selection mode
    114-138  生成自描述 runtime JSON

  test/registered/unit/test_misa_head_router.py
    pooled logical K、FP8 reference、dynamic head gather、MISA-set-restricted assignment
```

2026-09-07 在现有 `qyl/sglang-hisa:eval` 容器运行定向测试：

```text
3 passed, 1 skipped in 6.29s
```

跳过的是需要 CUDA 的 FP8 pooled-reference case；CPU 的逻辑 pooling、动态 candidate gather 和 assignment restriction contract 均通过。

## 14. 数据与结果索引

```text
# 4K shared DSA mass / per-head mapping
/DATA/disk0/qyl/data/headwise_minmax_e2e/exp1_dsa_mla_mass_recall/summary.json
/DATA/disk0/qyl/data/headwise_minmax_e2e/exp2_indexer_mla_head_map/summary.json

# 128K shared vs per-head、global budget、全 61 层 router
/DATA/disk0/qyl/data/headwise_minmax_e2e/exp3_128k_shared_vs_per_head/summary.json
/DATA/disk0/qyl/data/headwise_minmax_e2e/exp5_128k_multidecode_router/
/DATA/disk0/qyl/data/headwise_minmax_e2e/exp6_128k_all61_multidecode_router/

# exact MLA K to match shared DSA
/DATA/disk0/qyl/data/headwise_minmax_e2e/exp7_128k_exact_mla_budget/analysis/summary.json
/DATA/disk0/qyl/data/headwise_minmax_e2e/exp7_128k_exact_mla_budget/analysis/per_head_budget.csv

# static-M8/learned-assignment 的正式 E2E
/DATA/disk0/qyl/data/router_global_budget_accuracy_20260903/full/

# MISA 分析与 prefill microbenchmark
/DATA/disk0/qyl/data/headwise_minmax_e2e/misa_router_analysis.json
/DATA/disk0/qyl/data/ruler_eval_results/misa_routed_prefill.json
/DATA/disk0/qyl/code/dpskv32/indexer_analysis/results/experiment_hisa_misa_128k/
```

## 15. 一句话结论

我们已经证明了两件事：第一，逐 MLA consumer 的有效稀疏预算显著小于 shared 2048；第二，一次 query/layer 的高 utility 只集中在极少数 unique Indexer heads。当前四阶段实现已经把“无训练 MISA 选 M”和“trained assignment 选 slot”接到真实 decode forward，但现阶段最弱的环节仍是 candidate-aware assignment 与 per-Indexer candidate quality；在补齐 MISA 的完整端到端质量、owner-sharded scan 和 compact attention kernel 之前，不应把这个原型描述成已经获得端到端无损加速。
