# MISA Head Selection 与 Assignment Router 训练方法

## 摘要

本文描述一种面向 DSA 稀疏注意力的两阶段动态路由方法。第一阶段使用 MISA 根据当前 Indexer query、DSA gate 以及历史 KV context 的粗粒度摘要，从 64 个 Indexer heads 中动态选择 \(M=8\) 个 heads；第二阶段让这 8 个 heads 分别产生一组 720-token candidates，再由 Assignment Router 为 64 个 MLA pairs 选择最合适的 candidate set。这样设计的原因是，Indexer head 的价值不仅随 layer 变化，也强烈依赖当前 query 和上下文。如果使用固定的离线 head set，就会丢失这种动态性；如果每次扫描全部 64 个 Indexer heads，又会使计算量接近不可接受。MISA 负责以较低代价缩小候选范围，Assignment Router 则在已经缩小的真实候选集合中学习细粒度的 MLA-pair assignment。

整个部署路径可以概括为

\[
64\ \text{Indexer heads}
\xrightarrow{\text{MISA}}
8\ \text{candidate heads}
\xrightarrow{\text{retrieval}}
8\ \text{candidate sets}
\xrightarrow{\text{Assignment Router}}
64\ \text{MLA-pair assignments}.
\]

这一方法不再训练额外的 set scorer。候选集合完全由运行时 MISA 产生，Router 的训练目标与部署目标保持一致，只回答一个问题：对于当前 MLA pair，MISA 给出的 8 组真实 candidates 中哪一组能够保留最多的有效注意力信息。

## 1. 设计动机

DSA Indexer 包含 64 个 query heads，不同 Indexer heads 对不同类型的 query、不同上下文结构以及不同 layer 的敏感性并不相同。某些 heads 更擅长定位局部匹配，某些 heads 更擅长发现远距离依赖，也有一些 heads 只在特定任务或特定 layer 上有效。因此，一个固定的全局 Top-\(M\) head set 即使在平均意义上表现良好，也可能在单个 query 上遗漏真正重要的 head。Assignment Router 的旧训练方式如果先在 64 个 heads 上学习一个离线集合，再在该集合中做 assignment，也会产生明显的训练与部署错位，因为实际部署时的集合是由当前 query 动态决定的。

MISA 的核心作用是利用已经存在于 Indexer 路径中的信息做一次低成本的动态筛选。Indexer 本来就会产生当前 query 和 gate，KV cache 也可以持续维护局部 key sums，因此没有必要为了选 head 再执行一次完整的 token-level 扫描。我们只需先用粗粒度 context summaries 判断每个 Indexer head 是否与当前上下文中的若干区域存在显著正相关，就可以把 64 个 heads 缩减到 8 个。随后，计算量更高的 fine-grained retrieval 只在这 8 个 heads 上执行。

第二阶段仍然需要 Assignment Router，是因为 MISA importance 衡量的是某个 Indexer head 对当前 query 和整个 KV context 的总体相关程度，却没有直接区分 64 个 MLA pairs 的需求。同一个 Indexer head 产生的 token candidates 可能非常适合一部分 MLA pairs，却不适合另外一些 MLA pairs。MISA 解决 candidate set 的生成问题，Assignment Router 解决 MLA pair 与 candidate set 之间的匹配问题，两者不能简单地合并为同一个标量排序。

## 2. MISA 的动态 Head Selection

### 2.1 使用 coarse block 表示历史上下文

设历史 context 长度为 \(L\)。MISA 首先将历史 token 划分为大小为 \(B_{\mathrm{coarse}}\) 的 coarse blocks，当前首个 kernel 版本采用

\[
B_{\mathrm{coarse}}=512.
\]

对于第 \(b\) 个 coarse block，其 Indexer key summary 定义为 block 内 key 的均值

\[
\bar{\mathbf{k}}_b
=
\frac{1}{|B_b|}
\sum_{t\in B_b}\mathbf{k}_t.
\]

选择 512 tokens 是计算量与局部信息保真度之间的折中。如果 block 太小，例如 128 或 256 tokens，summary 的定位能力会更强，但 MISA 本身需要处理的 block 数量会明显增加，削弱第一阶段筛选的意义。如果 block 太大，例如 1024 tokens，一个局部非常重要的 16-token peak 可能被周围大量无关 token 平均掉，使相关 block 在 coarse ranking 中被错误淘汰。512 tokens 可以把 128K context 压缩为约 256 个 summaries，同时仍保留一定的局部区分能力。

实现上不需要重新遍历 KV cache 来计算这些 summaries。系统已经为后续 retrieval 维护了 16-token chunk sums，因此只需将相邻 32 个 chunk sums 聚合，就可以得到一个 512-token coarse block 的均值。这一点很重要，因为 MISA 的收益依赖于“复用已有统计量”，而不是额外引入一条代价接近完整 attention 的新路径。

### 2.2 计算 Indexer head importance

对于当前 decode query，第 \(h\) 个 Indexer head 具有 Indexer query \(\mathbf{q}^{I}_h\) 和 DSA gate \(g_h\)。该 head 与 coarse block \(b\) 的 affinity 定义为

\[
a_{h,b}
=
\left(\mathbf{q}^{I}_h\right)^\top\bar{\mathbf{k}}_b.
\]

随后，将所有可见 coarse blocks 上的正 affinity 聚合，并使用 gate 对其进行调节：

\[
I_h
=
|g_h|
\sum_{b=1}^{N_{\mathrm{block}}}
\operatorname{ReLU}(a_{h,b}).
\]

这里使用 \(\operatorname{ReLU}\) 的原因是，我们希望 importance 表示“上下文中存在多少与该 head 正向匹配的证据”。大量负 affinity 不应与少数强正 affinity 相互抵消，否则一个能够准确定位少量关键区域的 head 可能因为其他无关 blocks 的负响应而被压低。DSA gate 则提供当前 query 下该 head 的动态置信度。最终的 importance 同时包含“该 head 是否被当前 query 激活”和“它是否在历史 context 中找到了正向匹配区域”两类信息。

MISA 根据 importance 从 64 个 heads 中选择

\[
\mathcal{H}(q)
=
\operatorname{TopM}_{h\in\{1,\ldots,64\}} I_h,
\qquad M=8.
\]

因此，\(\mathcal{H}(q)\) 会随着 query、layer 和 KV context 改变。这里的 \(M=8\) 是计算预算，而不是固定的 head identity。不同 query 可以选择完全不同的 8 个 heads。

## 3. 利用 MISA Affinity 做分层剪枝

完成 head selection 后，最直接的做法是让选中的 8 个 heads 分别扫描全部 16-token chunks。然而，这会浪费 MISA 已经计算出的 coarse-block affinity。既然 \(a_{h,b}\) 已经近似描述了 head \(h\) 与 block \(b\) 的相关程度，就可以先淘汰一部分明显不相关的 coarse blocks，再对保留区域执行细粒度计算。

对于每个被选中的 head \(h\)，我们按照 \(a_{h,b}\) 对 coarse blocks 排序，并保留排名最高的 60%：

\[
\mathcal{B}_h
=
\operatorname{TopK}_{b}\ a_{h,b},
\qquad
K
=
\left\lceil0.6N_{\mathrm{block}}\right\rceil.
\]

首个 sink block 和当前 tail block始终保留。sink block 通常包含系统提示、任务定义或其他具有全局作用的信息；tail block 则包含离当前 decode query 最近的 tokens。将它们强制保留可以避免纯 affinity ranking 对这两类结构性位置造成不稳定影响。

只有属于 \(\mathcal{B}_h\) 的 coarse blocks 才会被进一步拆分为 16-token fine chunks。系统随后选择得分最高的 128 个 fine chunks，并在这些 chunks 内计算精细 token score。当前 720-token quota 不是简单地对整个 context 做一次全局 Top-720，而是在最重要的 52 个 fine chunks 中各保留 8 个 tokens，在其余 76 个 fine chunks 中各保留 4 个 tokens：

\[
52\times8+76\times4=720.
\]

这种 quota 设计有意保留一定的区域多样性。如果完全执行全局 token Top-\(k\)，少数高分局部区域可能占据绝大多数预算，导致其他相关区域完全消失。最终，第 \(i\) 个 MISA-selected head \(h_i\) 产生 candidate set

\[
\mathcal{C}_i
=
\operatorname{QuotaSelect720}
\left(
\mathbf{q}^{I}_{h_i},
\mathcal{B}_{h_i}
\right),
\qquad h_i\in\mathcal{H}(q).
\]

512-token coarse block 与 60% retention 是首个工程版本，而不是无需验证的理论最优值。它们是否合理，需要同时观察 baseline Top-128 fine chunks 在保留 coarse blocks 中的 recall、candidate attention mass、延迟收益和端到端准确率。这样可以避免只看到计算量下降，却忽略少量关键局部 peak 被 coarse mean 掩盖的问题。

## 4. Assignment Router 的监督信号

模型共有 128 个 MLA heads。当前运行时以相邻两个 MLA heads 为一个 assignment 单元，因此形成 64 个 MLA pairs：

\[
P_p=\{2p,2p+1\},
\qquad p=0,\ldots,63.
\]

以 pair 为单位并不是假设两个 MLA heads 的行为完全相同，而是遵循当前运行时的共享 assignment contract，并在路由开销与表达能力之间取得平衡。Teacher utility 会分别计算两个 MLA heads 的保留 attention mass，再取平均，因此两个 heads 的需求都参与监督，而不是只使用其中一个 head 作为代表。

对于每一个真实 query-layer context，训练过程首先运行 MISA 和分层 retrieval，得到 \(M=8\) 个真实 candidate sets \(\{\mathcal{C}_1,\ldots,\mathcal{C}_M\}\)。随后，对每个 MLA head 计算完整 context 上的 dense attention distribution：

\[
\alpha_{r,t}
=
\frac{
\exp\left(\gamma(\mathbf{q}^{M}_r)^\top\mathbf{k}^{M}_t\right)
}{
\sum_{u=1}^{L}
\exp\left(\gamma(\mathbf{q}^{M}_r)^\top\mathbf{k}^{M}_u\right)
},
\]

其中 \(\gamma\) 是模型实际使用的 attention scaling。Candidate set \(\mathcal{C}_i\) 对 MLA pair \(p\) 的 utility 定义为

\[
U_{p,i}
=
\frac{1}{2}
\sum_{r\in P_p}
\sum_{t\in\mathcal{C}_i}
\alpha_{r,t}.
\]

这个量直接回答了部署真正关心的问题：如果 MLA pair \(p\) 使用 candidate \(i\)，它能够保留多少 dense attention mass。每个 query-layer context 最终产生

\[
\mathbf{U}\in\mathbb{R}^{64\times M}.
\]

训练只构造这张 \(64\times8\) utility matrix，不再为 64 个 Indexer heads 构造一个与部署候选集合不一致的全局目标。这样可以避免旧方案最主要的问题：训练时优化的是离线选出的 head set，部署时面对的却是 MISA 动态生成的另一个 head set。

## 5. Candidate-wise Assignment Router

Router 需要同时理解 MLA pair 的需求和每个 candidate 的当前状态。对于 MLA pair \(p\)，我们将两个 MLA queries 拼接为

\[
\mathbf{x}_p
=
\left[
\mathbf{q}^{M}_{2p};
\mathbf{q}^{M}_{2p+1}
\right],
\]

并投影到低秩空间：

\[
\mathbf{z}_p
=
W_q\operatorname{LN}(\mathbf{x}_p)
+\mathbf{e}^{P}_p
+\mathbf{e}^{L}_{\ell}.
\]

\(\mathbf{e}^{P}_p\) 表示不同 MLA pairs 的稳定差异，\(\mathbf{e}^{L}_{\ell}\) 表示 layer-specific routing pattern，而当前 MLA query 则提供真正的 query-dependent 信息。只使用静态 pair 和 layer prior 容易记住训练集平均规律，却无法处理同一 layer 中不同 query 的变化；只使用当前 query 又可能忽略长期稳定的结构。因此三部分需要共同构成 pair representation。

候选 \(i\) 的表示为

\[
\mathbf{v}_i
=
\mathbf{e}^{H}_{h_i}
+W_I\operatorname{LN}\left(\mathbf{q}^{I}_{h_i}\right)
+W_C\operatorname{Norm}\left(\mathbf{c}_i\right).
\]

这里，\(\mathbf{e}^{H}_{h_i}\) 表示 Indexer head identity，当前 Indexer query \(\mathbf{q}^{I}_{h_i}\) 表示该 head 在当前 token 上正在寻找什么，\(\mathbf{c}_i\) 则描述该 candidate 与当前 KV context 的匹配形态。当前 \(\mathbf{c}_i\) 使用 coarse positive affinity 的均值、标准差、最大值和正值比例。静态 head embedding 能学习稳定的 head specialization，但它不能替代动态 Indexer query 和 context 特征；后两者是 Router 对当前样本做出不同 assignment 的主要依据。

最终 assignment score 为

\[
s_{p,i}
=
\frac{\mathbf{z}_p^\top\mathbf{v}_i}{\sqrt{d_r}}
+b_{\ell,h_i}
+\mathbf{w}_f^\top\mathbf{f}_i,
\]

其中 \(d_r\) 是低秩维度，当前默认 \(d_r=64\)；\(b_{\ell,h_i}\) 是较轻量的 layer-head prior；\(\mathbf{f}_i\) 包含标准化后的 DSA gate、MISA importance 和 context statistics。共享的 candidate-wise scorer 对每个 candidate 使用同一套参数，因此它不依赖 candidate 在 Top-8 中的槽位编号，也不会把“第 3 个 slot”错误地当成具有固定语义的类别。这种结构与 MISA candidate set 的动态性相匹配。

Router 输出

\[
\mathbf{S}\in\mathbb{R}^{64\times M},
\]

部署时每个 MLA pair 使用

\[
\hat{\pi}(p)=\arg\max_i s_{p,i}
\]

选择对应 candidate set。Router 不修改 candidate 内部的 token 顺序，也不重新计算 retrieval，只负责在 8 个现有集合之间做 assignment。

## 6. Training Objective

如果只把 \(\arg\max_i U_{p,i}\) 当作 hard label，会丢失 candidates 之间的 utility 差异。例如，两个 candidates 的 attention mass 可能只差 \(10^{-4}\)，它们在 hard classification 中却分别被视为完全正确和完全错误；反过来，一个 utility 明显更差的 candidate 与次优 candidate 也会受到相同的分类惩罚。这与我们的真实目标并不一致，因为部署关心的是 attention mass 损失，而不是候选编号是否精确命中。

因此，首先将 utility 转为 soft target：

\[
y_{p,i}
=
\frac{
\exp\left((U_{p,i}-U_p^\star)/\tau\right)
}{
\sum_j\exp\left((U_{p,j}-U_p^\star)/\tau\right)
},
\qquad
U_p^\star=\max_i U_{p,i}.
\]

Soft utility distillation loss 为

\[
\mathcal{L}_{\mathrm{soft}}
=
-\frac{1}{64}
\sum_p\sum_i
y_{p,i}
\log\operatorname{softmax}(\mathbf{s}_p)_i.
\]

Soft target 能表达 candidates 之间的相对质量，但它仍然是一个经过温度变换的分布。为了让优化目标更直接地对应部署损失，我们进一步加入 expected regret：

\[
\mathcal{L}_{\mathrm{regret}}
=
\frac{1}{64}
\sum_p\sum_i
\hat{y}_{p,i}
\left(U_p^\star-U_{p,i}\right),
\qquad
\hat{y}_{p,i}=\operatorname{softmax}(\mathbf{s}_p)_i.
\]

Expected regret 会根据实际 utility gap 调整惩罚强度。把概率分配给接近 oracle 的 candidate 只产生很小的损失，而把概率分配给明显较差的 candidate 会受到更强惩罚。最后保留一个低权重的 hard Top-1 cross entropy，用于维持清晰的最优候选判别边界：

\[
\mathcal{L}_{\mathrm{top1}}
=
\operatorname{CE}
\left(
\mathbf{s}_p,
\arg\max_i U_{p,i}
\right).
\]

总损失为

\[
\mathcal{L}
=
\mathcal{L}_{\mathrm{soft}}
+\lambda_r\mathcal{L}_{\mathrm{regret}}
+\lambda_1\mathcal{L}_{\mathrm{top1}}.
\]

当前默认使用 \(\lambda_r=1\)、\(\lambda_1=0.1\) 和 \(\tau=0.02\)。Top-1 loss 的权重刻意保持较低，因为它只用于辅助形成决策边界，不应覆盖 soft utility 和 regret 所表达的真实质量差异。

## 7. 数据采集与划分

训练数据必须完整复现部署路径。每条样本都来自真实 prompt 和真实 decode prefix；在每个目标 layer 上先运行 MISA，使用 512-token coarse blocks 动态选择 8 个 Indexer heads，再保留每个 head 的 60% coarse blocks并产生对应的 720-token candidate set，最后计算 \(\mathbf{U}\in\mathbb{R}^{64\times8}\)。只有这种采集方式才能保证 Router 训练时看到的 candidate 分布与部署一致。

数据来源需要同时覆盖长上下文检索、文档理解、数学推理、科学问答和代码任务。RULER 应使用完整的 task-length 网格，而不应只采集 128K。推荐长度至少覆盖

\[
4\mathrm{K},\ 8\mathrm{K},\ 16\mathrm{K},\ 32\mathrm{K},\ 64\mathrm{K},\ 128\mathrm{K}.
\]

LongBench-v2 用来补充真实长文档和多类型长上下文任务，AIME 2025 与 MATH-500 提供数学推理轨迹，GPQA Diamond 提供科学推理样本，后续还需要增加代码和多轮任务。不同数据集之间应采用平衡采样，避免样本量最大的任务完全主导梯度。

每个 prompt 应从真实生成轨迹中选择多个相互分散的 decode positions，而不是把连续几个 tokens 当作近似独立样本。连续 query 的表示高度相关，盲目增加相邻 token 数量会让样本数量看起来很大，却不能显著增加有效信息。数据划分必须以 prompt 为单位，使同一 prompt 的所有 decode positions 始终属于同一个 split，从而避免训练和评估之间发生隐性泄漏。对于 AIME 2025 等规模较小且常用于最终报告的数据集，还需要明确区分参与 Router 训练的题目和保留用于无污染端到端评测的题目。

## 8. 评估原则

Candidate Top-1 accuracy 不能单独反映 Router 质量，因为预测次优 candidate 可能几乎不损失 attention mass，而少量错误选择也可能产生非常大的 utility drop。首要指标应是 MISA candidate set 内的 oracle utility

\[
U_{\mathrm{oracle}}
=
\frac{1}{64}
\sum_p\max_i U_{p,i},
\]

以及 learned assignment utility

\[
U_{\mathrm{router}}
=
\frac{1}{64}
\sum_p U_{p,\hat{\pi}(p)}.
\]

两者之差

\[
R=U_{\mathrm{oracle}}-U_{\mathrm{router}}
\]

直接表示 Assignment Router 相对于当前 MISA set oracle 的 regret。评估还需要与静态 assignment baseline 比较，以确认动态 MLA query、Indexer query 和 context features 确实提供了超越 layer/pair/head 平均规律的收益。除了全局均值，还应报告逐 layer 结果和 query-level p10，防止平均值掩盖少量严重失败的 query。

最终判断仍然来自端到端任务准确率和运行性能。一个 Router 即使具有较高的平均 attention mass，也可能在少数关键推理步骤上造成不可恢复的错误；同样，如果动态路由本身增加了过多延迟，也会抵消减少 Indexer heads 所带来的收益。因此最终实验必须联合报告 MISA set-oracle utility、learned utility、regret、静态 baseline、tail quality、下游准确率以及 decode latency。

## 9. 总结

该方案的核心思想是让两个阶段分别解决不同粒度的问题。MISA 利用当前 Indexer query、DSA gate 和粗粒度 KV summaries，以较低成本从 64 个 Indexer heads 中动态找出 8 个值得继续计算的 heads，并复用已有 affinity 对 coarse blocks 做分层剪枝。Assignment Router 不再预测一个脱离运行时的固定 head set，而是在这 8 个真实 candidate sets 上，根据当前 MLA pair query、当前 Indexer query、MISA importance 和 context 特征预测 utility。训练目标使用 soft utility distillation 与 expected regret，使优化过程直接关注“选择错误 candidate 会损失多少 attention mass”，而不只是“是否命中某个离散标签”。

通过这种设计，head selection、candidate generation、Router supervision 和实际部署路径保持一致，从根本上减少旧 set scorer 方案中的训练—推理错位。
