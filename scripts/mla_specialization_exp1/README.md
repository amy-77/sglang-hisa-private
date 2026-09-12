# MLA specialization：Experiment 1

本目录只回答 Q1：在同一 token 预算下，MLA 是否受益于不同 head groups 使用不同的 KV 集合。Q2 candidate assignment、sharing-budget sweep、kernel 性能和 router 训练均未推进。

## 对照与目标

对 head 的真实 dense attention 分布 `alpha = softmax(qK * scale)`，每组 oracle 按 `mean_heads(alpha)` 排序。平均 logits 的 Top-K 不一定最大化组内平均 attention mass。

每个 query 分别比较 K=720 和 K=2048 下的四种选择：

1. DSA weighted Indexer shared Top-K（从重新计算的完整分数排序）。
2. 全 128 个 MLA heads 共享的 mass oracle。
3. 固定 TP-local 16-head 组的 mass oracle，共 8 组。
4. 固定相邻 2-head pair 的 mass oracle，共 64 pairs，仅作诊断上界。

每个组均保持原 head 顺序。离线分析集中读取 head 数据不改变部署的分组，也不引入运行时跨 GPU head 搬运。

`pair oracle - shared MLA oracle` 区分 specialization 与单纯更好的 token 排名；`shared MLA oracle - DSA` 衡量共享选择器的改进空间。group16 的收益决定当前部署约束是否仍保留可用空间。

同时记录 pair-vs-pair Top-K intersection/K、各 head/group attention mass，以及选择集合重归一化后的输出误差。输出误差为经过各 head 的 Wv 后、wo 前的 pooled relative L2，不是完整层输出或任务准确率。mass oracle 并非 output oracle。

## 当前数据覆盖与限制

旧 pilot_v2 和 exp4–7 保存了候选集 mass 汇总，缺少完整 MLA attention 或全历史 latent K/PE，无法还原本实验的 oracle。旧 Indexer logits 不能替代 MLA attention。

本轮新增采集使用真实 DeepSeek-V3.2 权重和 RULER 128K 数据的 `qa_1`、`niah_multikey_1`、`vt` 各第一条 prompt。在 4K/8K/16K/32K/64K/约128K 的每个端点抽取最后 4 个 query，共 72 个 query、3 个独立 prompt。末档实际前缀长度详见 summary.json，未补齐成 131072 tokens。

这些数据**仅覆盖 layer 0 的 teacher-forced prompt prefixes**。第一层输入可以准确地由 embedding→attn_norm 得到，不需要运行后续层。按 reference decode 代数重放，使用 FP8 模拟 latent cache、BF16 absorbed Q，FP32 logits/softmax 和误差统计；不是当前 SGLang BF16 KV 配置的等价重放。没有生成式 decode，没有中后层，没有任务 accuracy 评估。

RoPE/softmax 配置固定为 checkpoint 的 163840 上限，不随测试的可见前缀长度变化。初始 `smoke` 只用于 I/O 验证，配置不同，**未混入正式结果**。

同一 prompt 的邻近 query 高度相关；72 个 query 不能视为 72 个独立 prompt，也不能据此给出可靠的跨 prompt 置信区间。

## 文件与复现

- `scripts/collect_layer0_specialization.py`：独立采集第一层 absorbed Q、全历史 latent K/PE、Wv、排序后的 DSA IDs、样本来源和模型配置。
- `scripts/mla_specialization_metrics.py`：独立 NumPy 指标模块，适用于补齐后任意层的 attention probabilities。
- `scripts/analyze_specialization_captures.py`：逐 query 分析、保存 oracle indices、绘制 mass 曲线和完整 64×64 pair overlap 矩阵。
- `tests/test_mla_specialization_metrics.py`：概率均值与 logits 均值反例、共享与独立 oracle 的数值关系、已知 overlap、causal/duplicate indices 检查、稀疏输出重归一化。

远程代码位于 `/DATA/disk0/qyl/code/dpskv32/sglang-hisa/scripts/mla_specialization_exp1/`；数据和结果位于 `/DATA/disk0/qyl/data/mla_specialization_exp1_20260911/`。

```bash
# h20-9-57，sglang-hisa 目录
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 /root/miniconda3/envs/dpskv32/bin/python \
  scripts/mla_specialization_exp1/collect_layer0_specialization.py \
  --out /DATA/disk0/qyl/data/mla_specialization_exp1_20260911/captures

OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python \
  scripts/mla_specialization_exp1/analyze_specialization_captures.py \
  --captures /DATA/disk0/qyl/data/mla_specialization_exp1_20260911/captures \
  --out /DATA/disk0/qyl/data/mla_specialization_exp1_20260911/results

# 本地本目录
python3 -m unittest discover -s tests -v
```

## Q1 的判定仍待补齐

第一层结果只能决定是否值得继续采集，不能通过或否决完整 Q1。下一步仍是 Experiment 1：在真实全模型轨迹上补齐 layer 2/30/60 的若干独立 prompt 与 decode prefixes，保持相同 K、分组和数值配置。每个 query 需保留全部 8 ranks 的原始顺序 Q、同一请求/时刻的完整历史 latent K/PE、scale、DSA IDs（若比较多个 K，应保存排序分数或各 K 的集合）、必要的 V 投影和完整 provenance。

只有固定 group16 相对 shared MLA oracle 的 mass/输出收益在代表性数据上成立，才进入 Q2。低 overlap 本身不能证明收益；不能用这次 layer0 的结果启动 router 训练。
