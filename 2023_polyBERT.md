---
type: paper
field: AI4Science, Polymer Informatics, Transformer
year: 2023
journal: Nature Communications
authors: Christopher Kuenneth, Rampi Ramprasad
doi: 10.1038/s41467-023-39868-6
---
# polyBERT: a chemical language model to enable fully machine-driven ultrafast polymer informatics

## Claim
提出一个完全端到端的 ML-driven 聚合物信息学管线。核心是 polyBERT（DeBERTa-based Transformer），在 1 亿条假设聚合物上预训练，生成的 fingerprint 用于 29 个性质的预测。速度比手工指纹快 215 倍，精度持平/超越。

## Method
- **架构**：DeBERTa（解耦注意力 + 相对位置编码）
- **预训练**：MLM on 100M 假设聚合物（BRICS 碎片枚举组合）
- **Fingerprint**：对所有 token 的最后一层 hidden state 做平均池化
- **性质预测**：multitask DNN，按性质类别分组训练
- **共聚物处理**：组成加权求和 F = Σ(F_i × c_i)

## Dataset
- **预训练**：100M 假设聚合物（BRICS 分解 13,766 条已知聚合物 → 4,424 个碎片 → 枚举组合）
- **性质预测**：35,517 数据点，29 个性质（热学、电子、力学、光学等）
- 约 80% 均聚物，20% 共聚物

## Interesting Ideas
1. **100M 假设聚合物预训练**：用 BRICS 碎片组合生成训练数据，化学空间覆盖远超实验数据
2. **速度碾压**：1.06 ms/聚合物/GPU，比 Polymer Genome 快 215 倍——适合超大规模虚拟筛选
3. **端到端**：从 PSMILES 到性质预测完全无需人工特征工程，可部署在云端
4. **无监督 fingerprint**：学习到的 fingerprint 独立于下游性质（与 GNN 不同），可一次计算多次使用
5. **环境效益**：预测 1 亿聚合物仅排放 5.5 kg CO₂
6. **Attention 可视化**：与 TransPolymer 结论一致——对角线 attention 强（邻域化学键），远端弱

## Reproducibility
- 代码和预训练权重开源（论文引用 anonymous repo）
- 1 亿假设聚合物预测结果对学术界开放

## Impact
将聚合物信息学的速度提升了两个数量级，使大规模虚拟筛选变得实际可行。代表了"完全告别化学特征工程"的极端——模型仅从 PSMILES 字符串自学一切。与 Ramprasad 团队之前的手工指纹（Polymer Genome）工作形成鲜明对比。

## Critical Comparison
参见 `concepts/Polymer Graph Representation in ML.md` 的三篇对比。

## Relations
- [[explains]] [[Polymer Graph Representation in ML]]
- [[contrasts]] [[2023_PolymerGNN_Multitask_Property_Learning]]
- [[related-to]] [[2023_TransPolymer]]
