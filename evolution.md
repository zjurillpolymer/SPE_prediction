# Evolution — 踩坑记录

> PyCharm 全局共享，随项目扩容。分类 MOC 管理。

---

## Index ⊶ MOC

| # | 分类 | 踩坑点 | 速览 | 定位 |
|---|------|--------|------|------|
| 1 | Data | [归一化 split 顺序 → leak](#归一化-split-顺序--leak-风险) | 先 split 再 norm，否则 test 被 train 污染 | `spe_prediction.py:55-68` |
| 2 | Data | [SMILES polymer end cap](#smiles-特殊原子--polymer-end-cap) | `[Cu][Au]` 不在默认元素列表，one-hot 全落 other | `molecule_to_graph.py:8` |
| 3 | Data | [阴离子特征提取](#阴离子特征提取) | 离子液体可能无显式阴离子 fragment；3D volume 可能失败 | `dataloader.py:26-28` |
| 4 | Graph | [RDKit → PyG 双向边](#rdkit--pyg-无向图) | MessagePassing 默认单向，不加双向边表达能力减半 | `molecule_to_graph.py:83-85` |
| 5 | Graph | [原子特征维度一致性](#原子特征维度一致性) | 改 ATOM_TYPES 忘同步 node_in_dims → shape 不匹配 | `MPNN.py:41` |
| 6 | Graph | [键特征维度一致性](#键特征维度一致性) | 改 BOND_TYPES 忘同步 edge_dim | `molecule_to_graph.py:52-62` |
| 7 | Model | [MessagePassing message() 签名](#messagepassing-的-message-签名) | `x_i, x_j` 由 PyG 自动注入，额外参数需走 `**kwargs` | `MPNN.py:31` |
| 8 | Model | [Arrhenius 物理约束输出层](#arrhenius-物理约束输出层) | `inv_temp` 不参与归一化，否则破坏物理关系 | `spe_prediction.py:39-41` |
| 9 | Model | [额外特征拼接位置](#额外特征拼接位置) | extra 是全局特征，pooling 后拼，不能在 node level 拼 | `spe_prediction.py:36-37` |
| 10 | Training | [Gradient Clipping](#gradient-clipping) | 6 层 MPNN 消息传递梯度易爆，不加偶现 NaN | `spe_prediction.py:104` |
| 11 | Training | [loss scale 转换](#loss-scale-转换) | 归一化 MSE 转回原尺度需 `× y_std²`，忘则结果无意义 | `spe_prediction.py:130` |
| 12 | Training | [Cache 序列化兼容](#cache-文件的序列化兼容性) | 改 feature 定义后必须手动删 cache，不自检 | `dataloader.py:58-73` |
| 13 | Viz | [字体不支持中文](#matplotlib-默认字体不包含西里尔字母中文) | macOS serif → Times Roman，中文需指定字体路径 | `generate_spe_figure.py:9` |
| 14 | Viz | [numpy ↔ list 类型转换](#numpy-array--list-类型转换) | matplotlib 某些 API 不接受 ndarray | `generate_spe_figure.py` |
| 15 | Viz | [双轴 spines 残留](#双-y-轴--第二-x-轴的-spines-显示) | `twiny()` 后多一条 top spine 需手动隐藏 | `generate_spe_figure.py:113-118` |
| 16 | Env | [Boss 直聘反爬](#boss-直聘反爬) | SPA + JS cookie + IP 封禁，Playwright headless 也挡不住 | — |
| 17 | Env | [DeepSeek 不支持多模态](#autofigure--deepseek) | API 不能传图，改 matplotlib 代码生成 | `generate_spe_figure.py` |
| 18 | Env | [conda 系统级依赖](#conda-环境管理) | Cairo 需 conda install，pip 不够 | — |
| 19 | Env | [torch.load weights_only](#torchload-的-weights_only) | PyTorch 2.6+ 默认 True，cache 加载需设 False | `dataloader.py:60` / `compute_ea.py:40` |
| 20 | Data | [SMILES→PSMILES 转换](#smilespsmiles-转换) | `[Cu]`/`[Au]` → `[*]`，RDKit 验证 + canonicalize | `polybert/smiles_to_psmiles.py` |
| 21 | Env | [polyBERT 模型下载](#polybert-模型下载) | HF 被墙，用 `xushijie/polyBERT` 替代 `kuelumbus/polyBERT` | `train_polybert.py:15` |
| 22 | Training | [预训练编码器训更久更好](#预训练编码器的训练策略) | frozen + 500 epoch cosine 比 fine-tune 实用，loss 从 0.59→0.46 | `train_polybert.py` |
| 23 | Model | [冻结编码器 + 简单投影头](#冻结编码器--简单投影头-vs-深度回归头) | 数据少时简单头泛化更好，Ea 物理更合理 | `train_polybert.py:65` |

### 快速分类入口

- [→ 数据处理 (Data)](#数据处理-data)
- [→ 图构建 (Graph)](#图构建-graph)
- [→ 模型架构 (Model)](#模型架构-model)
- [→ 训练与调优 (Training)](#训练与调优-training)
- [→ 可视化 (Viz)](#可视化-viz)
- [→ 环境与工具 (Env)](#环境与工具-env)

---

## 数据处理 (Data)

### 归一化 split 顺序 —— leak 风险
`clean_train_data.csv` 里是全部数据，必须先 split 再计算 mean/std，否则 validation/test 的归一化参数被训练数据"看见"了。
- `spe_prediction.py:55-68` ✓ 正确做法：先 permute split，再从 train_df 算 mean/std
- `compute_ea.py:30-31` 推理脚本中虽然用了全体数据，但仅用于展示，不涉及模型评估，可接受

### SMILES 特殊原子 —— polymer end cap
聚合物用 `[Cu]` 和 `[Au]` 作端基 dummy 原子，**不在 RDKit 常见元素列表里**。
- `molecule_to_graph.py:8` `ATOM_TYPES` 必须包含 `79, 29`（Au, Cu）
- 否则 one-hot 编码全部落入 "other" 类别，信息丢失

### 阴离子特征提取
`compute_anion_features()` 靠 `GetFormalCharge < 0` 找阴离子片段：
- `dataloader.py:26-28` 遍历 `GetMolFrags` 取第一个带负电的
- **坑**：某些盐的 SMILES 写法（如离子液体）可能没有显式的阴离子 fragment，此时返回全零向量
- **坑**：3D conformer 生成可能失败（`EmbedMolecule` 返回 -1），需 try-except，失败时 volume 置 0

---

## 图构建 (Graph)

### RDKit → PyG 无向图
`molecule_to_graph.py:83-85` 每条 bond 生成 `[u,v]` 和 `[v,u]` 两个方向。
- MessagePassing 的 `flow='source_to_target'`（默认值）只从 src→dst 发消息
- 不加双向边，消息只能沿一个方向传播，表达能力受限

### 原子特征维度一致性
`atom_features()` 输出 31 维（11 + 6 + 1 + 5 + 3 + 1 + 1 + 1 + 2 = 31）
- `ATOM_TYPES` 有 12 种 → 11 one-hot + 1 other = 12 ✓（仔细算了一遍）
- 修改 `ATOM_TYPES` 时一定要同步更新 `node_in_dims`

### 键特征维度
`bond_features()` 输出 6 维（4 bond type one-hot + 1 conjugated + 1 in ring）
- 改 `BOND_TYPES` 时要同步改 `edge_dim`
- 注意 `Chem.rdchem.BondType` 枚举：SINGLE=1, DOUBLE=2, TRIPLE=3, AROMATIC=12

---

## 模型架构 (Model)

### MessagePassing 的 message() 签名
`MPNN.py:31` `def message(self, edge_attr, x_i, x_j):`
- `x_i` 和 `x_j` 由 PyG 自动注入，**不能显式传参**
- `edge_attr` 必须和 `propagate()` 调用时的命名一致
- 踩坑：如果想在 message 里用全局特征，需要通过 `forward()` 的 `**kwargs` 传递

### Arrhenius 物理约束输出层
`spe_prediction.py:39-41` 模型输出 `[A, Ea/R]`，计算 `log σ = A - (Ea/R) · 1/T`
- `inv_temp` 保持原始值（1/K），**不参与归一化**，否则 Arrhenius 物理关系被破坏
- `spe_prediction.py:73` 注释强调："inv_temp stays RAW"
- 推理时 `compute_ea.py:90` 同理，反归一化只在最后对 `log_sigma` 做

### 额外特征拼接位置
`MPNNEmbedding` 输出 graph embedding（经过 `global_mean_pool`）后再拼 `extra` 特征
- `spe_prediction.py:36-37` `emb = self.encoder(data)` → `torch.cat([emb, data.extra], dim=-1)`
- **不要**在 node level 拼 extra 特征（extra 是全局的，不是逐原子的）

---

## 训练与调优 (Training)

### Gradient Clipping
`spe_prediction.py:104` `clip_grad_norm_(model.parameters(), max_norm=1.0)`
- MPNN 在深层（6层）消息传递时梯度可能爆炸
- 不加 clipping 偶尔会遇到 NaN loss

### loss scale 转换
`spe_prediction.py:130` `test_loss * y_std**2` 把归一化的 loss 映射回原始尺度
- 因为 `y_norm = (y - y_mean) / y_std`，MSELoss 在归一化空间，反推原空间需要乘以 `y_std**2`
- 容易忘的一步，尤其在汇报结果时

### Cache 文件的序列化兼容性
`dataloader.py:58-73` Dataset 用 `torch.save` 缓存预处理结果。
- 如果 `molecule_to_graph.py` 的 feature 定义改了，**必须手动删 cache**
- cache 不会自动检测上游变化
- `weights_only=False` 用于 cache 文件（包含 data 对象），模型权重用 `weights_only=True`

---

## 可视化 (Viz)

### matplotlib 默认字体不包含西里尔字母/中文
`generate_spe_figure.py:9` 设置 `'font.family': 'serif'`
- macOS 上 'serif' 通常映射到 Times Roman，不支持中文
- 如果需要中文标注，需指定中文字体路径：`plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']`

### numpy array ↔ list 类型转换
`generate_spe_figure.py` 某版本踩坑：matplotlib 某些 API 要求 list，传入 numpy array 时报错
- 通用规则：给 `ax.set_xticks()`、`ax.set_xticklabels()` 等传参时，用 `list()` 包裹
- 或者在构建时就保证是 list 而非 ndarray

### 双 y 轴 / 第二 x 轴的 spines 显示
`generate_spe_figure.py:113-118` `ax2.twiny()` 后多余的 top spine
- `ax2_top.spines['top'].set_visible(False)` 隐藏
- 如果不隐藏，双 spine 重叠，图表不干净

---

## 环境与工具 (Env)

### Boss 直聘反爬
通过 Playwright 模拟浏览器访问 Boss 直聘：
- 静态 HTTP 请求拿不到数据（JS SPA）
- 直接调内部 API 返回 code 37（环境检测），连续请求后升级为 code 35（IP 封禁）
- Playwright headless Chromium 也会被检测，需要更多指纹伪装
- 最后结论：Boss 直聘反爬很强，常规手段很难稳定

### AutoFigure + DeepSeek
DeepSeek Chat API **不支持多模态视觉输入**
- image 文件无法传给 DeepSeek，需要改用 matplotlib 代码生成方案
- `generate_spe_figure.py` 就是这种场景下的产物

### conda 环境管理
- `autofigure` 环境自行安装了 Python 3.11 + Cairo 库
- AutoFigure-Edit 依赖系统级 Cairo 库，pip install 不够，还需 conda install cairo
- 经验：科学绘图类工具经常依赖 C 扩展库，单纯 pip 容易缺系统依赖

### torch.load 的 weights_only
- PyTorch 2.6+ 默认 `weights_only=True`，加载非权重 pickle 会报错
- 模型权重加载：`weights_only=True`（安全）
- cache 数据加载：`weights_only=False`（必须，因为包含 Data 对象）

### SMILES→PSMILES 转换
polyBERT 使用 PSMILES 格式（`[*]` 表示连接点），但数据集用的是带 `[Cu]`/`[Au]` 端基的 SMILES。

**转换策略**：`polybert/smiles_to_psmiles.py`
- 直接字符串替换：`smiles.replace('[Cu]', '[*]').replace('[Au]', '[*]')`
- RDKit 验证有效性：`Chem.MolFromSmiles(psmiles)`
- `canonicalize_psmiles` 库做规范化（减少重复单元）
- 229/229 全部转换成功

### polyBERT 模型下载
- 原模型 `kuelumbus/polyBERT` 在 HF 上，但国内网络 huggingface.co 连接超时
- 备选方案：`xushijie/polyBERT`——同一个 polyBERT 的镜像，可正常下载
- 也试过 `hf-mirror.com`，同样连不上
- `sentence-transformers` 版本 5.5.1 用 MPS 后端时有 lazy loading 问题，改用 `transformers.AutoModel` 直接加载解决

### 预训练编码器的训练策略
polyBERT 冻结编码器 + 简单 MLP 头，500 epoch 余弦退火后 test loss 从 0.5932 降到 **0.4600**，超过 MPNN baseline 的 0.54。

| 尝试 | Epoch | Test loss | 问题 |
|------|-------|-----------|------|
| 简单头 (600→128→64→2) | 100 | 0.5932 | 还没收敛 |
| Deep 头 (600→512→256→128→64→2) | 500 | 0.5995 | 过拟合，PP 的 Ea 反常 |
| 简单头，500 epoch + cosine annealing | 500 | **0.4600** | 收敛充分，Ea 物理合理 |

余弦退火对收敛很关键——原始固定 LR 在 epoch 100 处 plateau 了，cosine 让 LR 平滑下降后 val loss 继续改善到 epoch 400+。

### 冻结编码器 + 简单投影头 vs 深度回归头
在 ~28k 数据点上，frozen polyBERT（600 维 embedding）+ 简单两层投影头效果最好：
- 浅层头（85k 参数）：test loss 0.4600，Ea 物理正确
- 深层头（1.4M 参数）：test loss 0.5995，PP 的 Ea 低于 PEG（物理错误）

**教训**：预训练 embedding 容量已足够，在有限数据上简单模型的正则化效果更好。Fine-tune DeBERTa 最后 2 层理论上可以进一步提升，但训练速度慢 1000 倍（~30min/epoch vs ~1s/epoch），实际工程上不划算。
