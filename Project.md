# SPD — ShEPhERD 多模态扩散分子生成 + DPO 偏好优化

## 项目概述

**SPD (ShEPhERD with DPO)** 是一个基于 [ShEPhERD](https://github.com/coleygroup/shepherd)（Diffusing **Sh**ape, **E**lectrostatics, and **Ph**armacophores for **D**rug Design）的扩展项目。核心思路：

1. **预训练阶段** — 使用多模态异构图扩散模型，联合建模分子的 4 种表示（原子图、分子表面、静电势、药效团），从天然产物（NPs）数据集上学习分子分布。
2. **DPO 微调阶段** — 通过 Direct Preference Optimization（直接偏好优化），让模型在采样时偏向生成与参考分子更相似（Surface / ESP / Pharmacophore）的 3D 分子构型。

## 目录结构

```
SPD/
├── src/                         # 核心源码
│   ├── shepherd/                # 模型、数据集、推理逻辑
│   │   ├── model/               # 模型定义（EquiformerV2 + EGNN）
│   │   │   └── model.py         # 多模态异构图模型主文件（~1600行）
│   │   ├── lightning_module.py   # PyTorch Lightning 训练/DPO 损失逻辑
│   │   ├── inference.py          # 推理采样（扩散去噪、inpainting、harmonization）
│   │   ├── new_datasets.py       # 标准训练数据集（HeteroDataset）
│   │   ├── dpo_dataset.py        # DPO 偏好对数据集 + collate_dpo_batch
│   │   ├── dpo_utils.py          # DPO 辅助函数
│   │   └── extract.py            # 从模型输出提取 RDKit 分子
│   └── score/                   # Shepherd Score 评分库
│       └── shepherd_score/
│           ├── score/            # 3D 相似度评分
│           │   ├── gaussian_overlap_np.py      # Surface（形状）相似度
│           │   ├── electrostatic_scoring_np.py  # ESP（静电势）相似度
│           │   └── pharmacophore_scoring_np.py  # Pharmacophore（药效团）相似度
│           ├── container.py      # Molecule 容器（封装表面点、ESP、药效团）
│           └── evaluations/      # ConfEval / ConditionalEvalPipeline
├── training/                    # 训练脚本 & 参数
│   ├── new_train.py             # 标准扩散预训练脚本
│   ├── DPO1_0_surfOnly.py       # DPO 训练：仅优化 Surface Similarity
│   ├── DPO1_0_triSim.py         # DPO 训练：三指标（Surf×5 + ESP×3 + Pharm×2）
│   ├── DPO1_0_partlyFrozen.py   # DPO 训练：部分冻结 + 综合指标
│   ├── visualize_dpo_metrics.py  # DPO 训练指标可视化
│   ├── parameters/               # 模型/训练参数配置
│   │   └── params_x1x3x4_dpo_finetune_nps.py
│   └── jobs/                     # 训练产出（checkpoint、日志、指标）
├── evaluation/                  # 评估流程
│   └── experiment_SamEval/
│       ├── sample_NP.py          # 天然产物采样脚本（支持断点续传）
│       ├── eval_unified.py       # 统一评估（ConfEval + ConditionalEval）
│       └── union_eval.ipynb      # 评估结果 Notebook
├── data/
│   ├── conformers/               # 分子构象数据（pkl 格式）
│   └── shepherd_chkpts/          # 预训练模型 checkpoint
└── web/                         # Gradio Web 可视化界面
    ├── app.py                    # 主应用（支持采样 + 对接 + 可视化）
    └── backend_utils.py          # 后端工具函数
```

## 模型架构

### 多模态表示

模型同时建模分子的 4 种互补表示，每种表示作为异构图的一种节点类型：

| 模态 | 标识 | 内容 | 扩散类型 |
|------|------|------|----------|
| **x1** | 原子图 | 原子类型 + 3D 坐标 + 化学键 | 离散（类型）+ 连续（坐标） |
| **x2** | 分子表面 | 范德华表面点云（~75 点） | 连续（坐标） |
| **x3** | 静电势 | 表面点上的库仑势 | 连续（标量场） |
| **x4** | 药效团 | 药效团类型 + 位置 + 方向 | 离散（类型）+ 连续（位置/方向） |

### 网络结构

```
输入: 4 种模态的噪声数据 (t 时刻)
  │
  ├─ x1 Encoder (EquiformerV2)  ──┐
  ├─ x3 Encoder (EquiformerV2)  ──┤  各自独立编码
  └─ x4 Encoder (EquiformerV2)  ──┘
                                   │
                          ┌────────▼────────┐
                          │ Joint Hetero     │  跨模态信息交互
                          │ Graph Encoder    │  (EquiformerV2)
                          │ (6 层)           │
                          └────────┬────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
              x1 Decoder     x3 (隐式)     x4 Decoder
              (EGNN+E3NN     通过x2路由    (EGNN+E3NN
               +MLP)                        +MLP)
                    │                            │
              预测: 原子类型                 预测: 药效团类型
                    坐标噪声                     位置噪声
                    键类型                       方向噪声
```

- **EquiformerV2**: SO(3)-等变 Transformer，处理球谐展开的 l=0（标量）和 l=1（向量）特征
- **EGNN**: 用于坐标去噪（E(3)-等变）
- **E3NN**: 用于混合标量和向量通道的张量积

### DPO 训练

DPO（Direct Preference Optimization）在预训练模型基础上进行微调：

```
对同一参考分子采样 N 个生成分子
        │
        ▼
  评估 3D 相似度（Surface / ESP / Pharmacophore）
        │
        ▼
  构建偏好对: (winner, loser) — winner 是相似度更高的分子
        │
        ▼
  DPO 损失: L = -log σ(β · (log π(w) - log π_ref(w) - log π(l) + log π_ref(l)))
        │
        ▼
  总损失 = (1 - α) · L_标准去噪 + α · L_DPO
```

**三个 DPO 变体**：

| 脚本 | 评分策略 | 用途 |
|------|----------|------|
| `DPO1_0_surfOnly.py` | `total_score = Surface` | 仅优化形状 |
| `DPO1_0_triSim.py` | `total_score = Surf×5 + ESP×3 + Pharm×2` | 三指标联合优化 |
| `DPO1_0_partlyFrozen.py` | 三指标 + SA/LogP 惩罚 | 综合优化（含化学性质约束） |

**部分冻结策略**：DPO 微调时冻结 Encoder，只训练 Joint Encoder 最后 2 层、全局处理模块和 Decoder。

## 评估指标

### 构象评估（ConfEval）
- 分子有效性（is_valid）
- SA Score（合成可及性）
- LogP（亲脂性）

### 条件评估（ConditionalEval）— 3D 相似度

| 指标 | 函数 | 含义 |
|------|------|------|
| Surface Similarity | `get_overlap_np()` | 生成分子与参考分子的表面形状重叠度 |
| ESP Similarity | `get_overlap_esp_np()` | 静电势分布的重叠度（需 MMFF partial charges） |
| Pharmacophore Similarity | `get_overlap_pharm_np()` | 药效团特征点的 Tanimoto 重叠度 |

## 训练流程速览

```bash
# 1. 标准预训练（扩散模型）
cd training
python new_train.py params_x1x3x4_diffusion_mosesaq_20240824 42

# 2. DPO 微调（三指标优化，基于 NPs 数据集）
python DPO1_0_triSim.py params_x1x3x4_dpo_finetune_nps 42

# 3. 可视化 DPO 训练指标
python visualize_dpo_metrics.py jobs/33/x1x3x4_dpo_finetune_nps/dpo_round_metrics.json

# 4. 采样 & 评估
cd ../evaluation/experiment_SamEval
python sample_NP.py   # 采样生成分子
python eval_unified.py # 统一评估
```

## 关键技术细节

### 已解决的重要 Bug

1. **`collate_dpo_batch` 未生效**：PyG 的 `DataLoader` 会覆盖自定义 `collate_fn`，改用 `torch.utils.data.DataLoader`。
2. **`hasattr(batch, 'x1')` 在 HeteroDataBatch 上返回 False**：即使 `node_types` 包含 `'x1'`。改用 `'x1' in batch.node_types`。
3. **DPO 损失始终为 N/A**：上述两个 Bug 导致 `training_step` 从未进入 DPO 分支。

### DPO 训练参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `beta_dpo` | 0.2 | KL 散度约束强度 |
| `dpo_max_weight` | 0.5 | DPO 损失最大混合权重 |
| `dpo_ramp_up_epochs` | 5 | DPO 权重线性递增的 epoch 数 |
| `dpo_sampling_every_n_epochs` | 10 | 每 N 个 epoch 重新采样偏好对 |
| `lr` | 1e-5 | 学习率（比预训练低 10x） |

## 依赖环境

- Python 3.9
- PyTorch + PyTorch Geometric
- EquiformerV2 / e3nn
- RDKit
- Shepherd Score（内置于 `src/score/`）
- PyTorch Lightning
- Open3D

环境管理使用 **UV**。
