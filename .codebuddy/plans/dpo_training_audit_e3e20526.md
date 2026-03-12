---
name: dpo_training_audit
overview: 全面审查 DPO 训练流程，识别并修复所有潜在问题，包括 ref_model 同步、噪声共享、指标记录、训练稳定性等方面
todos:
  - id: fix-winner-double-forward
    content: 消除 lightning_module.py 中 winner 重复前向计算：重构 compute_dpo_loss 接收预计算的 input/output_winner，training_step 传入已有结果
    status: completed
  - id: fix-x3-dpo-dataset
    content: 修复 dpo_dataset.py 中 x3 数据缺失：_mol_to_hetero_data 独立生成 x3（去掉 and 'x2' in data_dict 条件，仿照 new_datasets.py 的两步生成逻辑）
    status: completed
  - id: fix-optimizer-ref-model
    content: 修复 configure_optimizers 排除 ref_model 参数，并在 params 中添加 dpo_sampling_every_n_epochs 配置
    status: completed
    dependencies:
      - fix-winner-double-forward
---

## 用户需求

对 DPO 训练流程中已发现的多个 Bug 进行系统性修复，确保训练正确、高效运行。

## 产品概述

SPD 项目是一个基于扩散模型的分子生成系统，使用 DPO（Direct Preference Optimization）对生成的分子进行偏好优化，使模型倾向于生成与参考分子具有更高 3D 表面形状相似度的分子。

## 核心修复内容

### P0 - 严重问题

**1. winner 分子在 training_step 中被前向计算两次（性能 + 正确性）**

在 `lightning_module.py` 的 DPO 训练分支中：

- 第一次：`forward_training(input_winner)` 用于计算标准损失
- 第二次：`compute_dpo_loss` 内部再次对 `batch_winner` 调用 `get_training_input_dict + model.forward`

导致每步 DPO 训练多消耗约 33% 的显存和计算量，且两次 forward 使用相同数据但结果可能因 dropout 等随机性略有不同，造成梯度不一致。

**2. DPODataset 中 x3 条件数据始终缺失（compute_x3=True, compute_x2=False）**

`dpo_dataset.py` 第282行：`if self.params['dataset'].get('compute_x3', False) and 'x2' in data_dict` — 因 compute_x2=False 导致 `'x2' in data_dict` 永远为 False，x3 表面静电势点云数据始终不生成。x3 作为模型条件输入参与编码，缺失时模型接收到的是空张量，导致条件编码不一致，影响生成质量。

### P1 - 中等问题

**3. configure_optimizers 中 ref_model 参数被纳入优化器**

`self.parameters()` 返回所有参数（含 ref_model），虽然 `requires_grad=False` 的参数不会被更新，但 Adam 优化器会为其分配一阶/二阶矩的状态内存，ref_model 与 model 等体量，造成约 2x 的优化器内存浪费。

**4. params 缺少 dpo_sampling_every_n_epochs 配置，默认值过大**

采样间隔默认为 10 epoch，但小数据集（batch_size=2，仅 3 个分子）每 10 epoch 才重采样一次，容易对少数偏好对过拟合，应加入显式配置并设置合理值（如 5）。

## 技术栈

- Python + PyTorch + PyTorch Lightning（现有项目）
- torch_geometric（HeteroData / Batch）
- 现有模块：`src/shepherd/lightning_module.py`、`src/shepherd/dpo_dataset.py`、`training/DPO1_0_surfOnly.py`、`training/parameters/params_x1x3x4_dpo_finetune_nps.py`

---

## 实现思路

### 修复1：消除 winner 的重复前向计算

**策略**：将 `compute_dpo_loss` 的签名改为同时接受预计算好的 `input_winner`、`output_winner`（来自 `training_step` 已有的计算结果），内部不再重复 `get_training_input_dict + model.forward(winner)`，仅对 loser 和 ref_model 做前向传播。

**关键点**：

- `compute_dpo_loss` 新增参数 `input_winner_precomputed, output_model_winner_precomputed`
- 内部直接复用传入的结果，跳过 winner 的 `get_training_input_dict` 和 `self.model.forward(input_winner)`
- `training_step` 传递已有的 `input_winner, output_winner`（由 `forward_training` 返回）
- 同时消除标准损失和 DPO 损失的 winner 计算不一致问题（两次计算可能因 dropout 不一致）

### 修复2：DPODataset 中 x3 数据生成

**策略**：仿照 `new_datasets.py` 中 `__getitem__` 的 x3 生成逻辑（先 `get_x2_data` 生成表面点云，再 `get_x3_data_electrostatics_only` 计算 ESP），在 `dpo_dataset.py` 的 `_mol_to_hetero_data` 中独立生成 x3，无需依赖 `data_dict['x2']`。

**关键点**：

- 条件改为 `if self.params['dataset'].get('compute_x3', False):`（去掉 `and 'x2' in data_dict`）
- 生成 x3 所需的原子坐标 `atom_centers` 从 x1 的已处理坐标获取（若 x1 已生成），否则使用 `mol_coordinates`
- 调用链：`get_x2_data(radii, atom_centers, ...)` → `get_x3_data_electrostatics_only(charges, atom_centers, x3_data, x3_pos, ...)`
- 用零电荷数组（`np.zeros(mol.GetNumAtoms())`）作为默认电荷（与当前代码保持一致）
- x3 数据的 `pos` 字段来自 `get_x2_data` 返回的 `x3_pos`，正确传入 `get_x3_data_electrostatics_only`

### 修复3：优化器排除 ref_model 参数

**策略**：`configure_optimizers` 中显式过滤掉 `ref_model` 的参数，只把 `self.model` 及其他可训练参数传给 Adam。

```python
trainable_params = [
    p for name, p in self.named_parameters()
    if not name.startswith('ref_model.')
]
optimizer = torch.optim.Adam(trainable_params, lr=self.lr)
```

**说明**：不改变外部 `apply_freeze_strategy` 的逻辑（它直接设置 `requires_grad=False`），此修改只影响优化器状态的内存分配，行为完全等价但节省约 50% 优化器内存。

### 修复4：params 添加 dpo_sampling_every_n_epochs

**策略**：在 `params['training']` 字典中显式添加 `'dpo_sampling_every_n_epochs': 5`，与 `dpo_ramp_up_epochs: 10` 配合，确保每轮 DPO 权重斜坡上升期间有足够的采样更新频率。

---

## 实现注意事项

1. **`compute_dpo_loss` 签名变更**：接口层面变化，`training_step` 是唯一调用方，同步修改即可，无其他调用点。
2. **x3 生成中的 `get_atomic_vdw_radii`**：已在 `dpo_dataset.py` 顶部区域以按需 import 的方式引入（compute_x2 块），x3 块需同样引入或提升到函数顶部（参考 compute_x2 的 import 位置）。
3. **`get_x3_data_electrostatics_only` 签名差异**：`new_datasets.py` 版本的签名为 `(charges, charge_centers, data, pos, t, alpha_dash_t, sigma_dash_t)`（6参数，无 `virtual_node_mask`），与 `datasets.py` 旧版不同，需对齐使用正确的版本签名。
4. **不破坏向后兼容**：`compute_dpo_loss` 保留旧参数 `shared_noise, shared_timestep`（接口兼容），仅新增两个预计算参数，或通过 `**kwargs` 兼容旧调用。

---

## 目录结构

```
SPD/
├── src/shepherd/
│   ├── lightning_module.py          # [MODIFY]
│   │   ├── configure_optimizers     → 过滤 ref_model 参数
│   │   ├── compute_dpo_loss         → 新增 input_winner/output_winner 参数，消除重复前向
│   │   └── training_step (dpo)      → 传入 input_winner/output_winner 给 compute_dpo_loss
│   └── dpo_dataset.py               # [MODIFY]
│       └── _mol_to_hetero_data      → 修复 x3 生成逻辑，独立于 x2
└── training/parameters/
    └── params_x1x3x4_dpo_finetune_nps.py  # [MODIFY]
        └── training 字典             → 添加 dpo_sampling_every_n_epochs: 5
```