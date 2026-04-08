# 离散特征扩散排查报告

## 排查范围

| 检查项 | 涉及文件 | 结论 |
|--------|----------|------|
| 训练前向加噪 | `new_datasets.py` | ✅ 内部自洽 |
| 损失函数 | `lightning_module.py` | ✅ 正确 |
| 推理去噪 | `inference.py` | ⚠️ 发现2个BUG |
| scale_atom_features 一致性 | 全链路 | ✅ 一致 |
| 噪声调度映射 | 全链路 | ✅ 设计合理 |
| 边际分布处理 | `sample_NP.py` / `inference.py` | ✅ 正确 |
| Origin vs SPD 差异 | `datasets.py` vs `new_datasets.py` | 📝 预期差异 |
| 模型输入值域 | 全链路 | ✅ 训练≡推理 |
| DPO 损失函数 | `lightning_module.py` | ✅ 正确 |
| **DPO 数据集噪声注入** | **`dpo_dataset.py`** | **🔴 发现3个严重BUG** |
| **DPO 在线采样** | **`dpo_utils.py`** | **🔴 发现1个严重BUG** |
| **参考模型更新** | **`lightning_module.py` / `callbacks.py`** | **⚠️ 未实现 EMA** |
| 后验采样公式 | `inference.py` | ✅ 正确 |
| 边际分布计算缓存 | `new_train.py` | ✅ 正确 |
| 混合 DataLoader | `mixed_dataloader.py` | ✅ 逻辑正确 |
| 推理虚拟节点重置 | `inference.py` | ⚠️ x1特征未重置 |

---

## 一、训练数据处理 (`new_datasets.py`)

### 数据流追踪

```
原子类型索引 → one-hot [0,1,0,...] → ×scale(0.25) → data['x'] / data['x_0']
                                                           ↓
                                    apply_noise(x_0[~vn], t) → matmul(scaled_onehot, Qtb) → normalize → multinomial → F.one_hot → pure one-hot [0,1,0,...]
                                                           ↓
                                    data['x_forward_noised'] = {虚拟节点: scaled(0.25), 真实节点: one-hot(0/1)}
```

### 关键发现

1. **`apply_noise` 输入是 scaled one-hot (0.25)**：`data['x_0']` 保存的是 scaled 值。`apply_noise` 内部做 `features_0 @ Qtb`，行和=0.25 而非1.0，但后续有归一化保护（line 330），**不影响采样正确性**。

2. **`x_forward_noised` 值域混合**：
   - 虚拟节点：`[0.25, 0, 0, ...]`（来自 `x_0` 的 clone，未被覆盖）
   - 真实节点：`[0, 1, 0, ...]`（来自 `apply_noise` 返回的 `F.one_hot`）

3. **键特征**：`scale_bond_features = 1.0`，所以 bond 的 `x_0` 和 `x_forward_noised` 都是纯 one-hot。

**结论：训练前向加噪逻辑内部自洽。** 模型看到的 x 输入是 "虚拟节点 scaled + 真实节点 one-hot" 的混合。

---

## 二、损失函数 (`lightning_module.py`)

### 标准去噪损失

| 特征 | 损失类型 | 预测目标 | 标签来源 |
|------|----------|----------|----------|
| x1 位置 | MSE | 噪声 ε | `pos_noise`（真实高斯噪声） |
| x1 原子类型 | CrossEntropy | x_0 logits | `argmax(x_0)` → 类别索引 |
| x1 键类型 | CrossEntropy | x_0 logits | `argmax(bond_x_0)` → 类别索引 |
| x4 位置 | MSE | 噪声 ε | `pos_noise` |
| x4 方向 | MSE | 噪声 ε | `direction_noise` |
| x4 类型 | CrossEntropy | x_0 logits | `argmax(x_0)` → 类别索引 |

**注意**：`true_atom_types_t0 = x_0` 是 scaled one-hot (0.25)，但 `argmax([0.25, 0, 0, ...]) = 0` 仍然正确。

### DPO 损失

DPO 损失函数结构正确：
- 连续特征（位置）：比较 winner/loser 的 MSE 损失差
- 离散特征（类型）：比较 winner/loser 的 CrossEntropy 损失差
- 使用 Bradley-Terry 模型 + sigmoid + log 计算最终 DPO loss

**结论：损失函数无 BUG。**

---

## 三、推理代码 (`inference.py`) — ⚠️ 发现2个BUG

### BUG 1：`forward_jump` 对离散 one-hot 特征使用高斯噪声（严重）

**位置**：`inference.py` line 1055-1056

```python
x1_x_t, x1_t_jump = forward_jump(x1_x_t, x1_t, harmonize_jump, x1_sigma_ts, ...)
x1_bond_edge_x_t, x1_t_jump = forward_jump(x1_bond_edge_x_t, x1_t, harmonize_jump, x1_sigma_ts, ...)
```

**问题**：`forward_jump` 内部执行 `x_jump = alpha * x + sigma * randn()`，对 one-hot 编码施加高斯噪声，破坏离散结构。跳跃后 `x1_x_t` 变成类似 `[0.12, 0.87, -0.03, ...]` 的连续值，而模型期望的是 one-hot。

**影响范围**：仅在 `harmonize=True` 时触发。当前 `sample_NP.py` 使用 `harmonize=False`，**暂不影响当前采样**。

**修复方案**：harmonize jump 后需要对离散特征重新执行离散扩散采样，而非高斯加噪。

```python
# 修复方案伪代码
if harmonize:
    # 对连续特征使用高斯 forward_jump
    x1_pos_t, x1_t_jump = forward_jump(x1_pos_t, ...)
    
    # 对离散特征使用离散扩散 forward_jump
    x1_x_t = discrete_forward_jump(x1_x_t, x1_atom_diffuser, x1_t, harmonize_jump)
    x1_bond_edge_x_t = discrete_forward_jump(x1_bond_edge_x_t, x1_bond_diffuser, x1_t, harmonize_jump)
```

---

### BUG 2：x4 离散去噪与 inpainting 连续轨迹冲突（已知，已回避）

**位置**：`inference.py` line 1592-1633

当 `pharm_marginals is not None` 且 `inpaint_x4_type=True` 时：
- inpainting 在每步用连续轨迹覆盖 `x4_x_t`（float 值）
- 离散后验采样期望 `x4_x_t` 是 one-hot
- 连续 ↔ one-hot 反复切换导致数值崩溃

**当前状态**：`sample_NP.py` 不传 `pharm_marginals`，默认 `None`，降级为连续 DDPM 去噪。**已正确回避。**

---

## 四、scale_atom_features 一致性

| 环节 | x1 atom (scale=0.25) | x1 bond (scale=1.0) | x4 pharm (scale=2.0) |
|------|----------------------|----------------------|----------------------|
| 训练 x_0 | ✅ scaled | ✅ 纯 one-hot | ✅ scaled |
| 训练 x_forward_noised (真实节点) | one-hot (0/1) | one-hot (0/1) | 连续高斯噪声 |
| 训练 虚拟节点 | scaled (0.25) | N/A | scaled (2.0) |
| 推理 虚拟节点 | scaled (0.25) | N/A | scaled (2.0) |
| 推理 真实节点 | one-hot (0/1) | one-hot (0/1) | 连续值 |

**结论：训练和推理的 scale 处理一致。** 虚拟节点都用 scaled one-hot，真实节点在离散扩散后都变成纯 one-hot。

---

## 五、噪声调度映射 — ✅ 设计合理

### 发现：两种调度的 alpha_bar 曲线高度一致

| 特征类型 | 噪声调度 | 来源 |
|----------|----------|------|
| x1 位置（连续） | `0.65*cosine + 0.35*linear` 混合调度 | `params['noise_schedules']['x1']` |
| x1 原子类型（离散） | `linear: 1e-4 → 0.02` | `PredefinedNoiseScheduleDiscrete` |
| x1 键类型（离散） | `linear: 1e-4 → 0.02` | `PredefinedNoiseScheduleDiscrete` |
| x4 位置（连续） | `0.65*cosine + 0.35*linear` 混合调度 | `params['noise_schedules']['x4']` |
| x4 类型（连续/离散） | 取决于是否传 `pharm_marginals` | 两种可能 |

### 量化对比

| t | 连续 α̅ | 离散 ᾱ | 比值 | 离散 I(x₀;x_t)/H(x₀) |
|---|---------|--------|------|----------------------|
| 50 | 0.938 | 0.936 | 0.997 | 83.6% |
| 100 | 0.782 | 0.773 | 0.989 | 56.2% |
| 200 | 0.373 | 0.362 | 0.971 | 14.5% |
| 300 | 0.100 | 0.102 | 1.023 | 1.7% |
| 400 | 0.008 | 0.017 | 2.117 | 0.07% |

**alpha_bar 曲线在 t=1~300 范围内比值始终在 0.97~1.02 之间，几乎完全对齐。**

### 为什么信息保留率不同但设计合理

离散扩散终态收敛到训练集边际分布 m（如 "45% H, 30% C, 10% N, ..."），而非无信息状态。
即使 ᾱ→0 时，P(碳保持碳) = m_C ≈ 30%，但这 30% 是边际先验的基础概率，不是未被破坏的信号。

**这是 DiGress 的核心设计**：通过边际分布作为终态先验，注入"什么原子类型更常见"的化学先验，
使生成偏向训练集中常见的原子组合。

**结论：不需要修改噪声调度。两者的参数化曲线高度一致，信息保留率的表面差异是离散扩散的本质特性。**

---

## 六、边际分布处理

### sample_NP.py 中的边际分布

```python
atom_marginals_x1[0] = 0.0  # 将虚拟节点(None)类型的概率清零
atom_marginals_x1 = atom_marginals_x1 / atom_marginals_x1.sum()  # 重新归一化
```

**bond_types**: `[None, 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']`，其中 None (index 0) 表示"无键"。大多数原子对之间无键，所以 `bond_marginals[0]` 应占主导地位，这是**正确的**。

**pharm_marginals**: `sample_NP.py` 不传此参数，默认 `None`。x4 作为条件模态使用连续 DDPM 去噪，**符合设计意图**。

**结论：边际分布处理正确。**

---

## 七、Origin_Shepherd vs SPD 的关键差异

| 方面 | Origin_Shepherd (`datasets.py`) | SPD (`new_datasets.py`) |
|------|--------------------------------|-------------------------|
| 原子类型噪声 | 连续高斯：`α*x + σ*noise` | 离散扩散：转移矩阵 + multinomial |
| 模型输入值域 | 连续 float（如 0.17, 0.08） | 离散 one-hot（0 或 1） |
| 损失函数 | MSE（预测噪声） | CrossEntropy（预测 x_0 类别） |
| 键类型噪声 | 连续高斯 | 离散扩散 |

**影响**：如果使用 Origin_Shepherd 预训练的 checkpoint 微调 SPD，模型的 embedding 层从处理连续值切换到处理 one-hot 值，是**根本性的分布迁移**。线性 embedding 对 one-hot 输入等效于 lookup table，功能上可行但需要充分重训练。

---

## 八、模型输入值域一致性（训练 vs 推理）

### x1 原子特征传入模型的路径

**训练**：
```
data['x1'].x_forward_noised → input_dict['x1']['decoder']['x'] → model.x1_decoder_encoder_embedding(x)
```
值域：虚拟节点 `[0.25, 0, ...]`，真实节点 `[0, 1, 0, ...]`

**推理**：
```
x1_x_t → input_dict['x1']['decoder']['x'] → model.x1_decoder_encoder_embedding(x)
```
值域：虚拟节点 `[0.25, 0, ...]`，真实节点 `[0, 1, 0, ...]`（来自 `F.one_hot`）

### x1 键特征传入模型的路径

**训练**：`data['x1','bond','x1'].x_forward_noised` → one-hot (0/1)
**推理**：`x1_bond_edge_x_t` → one-hot (0/1)（来自 `F.one_hot`）

**结论：模型输入值域在训练和推理之间完全一致。** ✅

---

## 九、DPO 数据集 (`dpo_dataset.py`) — 🔴 发现3个严重 BUG

### BUG 3：时间步类型不匹配导致离散扩散完全失效（致命）

**位置**：`dpo_dataset.py` line 177, 248 → `new_datasets.py` line 758, 292-308

**问题链条**：

```
DPODataset._mol_to_hetero_data:
  shared_timestep = np.random.uniform(0, 1)        # → float, 如 0.75
  ↓
get_x1_data(mol, t=0.75, alpha_dash_t, sigma_dash_t)
  ↓
  t_tensor = torch.tensor([0.75])                   # float tensor
  ↓
DiscreteFeatureDiffusion.apply_noise(features_0, t_int=tensor([0.75]), device)
  t_int_tensor = torch.tensor([tensor([0.75])])     # → tensor([[0.75]])
  ↓
get_params_for_t(tensor([[0.75]]), device):
  t_int = torch.clamp(0.75, 1, 400)                # → 1.0 (被强制 clamp 到最小值!)
  t_float = 1.0 / 400 = 0.0025                     # → 几乎是 t=0 的噪声水平!
```

**标准训练路径对比**：
```
HeteroDataset.__getitem__:
  ts = np.arange(1, 401)                            # → [1, 2, ..., 400]
  t = np.random.choice(ts)                          # → 整数, 如 300
  ↓
get_x1_data(mol, t=300, alpha_dash_t, sigma_dash_t)
  t_tensor = torch.tensor([300])                    # int tensor
  ↓
get_params_for_t(tensor([[300]]), device):
  t_int = clamp(300, 1, 400) = 300                  # → 正确!
  t_float = 300 / 400 = 0.75                        # → 正确的噪声水平
```

**影响**：DPO 训练中**所有偏好对的离散特征**（原子类型、键类型）**几乎没有噪声**（始终 t_float≈0.0025），离散扩散通道在 DPO 训练中**完全失效**。模型在 DPO 批次中看到的离散输入几乎是干净的，而标准批次中看到的是正常噪声水平，导致严重的训练不一致。

---

### BUG 4：timestep 存储截断导致模型时间嵌入错误（致命）

**位置**：`new_datasets.py` line 616

```python
data['timestep'] = torch.tensor([t], dtype=torch.long)
# 当 t=0.75 (DPO float) → torch.tensor([0.75], dtype=torch.long) = tensor([0])
# 当 t=300  (标准 int)  → torch.tensor([300],  dtype=torch.long) = tensor([300])
```

**影响**：DPO 批次中模型的时间嵌入（Time Embedding）**始终为 t=0**，模型误以为处于最干净状态。结合 BUG 3（离散噪声也接近 t=0），DPO 训练的离散通道在时间嵌入和实际噪声两方面都完全错位。

---

### BUG 5：连续噪声调度回退为错误的线性插值（严重）

**位置**：`dpo_dataset.py` line 222-231

```python
# 意图：从 HeteroDataset 获取正确的噪声调度参数
if hasattr(self.data_generator, 'x1_pos_diffuser'):   # ← 始终 False!
    noise_schedule = self.data_generator.x1_pos_diffuser.noise_schedule
    alpha_t = noise_schedule.get_alpha_bar(t_normalized).item()
    ...
else:
    # 回退到简单线性插值（错误！）
    alpha_dash_t = 1.0 - timestep       # t=0.75 → α̅=0.25
    sigma_dash_t = timestep              # t=0.75 → σ̅=0.75
```

**为什么始终 False**：`HeteroDataset` 没有 `x1_pos_diffuser` 属性。它只有 `x1_atom_diffuser`（离散）和 `x1_bond_diffuser`（离散）。连续噪声参数存储在 `noise_schedule_dict` 字典中，不是对象属性。

**实际 vs 回退对比**（以 t=0.75 为例）：

| 参数 | 回退值（错误） | 实际值（正确） | 偏差 |
|------|---------------|---------------|------|
| α̅ (alpha_dash) | 0.25 | ≈0.316 | -21% |
| σ̅ (sigma_dash) | 0.75 | ≈0.949 | -21% |

**影响**：DPO 训练中连续特征（原子坐标）的噪声水平也不正确。线性插值与实际 cosine-linear 混合调度的偏差随时间步变化，导致加噪分布与标准训练不一致。

---

## 十、DPO 在线采样 (`dpo_utils.py`) — 🔴 发现1个严重 BUG

### BUG 6：OnlineSampler 传入 pharm_marginals 导致采样崩溃

**位置**：`dpo_utils.py` line ~401（`_sample_one_molecule` 方法）

```python
result = inference_sample(
    ...
    inpaint_x4_type=True,           # x4 作为条件
    pharm_marginals=self.pharm_marginals,  # ← 不应传入!
    ...
)
```

**已知影响**：当 `inpaint_x4_type=True` 且 `pharm_marginals is not None` 时：
1. 初始化从连续高斯 → 离散 one-hot
2. 去噪步骤从连续 DDPM → 离散后验采样（multinomial）
3. 每步 inpainting 又用连续轨迹覆盖 → 连续值 ↔ one-hot 反复切换

**实测数据**：有效样本从 1519 → 44（**-97%**），已在采样脚本中回滚（`sample_NP.py` 不传 `pharm_marginals`）。

**但 OnlineSampler 中未修复**：DPO 在线采样仍然传入 `pharm_marginals`，导致每个 epoch 的在线采样生成极差，偏好对质量极低。

---

## 十一、参考模型更新 — ⚠️ 未实现 Doc 所述 EMA

### Doc 描述 vs 实际实现

**Doc (`02_DPO_Finetuning_Methodology.md`) 描述**：
> 冻结参考模型 Reference Model EMA: ref_model = 0.99 * ref_model + 0.01 * model

**`new_train.py` + `callbacks.py` (OnlineSamplingCallback) 实际行为**：
- `ref_model` 在 `load_state_dict` 中初始化为预训练权重
- **此后再无任何更新**
- `callbacks.py` 中的 `OnlineSamplingCallback` **不包含 ref_model 更新逻辑**

**`DPO1_0_triSim.py` (独立训练脚本) 行为**：
- 使用 **hard-replace**（非 EMA）：`ref_model.load_state_dict(model.state_dict())`
- 仅在当前模型分数超过历史最佳时更新
- 或超过 N 轮未更新时强制更新

**影响**：
- `new_train.py` 路径：ref_model 永远冻结在预训练权重，随训练推进越来越过时，DPO 约束可能逐渐失效或过强
- `DPO1_0_triSim.py` 路径：hard-replace 比 EMA 更激进，ref_model 可能跳变

---

## 十二、后验采样公式 — ✅ 正确

`compute_batched_over0_posterior_distribution` 实现了标准 D3PM 后验采样：

$$p(x_{t-1} | x_t, x_0) \propto p(x_t | x_{t-1}) \cdot p(x_{t-1} | x_0)$$

代码实现：
```python
numerator = (X_t @ Qt.T) * Qsb       # likelihood × prior
denominator = Qtb @ X_t.T             # evidence
posterior = numerator / denominator
```

然后通过模型预测 $p(x_0|x_t)$ 加权并边际化：
```python
weighted = pred_x0.unsqueeze(-1) * posterior_prob
prob = weighted.sum(dim=2)            # marginalize over x_0
sample = prob.multinomial(1)          # sample x_{t-1}
```

**结论：数学正确，与 DiGress 论文一致。** ✅

---

## 十三、推理虚拟节点特征重置 — ⚠️ 轻微不一致

**位置**：`inference.py` line 1696-1712

去噪循环中虚拟节点重置覆盖范围：
| 特征 | 重置 | 代码 |
|------|------|------|
| x1 位置 | ✅ | `x1_pos_t_1[vn_mask] = x1_pos_t[vn_mask]` |
| x1 原子类型 | ❌ 未重置 | 无对应代码 |
| x4 位置 | ✅ | `x4_pos_t_1[vn_mask] = x4_pos_t[vn_mask]` |
| x4 方向 | ✅ | `x4_direction_t_1[vn_mask] = x4_direction_t[vn_mask]` |
| x4 类型 | ✅ | `x4_x_t_1[vn_mask] = x4_x_t[vn_mask]` |

**问题**：离散后验采样后 `x1_x_t` 对所有节点更新为纯 one-hot。虚拟节点的 scaled one-hot（如 `[0.25, 0, ...]`）被覆盖为 unscaled 随机 one-hot（如 `[0, 0, 1, ...]`），后续步骤的模型输入中虚拟节点特征与训练时不一致。

**影响评估**：**较小**。虚拟节点输出已被 mask 置零（line 1403），不直接参与预测。但作为 GNN 输入节点，特征偏差可能通过消息传递间接影响其他节点。

---

## 十四、边际分布计算 (`new_train.py`) — ✅ 正确

`compute_and_cache_marginals` 函数：
- 并行遍历所有分子，统计原子/键/药效团类型频次
- 归一化为概率分布
- 结果缓存到磁盘，避免重复计算
- 包含空分布保护（fallback 为均匀分布）

**结论：实现正确，有缓存优化。** ✅

---

## 十五、混合 DataLoader (`mixed_dataloader.py`) — ✅ 逻辑正确

- `MixedDataLoader` 根据 `dpo_ratio` 交替提供标准/DPO 批次
- `update_dpo_dataset` 支持动态更新偏好对
- 偏好对为空时自动降级为纯标准训练
- 批次标记 `batch_type='dpo'/'standard'` 供 `training_step` 分派

**结论：DataLoader 混合逻辑正确。** 但其提供的 DPO 数据本身有 BUG 3/4/5 的问题。

---

## 十六、汇总与优先级建议

### 需要修复的 BUG

| # | 严重度 | 文件 | 描述 | 当前影响 |
|---|--------|------|------|----------|
| 3 | 🔴🔴 致命 | `dpo_dataset.py` | DPO 时间步 float[0,1] 传入期望 int[1,T] 的接口，离散扩散始终 t≈0 | **DPO 离散通道完全失效** |
| 4 | 🔴🔴 致命 | `new_datasets.py` | `torch.tensor([float], dtype=long)` 截断为 0，时间嵌入错误 | **DPO 模型时间感知完全错误** |
| 5 | 🔴 严重 | `dpo_dataset.py` | `x1_pos_diffuser` 属性不存在，连续噪声回退为线性插值 | **DPO 连续噪声分布错误** |
| 6 | 🔴 严重 | `dpo_utils.py` | OnlineSampler 传入 pharm_marginals 导致 x4 连续↔离散冲突 | **在线采样有效率 -97%** |
| 1 | ⚠️ 中 | `inference.py` | `forward_jump` 对离散 one-hot 特征使用高斯噪声 | 仅 `harmonize=True` 时触发 |

### 需要关注的设计问题

| # | 级别 | 文件 | 描述 | 建议 |
|---|------|------|------|------|
| E | ⚠️ 中 | `callbacks.py` | `new_train.py` 路径无 ref_model 更新，与 Doc EMA 描述不符 | 实现 EMA 或 iterative hard-replace |
| F | ⚠️ 低 | `inference.py` | x1 虚拟节点类型特征在离散采样后未重置为 scaled 值 | 添加 `x1_x_t[vn_mask] = x1_x_t_init[vn_mask]` |

### 设计备注（非 BUG，无需修改）

| # | 级别 | 描述 | 结论 |
|---|------|------|------|
| 1 | ✅ | 离散/连续噪声调度 alpha_bar 曲线高度一致（比值 0.97~1.02），信息保留率差异是离散扩散的本质特性（DiGress 设计） | 无需修改 |
| 2 | ✅ | `apply_noise` 输入是 scaled one-hot (0.25) 而非纯 one-hot | 有归一化保护，不影响正确性 |

### 已确认正确的部分

- ✅ 训练前向加噪逻辑（`new_datasets.py`）
- ✅ CrossEntropy 损失函数（x1 atom/bond, x4 pharm）
- ✅ MSE 位置损失函数
- ✅ DPO 损失函数公式（`lightning_module.py compute_dpo_loss`）
- ✅ 推理离散后验采样（D3PM 公式，`compute_batched_over0_posterior_distribution`）
- ✅ scale_atom_features 全链路一致
- ✅ 边际分布计算和缓存（`compute_and_cache_marginals`）
- ✅ 混合 DataLoader 交替逻辑（`mixed_dataloader.py`）
- ✅ DPO batch 检测和分派（`training_step`）
- ✅ sample_NP.py 不传 pharm_marginals（正确回避 x4 冲突）
- ✅ OnlineSamplingCallback 偏好对构建逻辑
- ✅ PreferencePairBuilder 分数差筛选
- ✅ DPOSamplingScheduler 种子选择策略

### BUG 3/4/5 联合影响分析

**DPO 训练的一个 batch 中，偏好对数据存在三重错误叠加**：

```
正确状态（应该是）:
  timestep = 300              # 中等噪声
  alpha_dash_t = 0.316        # cosine-linear 调度
  离散噪声 = t_float=0.75    # 中等转移概率

DPO 实际状态（三重错误）:
  timestep = 0                # BUG 4: 截断为 0
  alpha_dash_t = 0.25         # BUG 5: 线性回退
  离散噪声 = t_float=0.0025  # BUG 3: clamp 到最小值
```

**结果**：DPO 训练中模型看到的偏好对数据：
1. 时间嵌入说"我在 t=0（最干净）"
2. 连续噪声用了错误的 schedule（偏差 20%+）
3. 离散特征几乎是干净的（噪声水平 0.025%）

这导致 DPO 损失信号完全混乱 — 模型和 ref_model 在错误条件下比较 winner/loser 的预测能力，产生的梯度信号是无意义的。

---

## 十、用户释疑记录

### 1. 关于 `harmonize` 是否未启用以及是否是错误
**结论：这不是错误，`harmonize = False` 是正确的设定，且它完全不影响 `new_train.py` 和 DPO 训练。**

* **训练与 DPO 阶段**：`harmonize` 是一个**纯推理时（Inference-Only）**的启发式采样技巧（被称为协调跳跃 Harmonize Jump），即在反向去噪过程的某些时间步人为地向前跨越几步，以便让不同模态的生成进度重新对齐。因为它是专门为循环迭代采样（推理）设计的技巧，在 `new_train.py` 一次性前向加噪和 DPO 的损失计算中，**根本不存在也涉及不到这个参数**。
* **推理阶段(`sample_NP.py` 等)**：在原始 Origin Shepherd 中，`harmonize` 也是默认关闭的实验性功能。所以在推理脚本和 Web APP 界面中它默认是 `False`，这完全没问题。
* **为什么把它列为 BUG**：正因为目前整个框架默认关闭 `harmonize`，它**暂时不会报错**。但是代码 `inference.py` 中的 `forward_jump` 函数在底层写死了 `x = alpha * x + sigma * randn()`（连续高斯噪声公式）。如果以后在 UI 中打勾启用了 `harmonize`，这个连续加噪的公式就会强加在**离散 one-hot 特征**上，破坏特征结构并导致模型输出乱码。

### 2. 关于“离散和连续的噪声调度不同步”的设计风险
**结论：离散特征无法直接使用连续的高斯噪声参数，它们必须分别计算，但“进度不同步”会增加模型的学习难度。**

* **为什么离散不能直接套用“连续的噪声调度公式”？**
  连续的噪声调度分配的是高斯分布的**方差 $\sigma$**，用来控制坐标偏移的物理距离；而离散的噪声调度分配的是**转移概率**（矩阵 `Q` 里面的具体概率值），用来控制例如“有 1% 的概率从 C 变成 O”。这两者的数学机制完全不同，必须各自独立计算。`PredefinedNoiseScheduleDiscrete` (1e-4 -> 0.02) 的设计遵循了典型的离散扩散工作原理（如 DiGress 论文），是合理的。
* **为什么这会是一个风险？**
  风险出在**神经网络内部的学习难度**上。你可以这样直观地想象：
  假设目前时间步到了中期 `t = 200`（总步数 400）。
  此时，连续的高斯坐标调度使用的是 `cosine + linear` 混合调度，在这个阶段坐标可能已被加噪到一团糟，结构信息被大量破坏；
  而离散的原子类型调度用的是 `linear` 调度，在这个阶段原子类型还有相对较高的概率保留原样（实测 `alpha_bar ≈ 0.35`，原始信息依然明显）。
  **但是，在你的 UNet/Transformer 中，你只给模型喂进了一个统一的时间步编码（Time Embedding = 200）。**
  模型拿到这个统一的 `t=200` 后，它要去预测还原：一方面要还原高度模糊的坐标；同时另一方面要面对还算清晰的原子类型。要求网络凭借同一个时间步刻度，同时应对破坏进度（信息衰减速率）截然不同的两种模态变量，这在多模态扩散中是一个已知难题。它容易导致模型只学会了一种容易学的模态（比如坐标预测），而在另一种模态上表现糟糕（比如原子类型只依赖边际分布预测大量 C 和 H，这就是为什么你在评估报告见到了纯脂肪族问题）。这常常也是 DPO 微调急剧崩盘的原因，因为偏好学习可能过度强化了这种难度和梯度的失衡。
