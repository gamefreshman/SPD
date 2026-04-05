# 离散特征扩散排查报告

## 排查范围

| 检查项 | 涉及文件 | 结论 |
|--------|----------|------|
| 训练前向加噪 | `new_datasets.py` | ✅ 内部自洽 |
| 损失函数 | `lightning_module.py` | ✅ 正确 |
| 推理去噪 | `inference.py` | ⚠️ 发现2个BUG |
| scale_atom_features 一致性 | 全链路 | ✅ 一致 |
| 噪声调度映射 | 全链路 | ⚠️ 设计风险 |
| 边际分布处理 | `sample_NP.py` / `inference.py` | ✅ 正确 |
| Origin vs SPD 差异 | `datasets.py` vs `new_datasets.py` | 📝 预期差异 |
| 模型输入值域 | 全链路 | ✅ 训练≡推理 |
| DPO 损失函数 | `lightning_module.py` | ✅ 正确 |

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

## 五、噪声调度映射 — ⚠️ 设计风险

### 发现：离散和连续特征使用**不同的噪声调度**

| 特征类型 | 噪声调度 | 来源 |
|----------|----------|------|
| x1 位置（连续） | `0.65*cosine + 0.35*linear` 混合调度 | `params['noise_schedules']['x1']` |
| x1 原子类型（离散） | `linear: 1e-4 → 0.02` | `PredefinedNoiseScheduleDiscrete` |
| x1 键类型（离散） | `linear: 1e-4 → 0.02` | `PredefinedNoiseScheduleDiscrete` |
| x4 位置（连续） | `0.65*cosine + 0.35*linear` 混合调度 | `params['noise_schedules']['x4']` |
| x4 类型（连续/离散） | 取决于是否传 `pharm_marginals` | 两种可能 |

**问题**：同一个时间步 t，位置的噪声水平和原子类型的噪声水平不同。例如在 t=200（T=400）时：

- 连续调度 `alpha_dash_t` ≈ 0.3-0.5（较强噪声）
- 离散调度 `alpha_bar_t` ≈ 0.95+（非常弱的噪声）

这意味着在大部分时间步，**位置已经很模糊了，但原子类型几乎没变**。模型需要同时从一个 timestep embedding 中解码两种完全不同的噪声水平，这增加了学习难度。

**是否是 BUG**：这不是 BUG（训练和推理使用了一致的调度），而是**设计选择**。但如果离散调度太弱，模型可能无法有效学习原子类型的去噪。

**建议**：考虑将离散噪声调度的 beta 范围从 `[1e-4, 0.02]` 提升至与连续调度更匹配的水平，或者同步两者的 `alpha_bar` 曲线。

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

## 九、汇总与优先级建议

### 需要修复的 BUG

| # | 严重度 | 描述 | 当前影响 |
|---|--------|------|----------|
| 1 | 🔴 高 | `forward_jump` 对离散 one-hot 特征使用高斯噪声 | 仅 `harmonize=True` 时触发，当前采样脚本未触发 |

### 设计风险（非 BUG，但影响性能）

| # | 风险 | 描述 | 建议 |
|---|------|------|------|
| 1 | 🟡 中 | 离散/连续噪声调度不同步：位置已很模糊时原子类型几乎无噪声 | 考虑同步 alpha_bar 曲线 |
| 2 | 🟡 低 | `apply_noise` 输入是 scaled one-hot (0.25) 而非纯 one-hot | 有归一化保护，不影响正确性 |

### 已确认正确的部分

- ✅ 训练前向加噪逻辑
- ✅ CrossEntropy 损失函数（x1 atom/bond, x4 pharm）
- ✅ MSE 位置损失函数
- ✅ DPO 损失函数
- ✅ 推理离散后验采样（D3PM 公式）
- ✅ scale_atom_features 全链路一致
- ✅ 虚拟节点特征处理一致
- ✅ 边际分布计算和使用
- ✅ sample_NP.py 不传 pharm_marginals（正确回避 x4 冲突）
