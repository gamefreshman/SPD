# 离散特征扩散排查报告

## 排查范围

| 检查项 | 涉及文件 | 结论 |
|--------|----------|------|
| 训练前向加噪 | `new_datasets.py` | ✅ 内部自洽 |
| 损失函数 | `lightning_module.py` | ✅ 正确 |
| 推理去噪 | `inference.py` | ⚠️ 1个当前冲突 + 1个历史BUG已修复 |
| scale_atom_features 一致性 | 全链路 | ⚠️ x1一致；x4存在训练/推理分叉 |
| 噪声调度映射 | 全链路 | ✅ 设计合理 |
| 边际分布处理 | `sample_NP.py` / `inference.py` | ⚠️ x1/bond正确；x4是回避冲突的 workaround |
| Origin vs SPD 差异 | `datasets.py` vs `new_datasets.py` | 📝 预期差异 |
| 模型输入值域 | 全链路 | ⚠️ x1一致；x4不完全一致 |
| DPO 损失函数 | `lightning_module.py` | ✅ 正确 |
| **DPO 数据集噪声注入** | **`dpo_dataset.py`** | **🔴 发现3个严重BUG** |
| **DPO 在线采样** | **`dpo_utils.py`** | **🔴 发现1个严重BUG** |
| **参考模型更新** | **`lightning_module.py` / `callbacks.py`** | **⚠️ 未实现 EMA** |
| **DPO shared noise 语义** | **`dpo_dataset.py` / `lightning_module.py`** | **⚠️ 注释与实现不完全一致** |
| 后验采样公式 | `inference.py` | ✅ 正确 |
| 边际分布计算缓存 | `new_train.py` | ✅ 正确 |
| 混合 DataLoader | `mixed_dataloader.py` | ✅ 逻辑正确 |
| 推理虚拟节点重置 | `inference.py` | ⚠️ x1特征未重置 |

---

## 复核口径（2026-04-08）

本报告已按当前工作区源码再次复核，并补充如下口径：

1. **以当前代码为准，不以历史修复记录为准**。如果文档、历史报告和源码冲突，本报告优先记录当前树里的真实状态。
2. **区分“当前仍存在的 BUG”和“历史上出现过、但当前树已修复的问题”**。特别是 `harmonize` 的离散 jump 问题，在当前 `inference.py` 中已经修掉，不能再按“当前 bug”计入。
3. **区分“数学/实现正确”和“训练-推理分布一致”**。某条路径即使能跑通，也不代表与训练时的数据分布一致。`x4` 当前正属于这种“可运行但存在分叉”的情况。
4. **区分“workaround”和“最终正确设计”**。例如 `sample_NP.py` 不传 `pharm_marginals` 的做法，当前能绕开冲突，但它是回避性策略，不是严格的全链路一致解。

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

## 三、推理代码 (`inference.py`) — ⚠️ 1个当前冲突 + 1个历史BUG已修复

### 历史 BUG 1：`forward_jump` 对离散 one-hot 特征使用高斯噪声（当前树已修复）

**旧问题**：早期版本里，`harmonize jump` 对离散特征直接调用连续版 `forward_jump`，会把 one-hot 特征打成高斯连续值。

**当前状态**：已修复。当前源码在 harmonize 分支中：
- x1 位置仍走连续 `forward_jump`
- x1 原子类型、键类型改走 `discrete_forward_jump`
- x4 类型在 `pharm_marginals is not None` 时也走 `discrete_forward_jump`

**代码证据**：
```python
x1_pos_t, x1_t_jump = forward_jump(...)
x1_x_t, x1_t_jump = discrete_forward_jump(...)
x1_bond_edge_x_t, x1_t_jump = discrete_forward_jump(...)
...
if pharm_marginals is not None:
    x4_x_t, x4_t_jump = discrete_forward_jump(...)
```

**结论**：这条应从“当前 bug”降级为“历史 bug，当前树已修复”。如果未来再次回退到连续 `forward_jump`，才会重新出现。

---

### BUG 2：x4 离散去噪与 inpainting 连续轨迹冲突（当前仍存在，调用侧用 workaround 回避）

**位置**：`inference.py` line 1592-1633

当 `pharm_marginals is not None` 且 `inpaint_x4_type=True` 时：
- inpainting 在每步用连续轨迹覆盖 `x4_x_t`（float 值）
- 离散后验采样期望 `x4_x_t` 是 one-hot
- 连续 ↔ one-hot 反复切换导致数值崩溃

**当前状态**：
- `sample_NP.py` 不传 `pharm_marginals`，所以当前采样脚本确实绕开了这条冲突
- 但这只是 **回避**，不是“严格正确”
- 因为训练侧 `x4` 类型本身已经是离散 one-hot 加噪，而这里推理退回成连续 DDPM 路径，训练-推理分布并不一致

**更准确的表述**：应写成“已通过调用侧 workaround 回避，但代价是 x4 推理路径与训练路径分叉”，而不是“完全正确”。

---

## 四、scale_atom_features 一致性

| 环节 | x1 atom (scale=0.25) | x1 bond (scale=1.0) | x4 pharm (scale=2.0) |
|------|----------------------|----------------------|----------------------|
| 训练 x_0 | ✅ scaled | ✅ 纯 one-hot | ✅ scaled |
| 训练 x_forward_noised (真实节点) | one-hot (0/1) | one-hot (0/1) | one-hot (0/1)，不是连续高斯 |
| 训练 虚拟节点 | scaled (0.25) | N/A | scaled (2.0) |
| 推理 虚拟节点 | scaled (0.25) | N/A | scaled (2.0) |
| 推理 真实节点 | one-hot (0/1) | one-hot (0/1) | `pharm_marginals!=None` 时 one-hot；否则连续值 |

**更精确的结论**：
- **x1 链路是一致的**
- **x4 训练链路是离散 one-hot，加噪后真实节点仍是 one-hot**
- **x4 推理链路是否一致，取决于是否传 `pharm_marginals`**
  - 传入：更接近训练分布，但会和 `inpaint_x4_type` 冲突
  - 不传：可运行，但回退成连续值路径，与训练分布分叉

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

**pharm_marginals**: `sample_NP.py` 不传此参数，默认 `None`。这在当前实现里是一个 **workaround**：
- 好处：避免 `inpaint_x4_type=True` 时的连续轨迹 / 离散后验冲突
- 代价：x4 type 推理退回连续 DDPM，而训练端 x4 type 是离散 one-hot 加噪

**结论**：
- `atom_marginals` / `bond_marginals` 的处理是正确的
- `pharm_marginals` 在 `sample_NP.py` 中“不传”不是遗漏，但也不能简单记为“完全正确”，应记为“为了避开当前实现冲突而采用的回避策略”

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

## 八、模型输入值域一致性（x1 一致；x4 分叉）

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

### x4 药效团类型输入的额外说明

这一点是原报告最容易被误读的地方，单独说明如下：

**训练**：
```
data['x4'].x_forward_noised
```
值域：
- 虚拟节点：scaled one-hot（如 `[2.0, 0, ...]`）
- 真实节点：离散扩散采样后的 one-hot（0/1）

**推理**：
```
x4_x_t → input_dict['x4']['decoder']['x']
```
值域分两种：
- `pharm_marginals is not None`：真实节点是 one-hot，和训练更接近
- `pharm_marginals is None`：真实节点是连续值，走回退 DDPM 路径

**修订后的结论**：
- `x1` 输入值域训练 ≡ 推理，这条判断正确
- `x4` 不能写成“训练 ≡ 推理”，只能写成“取决于推理分支；当前 `sample_NP.py` 使用的是回退连续分支”

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

### BUG 4：timestep 存储截断导致模型时间嵌入错误（致命，影响范围比原文更广）

**位置**：
- `new_datasets.py` line 616（x1）
- `new_datasets.py` line 808（x2/x3 载体）
- `new_datasets.py` line 939（x4）

```python
data['timestep'] = torch.tensor([t], dtype=torch.long)
# 当 t=0.75 (DPO float) → torch.tensor([0.75], dtype=torch.long) = tensor([0])
# 当 t=300  (标准 int)  → torch.tensor([300],  dtype=torch.long) = tensor([300])
```

**影响范围补充**：
- 在当前 NPs DPO 配置里，`compute_x1=True, compute_x3=True, compute_x4=True`，因此至少 **x1 / x3 / x4** 的时间嵌入都会被错误喂成 `t=0`
- 如果未来启用 `x2`，x2 也会受同样影响

**影响**：DPO 批次中模型的时间嵌入（Time Embedding）被系统性错误压到最干净端。结合 BUG 3（离散噪声也接近 t=0），这不是单一模态的小偏差，而是整批 DPO 数据的条件标签错位。

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

**问题本质**：
- 这条判断“存在 bug”是对的
- 但旧版报告中“正确值 ≈ 0.316 / 0.949”的举例需要谨慎，因为当前代码并没有真正存在一个 `x1_pos_diffuser` 来定义“float 0.75 应该如何映射到连续 schedule”
- 在当前参数文件中，`0.316 / 0.949` 更接近离散索引 `t≈216` 的连续调度值，而不是 `t=300`

**因此这一节更准确的写法应是**：
- 回退路径 `alpha_dash_t = 1 - timestep`, `sigma_dash_t = timestep` **肯定不对**
- 正确实现应当 **显式复用 `noise_schedule_dict['x1']` 的查表结果**，而不是手写线性替代
- 数值示意应标明“取决于 float→schedule 的映射约定”，避免给出会被误解为精确 ground truth 的单一数值

**影响**：DPO 训练中连续特征（原子坐标）的噪声水平也不正确。线性插值与实际 cosine-linear 混合调度的偏差随时间步变化，导致加噪分布与标准训练不一致。

### 补充观察 S1：`shared_noise` 语义与注释不一致（中等）

**位置**：
- `dpo_dataset.py`
- `lightning_module.py compute_dpo_loss`

**现象**：
- `compute_dpo_loss` 的注释写的是“shared_noise 和 shared_timestep 已由 dataset 端保证”
- 但 `DPODataset.get()` 实际只共享了 `shared_timestep`
- winner / loser 使用的是不同 seed
- 返回的 `shared_noise` 还是空字典

**结论**：
- 这不一定是致命 bug，因为 DPO 未必必须共享逐元素噪声
- 但它确实意味着 **注释、接口语义和实际行为不完全一致**
- 后续若要继续主打“shared noise mapping”这个说法，应该要么真正共享噪声，要么把注释改掉

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

### 需要修复的 BUG（当前树）

| # | 严重度 | 文件 | 描述 | 当前影响 |
|---|--------|------|------|----------|
| 3 | 🔴🔴 致命 | `dpo_dataset.py` | DPO 时间步 float[0,1] 传入期望 int[1,T] 的接口，离散扩散始终 t≈0 | **DPO 离散通道完全失效** |
| 4 | 🔴🔴 致命 | `new_datasets.py` | `torch.tensor([float], dtype=long)` 截断为 0，x1/x3/x4（以及未来的 x2）时间嵌入错误 | **DPO 条件标签系统性错位** |
| 5 | 🔴 严重 | `dpo_dataset.py` | `x1_pos_diffuser` 属性不存在，连续噪声回退为线性插值替代 | **DPO 连续噪声分布错误** |
| 6 | 🔴 严重 | `dpo_utils.py` | OnlineSampler 传入 pharm_marginals 导致 x4 连续↔离散冲突 | **在线采样有效率 -97%** |

### 需要关注的设计问题

| # | 级别 | 文件 | 描述 | 建议 |
|---|------|------|------|------|
| E | ⚠️ 中 | `callbacks.py` | `new_train.py` 路径无 ref_model 更新，与 Doc EMA 描述不符 | 实现 EMA 或 iterative hard-replace |
| F | ⚠️ 低 | `inference.py` | x1 虚拟节点类型特征在离散采样后未重置为 scaled 值 | 添加 `x1_x_t[vn_mask] = x1_x_t_init[vn_mask]` |
| G | ⚠️ 中 | `dpo_dataset.py` / `lightning_module.py` | `shared_noise` 注释语义与实际实现不一致，当前只共享 timestep 不共享噪声 | 统一接口语义或补齐实现 |
| H | ⚠️ 中 | `inference.py` / `sample_NP.py` | x4 通过“不传 pharm_marginals”回避冲突，但因此与训练路径分叉 | 明确将其记录为 workaround，而非最终一致方案 |

### 历史问题（当前树已修复）

| # | 级别 | 文件 | 描述 | 当前状态 |
|---|------|------|------|----------|
| 1 | ✅ 已修复 | `inference.py` | `harmonize jump` 曾对离散 one-hot 特征使用连续 `forward_jump` | 当前已改为 `discrete_forward_jump` |

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
- ✅ x1 的 `scale_atom_features` 链路在训练/推理之间一致
- ✅ 边际分布计算和缓存（`compute_and_cache_marginals`）
- ✅ 混合 DataLoader 交替逻辑（`mixed_dataloader.py`）
- ✅ DPO batch 检测和分派（`training_step`）
- ✅ `sample_NP.py` 不传 `pharm_marginals` 能有效回避当前 x4 冲突
- ✅ OnlineSamplingCallback 偏好对构建逻辑
- ✅ PreferencePairBuilder 分数差筛选
- ✅ DPOSamplingScheduler 种子选择策略

**但需补充限定语**：
- 上面这条关于 `sample_NP.py` 的判断，准确说法应是“有效 workaround”，而不是“严格意义上的全链路正确”

### BUG 3/4/5 联合影响分析

**DPO 训练的一个 batch 中，偏好对数据存在三重错误叠加**：

```
正确状态（应该是）:
  timestep = 一个中高噪声时间步   # 例如 float 0.75 应被正确映射到配置 schedule
  alpha_dash_t = 来自 noise_schedule_dict 的查表值
  离散噪声 = 与该时间步匹配的正常转移概率

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

**影响范围再补充一句**：
- 这里的 “timestep = 0” 不只是 x1 的本地时间嵌入错误，在当前配置里还会同步污染 x3 / x4 的时间条件编码
- 因此它更像是“整条 DPO 条件化输入被错误标注”，而不是单个 head 的离散通道 bug

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
