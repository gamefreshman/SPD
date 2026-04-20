# DPO 微调代码迁移至 Shepherd 项目计划

将 SPD 项目的 DPO 微调功能迁移到 Shepherd 项目，核心适配点是 Shepherd 使用**全连续扩散**（所有特征均用 Gaussian noise + MSE loss），而 SPD 对原子/键/药效团类型使用离散扩散（Markov transition + CrossEntropy）。从零编写正确的 DPO 数据管道，避免迁入 SPD 的已知 BUG。

---

## 0. 背景与关键差异

| 维度 | SPD (源) | Shepherd (目标) |
|------|---------|----------------|
| x1 原子类型 loss | CrossEntropy (离散) | **MSE** (连续) |
| x1 键类型 loss | CrossEntropy (离散) | **MSE** (连续) |
| x4 药效团类型 loss | CrossEntropy (离散) | **MSE** (连续) |
| x1/x4 位置 & 方向 loss | MSE | MSE (不变) |
| 数据集噪声注入 | `DiscreteFeatureDiffusion` + 连续 | 全连续 (`alpha*x + sigma*noise`) |
| 推理去噪 | 离散后验采样 + DDPM | 全 DDPM |
| `true_atom_types_t0` 字段 | 存在 (离散 GT label) | **不存在** — 用 `x_noise` |
| `true_pharm_types_t0` 字段 | 存在 (离散 GT label) | **不存在** — 用 `x_noise` |
| lightning_module DPO | 有 `compute_dpo_loss`, `ref_model` | **不存在**，需新增 |
| 参数配置 (ckpt) | `params_x1x3x4_dpo_finetune_nps.py` | 基于 `params_x1x3x4_diffusion_mosesaq_20240824.py` |

**SPD BUG 不再适用**：BUG 3（离散扩散失效）、BUG 6（pharm_marginals 冲突）在全连续架构下自然消失。BUG 4（timestep 截断）、BUG 5（噪声调度回退）通过从零设计避免。

---

## 1. 新增文件清单

在 Shepherd 项目中新建以下文件：

| 文件 | 职责 |
|------|------|
| `src/shepherd/dpo_dataset.py` | DPO 偏好对数据集（全连续版） |
| `src/shepherd/dpo_utils.py` | OnlineSampler + ShepherdScorer + PreferencePairBuilder |
| `src/shepherd/mixed_dataloader.py` | MixedDPODataset + collate_mixed_batch |
| `training/dpo_train.py` | DPO 训练主脚本（含 DPOSamplingCallback） |
| `training/parameters/params_x1x3x4_dpo_finetune_nps.py` | DPO 超参数配置 |

**修改文件**：
| 文件 | 改动 |
|------|------|
| `src/shepherd/lightning_module.py` | 新增 `ref_model`、`compute_dpo_loss()`、修改 `training_step()` 支持 DPO batch |

---

## 2. 分步实施

### Phase 1: DPO LightningModule 扩展

**目标**：在 `src/shepherd/lightning_module.py` 中添加 DPO 训练能力。

**2.1 新增 `ref_model` 初始化** (`__init__`)
- 当 `params['training'].get('enable_dpo', False)` 时，创建 `self.ref_model = deepcopy(self.model)`
- 冻结 `ref_model` 全部参数：`p.requires_grad = False`
- 新增 `self.beta_dpo` 和 `self.dpo_optimize_x4` 等 DPO 超参数

**2.2 新增 `compute_dpo_loss()` — 全 MSE 版**

这是与 SPD 最大的不同点。SPD 版 `compute_dpo_loss` 混合了 MSE（位置）和 CrossEntropy（类型），Shepherd 版全部使用 **MSE**：

```
对每个 active 的 xi (x1, x3, x4)：
  对每个 output channel (pos_out, x_out, bond_edge_x_out, direction_out)：
    model_loss_w = MSE(model_pred_w, true_noise_w)
    model_loss_l = MSE(model_pred_l, true_noise_l)
    ref_loss_w   = MSE(ref_pred_w, true_noise_w)  # no_grad
    ref_loss_l   = MSE(ref_pred_l, true_noise_l)  # no_grad
    
    model_diff = model_loss_w - model_loss_l
    ref_diff   = ref_loss_w   - ref_loss_l
    inside_term = -beta_dpo * (model_diff - ref_diff)
    loss_dpo_channel = -log(sigmoid(inside_term) + 1e-8)
    acc_channel = (model_diff < 0).float()
```

**关键**：Shepherd 的 `x_noise` 是连续高斯噪声，`x_out` 是模型对噪声的预测。DPO loss 比较的是「模型在 winner vs loser 上预测噪声的 MSE」——winner 上 MSE 更小说明模型更偏好 winner。

**x1 DPO channels**:
- `pos_out` vs `pos_noise` (位置噪声)
- `x_out` vs `x_noise` (原子类型噪声) ← SPD 用 CrossEntropy，这里改为 MSE
- `bond_edge_x_out` vs `bond_edge_x_noise` (键类型噪声) ← SPD 用 CrossEntropy，这里改为 MSE

**x4 DPO channels** (可选，由 `dpo_optimize_x4` 控制):
- `pos_out` vs `pos_noise`
- `x_out` vs `x_noise` (药效团类型噪声) ← SPD 用 CrossEntropy，这里改为 MSE
- `direction_out` vs `direction_noise`

**2.3 修改 `training_step()`**
- 检测 DPO batch（`'winner' in train_batch`）
- DPO path：计算标准去噪 loss + DPO loss，加权合并
- Standard path：保持原逻辑不变

**2.4 新增 `get_dpo_weight()`**
- 实现 DPO 权重 ramp-up：前 N epoch 线性从 0 增到 `dpo_max_weight`

---

### Phase 2: DPO 数据集（全连续版）

**目标**：`src/shepherd/dpo_dataset.py` — 从零编写，无离散扩散依赖。

**核心逻辑**：
1. 接收 `preference_pairs: List[(winner_mol, loser_mol, score_w, score_l)]`
2. 对每对，**复用 Shepherd `HeteroDataset` 的数据生成逻辑**生成 `HeteroData`
3. 共享 timestep 和噪声种子：winner 和 loser 使用**相同的 timestep**（从噪声调度中随机采样），通过设置相同的随机种子确保噪声模式一致
4. 返回 `{'winner': HeteroData, 'loser': HeteroData, 'shared_timestep': float}`

**与 SPD 版的关键区别**：
- **无 `DiscreteFeatureDiffusion`**：所有特征都用连续高斯前向加噪
- **无 `atom_marginals` / `bond_marginals` / `pharm_marginals`**
- **timestep 直接用 float**：从 `noise_schedule_dict` 的 `ts` 数组中随机选择一个 index，获取对应的 `alpha_dash_t`, `sigma_dash_t` 进行前向加噪
- **实现方式**：直接调用 `HeteroDataset.__getitem__()` 的内部方法（`get_x1_data`, `get_x4_data` 等），或创建一个轻量化的数据生成器

**防 BUG 设计**：
- timestep 存储为 `torch.float32`（避免 SPD BUG 4 的 int16 截断）
- 噪声调度参数从 `noise_schedule_dict` 直接索引（避免 SPD BUG 5 的线性插值回退）

---

### Phase 3: 混合 DataLoader

**目标**：`src/shepherd/mixed_dataloader.py`

**内容**（与 SPD 版逻辑基本一致，简化离散相关代码）：
- `MixedDPODataset`：按 `real_data_ratio` 随机返回标准样本或 DPO 偏好对
- `collate_mixed_batch()`：区分 standard batch（PyG collate）和 DPO batch（dict with 'winner'/'loser'）
- 使用 `torch.utils.data.DataLoader`（非 PyG DataLoader），自定义 collate_fn

---

### Phase 4: OnlineSampler + 评分器

**目标**：`src/shepherd/dpo_utils.py`

**4.1 OnlineSampler**
- 调用 Shepherd 的 `inference/sampler.py` 的 `generate()` 函数（而非 SPD 的 `inference_sample()`）
- 参数适配：Shepherd `generate()` 签名与 SPD `inference_sample()` 不同
- **无 `pharm_marginals` 参数**：Shepherd 推理是全连续的，无需传入边际分布

**4.2 ShepherdScorer**
- 复用 Shepherd 自带的 `shepherd_score_utils/` 评分工具
- 评分公式沿用 SPD 的三元相似度：`total = surf*1 + esp*3 + pharm*3 - sa*1.5 + 2.0`
- 需要导入 `shepherd_score` 包（`ConfEval`, `ConditionalEvalPipeline`, `Molecule`）

**4.3 PreferencePairBuilder**
- 逻辑与 SPD `evaluate_and_build_pairs()` 基本一致
- `create_rdkit_molecule` 使用 Shepherd 的 `src/shepherd/extract.py`（非 SPD 的 `extract_shepherd.py`）

---

### Phase 5: DPO 训练主脚本

**目标**：`training/dpo_train.py`

**结构**（参照 SPD 的 `DPO1_0_triSim.py`，但简化）：

```
main():
  1. 加载参数 (params_x1x3x4_dpo_finetune_nps)
  2. 加载 NPs 参考分子数据
  3. 创建 HeteroDataset (标准训练数据，仅 1 个 NPs 分子)
  4. 创建空 DPODataset
  5. 加载预训练 ckpt → LightningModule
  6. 首次采样 → 生成初始偏好对
  7. 创建混合 DataLoader
  8. 注册 DPOSamplingCallback + CheckpointCallback
  9. pl.Trainer.fit()
```

**DPOSamplingCallback** (与 SPD 版结构一致)：
- `on_train_epoch_end()`：每 N epoch 执行在线采样 → 评估 → 构建偏好对 → 更新 DPO 数据集
- Iterative DPO：可选的 ref_model 更新（score 超过历史最佳时）
- 质量门控：validity rate / pairs 数量 / score_failed_count
- 保护性停训：连续低有效率轮次触发停止

---

### Phase 6: DPO 参数配置

**目标**：`training/parameters/params_x1x3x4_dpo_finetune_nps.py`

基于 `params_x1x3x4_diffusion_mosesaq_20240824.py`，新增：

```python
# DPO 训练配置
'enable_dpo': True,
'beta_dpo': 0.3,               # KL 约束强度
'dpo_max_weight': 0.3,         # DPO loss 最大权重
'dpo_ramp_up_epochs': 10,      # DPO 权重线性增长期
'dpo_optimize_x4': True,       # 是否对 x4 也计算 DPO loss
'real_data_ratio': 0.5,        # 混合训练中真实数据占比
'dpo_min_score_gap': 0.15,     # 最小分数差距
'dpo_sampling_every_n_epochs': 3,  # 采样间隔

# Iterative DPO
'iterative_dpo_enabled': False, # 初始禁用，先跑稳定基线

# 预训练 checkpoint 路径
'pretrained_checkpoint_path': '<Shepherd ckpt 路径>',

# 学习率（DPO 微调用更小 lr）
'lr': 2e-6,
'min_lr': 2e-6,

# NPs 参考分子数据路径
'nps_data_path': '<NPs molblock pickle 路径>',
```

---

## 3. 验证计划

### 单元测试
1. **DPO 数据集**：验证 `DPODataset.__getitem__()` 返回的 winner/loser 的 timestep 一致、噪声调度参数正确
2. **DPO Loss**：构造简单的 mock input/output，验证 `compute_dpo_loss()` 梯度流正确、implicit_acc 合理
3. **混合 DataLoader**：验证 standard/dpo batch 交替出现

### 集成测试
4. **Smoke test**：用 1 个 NPs 分子，batch_size=1，跑 1 epoch 标准训练 + 1 epoch DPO 训练，确认无崩溃
5. **推理一致性**：对比 Shepherd `generate()` 在有/无 DPO 微调权重下的输出分布

### 指标验证
6. **DPO weight ramp-up**：确认前 N epoch DPO 权重从 0 线性增长
7. **ref_model 冻结**：确认 ref_model 参数梯度始终为 None
8. **有效率监控**：首轮 valid% 应与 Shepherd 基模型一致（~38%）

---

## 4. 风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| Shepherd ckpt 与 DPO LightningModule 不兼容 | `strict=False` 加载，仅 `ref_model` 是新增参数 |
| 全 MSE DPO loss 信号太弱 | 按 channel 独立计算 DPO loss，而非汇总后再算；调整 `beta_dpo` |
| NPs 分子在 MOSES_aq 训练集分布外 | 预期行为（DPO 的目的就是引导向 NPs 方向）|
| 有效率崩塌 | 混合训练 + 保护性停训 + 健康 checkpoint 机制 |
| `generate()` 函数签名变化 | 仔细对比 Shepherd sampler.py 的参数，适配 OnlineSampler |

---

## 5. 实施顺序

1. **Phase 1** → 修改 `lightning_module.py`（DPO loss 核心）
2. **Phase 6** → 创建参数配置文件
3. **Phase 2** → 创建 `dpo_dataset.py`
4. **Phase 3** → 创建 `mixed_dataloader.py`
5. **Phase 4** → 创建 `dpo_utils.py`（OnlineSampler + Scorer）
6. **Phase 5** → 创建 `dpo_train.py`（训练主脚本）
7. **验证** → 按 §3 顺序验证

---

当前 DPODataset 会把 winner/loser 各自重新提取成完整 x1/x3/x4 数据，不会把“同一个参考 NP 的固定 x3/x4 条件”显式存进 pair 里参与训练。


**核心区别**

DPO 本质上是在学这个不等式：

`p_theta(winner | condition) > p_theta(loser | condition)`

这里最重要的是：`condition` 必须对 winner 和 loser 相同。

拿 LLM 类比就是：

- `condition` = 同一个 prompt
- `winner/loser` = 两个候选回答

如果你把 winner 和 loser 各自换成不同 prompt，那就不是标准 DPO 了。

**放到你这个任务里**

在 Shepherd 的 NP 条件生成里，比较自然的定义是：

- `condition` = 同一个参考 NP 提供的 `x3/x4`
  - `x3` 是参考分子的表面/ESP 条件
  - `x4` 是参考分子的药效团条件
- `winner/loser` = 在这个同一条件下采样出来的两个 `x1` 候选分子

也就是说，正确语义更像：

- 在“参考 NP 条件”下，A 比 B 更好
- 所以训练模型更偏向 `A | 同一个参考条件`

**当前实现做了什么**

采样和打分阶段，确实是按“同一个参考 NP 条件”来的：

- 在 [dpo_utils.py](/Users/mutsumi/Desktop/WorkSpace/Shepherd/src/shepherd/dpo_utils.py#L82) 到 [dpo_utils.py](/Users/mutsumi/Desktop/WorkSpace/Shepherd/src/shepherd/dpo_utils.py#L117)，我先从参考分子提取 `surface/electrostatics/pharm`
- 在 [dpo_utils.py](/Users/mutsumi/Desktop/WorkSpace/Shepherd/src/shepherd/dpo_utils.py#L137) 到 [dpo_utils.py](/Users/mutsumi/Desktop/WorkSpace/Shepherd/src/shepherd/dpo_utils.py#L175)，`generate()` 也是拿这同一套参考条件去采样
- 然后用这些样本对同一个参考分子打分，选出 winner/loser

问题出在训练数据重建阶段：

- 在 [dpo_dataset.py](/Users/mutsumi/Desktop/WorkSpace/Shepherd/src/shepherd/dpo_dataset.py#L50) 到 [dpo_dataset.py](/Users/mutsumi/Desktop/WorkSpace/Shepherd/src/shepherd/dpo_dataset.py#L75)，pair 里只保留了 `winner_mol` 和 `loser_mol`
- 接着在 [dpo_dataset.py](/Users/mutsumi/Desktop/WorkSpace/Shepherd/src/shepherd/dpo_dataset.py#L77) 到 [dpo_dataset.py](/Users/mutsumi/Desktop/WorkSpace/Shepherd/src/shepherd/dpo_dataset.py#L133)，我把它们各自重新转成完整 `x1/x3/x4`
- 其中 `x3` 是从 winner 自己、loser 自己各自重新算的，见 [dpo_dataset.py](/Users/mutsumi/Desktop/WorkSpace/Shepherd/src/shepherd/dpo_dataset.py#L212)
- `x4` 也是从 winner 自己、loser 自己各自重新算的，见 [dpo_dataset.py](/Users/mutsumi/Desktop/WorkSpace/Shepherd/src/shepherd/dpo_dataset.py#L253)

所以现在训练时实际比较的是：

- `p_theta(winner | winner自己的x3/x4)`
- `p_theta(loser | loser自己的x3/x4)`

而不是：

- `p_theta(winner | 同一个参考NP的x3/x4)`
- `p_theta(loser | 同一个参考NP的x3/x4)`

**为什么这会有偏差**

因为这样一来，winner 和 loser 的“条件”也变了。

模型学到的更像是：

- winner 在“它自己的条件”下更自洽
- loser 在“它自己的条件”下没那么好

但我们真正想学的是：

- 在同一个参考 NP 条件下，winner 比 loser 更值得偏好

这两者不是一回事。

更直白一点：

- 你打分时问的是：“谁更像参考 NP？”
- 但你训练时变成在问：“谁更像它自己？”

所以信号会被稀释。

**为什么我没有直接判它是错的**

因为 Shepherd 的原始预训练数据就是“同一个分子的 `x1/x3/x4` 绑定在一起”的自条件样本，所以我先用了这个版本，优点是：

- 实现简单
- 跟现有 `HeteroDataset` 兼容
- 不容易把训练管线改崩

但如果你的目标明确是“针对固定 NP 条件做 conditional DPO”，那现在这版语义就不够准。

**如果改成你真正想要的“共享参考条件版”**

训练 pair 应该改成：

- `winner`:
  - `x1` 来自 winner 分子
  - `x3/x4` 来自参考 NP
- `loser`:
  - `x1` 来自 loser 分子
  - `x3/x4` 也来自同一个参考 NP

也就是 pair 里要显式存一份 reference condition，而不是只存 `winner_mol/loser_mol`。

这会进一步引出一个设计选择：

- DPO loss 只比 `x1`
- `x3/x4` 只作为固定条件输入，不参与 winner/loser 偏好差分

我个人认为这才是更合理的版本。
