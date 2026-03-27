# 离散扩散代码修复计划

## 背景

SPD 将 Shepherd 的原子类型/键类型从连续扩散改为离散扩散（Marginal Uniform Transition），但推理代码 `inference.py` 存在多处 Bug，导致离散扩散的采样过程被破坏。**训练代码本身是正确的，预训练权重无需重新训练**，只需修复推理代码。

---

## Bug 清单

### Bug 1（致命）：原子类型后验计算传入了模型 logits 而非 one-hot 噪声状态

**文件**: `src/shepherd/inference.py:1443`
**现状**:
```python
x1_x_out_batch = x1_x_out.unsqueeze(0)  # ❌ 模型输出的 raw logits
atom_posterior_prob = compute_batched_over0_posterior_distribution(
    x1_x_out_batch,  # 应该是 x1_x_t（当前 one-hot 噪声状态）
    Qt_batch, Qtb_atom, Qsb_atom
)
```
**对比同文件的正确写法**（键类型，line 1475）:
```python
x1_bond_edge_x_t_batch = x1_bond_edge_x_t.unsqueeze(0)  # ✅ 当前 one-hot 噪声状态
bond_posterior_prob = compute_batched_over0_posterior_distribution(
    x1_bond_edge_x_t_batch, ...
)
```
**影响**: 后验公式 `p(x_{t-1}|x_t, x_0) ∝ (x_t · Qt^T) · Qsb` 中 `x_t` 必须是 one-hot。传入 logits 导致贝叶斯更新完全错误，原子类型采样质量极差。
**修复**:
```python
x1_x_t_batch = x1_x_t.unsqueeze(0)  # 使用当前 one-hot 噪声状态
atom_posterior_prob = compute_batched_over0_posterior_distribution(
    x1_x_t_batch, Qt_batch, Qtb_atom, Qsb_atom
)
```

---

### Bug 2（致命）：X4 药效团类型在推理时使用连续 DDPM 公式而非离散后验采样

**文件**: `src/shepherd/inference.py:1542`
**现状**:
```python
# ❌ 连续 DDPM 去噪公式 — 对离散分类变量无意义
x4_x_t_1 = ((1./x4_alpha_t) * x4_x_t) - ((x4_var_dash_t/(x4_alpha_t * x4_sigma_dash_t)) * x4_x_out) + (x4_c_t * x4_x_epsilon)
```
**但训练时** (`lightning_module.py:512-516`) 使用的是 `F.cross_entropy`（离散分类），存在训练-推理不一致。
**影响**: 药效团类型的去噪完全错误，连续公式产出的值不是有效概率分布，直接导致 `sims_pharm_target` 无法提升。
**修复**: 需要为 x4 实现与 x1 相同的离散后验采样流程（依赖 Bug 3 的修复）。

---

### Bug 3（致命）：推理代码缺少 X4 药效团的离散扩散处理器

**文件**: `src/shepherd/inference.py:540-542`
**现状**:
```python
x1_atom_diffuser = DiscreteFeatureDiffusion(timesteps=T, marginals=atom_marginals)
x1_bond_diffuser = DiscreteFeatureDiffusion(timesteps=T, marginals=bond_marginals)
# ❌ 缺少: x4_pharm_diffuser = DiscreteFeatureDiffusion(timesteps=T, marginals=pharm_marginals)
```
**同时**，函数签名 (`inference.py:431-432`) 只接收 `atom_marginals` 和 `bond_marginals`，**缺少 `pharm_marginals` 参数**。
**影响**: 没有 x4 的转移矩阵 (Qt, Qsb, Qtb)，无法实现 x4 的离散后验采样。
**修复**:
1. 函数签名新增 `pharm_marginals=None` 参数
2. 初始化 `x4_pharm_diffuser = DiscreteFeatureDiffusion(timesteps=T, marginals=pharm_marginals)`
3. 所有调用 `inference_sample()` 的地方传入 `pharm_marginals`

---

### Bug 4（致命）：X4 药效团类型的初始噪声用了高斯采样而非离散极限分布

**文件**: `src/shepherd/inference.py:862`
**现状**:
```python
# ❌ 高斯噪声 — 对离散变量无意义
x4_x_T = torch.randn(N_x4, num_pharm_types).to(device)
```
**对比 x1 的正确做法** (`inference.py:805`):
```python
# ✅ 从边际分布采样 one-hot
x1_x_T, x1_bond_edge_x_T = initial_sample_discrete_feature_noise(atom_marginals, bond_marginals, N_x1)
```
**对比训练时** (`new_datasets.py:1026`):
```python
# ✅ 训练时使用离散扩散
x_noised_no_vn = self.x4_pharm_diffuser.apply_noise(x_clean_no_vn, t_tensor, device)
```
**影响**: t=T 时的初始分布完全错误。离散扩散的极限分布应该是边际分布（每个类别的先验概率），而非高斯分布。
**修复**:
```python
# 从药效团边际分布采样 one-hot
pharm_limit = pharm_marginals[None, :].expand(N_x4, -1)
x4_x_T_indices = pharm_limit.multinomial(1).squeeze(-1)
x4_x_T = F.one_hot(x4_x_T_indices, num_classes=num_pharm_types).float().to(device)
```

---

### Bug 5（中等）：X4 最终输出使用 argmin 距离匹配而非 argmax

**文件**: `src/shepherd/inference.py:1591`
**现状**:
```python
# ❌ 使用连续值的距离匹配 — 因为 x4_x_t 是连续 DDPM 产物
x4_x_final = np.argmin(np.abs(x4_x_t[...] - params['dataset']['x4']['scale_node_features']), axis=-1)
```
**对比 x1 的正确做法** (`inference.py:1619`):
```python
# ✅ 对 one-hot 编码取 argmax
x1_x_final_indices = torch.argmax(x1_x_t[~virtual_node_mask_x1], dim=-1).cpu().numpy()
```
**影响**: 修复 Bug 2/4 后，x4_x_t 将变为 one-hot 编码，需要同步修改为 argmax。
**修复**:
```python
x4_x_final = torch.argmax(x4_x_t[~virtual_node_mask_x4], dim=-1).cpu().numpy()
# 注意: 需要确认是否需要 -1 的索引偏移（取决于虚拟节点的药效团类型编码）
```

---

### Bug 6（低）：调用侧需要传递 pharm_marginals

**涉及文件**: 所有调用 `inference_sample()` 的地方
**需要排查的文件**:
- `training/DPO1_0_triSim.py` (DPO 训练中的采样)
- `evaluation/` 目录下的评估脚本
- 其他调用推理的入口

**修复**: 找到所有调用点，从 dataset 对象获取 `pharm_marginals` 并传入。训练代码中已有 `dataset.x4_pharm_diffuser.transition_model.x_marginals`（见 `dpo_dataset.py:77`），可直接使用。

---

## 不需要修改的部分（已验证正确）

| 组件 | 结论 |
|------|------|
| **x1 训练 loss** (`cross_entropy` 预测 x_0) | 正确，D3PM 论文的 x_0 参数化是有效方法 |
| **x4 训练 loss** (`cross_entropy` 预测 x_0) | 正确，与 x1 一致 |
| **x1 键类型后验采样** (inference.py:1474-1525) | 正确，使用了 one-hot 噪声状态 |
| **x1 初始噪声** (`initial_sample_discrete_feature_noise`) | 正确，从边际分布采样 one-hot |
| **训练时的前向加噪** (new_datasets.py) | 正确，x1/x4 都通过转移矩阵加噪 |
| **模型输出格式** (model.py:1306 "continuous for now") | 注释误导但代码正确，MLP 输出的是 logits |
| **连续特征** (pos_loss MSE) | 正确，标准噪声预测范式 |

---

## 修复优先级与依赖关系

```
Bug 3 (添加 pharm_diffuser + 参数传递)
  ├── Bug 4 (x4 初始噪声: randn → 边际分布采样)
  ├── Bug 2 (x4 去噪: DDPM → 离散后验采样)
  │     └── Bug 5 (x4 最终输出: argmin → argmax)
  └── Bug 6 (调用侧传递 pharm_marginals)

Bug 1 (x1 原子类型后验: logits → one-hot) — 独立，可并行修复
```

**建议执行顺序**:
1. **第一步**: 修复 Bug 1（独立，最简单，立即可验证）
2. **第二步**: 修复 Bug 3 + Bug 6（添加 pharm_diffuser 基础设施）
3. **第三步**: 修复 Bug 4（x4 初始噪声）
4. **第四步**: 修复 Bug 2 + Bug 5（x4 离散后验采样 + 最终输出）

---

## 验证计划

修复后运行以下验证:
1. **基础验证**: 用修复后的 `inference.py` + 现有预训练 checkpoint 生成 750 个分子
2. **指标对比**: 与修复前的 SPD 基模型指标对比（sims_pharm_target, sims_esp_target, sims_surf_target, Valid%）
3. **预期结果**:
   - `sims_pharm_target` 应显著提升（Bug 2/3/4/5 修复后药效团类型采样正确）
   - `sims_surf_target` / `sims_esp_target` 可能提升（Bug 1 修复后原子类型更准确 → 分子结构更合理）
   - Valid% 应保持或提升
4. **如果 SPD 指标接近或超过 OriginShepherd**: 说明预训练本身没有问题，可以在此基础上重新进行 DPO 微调
