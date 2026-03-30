# SPD 项目问题修复日志

> 本文档记录项目运行中遇到的报错及有效的修复方法，便于后续快速定位问题。

---

## 目录

- [BF-001 prob_X 归一化断言失败 (AssertionError)](#bf-001-prob_x-归一化断言失败)
- [BF-002 离散扩散推理 6 个关键 Bug (v1.9)](#bf-002-离散扩散推理-6-个关键-bug)

---

## BF-001 prob_X 归一化断言失败

- **日期**: 2026-03-27
- **严重程度**: 阻断（运行时崩溃）
- **关联版本**: v1.9（离散扩散 Bug 修复后触发）

### 报错信息

```
File "training/DPO1_0_triSim.py", line 654, in _sample_single_group
    sub_samples = inference_sample(model_pl, **inference_kwargs)
File "src/shepherd/inference.py", line 1470, in inference_sample
    assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-3).all()
AssertionError
```

### 根因分析

后验概率归一化链中，`unnormalized_prob_X` 的某些行概率总和**趋近于零**而非精确为零：

1. `weighted_X = pred_x1_x * posterior_prob`——当模型预测和后验分布高度不重叠时，乘积结果趋近零
2. 原有检测 `== 0` 只能捕获**精确零**，无法检测近零值（如 `1e-10`）
3. 归一化时 `S / (S + 1e-8)` 中 epsilon 占主导，导致归一化后概率和远小于 1.0

**实际运行数据证实**：偏差从 timestep 108 的 0.01 快速增长到 timestep 73 的 0.91，因为低时间步时模型预测更确定（softmax 输出更尖锐），与后验分布的不重叠度增加。

### 修复方法

**文件**: `src/shepherd/inference.py`（三处统一修复：x1 原子、x1 键、x4 药效团）

将精确零检测 `== 0` 替换为阈值检测 `< 1e-5`，近零行使用均匀分布作为 fallback，归一化除法不再加 epsilon（因为零值已被正确处理）：

```diff
 # x1 原子类型、x1 键类型、x4 药效团类型——统一模式：
 unnormalized_prob = weighted.sum(dim=2)
 unnormalized_prob = torch.clamp(unnormalized_prob, min=0.0)
-unnormalized_prob[torch.sum(unnormalized_prob, dim=-1) == 0] = 1e-6
-prob = unnormalized_prob / (torch.sum(unnormalized_prob, dim=-1, keepdim=True) + 1e-8)
-assert ((prob.sum(dim=-1) - 1).abs() < 1e-3).all()
+row_sums = torch.sum(unnormalized_prob, dim=-1)
+near_zero_mask = row_sums < 1e-5
+if near_zero_mask.any():
+    num_classes = unnormalized_prob.shape[-1]
+    unnormalized_prob[near_zero_mask] = 1.0 / num_classes  # 均匀分布 fallback
+prob = unnormalized_prob / torch.sum(unnormalized_prob, dim=-1, keepdim=True)
```

**设计决策**：
- **阈值 `1e-5`**：比 `compute_batched_over0_posterior_distribution` 内部的 `1e-6` 和 `1e-8` 大，确保能捕获所有近零情况
- **均匀分布 fallback**：比 `1e-6` 常数更有意义——当模型无法确定类别时，应用均匀采样而非低概率采样
- **无 epsilon 归一化**：零值已正确处理，不需要额外的防除零保护，结果数学上精确归一化

### 验证

运行 DPO 采样循环，观察：
- [ ] 不再出现 AssertionError
- [ ] Warning 频率应很低（偶尔出现可接受）
- [ ] 生成分子质量不受影响

---

## BF-002 离散扩散推理 6 个关键 Bug

- **日期**: 2026-03-27
- **严重程度**: 致命（生成质量严重劣化）
- **关联版本**: v1.9
- **详细记录**: 见 [DpoAndVisual.md v1.9 条目](../training/DpoAndVisual.md#v19-2026-03-27-修复离散扩散推理-6-个关键-bug-bug修复)

### 概要

`inference.py` 中 `inference_sample()` 函数在处理离散特征时存在 6 个 Bug，导致训练-推理不一致：

| Bug | 严重程度 | 问题 | 修复 |
|-----|---------|------|------|
| **1** | 致命 | x1 原子类型后验传入 logits 而非 one-hot | `x1_x_out` → `x1_x_t` |
| **2** | 致命 | x4 去噪使用连续 DDPM 而非离散后验采样 | 替换为转移矩阵+multinomial |
| **3** | 致命 | 缺少 `pharm_marginals` 参数和 `x4_pharm_diffuser` 初始化 | 新增参数和初始化 |
| **4** | 致命 | x4 初始噪声用 `randn` 而非边际分布 | `randn` → `marginals.multinomial` + one-hot |
| **5** | 中等 | x4 最终输出用 `argmin` 而非 `argmax` | `argmin` → `argmax` |
| **6** | 低 | 调用侧未传入 `pharm_marginals` | 3 个文件新增传参 |

### 涉及文件

- `src/shepherd/inference.py` — Bug 1-5 的核心修复
- `src/shepherd/dpo_utils.py` — Bug 6（调用侧）
- `training/dpo_trainer.py` — Bug 6（调用侧）
- `evaluation/experiment_SamEval/sample_discrete.py` — Bug 6（调用侧）

### 验证

- [ ] x4 药效团类型在去噪过程中保持为有效 one-hot 编码
- [ ] 生成分子的药效团类型分布与训练集边际分布接近
- [ ] `sims_pharm_target` 指标显著提升
- [ ] `pharm_marginals=None` 时推理管线仍正常运行（向后兼容）

---

## BF-003 训练与评估条件语义漂移 (x2/x3/x4 角色混淆)

- **日期**: 2026-03-30
- **严重程度**: 致命（导致模型生成能力崩塌，Valid% 跌破 15%）
- **关联版本**: v1.6, v1.7, v1.9

### 概要

在近 10 次的 commit 更新中，扩散推理的条件语义发生了严重的偏移。导致模型在 DPO 微调时性能崩溃、推理时遭遇严重的 OOD (分布外) 漂移：

1. **评估时 x2/x3 语义破损**：早前在 `ref_NP.py` 和 `sample_NP.py` 中激活了 `inpaint_x3_pos=True` 和 `inpaint_x3_x=True` 等标志，这为理应**完全静态**作为轮廓的 x2 和 x3 引入了扩散去噪过程的噪声。但模型训练时是从未见过 x2 和 x3 添加噪声的（始终是固定条件），这就导致了评估输入与训练分布脱节。
2. **训练时 x4 目标越权**：x4 被设计为伴随前向加噪的“带噪条件”，DPO 的生成任务理应是条件生成 $P(x_1 | x_2, x_3, x_4)$。但在以前的代码中，计算 DPO 标准损失时一并优化了 x4，错误地将其升格为了生成任务的优化对象。

### 修复方法（当前工作区已完成）

1. **恢复 x2/x3 绝对静态**：
   - 检查并验证 `inference.py` 中反向去噪循环已撤除所有对 x2 和 x3 的后向更新步骤，它们只作为前向图传递属性。
   - 在 `ref_NP.py` / `sample_NP.py` 强制置空所有的 `inpaint_x2_*` 与 `inpaint_x3_*` 变量。
2. **约束 x4 仅作为带噪条件**：
   - 在 `src/shepherd/lightning_module.py` 中引入 `dpo_optimize_x4` 卡点。
   - `params_*.py` 设定 `dpo_optimize_x4 = False`，在计算 `loss_std` 仅回传 `loss_std_x1` 梯度。
3. **增加严格的语义检验**：
   - `eval_unified.py` 中引入 `sidecar metadata` 指纹校验，确保样本对应的生成语义高度一致。

### 验证

已经执行过工作区代码审查：
- [x] 确认 `inference.py` 中的 `x2_pos_t` / `x3_pos_t` 仅接收初始 `target_inpaint` 值，未被添加前向噪声且不进入更新循环。
- [x] 确认 `loss_std` 脱离了对 x4 loss 的绑定。
- 后续运行需观察崩塌现象是否停止。
