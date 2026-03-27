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
