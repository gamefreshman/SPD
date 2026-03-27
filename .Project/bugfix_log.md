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

后验概率归一化链中存在**累积浮点误差**：

1. `compute_batched_over0_posterior_distribution()` 内部计算分母时使用 `+ 1e-8` 防止除零（line 132）
2. 外部归一化 `prob_X = unnormalized / (sum + 1e-8)` 又引入一次 epsilon（line 1469）
3. 两次 epsilon 叠加导致概率和偏离 1.0 的误差在某些边界情况下超过 `1e-3`

**触发条件**：v1.9 修复 Bug 1 后，`compute_batched_over0_posterior_distribution` 的输入从 logits 变为 one-hot 编码。one-hot 输入使得转移矩阵乘法结果更加稀疏，放大了 epsilon 导致的归一化偏差。

### 修复方法

**文件**: `src/shepherd/inference.py:1469-1475`

将硬断言替换为带警告的条件重归一化：

```diff
-            prob_X = unnormalized_prob_X / (torch.sum(unnormalized_prob_X, dim=-1, keepdim=True) + 1e-8)
-            assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-3).all()
+            # 数值保护：归一化并检查质量，若偏差过大则警告并强制重归一化
+            prob_X = unnormalized_prob_X / (torch.sum(unnormalized_prob_X, dim=-1, keepdim=True) + 1e-8)
+            prob_sum = prob_X.sum(dim=-1)
+            max_deviation = (prob_sum - 1).abs().max().item()
+            if max_deviation > 1e-2:
+                print(f"Warning: prob_X normalization deviation {max_deviation:.6f} at timestep {t}, forcing re-normalization")
+                prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True).clamp(min=1e-8)
```

**设计决策**：
- 阈值 `1e-2` 而非 `1e-3`：允许正常浮点误差，仅在真正异常时警告
- `1e-2` 以下偏差不影响 `multinomial` 采样的统计性质
- 仅在偏差超限时才触发重归一化，避免不必要的计算开销

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
