# 键确定方法更新说明

## 修改概述

将 `convert_data.py` 中的键确定方法完全替换为 `extract.py` 中的 `build_3d_mol_from_arrays` 方法。
**不再使用 `rdDetermineBonds`**，必须提供键信息才能构建分子。

## 修改的文件

1. **`src/score/shepherd_score/evaluations/utils/convert_data.py`**
   - 导入 `build_3d_mol_from_arrays` 和 `edge_list_to_adjacency_matrix`
   - 修改 `extract_mol_from_xyz_block()` 添加可选的键信息参数
   - 修改 `get_mol_from_atom_pos()` 添加可选的键信息参数

2. **`src/score/shepherd_score/evaluations/evaluate/evals.py`**
   - 更新 `ConfEval`、`ConsistencyEval`、`ConditionalEval` 类以支持可选的键参数

## 使用方法

**⚠️ 重要：现在必须提供键信息，否则将返回 None**

```python
from shepherd_score.evaluations.utils.convert_data import get_mol_from_atom_pos
from shepherd_score.evaluations.evaluate import ConfEval
import numpy as np

# 假设您有以下数据
atoms = np.array([6, 6, 8, 1, 1, 1, 1])  # C, C, O, H, H, H, H
positions = np.array([...])  # 3D 坐标
bonds = np.array([...])  # 键类型（edge list 格式）- 必需

# 使用 get_mol_from_atom_pos
mol, charge, xyz_block = get_mol_from_atom_pos(
    atoms=atoms,
    positions=positions,
    bonds=bonds  # 必须提供
)

# 或者使用 ConfEval
conf_eval = ConfEval(
    atoms=atoms,
    positions=positions,
    bonds=bonds  # 必须提供
)

# ❌ 不提供键信息将返回 None
mol, charge, xyz_block = get_mol_from_atom_pos(
    atoms=atoms,
    positions=positions
    # 缺少 bonds - 将返回 None
)
```

## 键数据格式

键数据应该是一个 numpy 数组，格式为 edge list：
- 长度应为 `N * (N-1) / 2`，其中 N 是原子数
- 每个元素表示键类型索引：
  - 0: 无键
  - 1: 单键 (SINGLE)
  - 2: 双键 (DOUBLE)
  - 3: 三键 (TRIPLE)
  - 4: 芳香键 (AROMATIC)

## 优势

1. **更可靠**：使用模型预测的键信息，完全避免 `rdDetermineBonds` 可能出现的问题
2. **明确性**：强制要求键信息，避免不确定的回退行为
3. **一致性**：所有分子都使用相同的键确定方法

## 注意事项

- ⚠️ **必须提供键信息**：如果不提供 `bonds` 参数，函数将直接返回 `None`
- 如果键信息格式不正确（长度不匹配），函数将返回 `None` 而不是回退
- 确保您的数据源包含键信息（如模型输出的 `sample['x1']['bonds']`）
- 原有的调试打印语句（"bond" 和 "bond_end"）已被移除
