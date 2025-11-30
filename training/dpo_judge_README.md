# DPO分子评估模块 (dpo_judge.py)

从 `dpo_trainer.py` 提取的独立评估模块，用于评估生成的分子质量。

## 功能特性

### 核心评估指标

1. **QED (Quantitative Estimate of Drug-likeness)**: 药物相似性评分
2. **LogP**: 脂水分配系数
3. **Strain Energy**: 应变能（需要xtb，可选）
4. **SA Score**: 合成可达性评分

### 综合评分公式

```python
total_score = qed * 2.0 - |logp - 1.5| * 0.3 - strain_energy * 0.5 - sa_score * 0.3
```

### 两种评估模式

- **Shepherd Score模式**（`--use-shepherd`）: 使用xtb计算应变能，更准确但需要额外依赖
- **RDKit模式**（默认）: 仅使用RDKit，速度快但无应变能计算

## 安装依赖

### 使用UV环境管理（推荐）

本项目使用UV进行Python环境管理。

```bash
# 确保已安装UV
# 如未安装: curl -LsSf https://astral.sh/uv/install.sh | sh

# UV会自动管理依赖，无需手动安装
# 项目依赖已在pyproject.toml或requirements.txt中定义
```

### 基础依赖（必需）
如果不使用UV，需要手动安装：
```bash
pip install rdkit numpy
```

### Shepherd Score依赖（可选，用于应变能计算）
```bash
# 需要安装xtb和shepherd_score包
conda install -c conda-forge xtb
# 或根据项目配置安装shepherd_score
```

## 使用方法

### 基础用法（使用UV环境）

```bash
# 使用RDKit模式评估（默认）
uv run python dpo_judge.py output_all_mols0.json

# 使用Shepherd Score模式（需要xtb）
uv run python dpo_judge.py output_all_mols0.json --use-shepherd

# 指定输出文件
uv run python dpo_judge.py output_all_mols0.json -o results.json

# 调整显示的最佳分子数量
uv run python dpo_judge.py output_all_mols0.json --top-k 50
```

### 不使用UV（传统方式）

```bash
# 需要先激活虚拟环境
python dpo_judge.py output_all_mols0.json
```

### 快速测试

```bash
# 使用测试脚本（自动使用UV环境）
chmod +x test_judge.sh
./test_judge.sh
```

## 输入文件格式

JSON文件应包含分子列表，每个分子具有以下结构：

```json
[
    {
        "x1": {
            "atoms": [6, 1, 6, 1, ...],        // 原子序数列表
            "positions": [[x, y, z], ...]      // 3D坐标
        }
    },
    ...
]
```

## 输出文件格式

评估结果JSON文件：

```json
[
    {
        "rank": 1,
        "total_score": 2.456,
        "qed": 0.712,
        "logp": 2.34,
        "strain_energy": 0.123,
        "sa_score": 2.45,
        "smiles": "CCO...",
        "num_atoms": 23
    },
    ...
]
```

## API使用示例

### Python代码调用

```python
from dpo_judge import MoleculeJudge, load_molecules_from_json

# 加载分子
molecules = load_molecules_from_json('output_all_mols0.json')

# 创建评估器
judge = MoleculeJudge(use_shepherd_score=False, verbose=True)

# 批量评估
evaluated_results = judge.evaluate_batch(molecules)

# 排名
ranked_results = judge.rank_molecules(evaluated_results)

# 打印报告
judge.print_ranking_report(ranked_results)

# 保存结果
judge.save_results(ranked_results, 'results.json')
```

### 单分子评估

```python
# 评估单个分子
sample = molecules[0]
result = judge.evaluate_single_molecule(sample, sample_idx=1)

if result['status'] == 'success':
    conf_scores = result['conf_scores']
    print(f"QED: {conf_scores['qed']:.3f}")
    print(f"LogP: {conf_scores['logp']:.2f}")
    print(f"SMILES: {conf_scores['smiles']}")
```

## 评估输出示例

```
================================================================================
📊 分子质量排名
================================================================================

排名   总分     QED      LogP     应变能      SA分数    SMILES
--------------------------------------------------------------------------------
1      2.456    0.712    2.340    0.123      2.450     CCO...
2      2.234    0.680    1.890    0.234      2.670     CC(C)...
3      2.112    0.695    2.100    0.345      2.890     CCC...
...

💾 评估结果已保存到: output_all_mols0_evaluated.json
```

## 单元测试

模块包含以下可测试的组件：

1. **MoleculeJudge类**: 核心评估器
2. **evaluate_single_molecule()**: 单分子评估
3. **evaluate_batch()**: 批量评估
4. **compute_total_score()**: 综合评分计算
5. **rank_molecules()**: 分子排名

### 运行单元测试

使用UV环境运行pytest：

```bash
# 运行所有测试
uv run pytest test_dpo_judge.py -v

# 运行特定测试
uv run pytest test_dpo_judge.py -v -k "test_init"

# 运行集成测试
uv run pytest test_dpo_judge.py -v -k "test_integration"
```

### 单元测试示例代码

```python
import pytest
from dpo_judge import MoleculeJudge

def test_evaluate_single_molecule():
    judge = MoleculeJudge(use_shepherd_score=False)
    
    # 测试样本
    sample = {
        'x1': {
            'atoms': [6, 1, 1, 1, 1],  # CH4
            'positions': [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]]
        }
    }
    
    result = judge.evaluate_single_molecule(sample, 1)
    assert result['status'] == 'success'
    assert 'conf_scores' in result
    assert 'qed' in result['conf_scores']
```

## 性能说明

- **RDKit模式**: ~0.1-0.5秒/分子
- **Shepherd Score模式**: ~1-5秒/分子（取决于xtb计算）

## 故障排除

### 常见问题

1. **ImportError: shepherd_score**
   - 解决：使用默认RDKit模式，不要加 `--use-shepherd` 参数

2. **RDKit分子创建失败**
   - 原因：分子结构不合理或坐标问题
   - 解决：检查输入JSON文件的atoms和positions字段

3. **SA Score计算失败**
   - 原因：sascorer模块未安装
   - 解决：系统会自动使用默认值5.0

## 与dpo_trainer.py的关系

`dpo_judge.py` 提取了以下函数：

- `evaluate_and_build_pairs()` → `MoleculeJudge.evaluate_batch()`
- 分子评估逻辑 → `MoleculeJudge.evaluate_single_molecule()`
- 综合评分计算 → `MoleculeJudge.compute_total_score()`

主要改进：
- ✅ 独立运行，不依赖训练框架
- ✅ 支持命令行调用
- ✅ 更好的错误处理
- ✅ 灵活的评估模式切换
- ✅ 详细的评估报告

## 许可证

遵循SPD项目的许可证。
