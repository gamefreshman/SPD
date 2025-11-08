# 分子生成与评估脚本

## 概述

这个脚本将notebook转换为Python脚本，用于对3个天然产物分子各生成200个样本，并进行全面的分子质量评估。

## 评估流程

1. **RDKit分子创建** (`create_rdkit_molecule`)
   - 从生成的原子坐标创建RDKit分子对象
   - 如果创建失败，直接标记为无效

2. **ConfEval评估** (无条件构象评估)
   - 分子有效性验证
   - 基本分子属性计算 (SA_score, QED, logP, fsp3)
   - 结构稳定性分析 (strain_energy, RMSD)

3. **ConditionalEval评估** (条件相似性评估)
   - 仅当ConfEval判断分子有效时才进行
   - 3D结构相似性评估 (表面、静电势、药效团)
   - 2D图结构相似性评估

## 文件结构

```
evaluation/core/
├── molecular_evaluation.py    # 主评估脚本
├── run_evaluation.py         # 运行入口脚本
├── config.json              # 配置文件
├── README.md               # 使用说明
├── data/                   # 输出目录
│   ├── molecule_0_evaluation_results.json
│   ├── molecule_1_evaluation_results.json
│   └── molecule_2_evaluation_results.json
└── evaluation.log          # 日志文件
```

## 使用方法

### 方法1: 直接运行主脚本
```bash
cd /home1/zhh/workspace/SPD/evaluation/core
python molecular_evaluation.py
```

### 方法2: 使用入口脚本
```bash
cd /home1/zhh/workspace/SPD/evaluation/core
python run_evaluation.py
```

## 配置参数

编辑 `config.json` 文件来调整参数：

```json
{
    "model": {
        "checkpoint_path": "模型检查点路径"
    },
    "data": {
        "molblocks_path": "天然产物数据路径"
    },
    "sampling": {
        "n_atoms": 70,           # 生成分子原子数
        "batch_size": 10         # 采样批次大小
    },
    "evaluation": {
        "samples_per_molecule": 200,  # 每个分子采样数
        "num_surf_points": 400        # 表面点数
    }
}
```

## 输出格式

每个分子的评估结果保存为独立的JSON文件，包含：

### 统计信息
- `total_samples`: 总样本数
- `success_statistics`: 各阶段成功数量
- `success_rates`: 各阶段成功率

### 详细结果
每个样本包含：
- `rdkit_creation_success`: RDKit分子创建是否成功
- `conf_evaluation_success`: ConfEval评估是否成功
- `conf_is_valid`: ConfEval判断分子是否有效
- `cond_evaluation_success`: ConditionalEval评估是否成功
- `conf_results`: ConfEval详细结果
- `cond_results`: ConditionalEval详细结果 (仅当分子有效时)
- `error_messages`: 错误信息列表

## 性能监控

脚本会实时输出进度信息和成功率统计：
- RDKit创建成功率
- Conf评估成功率  
- Conf有效率
- Cond评估成功率

## 系统要求

- Python 3.7+
- PyTorch
- RDKit
- shepherd_score包
- 足够的GPU内存 (推荐8GB+)
- 足够的存储空间 (预估每个分子结果文件~10-50MB)

## 预期运行时间

- 每个分子采样: ~10-30分钟
- 每个样本评估: ~5-15秒
- 总预期时间: 2-6小时 (取决于硬件配置)

## 错误处理

脚本具有完善的错误处理机制：
- 单个样本失败不会影响整体进程
- 详细的错误信息记录
- 自动跳过问题分子继续处理
- 完整的日志记录

## 注意事项

1. 确保有足够的GPU内存进行批量采样
2. 定期检查日志文件了解运行状态
3. 可以随时中断并重新运行 (会覆盖之前结果)
4. 结果文件较大，注意存储空间
