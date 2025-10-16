# SPD 分子生成模型评估系统使用手册

## 📋 项目概述

本评估系统专门用于评估SPD（Structure-Pharmacophore Diffusion）分子生成模型的性能。系统提供了完整的条件分子生成、变量追踪和多维度评估功能，帮助研究人员深入了解模型的生成质量和推理过程。

## 🎯 主要功能

### 1. 条件分子生成
- 固定条件变量x2（表面点云）和x3（静电势），生成x1（原子图模态）和x4（药效团模态）
- 支持批量生成和单样本生成
- 提供详细的推理过程追踪

### 2. 变量追踪系统
- 实时追踪推理过程中5个关键变量的数值变化：
  - `x1_x_out`: 原子图节点特征 (形状: N×12)
  - `x1_bond_edge_x_out`: 原子图边特征 (形状: M×5)
  - `x4_pos_out`: 药效团位置坐标 (形状: K×3)
  - `x4_direction_out`: 药效团方向向量 (形状: K×3)
  - `x4_x_out`: 药效团特征 (形状: K×10)
- 自动检测和报告NaN值
- 生成详细的追踪报告和统计信息

### 3. 多维度评估指标
- **化学结构合理性评估（x1维度）**：使用RDKit验证生成分子的化学合理性
- **分子构象正确性评估（x4维度）**：采用xTB方法计算RMSD评估构象质量
- **条件一致性评估**：评估生成结果与输入条件的一致性

## 📁 文件结构

```
evaluation/
├── README.md                              # 本使用手册
├── run_variable_tracking.py               # 变量追踪主程序
├── conditional_generation_evaluation.py   # 条件生成和评估核心模块
├── evaluation.py                          # 增强版综合评估系统
├── simplified_variable_tracker.py         # 简化变量追踪器 1
├── complete_variable_tracking_summary.json # 完整变量追踪汇总数据
├── final_complete_tracking_summary.json   # 最终完整追踪数据
└── detailed_tracking_results/             # 详细追踪结果目录
    ├── variable_tracking_batch_0.json
    ├── variable_tracking_batch_1.json
    ├── variable_tracking_batch_2.json
    └── variable_tracking_batch_3.json
```

## 🚀 快速开始

### 环境要求

```bash
# Python 3.8+
# PyTorch 1.9+
# RDKit
# NumPy
# tqdm
```

### 基本使用

#### 1. 运行变量追踪评估

```bash
cd /home1/zhh/workspace/SPD/evaluation
python run_variable_tracking.py
```

**功能说明：**
- 使用训练好的模型 `last.ckpt` 进行条件分子生成
- 追踪400个推理步骤中5个关键变量的实际数值
- 生成详细的追踪报告和统计信息

**输出文件：**
- `final_complete_tracking_summary.json`: 完整的变量追踪数据（约14MB）
- `complete_variable_tracking_summary.json`: 汇总的追踪数据
- `detailed_tracking_results/`: 详细的批次追踪结果
- `variable_tracking.log`: 详细的运行日志

#### 2. 运行综合评估

```bash
python evaluation.py
```

**功能说明：**
- 执行完整的分子生成和评估流程
- 包含化学合理性、构象质量和条件一致性评估
- 提供详细的诊断信息和错误追踪

#### 3. 自定义条件生成

```python
from conditional_generation_evaluation import ConditionalMoleculeGenerator

# 初始化生成器
generator = ConditionalMoleculeGenerator(
    model_path='/path/to/last.ckpt',
    device='cuda'
)

# 从分子数据提取条件
conditions = generator.extract_conditions_from_molecule(mol_data)

# 生成分子
results = generator.generate_conditional_molecules(
    conditions=conditions,
    n_atoms=20,
    batch_size=1,
    num_samples=5
)
```

## 📊 输出数据格式

### 变量追踪数据格式

```json
{
  "metadata": {
    "batch_idx": 0,
    "batch_size": 1,
    "n_atoms": 20,
    "model_path": "/path/to/model.ckpt",
    "generation_mode": "conditional",
    "tracking_method": "forward_replacement_tracking"
  },
  "variables": {
    "x1_x_out": [
      {
        "step": 0,
        "data": [[...]], // 实际数值数组
        "modality": "x1",
        "key": "x_out",
        "found": true,
        "path": "x1.decoder.denoiser.x_out"
      }
    ],
    // 其他变量...
  },
  "warnings": [
    {
      "step": 159,
      "variable": "x1_x_out",
      "message": "变量 x1_x_out 在步骤 159 包含 NaN 值"
    }
  ],
  "step_count": 400
}
```

### 评估结果格式

```json
{
  "chemical_validity": {
    "is_valid": true,
    "smiles": "CCO",
    "error_message": null
  },
  "conformer_quality": {
    "rmsd": 1.23,
    "is_reasonable": true
  },
  "conditional_consistency": {
    "similarity_score": 0.85
  }
}
```

## ⚙️ 配置参数

### 模型配置

```python
# 模型路径（必须使用指定的检查点文件）
model_path = '/home1/zhh/workspace/SPD/training/jobs/x1x3x4_diffusion_mosesaq_20240824/last.ckpt'

# 设备配置
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 生成参数
n_atoms = 20        # 原子数量
batch_size = 1      # 批次大小
num_samples = 1     # 样本数量
```

### 追踪配置

```python
# 目标变量列表
target_variables = [
    'x1_x_out',
    'x1_bond_edge_x_out',
    'x4_pos_out',
    'x4_direction_out',
    'x4_x_out'
]

# 追踪参数
max_steps = 400     # 最大追踪步数
log_level = 'INFO'  # 日志级别
```

## 🔧 高级功能

### 1. 自定义变量追踪

```python
from simplified_variable_tracker import SimplifiedVariableTracker

# 创建追踪器
tracker = SimplifiedVariableTracker(model, device)

# 设置元数据
tracker.set_metadata({
    'experiment_name': 'custom_tracking',
    'timestamp': datetime.now().isoformat()
})

# 追踪推理步骤
step_result = tracker.track_inference_step(step_number, model_output)

# 保存追踪数据
tracker.save_tracking_data('custom_tracking_results.json')
```

### 2. 批量评估

```python
from evaluation import EnhancedMolecularEvaluator

# 创建评估器
evaluator = EnhancedMolecularEvaluator(model_path, device)

# 运行综合评估
results = evaluator.run_comprehensive_evaluation(
    data_file='path/to/molecules.pkl',
    max_molecules=10,
    samples_per_molecule=3
)
```

### 3. 条件数据提取

```python
# 从分子数据提取表面和药效团条件
conditions = generator.extract_conditions_from_molecule(mol_data)

# 条件包含：
# - surface: 表面点云坐标 (N×3)
# - electrostatics: 静电势值 (N,)
# - pharm_types: 药效团类型 (M,)
# - pharm_pos: 药效团位置 (M×3)
# - pharm_direction: 药效团方向 (M×3)
```

## 📈 性能监控

### 日志文件

- `variable_tracking.log`: 变量追踪详细日志
- `conditional_generation.log`: 条件生成过程日志
- `evaluation.log`: 评估过程日志

### 关键指标

- **推理步数**: 通常为400步
- **变量数据量**: 每个变量在每步都被记录
- **NaN检测**: 自动检测和报告异常值
- **内存使用**: 追踪数据文件约14MB
- **运行时间**: 单样本生成约40-50秒

## ⚠️ 注意事项

### 1. 模型文件
- **必须使用指定的模型文件**: `/home1/zhh/workspace/SPD/training/jobs/x1x3x4_diffusion_mosesaq_20240824/last.ckpt`
- 禁止调用其他模型文件
- 确保模型文件存在且可访问

### 2. 数据格式
- 中间文件不要保存为pkl格式，避免依赖库问题
- 直接存储包含numpy或tensor的字典/列表
- 使用JSON格式保存追踪数据

### 3. 内存管理
- 追踪数据可能较大（14MB+），注意内存使用
- 大批量评估时建议分批处理
- 及时清理临时文件

### 4. GPU使用
- 推荐使用CUDA加速
- 确保GPU内存充足
- 监控GPU使用率

## 📞 技术支持

如遇到问题，请：
1. 查看相关日志文件获取详细错误信息
2. 检查配置参数是否正确
3. 确认环境依赖是否完整
4. 参考本手册的常见问题部分

## 📝 更新日志

### v1.0.0 (2024-09-02)
- ✅ 实现完整的变量追踪功能
- ✅ 支持5个关键变量的实时追踪
- ✅ 添加NaN值检测和警告
- ✅ 生成详细的追踪报告
- ✅ 修复forward方法替换问题
- ✅ 优化变量名映射逻辑
- ✅ 简化数据提取流程
- ✅ 添加详细的调试信息

### 已知问题
- 推理过程中可能出现NaN值（属于正常现象）
- 大批量处理时内存使用较高
- 某些复杂分子的评估可能耗时较长

### 计划改进
- [ ] 添加更多评估指标
- [ ] 优化内存使用效率
- [ ] 支持更多模型格式
- [ ] 添加可视化功能
- [ ] 改进错误处理机制

---

**最后更新**: 2024年9月2日  
**版本**: v1.0.0  
**维护者**: AI助手