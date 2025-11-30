# 快速运行指南 - DPO分子评估

## 使用UV环境运行评估

### 1️⃣ 直接评估分子文件

```bash
# 评估output_all_mols0.json
uv run python dpo_judge.py output_all_mols0.json
```

### 2️⃣ 使用测试脚本

```bash
# 一键运行评估（推荐）
./test_judge.sh
```

### 3️⃣ 运行单元测试

```bash
# 运行所有单元测试
uv run pytest test_dpo_judge.py -v

# 只运行快速测试（跳过集成测试）
uv run pytest test_dpo_judge.py -v -k "not integration"

# 运行集成测试（使用真实文件）
uv run pytest test_dpo_judge.py -v -k "integration"
```

## 常用命令参数

```bash
# 指定输出文件
uv run python dpo_judge.py output_all_mols0.json -o my_results.json

# 显示更多最佳分子
uv run python dpo_judge.py output_all_mols0.json --top-k 50

# 使用Shepherd Score模式（需要xtb）
uv run python dpo_judge.py output_all_mols0.json --use-shepherd

# 查看帮助
uv run python dpo_judge.py --help
```

## 查看结果

评估完成后，会生成两个输出：

1. **终端输出**: 分子质量排名表格
2. **JSON文件**: 详细评估结果（默认：`output_all_mols0_evaluated.json`）

查看JSON结果：
```bash
cat output_all_mols0_evaluated.json | jq '.[0:5]'  # 查看前5个最佳分子
```

## 故障排查

### UV环境问题

```bash
# 检查UV是否安装
uv --version

# 如果未安装UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# 重新同步依赖
uv sync
```

### 依赖问题

```bash
# 如果提示缺少rdkit或其他包
uv add rdkit numpy

# 或者使用pip在UV环境中安装
uv pip install rdkit numpy
```

### 权限问题

```bash
# 给测试脚本添加执行权限
chmod +x test_judge.sh
```

## 性能提示

- 使用RDKit模式（默认）: 速度快，适合大批量评估
- 使用Shepherd Score模式: 更准确但较慢，适合精确评估

## 完整流程示例

```bash
# 1. 确保在项目目录
cd /home1/zhh/workspace/SPD/training

# 2. 运行评估
uv run python dpo_judge.py output_all_mols0.json --top-k 30

# 3. 查看结果
cat output_all_mols0_evaluated.json | jq '.[0:3]'

# 4. （可选）运行单元测试验证
uv run pytest test_dpo_judge.py -v -k "test_compute_total_score"
```
