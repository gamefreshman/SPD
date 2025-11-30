# XTB 错误诊断报告：无效原子序数问题

## 🔴 问题描述

### 错误表现
```
[ERROR] Program stopped due to fatal error
-2- reading geometry input 'input_mol.xyz' failed
-1- Error: Cannot map symbol to atomic number
6 | *     67.12503815    -97.43159485   -137.27485657
  | ^ unknown element
```

### 次生错误
- **NCCL 超时**：分布式训练进程等待 XTB 完成，30分钟后超时终止

---

## 🎯 根本原因分析

### 错误链条
1. **模型生成阶段**：离散扩散模型采样时产生了无效的原子序数（0 或负数）
2. **XYZ 转换阶段**：`write_xyz_file()` 将原子序数转换为元素符号时
   - `rdkit.Chem.Atom(0).GetSymbol()` 返回 `*`（星号）
   - XYZ 文件包含无效元素符号
3. **XTB 计算阶段**：XTB 无法识别 `*` 元素，报错退出（退出码 128）
4. **分布式训练阶段**：评估进程卡住，NCCL 等待超时

### 核心代码位置

**原始代码**（`convert_data.py:38-40`）：
```python
a = int(atomic_numbers[i])
p = positions[i]
xyz+= f'{rdkit.Chem.Atom(a).GetSymbol()} {p[0]:>15.8f} {p[1]:>15.8f} {p[2]:>15.8f}\n'
```

**问题**：未验证 `atomic_numbers[i]` 的有效性（必须在 1-118 之间）

---

## ✅ 解决方案

### 1. 修复 XYZ 文件生成（`convert_data.py`）

**添加原子序数过滤**：
```python
# 过滤掉无效的原子（原子序数 <= 0 或 > 118）
valid_mask = (atomic_numbers > 0) & (atomic_numbers <= 118)
valid_atomic_numbers = atomic_numbers[valid_mask]
valid_positions = positions[valid_mask]
```

**添加异常处理**：
```python
try:
    symbol = rdkit.Chem.Atom(a).GetSymbol()
    if symbol == '*':
        continue  # 跳过无效原子
    xyz += f'{symbol} {p[0]:>15.8f} {p[1]:>15.8f} {p[2]:>15.8f}\n'
except Exception as e:
    print(f"警告: 无法处理原子序数 {a}: {e}")
    continue
```

### 2. 增强评估前验证（`dpo_trainer.py`）

**添加原子有效性检查**：
```python
# 验证原子序数的有效性
valid_atoms = (atoms > 0) & (atoms <= 118)
num_valid = np.sum(valid_atoms)
num_invalid = len(atoms) - num_valid

if num_invalid > 0:
    print(f"⚠️  {num_invalid} 个无效原子，原子序数范围: [{atoms.min():.0f}, {atoms.max():.0f}]")
    if num_valid == 0:
        print(f"✗ 所有原子都无效，跳过此样本")
        continue
```

### 3. 增强 XTB 错误诊断（`conformer_generation.py`）

**捕获并显示完整 XTB 输出**：
```python
try:
    output = subprocess.check_output([...], stderr=subprocess.STDOUT)
except subprocess.CalledProcessError as e:
    print(f"❌ XTB计算失败 (退出码 {e.returncode}):")
    print(f"   命令: {' '.join(e.cmd)}")
    if e.output:
        print(f"   XTB输出:\n{e.output}")
    raise
```

---

## 🔍 深层问题：模型为何生成无效原子？

### 可能原因

1. **虚拟节点处理不当**
   - 如果模型使用虚拟节点（原子序数=0），需要在后处理中移除

2. **离散扩散去噪错误**
   - 边际分布可能包含无效索引
   - 转移矩阵配置错误

3. **原子类型映射问题**
   - One-hot 编码到原子序数的转换逻辑错误

### 建议排查步骤

1. **检查边际分布**：
   ```python
   print("Atom marginals:", atom_marginals)
   # 应该只包含有效原子序数的概率
   ```

2. **检查采样输出**：
   ```python
   print("Generated atoms:", sample['x1']['atoms'])
   print("Unique values:", np.unique(sample['x1']['atoms']))
   # 查看是否包含 0 或负数
   ```

3. **检查虚拟节点配置**：
   ```python
   # 在 parameters 文件中
   params['dataset']['x1']['add_virtual_node'] = False  # 尝试关闭
   ```

---

## 📊 测试验证

### 运行修复后的代码
```bash
cd /home1/zhh/workspace/SPD/training
python dpo_trainer.py params_x1x3x4_dpo_finetune_nps 0
```

### 预期输出
```
🔬 样本 1: 70 个原子
  或
🔬 样本 1: 70 个原子 (⚠️  5 个无效原子，原子序数范围: [0, 8])
✓ ConfEval完成: QED=0.523, LogP=2.34
```

---

## 📝 总结

### 修复内容
✅ XYZ 文件生成增加原子序数过滤  
✅ 评估前增加原子有效性验证  
✅ XTB 错误增强诊断输出  
✅ 完整的错误堆栈跟踪  

### 待解决问题
⚠️  模型为何生成无效原子序数？需要进一步排查采样逻辑

### 影响范围
- 所有使用 `write_xyz_file()` 的代码路径
- 所有依赖 XTB 的评估流程
- DPO 训练的在线采样评估
