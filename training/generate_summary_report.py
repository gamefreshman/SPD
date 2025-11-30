#!/usr/bin/env python3
"""生成简洁的汇总报告"""

import json
from pathlib import Path

# 读取汇总数据
with open('batch_evaluation_results/summary_report.json', 'r') as f:
    data = json.load(f)

# 原子类型映射
ATOM_MAP = {
    0: 'None', 1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}

print("=" * 80)
print("📊 DPO训练生成分子的原子类型分析报告")
print("=" * 80)

print(f"\n🔍 分析时间: {data['timestamp']}")
print(f"📁 分析文件数: {data['total_files']}")
print(f"🧬 总样本数: {data['total_samples']}")
print(f"⚛️  总原子数: {data['total_atoms']}")

print(f"\n{'='*80}")
print("🧪 全局原子类型分布 (所有文件合计)")
print(f"{'='*80}")

print(f"\n{'序号':<6} {'原子':<8} {'数量':<10} {'占比':<8} {'图示'}")
print("-" * 70)

atom_dist = data['global_atom_distribution']
total = sum(atom_dist.values())

# 按数量排序
sorted_atoms = sorted(atom_dist.items(), key=lambda x: int(x[1]), reverse=True)

for i, (atom_num_str, count) in enumerate(sorted_atoms, 1):
    atom_num = int(atom_num_str)
    atom_name = ATOM_MAP.get(atom_num, f"?({atom_num})")
    percentage = count / total * 100
    bar = '█' * int(percentage)
    
    symbol = "⚠️" if atom_num == 0 else f"{i:2d}"
    print(f"{symbol:<6} [{atom_num:2d}] {atom_name:<4} {count:<10} {percentage:5.2f}%  {bar}")

print(f"\n{'='*80}")
print("⚠️  关键发现")
print(f"{'='*80}")

invalid_count = atom_dist.get('0', 0)
invalid_percentage = invalid_count / total * 100

print(f"\n🚨 无效原子(0)问题:")
print(f"   - 出现次数: {invalid_count} / {total} ({invalid_percentage:.2f}%)")
print(f"   - 受影响文件: {data['files_with_invalid']}/{data['total_files']} (100%)")
print(f"   - 每个文件平均: {invalid_count/data['total_files']:.1f} 个")
print(f"   - 每个样本平均: {invalid_count/data['total_samples']:.1f} 个")

print(f"\n📊 原子类型覆盖:")
print(f"   - 理论上应该只有前5种: [None, H, C, N, O]")
print(f"   - 实际生成了全部12种原子类型")
print(f"   - 说明模型使用了预训练权重中学习的所有原子类型")

print(f"\n🎯 各原子类型出现情况:")
expected_atoms = [0, 1, 6, 7, 8]  # None, H, C, N, O
unexpected_atoms = [9, 14, 15, 16, 17, 35, 53]  # F, Si, P, S, Cl, Br, I

print(f"\n   预期原子 (训练数据中有):")
for atom_num in expected_atoms:
    atom_name = ATOM_MAP[atom_num]
    count = atom_dist.get(str(atom_num), 0)
    pct = count / total * 100
    print(f"      [{atom_num:2d}] {atom_name:<4}: {count:5d} ({pct:5.2f}%)")

print(f"\n   非预期原子 (训练数据中无，但模型仍生成):")
for atom_num in unexpected_atoms:
    atom_name = ATOM_MAP[atom_num]
    count = atom_dist.get(str(atom_num), 0)
    pct = count / total * 100
    print(f"      [{atom_num:2d}] {atom_name:<4}: {count:5d} ({pct:5.2f}%)")

# 文件级别统计
print(f"\n{'='*80}")
print("📂 各文件详细统计")
print(f"{'='*80}")

print(f"\n{'文件名':<40} {'样本':<6} {'原子':<6} {'无效(0)':<8} {'无效率'}")
print("-" * 80)

for result in data['file_results']:
    filename = Path(result['file']).name
    num_samples = result['stats']['num_samples']
    total_atoms_file = result['stats']['total_atoms']
    invalid_in_file = result['stats']['atom_counter'].get('0', 0)
    invalid_rate = invalid_in_file / total_atoms_file * 100
    
    print(f"{filename:<40} {num_samples:<6} {total_atoms_file:<6} {invalid_in_file:<8} {invalid_rate:5.2f}%")

print(f"\n{'='*80}")
print("✅ 报告生成完成")
print(f"详细数据保存在: batch_evaluation_results/summary_report.json")
print(f"{'='*80}")
