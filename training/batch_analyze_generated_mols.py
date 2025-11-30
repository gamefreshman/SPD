#!/usr/bin/env python3
"""
批量分析生成的分子文件
1. 统计每个文件中的原子类型
2. 运行test_judge.sh评估
"""

import json
import subprocess
import os
from pathlib import Path
from collections import Counter

# 原子类型映射
atom_types_map = {
    0: 'None',
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
    14: 'Si',
    15: 'P',
    16: 'S',
    17: 'Cl',
    35: 'Br',
    53: 'I'
}

# 所有要处理的文件
json_files = [
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251122_171041.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251122_175617.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251124_033409.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251124_035631.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251124_041344.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251124_044121.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251124_044127.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251124_061425.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251124_063125.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251124_065741.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251124_070909.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251124_072746.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251124_124534.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251124_131402.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251124_165258.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251125_063120.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251125_065911.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251125_072043.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251125_092544.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251125_094407.json",
    "jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251125_100725.json",
]

def analyze_atom_types(json_file):
    """分析JSON文件中的原子类型"""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        all_atoms = []
        for sample in data:
            if 'x1' in sample and 'atoms' in sample['x1']:
                atoms = sample['x1']['atoms']
                all_atoms.extend(atoms)
        
        # 统计原子类型
        unique_atoms = sorted(set(all_atoms))
        atom_counter = Counter(all_atoms)
        
        return unique_atoms, atom_counter, len(data)
    
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None, None, 0

def run_judge_script(json_file):
    """运行test_judge.sh脚本"""
    try:
        # 使用test_judge.sh的逻辑 - 假设它接受JSON文件作为参数
        result = subprocess.run(
            ['bash', 'test_judge.sh', json_file],
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "超时"
    except Exception as e:
        return -1, "", str(e)

def main():
    print("=" * 80)
    print("📊 批量分析生成的分子文件")
    print("=" * 80)
    
    results = []
    
    for i, json_file in enumerate(json_files, 1):
        print(f"\n{'='*80}")
        print(f"📁 [{i}/{len(json_files)}] 处理文件: {Path(json_file).name}")
        print(f"{'='*80}")
        
        # 检查文件是否存在
        if not os.path.exists(json_file):
            print(f"⚠️  文件不存在，跳过")
            continue
        
        # 分析原子类型
        unique_atoms, atom_counter, num_samples = analyze_atom_types(json_file)
        
        if unique_atoms is not None:
            print(f"\n📊 文件统计:")
            print(f"  - 样本数量: {num_samples}")
            print(f"  - 原子总数: {sum(atom_counter.values())}")
            print(f"  - 原子类型数: {len(unique_atoms)}")
            
            print(f"\n🧪 原子类型分布:")
            print(f"  原子序数 -> 元素 -> 数量")
            print(f"  {'-'*40}")
            for atom_num in unique_atoms:
                atom_name = atom_types_map.get(atom_num, f"Unknown({atom_num})")
                count = atom_counter[atom_num]
                percentage = count / sum(atom_counter.values()) * 100
                print(f"  [{atom_num:2d}] {atom_name:4s} -> {count:5d} ({percentage:5.2f}%)")
            
            # 检查是否有无效原子(0)
            if 0 in unique_atoms:
                count_0 = atom_counter[0]
                print(f"\n  ⚠️  包含无效原子(0): {count_0} 个")
            
            results.append({
                'file': json_file,
                'num_samples': num_samples,
                'unique_atoms': unique_atoms,
                'atom_counter': dict(atom_counter),
                'has_invalid': 0 in unique_atoms
            })
        
        # 运行judge脚本（如果需要的话）
        # returncode, stdout, stderr = run_judge_script(json_file)
        # if returncode == 0:
        #     print(f"\n✅ Judge脚本执行成功")
        # else:
        #     print(f"\n❌ Judge脚本执行失败: {stderr}")
    
    # 输出汇总统计
    print(f"\n{'='*80}")
    print("📈 汇总统计")
    print(f"{'='*80}")
    
    total_files = len(results)
    files_with_invalid = sum(1 for r in results if r['has_invalid'])
    
    print(f"\n总文件数: {total_files}")
    print(f"包含无效原子(0)的文件: {files_with_invalid} ({files_with_invalid/total_files*100:.1f}%)")
    
    # 统计所有文件中出现的原子类型
    all_atom_types = set()
    for r in results:
        all_atom_types.update(r['unique_atoms'])
    
    print(f"\n所有文件中出现的原子类型: {len(all_atom_types)} 种")
    print(f"原子序数: {sorted(all_atom_types)}")
    print(f"元素符号: {[atom_types_map.get(a, f'?({a})') for a in sorted(all_atom_types)]}")
    
    # 保存结果
    output_file = 'batch_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 详细结果已保存到: {output_file}")

if __name__ == "__main__":
    main()
