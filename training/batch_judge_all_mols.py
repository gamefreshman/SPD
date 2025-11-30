#!/usr/bin/env python3
"""
批量评估所有生成的分子文件
1. 分析原子类型分布
2. 运行dpo_judge.py评估每个文件
3. 生成汇总报告
"""

import json
import subprocess
import os
from pathlib import Path
from collections import Counter
from datetime import datetime

# 原子类型映射
ATOM_TYPES_MAP = {
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
JSON_FILES = [
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
        sample_atom_types = []  # 每个样本的原子类型
        
        for sample in data:
            if 'x1' in sample and 'atoms' in sample['x1']:
                atoms = sample['x1']['atoms']
                all_atoms.extend(atoms)
                sample_atom_types.append(sorted(set(atoms)))
        
        # 统计原子类型
        unique_atoms = sorted(set(all_atoms))
        atom_counter = Counter(all_atoms)
        
        return {
            'unique_atoms': unique_atoms,
            'atom_counter': dict(atom_counter),
            'num_samples': len(data),
            'total_atoms': len(all_atoms),
            'sample_atom_types': sample_atom_types
        }
    
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return None


def run_judge(json_file, output_file):
    """运行dpo_judge.py评估"""
    try:
        print(f"\n🔍 开始评估...")
        result = subprocess.run(
            ['python', 'dpo_judge.py', json_file, 
             '--output', output_file,
             '--top-k', '20'],
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )
        
        if result.returncode == 0:
            print(f"✅ 评估成功")
            # 尝试读取评估结果
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    eval_data = json.load(f)
                return True, eval_data
            return True, None
        else:
            print(f"❌ 评估失败")
            if result.stderr:
                print(f"错误信息: {result.stderr[:500]}")
            return False, None
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 评估超时")
        return False, None
    except Exception as e:
        print(f"❌ 评估异常: {e}")
        return False, None


def print_atom_stats(stats, file_name):
    """打印原子统计信息"""
    print(f"\n📊 文件: {file_name}")
    print(f"  {'='*60}")
    print(f"  样本数量: {stats['num_samples']}")
    print(f"  原子总数: {stats['total_atoms']}")
    print(f"  原子类型数: {len(stats['unique_atoms'])}")
    
    print(f"\n  🧪 原子类型分布:")
    print(f"  {'原子序数':<8} {'元素':<8} {'数量':<10} {'占比'}")
    print(f"  {'-'*50}")
    
    total = sum(stats['atom_counter'].values())
    for atom_num in stats['unique_atoms']:
        atom_name = ATOM_TYPES_MAP.get(atom_num, f"?({atom_num})")
        count = stats['atom_counter'][atom_num]
        percentage = count / total * 100
        symbol = "⚠️ " if atom_num == 0 else "  "
        print(f"  {symbol}[{atom_num:2d}]     {atom_name:<8} {count:<10} {percentage:5.2f}%")
    
    # 检查无效原子
    if 0 in stats['unique_atoms']:
        count_0 = stats['atom_counter'][0]
        samples_with_0 = sum(1 for s in stats['sample_atom_types'] if 0 in s)
        print(f"\n  ⚠️  警告: 包含无效原子(0)")
        print(f"      - 总数量: {count_0}")
        print(f"      - 受影响样本: {samples_with_0}/{stats['num_samples']}")


def main():
    print("=" * 80)
    print("📊 批量评估生成的分子文件")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建输出目录
    output_dir = Path("batch_evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    for i, json_file in enumerate(JSON_FILES, 1):
        print(f"\n{'='*80}")
        print(f"📁 [{i}/{len(JSON_FILES)}] {Path(json_file).name}")
        print(f"{'='*80}")
        
        # 检查文件是否存在
        if not os.path.exists(json_file):
            print(f"⚠️  文件不存在，跳过")
            continue
        
        # 分析原子类型
        stats = analyze_atom_types(json_file)
        
        if stats is None:
            continue
        
        # 打印统计
        print_atom_stats(stats, Path(json_file).name)
        
        # 运行评估
        output_file = output_dir / f"{Path(json_file).stem}_evaluated.json"
        success, eval_data = run_judge(json_file, str(output_file))
        
        # 保存结果
        result = {
            'file': json_file,
            'timestamp': Path(json_file).stem.split('_')[-2:],
            'stats': stats,
            'evaluated': success,
            'eval_file': str(output_file) if success else None
        }
        all_results.append(result)
    
    # 生成汇总报告
    print(f"\n{'='*80}")
    print("📈 汇总统计报告")
    print(f"{'='*80}")
    
    total_files = len(all_results)
    files_with_invalid = sum(1 for r in all_results if 0 in r['stats']['unique_atoms'])
    total_samples = sum(r['stats']['num_samples'] for r in all_results)
    total_atoms = sum(r['stats']['total_atoms'] for r in all_results)
    
    print(f"\n📊 基本统计:")
    print(f"  - 处理文件数: {total_files}")
    print(f"  - 总样本数: {total_samples}")
    print(f"  - 总原子数: {total_atoms}")
    print(f"  - 包含无效原子的文件: {files_with_invalid} ({files_with_invalid/total_files*100:.1f}%)")
    
    # 统计所有原子类型
    all_atom_types = set()
    global_atom_counter = Counter()
    
    for r in all_results:
        all_atom_types.update(r['stats']['unique_atoms'])
        global_atom_counter.update(r['stats']['atom_counter'])
    
    print(f"\n🧪 全局原子类型分布:")
    print(f"  共 {len(all_atom_types)} 种原子类型")
    print(f"\n  {'原子序数':<8} {'元素':<8} {'总数量':<12} {'占比'}")
    print(f"  {'-'*50}")
    
    for atom_num in sorted(all_atom_types):
        atom_name = ATOM_TYPES_MAP.get(atom_num, f"?({atom_num})")
        count = global_atom_counter[atom_num]
        percentage = count / total_atoms * 100
        symbol = "⚠️ " if atom_num == 0 else "  "
        print(f"  {symbol}[{atom_num:2d}]     {atom_name:<8} {count:<12} {percentage:5.2f}%")
    
    # 保存汇总结果
    summary_file = output_dir / "summary_report.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_files': total_files,
            'total_samples': total_samples,
            'total_atoms': total_atoms,
            'files_with_invalid': files_with_invalid,
            'global_atom_distribution': dict(global_atom_counter),
            'all_atom_types': sorted(all_atom_types),
            'file_results': all_results
        }, f, indent=2)
    
    print(f"\n💾 汇总报告已保存到: {summary_file}")
    print(f"\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
