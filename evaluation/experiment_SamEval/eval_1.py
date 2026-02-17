# %%
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import json
import pickle

import numpy as np
import pandas as pd
import torch
import rdkit
from tqdm import tqdm

from lightning_fabric.utilities.seed import seed_everything
seed_everything(0)

from shepherd.lightning_module import LightningModule
from shepherd.inference import *
from shepherd.extract import create_rdkit_molecule_from_mol

from shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates
from shepherd.shepherd_score_utils.generate_point_cloud import (
    get_atomic_vdw_radii, 
    get_molecular_surface,
    get_electrostatics_given_point_charges,
)
from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores

from shepherd_score.container import Molecule
from shepherd_score.conformer_generation import embed_conformer_from_smiles
from shepherd_score.evaluations.evaluate import (
    ConfEval,
    UnconditionalEvalPipeline,
    ConsistencyEvalPipeline,
    ConditionalEvalPipeline,
)

# %% [markdown]
# # 加载分子

# %%
# ==================== 加载三种模型的采样结果并按参考分子分组 ====================

def convert_sample_format(sample):
    """将单个样本的JSON数据转换为numpy数组格式"""
    modal_keys = ['x1', 'x2', 'x3', 'x4']
    
    for modal_key in modal_keys:
        if modal_key in sample and isinstance(sample[modal_key], dict):
            for data_key in sample[modal_key]:
                if isinstance(sample[modal_key][data_key], list):
                    sample[modal_key][data_key] = np.array(sample[modal_key][data_key])
    
    return sample

def load_and_group_nested_format(data):
    """
    加载嵌套字典格式的数据（按分子分组）
    格式: {molecule_0: {n_atoms_X: {samples: [...]}}, molecule_1: {...}, ...}
    """
    grouped = {0: [], 1: [], 2: []}
    all_samples = []
    
    for mol_key in sorted(data.keys()):  # molecule_0, molecule_1, molecule_2
        mol_idx = int(mol_key.split('_')[1])  # 提取分子索引
        mol_data = data[mol_key]
        
        # 遍历该分子下的所有 n_atoms_X 子结构
        for n_atoms_key in mol_data.keys():
            samples = mol_data[n_atoms_key].get('samples', [])
            
            for sample in samples:
                sample = convert_sample_format(sample)
                sample['ref_mol_index'] = mol_idx
                if mol_idx in grouped:
                    grouped[mol_idx].append(sample)
                all_samples.append(sample)
    
    return all_samples, grouped

# ========== 定义文件路径 ==========
dpo_json_file = '/home1/zhh/workspace/SPD/evaluation/core_data/data/1/dpo/generated_samples_all_molecules_last.ckpt.json'
origin_json_file = '/home1/zhh/workspace/SPD/evaluation/core_data/data/1/origin/generated_samples_all_molecules.json'
spd_json_file = '/home1/zhh/workspace/SPD/evaluation/core_data/data/1/DIS/33/generated_samples_all_molecules_last_33epoch.ckpt.json'

# ========== 1. 加载DPO模型采样结果 ==========
with open(dpo_json_file, 'r', encoding='utf-8') as f:
    dpo_data = json.load(f)

dpo_samples, dpo_grouped = load_and_group_nested_format(dpo_data)
print(f"📊 DPO模型: 总共 {len(dpo_samples)} 个样本")
for ref_idx, samples in dpo_grouped.items():
    print(f"   - 参考分子{ref_idx}: {len(samples)} 个样本")

# ========== 2. 加载原始Shepherd模型采样结果 ==========
with open(origin_json_file, 'r', encoding='utf-8') as f:
    origin_data = json.load(f)

origin_samples, origin_grouped = load_and_group_nested_format(origin_data)
print(f"📊 原始Shepherd模型: 总共 {len(origin_samples)} 个样本")
for ref_idx, samples in origin_grouped.items():
    print(f"   - 参考分子{ref_idx}: {len(samples)} 个样本")

# ========== 3. 加载自训练SPD模型采样结果 ==========
with open(spd_json_file, 'r', encoding='utf-8') as f:
    spd_data = json.load(f)

spd_samples, spd_grouped = load_and_group_nested_format(spd_data)
print(f"📊 自训练SPD模型: 总共 {len(spd_samples)} 个样本")
for ref_idx, samples in spd_grouped.items():
    print(f"   - 参考分子{ref_idx}: {len(samples)} 个样本")

# ========== 汇总 ==========
all_model_samples = {
    'DPO': dpo_samples,
    'Origin_Shepherd': origin_samples,
    'SPD': spd_samples,
}

all_model_grouped = {
    'DPO': dpo_grouped,
    'Origin_Shepherd': origin_grouped,
    'SPD': spd_grouped,
}

print(f"\n{'='*60}")
print("📋 三种模型采样数据加载并分组完成:")
for model_name, grouped in all_model_grouped.items():
    total = sum(len(s) for s in grouped.values())
    print(f"  - {model_name}: {total} 个样本")
    for ref_idx, samples in grouped.items():
        print(f"      参考分子{ref_idx}: {len(samples)} 个")
print(f"{'='*60}")

# %%
# ==================== 对三种模型分别进行ConfEval评估 ====================

import json
from datetime import datetime
import pandas as pd  # 添加pandas导入

def evaluate_samples(samples, model_name):
    """对一组样本进行ConfEval评估"""
    evaluation_results = []
    
    print(f"\n{'='*60}")
    print(f"🔬 开始评估 {model_name} 模型的 {len(samples)} 个样本")
    print(f"{'='*60}")
    
    for i, structure in enumerate(samples):
        print(f"正在评估第 {i+1}/{len(samples)} 个生成结构...")
        
        try:
            atoms = structure['x1']['atoms']
            positions = structure['x1']['positions']
            bonds = structure['x1'].get('bonds', None)

            if isinstance(atoms, np.ndarray):
                atoms = atoms.flatten()
            if isinstance(positions, np.ndarray) and positions.ndim == 2:
                if positions.shape[1] != 3:
                    print(f"警告：第 {i+1} 个结构的位置坐标维度不正确: {positions.shape}")
                    continue
            
            conf_eval = ConfEval(atoms, positions, solvent='water', bonds=bonds)
            eval_df = conf_eval.to_pandas()
            
            # 提取所有指标值
            result_dict = {
                'structure_id': i,
                'model_name': model_name,
                'num_atoms': len(atoms),
                'is_valid': conf_eval.is_valid,
                'evaluation_data': {}  # 存储所有eval_df的内容
            }
            
            # 将eval_df的所有内容转换为可序列化的格式
            for key, value in eval_df.items():
                # 处理不同类型的值
                if isinstance(value, (int, float, bool, str)):
                    result_dict['evaluation_data'][key] = value
                elif isinstance(value, np.ndarray):
                    result_dict['evaluation_data'][key] = value.tolist()
                elif hasattr(value, 'item'):  # numpy scalar
                    result_dict['evaluation_data'][key] = value.item()
                elif value is None or pd.isna(value):
                    result_dict['evaluation_data'][key] = None
                else:
                    # 对于其他类型，尝试转换为字符串
                    try:
                        result_dict['evaluation_data'][key] = str(value)
                    except:
                        result_dict['evaluation_data'][key] = None
            
            # 为了向后兼容，仍然在顶层保留主要指标
            for metric in ['QED', 'SA_score', 'logP', 'strain_energy']:
                val = eval_df.get(metric, None)
                if val is not None:
                    try:
                        result_dict[metric] = float(val)
                    except:
                        result_dict[metric] = None
                else:
                    result_dict[metric] = None
            
            evaluation_results.append(result_dict)
            
            # 安全打印
            qed_str = f"{result_dict['QED']:.3f}" if result_dict['QED'] is not None else "N/A"
            sa_str = f"{result_dict.get('SA_score', 'N/A'):.3f}" if result_dict.get('SA_score') is not None else "N/A"
            print(f"  ✓ 第 {i+1} 个结构评估完成: QED={qed_str}, SA_score={sa_str}")
            
        except Exception as e:
            print(f"  ✗ 第 {i+1} 个结构评估失败: {str(e)}")
            # 即使失败也记录基本信息
            result_dict = {
                'structure_id': i,
                'model_name': model_name,
                'num_atoms': len(atoms) if 'atoms' in locals() else None,
                'is_valid': False,
                'evaluation_data': None,
                'error': str(e)
            }
            evaluation_results.append(result_dict)
            continue
    
    print(f"\n✅ {model_name} 模型: 成功评估 {len([r for r in evaluation_results if r['evaluation_data'] is not None])}/{len(samples)} 个结构")
    return evaluation_results

# 对三种模型分别进行评估
all_conf_eval_results = {}

for model_name, samples in all_model_samples.items():
    all_conf_eval_results[model_name] = evaluate_samples(samples, model_name)

print(f"\n{'='*60}")
print("📊 所有模型ConfEval评估完成!")
print(f"{'='*60}")

# 保存所有评估结果到JSON文件
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"conf_eval_results_{timestamp}.json"

# 将结果转换为可序列化的格式
serializable_results = {}
for model_name, results in all_conf_eval_results.items():
    serializable_results[model_name] = results

# 保存到JSON文件
with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(serializable_results, f, ensure_ascii=False, indent=2)

print(f"\n💾 评估结果已保存到: {output_filename}")
print(f"   - 文件包含 {len(serializable_results)} 个模型的评估数据")
for model_name, results in serializable_results.items():
    print(f"   - {model_name}: {len(results)} 个样本")

# %%
# ==================== 对评估数据进行统计分析 ====================

import json
import pandas as pd
import numpy as np
from collections import defaultdict

# 加载评估结果JSON文件
json_file_path = "/home1/zhh/workspace/SPD/evaluation/experiment_SamEval/conf_eval_results_20260215_181944.json"

print(f"📊 加载评估数据: {json_file_path}")
with open(json_file_path, 'r', encoding='utf-8') as f:
    eval_results = json.load(f)

print(f"✅ 成功加载 {len(eval_results)} 个模型的评估数据")
print(f"   模型列表: {list(eval_results.keys())}")

# ==================== 1. 基础统计信息 ====================
print("\n" + "="*80)
print("📈 1. 基础统计信息")
print("="*80)

for model_name, results in eval_results.items():
    print(f"\n🔹 {model_name} 模型:")
    
    # 统计有效样本数
    total_samples = len(results)
    valid_samples = sum(1 for r in results if r.get('is_valid', False))
    valid_post_opt = sum(1 for r in results if r.get('evaluation_data') is not None and r.get('evaluation_data', {}).get('is_valid_post_opt', False))
    graph_consistent = sum(1 for r in results if r.get('evaluation_data') is not None and r.get('evaluation_data', {}).get('is_graph_consistent', False))
    
    print(f"   总样本数: {total_samples}")
    print(f"   初始有效: {valid_samples} ({valid_samples/total_samples*100:.1f}%)")
    print(f"   优化后有效: {valid_post_opt} ({valid_post_opt/total_samples*100:.1f}%)")
    print(f"   图一致性: {graph_consistent} ({graph_consistent/total_samples*100:.1f}%)")

# ==================== 2. 关键指标统计 ====================
print("\n" + "="*80)
print("📊 2. 关键化学性质指标统计")
print("="*80)

# 定义要统计的关键指标
key_metrics = ['QED', 'SA_score', 'logP', 'strain_energy', 'fsp3', 'energy']

# 为每个模型计算统计信息
model_stats = {}

for model_name, results in eval_results.items():
    stats = defaultdict(list)
    
    # 收集每个指标的值
    for result in results:
        # 优先使用evaluation_data中的值
        eval_data = result.get('evaluation_data', {})
        
        for metric in key_metrics:
            # 尝试多个可能的键名
            value = None
            
            # 从evaluation_data中获取（先检查eval_data是否存在）
            if eval_data is not None and metric in eval_data and eval_data[metric] is not None:
                value = eval_data[metric]
            # 从顶层获取
            elif metric in result and result[metric] is not None:
                value = result[metric]
            # 尝试post_opt版本
            elif eval_data is not None and f"{metric}_post_opt" in eval_data and eval_data[f"{metric}_post_opt"] is not None:
                value = eval_data[f"{metric}_post_opt"]
            
            # 添加有效值
            if value is not None and isinstance(value, (int, float)) and not np.isnan(value):
                stats[metric].append(float(value))
    
    model_stats[model_name] = stats

# 创建统计表格
summary_data = []

for model_name in eval_results.keys():
    for metric in key_metrics:
        values = model_stats[model_name][metric]
        if len(values) > 0:
            summary_data.append({
                '模型': model_name,
                '指标': metric,
                '样本数': len(values),
                '平均值': np.mean(values),
                '标准差': np.std(values),
                '最小值': np.min(values),
                '25%分位': np.percentile(values, 25),
                '中位数': np.median(values),
                '75%分位': np.percentile(values, 75),
                '最大值': np.max(values)
            })

# 创建DataFrame并显示
if summary_data:
    df_summary = pd.DataFrame(summary_data)
    
    # 按模型分组显示
    for model in eval_results.keys():
        print(f"\n📌 {model} 模型指标统计:")
        model_df = df_summary[df_summary['模型'] == model]
        
        # 格式化显示
        for _, row in model_df.iterrows():
            print(f"\n   {row['指标']}:")
            print(f"      样本数: {row['样本数']}")
            print(f"      平均值±标准差: {row['平均值']:.4f} ± {row['标准差']:.4f}")
            print(f"      范围: [{row['最小值']:.4f}, {row['最大值']:.4f}]")
            print(f"      四分位数: Q1={row['25%分位']:.4f}, Q2={row['中位数']:.4f}, Q3={row['75%分位']:.4f}")

# ==================== 3. 模型间比较 ====================
print("\n" + "="*80)
print("🔍 3. 模型间关键指标对比")
print("="*80)

# 创建对比表
comparison_data = []
for model_name in eval_results.keys():
    row = {'模型': model_name}
    
    for metric in ['QED', 'SA_score', 'logP', 'strain_energy']:
        values = model_stats[model_name][metric]
        if len(values) > 0:
            row[f'{metric}_mean'] = np.mean(values)
            row[f'{metric}_std'] = np.std(values)
        else:
            row[f'{metric}_mean'] = np.nan
            row[f'{metric}_std'] = np.nan
    
    comparison_data.append(row)

df_comparison = pd.DataFrame(comparison_data)

# 格式化显示
print("\n平均值对比:")
print("-" * 70)
print(f"{'模型':<20} {'QED':>12} {'SA_score':>12} {'logP':>12} {'strain_energy':>14}")
print("-" * 70)

for _, row in df_comparison.iterrows():
    model = row['模型']
    qed = f"{row['QED_mean']:.3f}±{row['QED_std']:.3f}" if not np.isnan(row['QED_mean']) else "N/A"
    sa = f"{row['SA_score_mean']:.2f}±{row['SA_score_std']:.2f}" if not np.isnan(row['SA_score_mean']) else "N/A"
    logp = f"{row['logP_mean']:.2f}±{row['logP_std']:.2f}" if not np.isnan(row['logP_mean']) else "N/A"
    strain = f"{row['strain_energy_mean']:.2f}±{row['strain_energy_std']:.2f}" if not np.isnan(row['strain_energy_mean']) else "N/A"
    
    print(f"{model:<20} {qed:>12} {sa:>12} {logp:>12} {strain:>14}")

# ==================== 4. 分子大小分布 ====================
print("\n" + "="*80)
print("🧬 4. 分子大小分布")
print("="*80)

for model_name, results in eval_results.items():
    atom_counts = [r['num_atoms'] for r in results if 'num_atoms' in r]
    
    if atom_counts:
        print(f"\n{model_name}:")
        print(f"  平均原子数: {np.mean(atom_counts):.1f} ± {np.std(atom_counts):.1f}")
        print(f"  范围: {min(atom_counts)} - {max(atom_counts)}")
        
        # 分布统计
        bins = [0, 30, 50, 70, 90, 1000]
        hist, _ = np.histogram(atom_counts, bins=bins)
        print("  分布:")
        for i in range(len(bins)-1):
            if bins[i+1] == 1000:
                label = f"    >{bins[i]}原子"
            else:
                label = f"    {bins[i]}-{bins[i+1]}原子"
            print(f"{label}: {hist[i]} ({hist[i]/len(atom_counts)*100:.1f}%)")

# ==================== 5. 失败样本分析 ====================
print("\n" + "="*80)
print("❌ 5. 失败样本分析")
print("="*80)

for model_name, results in eval_results.items():
    print(f"\n{model_name}:")
    
    # 统计失败原因
    failed_samples = [r for r in results if r.get('evaluation_data') is None or not r.get('is_valid', True)]
    
    if failed_samples:
        print(f"  失败样本数: {len(failed_samples)} ({len(failed_samples)/len(results)*100:.1f}%)")
        
        # 分析错误信息
        error_types = defaultdict(int)
        for sample in failed_samples:
            error_msg = sample.get('error', 'Unknown error')
            # 简化错误信息
            if 'rdkit' in error_msg.lower():
                error_types['RDKit错误'] += 1
            elif 'atom' in error_msg.lower():
                error_types['原子类型错误'] += 1
            elif 'bond' in error_msg.lower():
                error_types['键错误'] += 1
            else:
                error_types['其他错误'] += 1
        
        print("  错误类型分布:")
        for error_type, count in error_types.items():
            print(f"    {error_type}: {count}")
    else:
        print("  所有样本评估成功 ✓")

# ==================== 6. 保存统计报告 ====================
print("\n" + "="*80)
print("💾 6. 保存统计报告")
print("="*80)

# 创建详细的统计报告
report = {
    'summary': {
        'total_models': len(eval_results),
        'models': list(eval_results.keys()),
        'evaluation_date': json_file_path.split('_')[-1].split('.')[0]
    },
    'model_statistics': {}
}

for model_name, results in eval_results.items():
    model_report = {
        'total_samples': len(results),
        'valid_samples': sum(1 for r in results if r.get('is_valid', False)),
        'metrics': {}
    }
    
    # 添加每个指标的统计信息
    for metric in key_metrics:
        values = model_stats[model_name][metric]
        if values:
            model_report['metrics'][metric] = {
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
    
    report['model_statistics'][model_name] = model_report

# 保存报告
report_file = f"evaluation_statistics_report_{json_file_path.split('_')[-1]}"
with open(report_file, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"✅ 统计报告已保存至: {report_file}")

print("\n" + "="*80)
print("✨ 评估数据统计分析完成！")
print("="*80)

# %%
# ==================== 创建所有指标的完整对比表格 ====================

# 获取所有可能的数值指标
all_metrics = set()
for model_name in model_stats.keys():
    all_metrics.update(model_stats[model_name].keys())

# 过滤出有数据的指标
valid_metrics = []
for metric in sorted(all_metrics):
    has_data = any(len(model_stats[model][metric]) > 0 for model in model_stats.keys())
    if has_data:
        valid_metrics.append(metric)

print("="*100)
print("📊 所有数值指标的完整对比表")
print("="*100)

# 按指标分组创建表格
for metric in valid_metrics:
    print(f"\n{metric}:")
    print("-" * 80)
    print(f"{'模型':<20} {'平均值':>10} {'标准差':>10} {'最小值':>10} {'最大值':>10} {'中位数':>10}")
    print("-" * 80)
    
    for model_name in eval_results.keys():
        values = model_stats[model_name][metric]
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            median_val = np.median(values)
            
            print(f"{model_name:<20} {mean_val:>10.3f} {std_val:>10.3f} {min_val:>10.3f} {max_val:>10.3f} {median_val:>10.3f}")
        else:
            print(f"{model_name:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

# 创建汇总表（平均值±标准差格式）
print("\n" + "="*100)
print("📋 综合对比表（平均值±标准差）")
print("="*100)

# 准备表头
header = "模型" + " " * 16
for metric in valid_metrics:
    if len(metric) < 12:
        header += f"{metric:>14}"
    else:
        header += f"{metric[:11]:>14}"

print(header)
print("-" * len(header))

# 填充数据
for model_name in eval_results.keys():
    row = f"{model_name:<20}"
    
    for metric in valid_metrics:
        values = model_stats[model_name][metric]
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            # 根据数值大小选择合适的精度
            if abs(mean_val) < 0.01:
                cell = f"{mean_val:.1e}±{std_val:.1e}"
            elif abs(mean_val) < 1:
                cell = f"{mean_val:.3f}±{std_val:.3f}"
            elif abs(mean_val) < 10:
                cell = f"{mean_val:.2f}±{std_val:.2f}"
            else:
                cell = f"{mean_val:.1f}±{std_val:.1f}"
        else:
            cell = "N/A"
        
        row += f"{cell:>14}"
    
    print(row)

# 创建样本数对比
print("\n" + "="*100)
print("📊 有效样本数对比")
print("="*100)

header = "模型" + " " * 16
for metric in valid_metrics:
    if len(metric) < 10:
        header += f"{metric:>12}"
    else:
        header += f"{metric[:9]:>12}"

print(header)
print("-" * len(header))

for model_name in eval_results.keys():
    row = f"{model_name:<20}"
    
    for metric in valid_metrics:
        count = len(model_stats[model_name][metric])
        row += f"{count:>12}"
    
    print(row)

# %%
# 条件评估参考分子准备算法

# %%
with open('/home1/zhh/workspace/SPD/data/conformers/np/molblock_charges_NPs.pkl', 'rb') as f:
    molblocks_and_charges = pickle.load(f)

# 为每个天然产物分子创建参考分子
ref_molecules = {}  # 存储每个源分子索引对应的参考分子

print("🔬 创建参考分子对象...")

for mol_index in range(len(molblocks_and_charges)):
    print(f"\n📋 处理天然产物分子 {mol_index}...")
    
    # 从molblock创建RDKit分子对象,保留氢原子
    mol = rdkit.Chem.MolFromMolBlock(molblocks_and_charges[mol_index][0], removeHs=False)
    charges = np.array(molblocks_and_charges[mol_index][1])
    
    # 创建标准化的参考分子对象
    ref_molec = Molecule(
        mol, 
        num_surf_points=200,
        probe_radius=1.2,
        pharm_multi_vector=False
    )
    
    # 存储参考分子
    ref_molecules[mol_index] = ref_molec
    print(f"  ✅ 分子 {mol_index} 参考对象创建完成")

print(f"\n✅ 共创建了 {len(ref_molecules)} 个参考分子对象")

# 显示样本分布统计（使用已加载的数据）
from collections import Counter
print(f"\n📊 样本分布统计:")
for model_name, grouped in all_model_grouped.items():
    print(f"\n  {model_name}:")
    for ref_idx, samples in grouped.items():
        if ref_idx in ref_molecules:
            print(f"    - 参考分子 {ref_idx}: {len(samples)} 个生成样本 -> 参考分子已创建 ✅")
        else:
            print(f"    - 参考分子 {ref_idx}: {len(samples)} 个生成样本 -> 参考分子缺失 ❌")

# %%
# 条件评估管道核心算法

# %%
# ==================== 条件评估：对三种模型分别按参考分子分组评估 ====================
from collections import defaultdict
from shepherd.extract import create_rdkit_molecule

def build_generated_mols_for_group(samples):
    """将样本列表转换为ConditionalEvalPipeline所需的格式"""
    generated_mols = []
    
    for sample in samples:
        try:
            rdkit_mol = create_rdkit_molecule(sample)
            
            if rdkit_mol is not None:
                atoms = np.array([a.GetAtomicNum() for a in rdkit_mol.GetAtoms()])
                positions = rdkit_mol.GetConformer().GetPositions()
                generated_mols.append((atoms, positions))
        except Exception as e:
            continue
    
    return generated_mols

# 存储所有模型的条件评估结果
all_cond_eval_results = {}  # {model_name: {ref_mol_idx: (properties_df, global_attr)}}

print("="*80)
print("🎯 开始条件评估：对三种模型分别按参考分子分组评估")
print("="*80)

for model_name, grouped in all_model_grouped.items():
    print(f"\n{'='*60}")
    print(f"🔬 评估模型: {model_name}")
    print(f"{'='*60}")
    
    model_results = {}
    
    for ref_mol_idx, samples in grouped.items():
        print(f"\n📋 参考分子 {ref_mol_idx}: {len(samples)} 个样本")
        
        # 检查参考分子是否存在
        if ref_mol_idx not in ref_molecules:
            print(f"  ❌ 缺少参考分子 {ref_mol_idx}，跳过")
            continue
        
        # 构建生成分子列表
        generated_mols = build_generated_mols_for_group(samples)
        
        if len(generated_mols) == 0:
            print(f"  ⚠️ 没有有效的生成分子，跳过")
            continue
        
        print(f"  📊 有效生成分子: {len(generated_mols)}/{len(samples)}")
        
        try:
            # 初始化条件评估管道
            cond_pipe = ConditionalEvalPipeline(
                ref_molecules[ref_mol_idx],
                generated_mols=generated_mols,
                condition='all',
                num_surf_points=200,
                pharm_multi_vector=False,
                solvent=None
            )
            
            # 执行评估
            cond_pipe.evaluate(verbose=False)
            
            # 获取结果
            properties_df, global_attr = cond_pipe.to_pandas()
            
            model_results[ref_mol_idx] = {
                'properties_df': properties_df,
                'global_attr': global_attr,
                'num_samples': len(samples),
                'num_valid': len(generated_mols),
            }
            
            print(f"  ✅ 评估完成")
            
        except Exception as e:
            print(f"  ❌ 评估失败: {str(e)}")
            continue
    
    all_cond_eval_results[model_name] = model_results
    print(f"\n✅ {model_name} 模型: 完成 {len(model_results)}/3 组评估")

print(f"\n{'='*80}")
print("🎉 所有模型条件评估完成!")
print(f"{'='*80}")

# %%
# ==================== 条件评估结果统计与打印 ====================
import pandas as pd

print("="*80)
print("📊 条件评估结果统计")
print("="*80)

# 收集所有模型的统计数据
cond_stats_all = []

for model_name, model_results in all_cond_eval_results.items():
    print(f"\n{'='*60}")
    print(f"🔹 {model_name} 模型条件评估结果")
    print(f"{'='*60}")
    
    model_stats = {
        'Model': model_name,
        'Total_Groups': len(model_results),
    }
    
    # 收集各组的关键指标
    all_sims_surf = []
    all_sims_esp = []
    all_sims_pharm = []
    all_rmsds = []
    total_samples = 0
    total_valid = 0
    
    for ref_mol_idx, result in model_results.items():
        properties_df = result['properties_df']
        global_attr = result['global_attr']
        num_samples = result['num_samples']
        num_valid = result['num_valid']
        
        total_samples += num_samples
        total_valid += num_valid
        
        print(f"\n  📋 参考分子 {ref_mol_idx}:")
        print(f"     样本数: {num_samples} | 有效: {num_valid}")
        
        # 提取关键相似度指标
        if hasattr(properties_df, 'index'):
            # properties_df 是 Series
            for key in properties_df.index:
                value = properties_df[key]
                if 'sims_surf' in str(key).lower():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        all_sims_surf.append(value)
                        print(f"     {key}: {value:.4f}")
                elif 'sims_esp' in str(key).lower():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        all_sims_esp.append(value)
                        print(f"     {key}: {value:.4f}")
                elif 'sims_pharm' in str(key).lower():
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        all_sims_pharm.append(value)
                        print(f"     {key}: {value:.4f}")
        
        # 从global_attr提取RMSD
        if hasattr(global_attr, 'index') and 'rmsds' in global_attr.index:
            rmsd_values = global_attr['rmsds']
            if hasattr(rmsd_values, '__iter__'):
                for v in rmsd_values:
                    if isinstance(v, (int, float)) and not np.isnan(v):
                        all_rmsds.append(v)
            elif isinstance(rmsd_values, (int, float)) and not np.isnan(rmsd_values):
                all_rmsds.append(rmsd_values)
    
    # 计算模型整体统计
    model_stats['Total_Samples'] = total_samples
    model_stats['Total_Valid'] = total_valid
    model_stats['Valid_Rate'] = f"{total_valid/total_samples*100:.1f}%" if total_samples > 0 else "N/A"
    
    if all_sims_surf:
        model_stats['Sims_Surf_Mean'] = np.mean(all_sims_surf)
        model_stats['Sims_Surf_Std'] = np.std(all_sims_surf)
    if all_sims_esp:
        model_stats['Sims_ESP_Mean'] = np.mean(all_sims_esp)
        model_stats['Sims_ESP_Std'] = np.std(all_sims_esp)
    if all_sims_pharm:
        model_stats['Sims_Pharm_Mean'] = np.mean(all_sims_pharm)
        model_stats['Sims_Pharm_Std'] = np.std(all_sims_pharm)
    if all_rmsds:
        model_stats['RMSD_Mean'] = np.mean(all_rmsds)
        model_stats['RMSD_Std'] = np.std(all_rmsds)
    
    cond_stats_all.append(model_stats)
    
    # 打印模型汇总
    print(f"\n  📊 {model_name} 模型汇总:")
    print(f"     总样本: {total_samples} | 有效: {total_valid} ({model_stats['Valid_Rate']})")
    if all_sims_surf:
        print(f"     表面相似度: {np.mean(all_sims_surf):.4f} ± {np.std(all_sims_surf):.4f}")
    if all_sims_esp:
        print(f"     静电势相似度: {np.mean(all_sims_esp):.4f} ± {np.std(all_sims_esp):.4f}")
    if all_sims_pharm:
        print(f"     药效团相似度: {np.mean(all_sims_pharm):.4f} ± {np.std(all_sims_pharm):.4f}")
    if all_rmsds:
        print(f"     RMSD: {np.mean(all_rmsds):.4f} ± {np.std(all_rmsds):.4f}")

# 打印对比表格
print(f"\n{'='*80}")
print("📋 三种模型条件评估对比表格")
print(f"{'='*80}")

comparison_data = []
for stats in cond_stats_all:
    row = {
        'Model': stats['Model'],
        'Samples': stats.get('Total_Samples', 'N/A'),
        'Valid_Rate': stats.get('Valid_Rate', 'N/A'),
    }
    if 'Sims_Surf_Mean' in stats:
        row['Surf_Sim'] = f"{stats['Sims_Surf_Mean']:.3f}±{stats['Sims_Surf_Std']:.3f}"
    else:
        row['Surf_Sim'] = 'N/A'
    if 'Sims_ESP_Mean' in stats:
        row['ESP_Sim'] = f"{stats['Sims_ESP_Mean']:.3f}±{stats['Sims_ESP_Std']:.3f}"
    else:
        row['ESP_Sim'] = 'N/A'
    if 'RMSD_Mean' in stats:
        row['RMSD'] = f"{stats['RMSD_Mean']:.3f}±{stats['RMSD_Std']:.3f}"
    else:
        row['RMSD'] = 'N/A'
    comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

print(f"\n{'='*80}")
print("✅ 条件评估统计完成!")
print(f"{'='*80}")


