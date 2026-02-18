# %%
"""
统一评估脚本：一次性完成 ConfEval（构象评估）和 CondEval（条件评估），
只有同时通过两种评估的分子才参与最终指标统计。
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

import json
import glob
import pickle
import multiprocessing
from datetime import datetime
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd
import torch
import rdkit
from tqdm import tqdm

from lightning_fabric.utilities.seed import seed_everything
seed_everything(0)

from shepherd.lightning_module import LightningModule
from shepherd.inference import *
from shepherd.extract import create_rdkit_molecule_from_mol, create_rdkit_molecule

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

# =============================================================================
# 第一阶段：数据加载
# =============================================================================

# %%

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

    for mol_key in sorted(data.keys()):
        mol_idx = int(mol_key.split('_')[1])
        mol_data = data[mol_key]

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
MODEL_FILES = {
    'DPO': '/home1/zhh/workspace/SPD/evaluation/core_data/data/1/dpo/generated_samples_all_molecules_last.ckpt.json',
    'Origin_Shepherd': '/home1/zhh/workspace/SPD/evaluation/core_data/data/1/origin/generated_samples_all_molecules.json',
    'SPD': '/home1/zhh/workspace/SPD/evaluation/core_data/data/1/DIS/33/generated_samples_all_molecules_last_33epoch.ckpt.json',
}

REF_MOL_PKL = '/home1/zhh/workspace/SPD/data/conformers/np/molblock_charges_NPs.pkl'

# ========== 并行配置 ==========
# NUM_WORKERS: ConfEval 并行 worker 数（多进程并行评估多个样本）
# NUM_PROCESSES: 每个 worker 内部 xtb 优化的进程数
# 约束：NUM_WORKERS * NUM_PROCESSES <= 可用 CPU 核心数
_AVAILABLE_CPUS = multiprocessing.cpu_count() or 1
NUM_WORKERS = min(8, _AVAILABLE_CPUS)     # ConfEval 并行 worker 数
NUM_PROCESSES = 1                          # 每个 worker 的 xtb 进程数
COND_NUM_WORKERS = min(8, _AVAILABLE_CPUS) # CondEval 并行 worker 数
print(f"\n⚙️  并行配置: CPU核心数={_AVAILABLE_CPUS}, "
      f"ConfEval workers={NUM_WORKERS}, CondEval workers={COND_NUM_WORKERS}, "
      f"xtb processes/worker={NUM_PROCESSES}")

# =============================================================================
# 缓存检测：查找上一次的评估结果，判断哪些模型需要重新评估
# =============================================================================

def find_latest_cache(pattern='unified_eval_results_*.json'):
    """查找当前目录下最新的评估结果文件"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = sorted(glob.glob(os.path.join(script_dir, pattern)))
    if candidates:
        return candidates[-1]  # 按文件名排序，最后一个即为最新
    return None


def detect_changed_models(model_files, cache_path):
    """
    对比当前 MODEL_FILES 与缓存中记录的文件路径，
    返回 (models_to_eval, cached_data):
      - models_to_eval: set of model names that need re-evaluation
      - cached_data: the loaded cache dict (or None if no cache)
    """
    if cache_path is None:
        return set(model_files.keys()), None

    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
    except Exception as e:
        print(f"⚠️ 无法加载缓存文件 {cache_path}: {e}")
        return set(model_files.keys()), None

    cached_model_files = cached_data.get('metadata', {}).get('model_files', {})

    models_to_eval = set()
    for model_name, file_path in model_files.items():
        cached_path = cached_model_files.get(model_name)
        if cached_path != file_path:
            models_to_eval.add(model_name)
            print(f"  🔄 {model_name}: 路径已变化，需要重新评估")
            print(f"     旧: {cached_path}")
            print(f"     新: {file_path}")
        else:
            print(f"  ✅ {model_name}: 路径未变化，将复用缓存")

    # 检查是否有新增的模型
    for model_name in model_files:
        if model_name not in cached_model_files:
            models_to_eval.add(model_name)
            print(f"  🆕 {model_name}: 新增模型，需要评估")

    return models_to_eval, cached_data


# 检测缓存
print("\n🔍 检测缓存...")
cache_path = find_latest_cache()
if cache_path:
    print(f"  找到缓存文件: {cache_path}")
else:
    print("  未找到缓存文件，将对所有模型进行完整评估")

models_to_eval, cached_data = detect_changed_models(MODEL_FILES, cache_path)

if len(models_to_eval) == 0:
    print("\n✅ 所有模型路径均未变化，无需重新评估。")
    print("   如需强制重新评估，请删除缓存文件后重新运行。")
    # 直接使用缓存数据进入统计阶段
    all_unified_results = cached_data['per_sample_results']
    all_cond_group_results = {
        model_name: {int(k): v for k, v in groups.items()}
        for model_name, groups in cached_data['cond_group_results'].items()
    }
    SKIP_EVALUATION = True
else:
    print(f"\n📋 需要评估的模型: {models_to_eval}")
    SKIP_EVALUATION = False

# 仅加载需要评估的模型数据（未变化的模型无需加载原始采样数据）
all_model_samples = {}
all_model_grouped = {}

for model_name, json_file in MODEL_FILES.items():
    if model_name not in models_to_eval:
        print(f"\n⏭️  跳过加载 {model_name}（将使用缓存）")
        continue
    print(f"\n加载 {model_name} 模型数据: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples, grouped = load_and_group_nested_format(data)
    all_model_samples[model_name] = samples
    all_model_grouped[model_name] = grouped
    print(f"  📊 {model_name}: 总共 {len(samples)} 个样本")
    for ref_idx, s_list in grouped.items():
        print(f"     - 参考分子{ref_idx}: {len(s_list)} 个样本")

print(f"\n{'='*60}")
print("📋 数据加载完成")
print(f"{'='*60}")

# =============================================================================
# 第二阶段：准备参考分子（用于条件评估，仅在需要评估时加载）
# =============================================================================

# %%
if not SKIP_EVALUATION:
    print("\n🔬 加载参考分子...")
    with open(REF_MOL_PKL, 'rb') as f:
        molblocks_and_charges = pickle.load(f)

    ref_molecules = {}
    for mol_index in range(len(molblocks_and_charges)):
        print(f"  处理参考分子 {mol_index}...")
        mol = rdkit.Chem.MolFromMolBlock(molblocks_and_charges[mol_index][0], removeHs=False)
        ref_molec = Molecule(
            mol,
            num_surf_points=200,
            probe_radius=1.2,
            pharm_multi_vector=False
        )
        ref_molecules[mol_index] = ref_molec
        print(f"  ✅ 参考分子 {mol_index} 创建完成")

    print(f"✅ 共创建了 {len(ref_molecules)} 个参考分子对象")
else:
    print("\n⏭️  跳过参考分子加载（全部使用缓存）")
    ref_molecules = {}

# =============================================================================
# 第三阶段：统一评估（ConfEval + CondEval 一次性完成）
# =============================================================================

# %%

def serialize_value(value):
    """将值转换为 JSON 可序列化的格式"""
    if isinstance(value, (int, float, bool, str)):
        return value
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif hasattr(value, 'item'):  # numpy scalar
        return value.item()
    elif value is None:
        return None
    elif isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    else:
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        try:
            return str(value)
        except Exception:
            return None


def _conf_eval_worker(args):
    """
    multiprocessing worker 函数：对单个样本执行 ConfEval。
    必须是模块顶级函数才能被 pickle 序列化。
    
    args: (structure_id, model_name, atoms, positions, bonds, num_processes)
    返回: (structure_id, result_dict, is_valid)
    """
    structure_id, model_name, atoms, positions, bonds, num_processes = args
    try:
        if isinstance(atoms, list):
            atoms = np.array(atoms)
        if isinstance(positions, list):
            positions = np.array(positions)
        if isinstance(bonds, list):
            bonds = np.array(bonds)

        if isinstance(atoms, np.ndarray):
            atoms = atoms.flatten()
        if isinstance(positions, np.ndarray) and positions.ndim == 2:
            if positions.shape[1] != 3:
                return structure_id, {
                    'structure_id': structure_id,
                    'model_name': model_name,
                    'num_atoms': len(atoms) if hasattr(atoms, '__len__') else None,
                    'conf_valid': False,
                    'conf_data': None,
                    'conf_error': f'位置坐标维度不正确: {positions.shape}'
                }, False

        conf_eval = ConfEval(atoms, positions, solvent='water', bonds=bonds,
                             num_processes=num_processes)
        eval_df = conf_eval.to_pandas()

        # 构建结果字典
        conf_data = {}
        for key, value in eval_df.items():
            conf_data[key] = serialize_value(value)

        result = {
            'structure_id': structure_id,
            'model_name': model_name,
            'num_atoms': len(atoms),
            'conf_valid': bool(conf_eval.is_valid),
            'conf_data': conf_data,
        }

        # 顶层保留主要指标，方便后续统计
        for metric in ['QED', 'SA_score', 'logP', 'strain_energy']:
            val = eval_df.get(metric, None)
            if val is not None:
                try:
                    result[metric] = float(val)
                except (TypeError, ValueError):
                    result[metric] = None
            else:
                result[metric] = None

        return structure_id, result, bool(conf_eval.is_valid)

    except Exception as e:
        return structure_id, {
            'structure_id': structure_id,
            'model_name': model_name,
            'num_atoms': None,
            'conf_valid': False,
            'conf_data': None,
            'conf_error': str(e),
        }, False


def run_conf_eval_parallel(samples, model_name, num_workers, num_processes):
    """
    并行执行 ConfEval：对一组样本使用多进程评估。
    
    返回:
        results_list: list of result_dict（按原始顺序）
        conf_valid_indices: list of (index, sample) -- conf 有效的样本
    """
    # 准备 worker 输入（提取 numpy 数据，避免传递复杂对象）
    worker_inputs = []
    sample_mapping = {}  # worker_id -> (original_index, sample)

    for i, sample in enumerate(samples):
        try:
            atoms = sample['x1']['atoms']
            positions = sample['x1']['positions']
            bonds = sample['x1'].get('bonds', None)

            # 转换为 list 以确保可 pickle
            atoms_data = atoms.tolist() if isinstance(atoms, np.ndarray) else atoms
            positions_data = positions.tolist() if isinstance(positions, np.ndarray) else positions
            bonds_data = bonds.tolist() if isinstance(bonds, np.ndarray) else bonds

            worker_inputs.append((i, model_name, atoms_data, positions_data, bonds_data, num_processes))
            sample_mapping[i] = sample
        except Exception as e:
            # 数据提取失败的样本直接标记为无效
            worker_inputs.append(None)
            sample_mapping[i] = sample

    # 过滤有效输入
    valid_inputs = [inp for inp in worker_inputs if inp is not None]
    failed_indices = {i for i, inp in enumerate(worker_inputs) if inp is None}

    # 初始化结果列表
    results_list = [None] * len(samples)
    conf_valid_items = []  # (index, sample)

    # 对失败的样本直接填充
    for idx in failed_indices:
        results_list[idx] = {
            'structure_id': idx,
            'model_name': model_name,
            'num_atoms': None,
            'conf_valid': False,
            'conf_data': None,
            'conf_error': '数据提取失败',
        }

    if len(valid_inputs) == 0:
        return results_list, conf_valid_items

    # 并行执行
    effective_workers = min(num_workers, len(valid_inputs))

    if effective_workers <= 1:
        # 单进程模式（样本太少时避免进程启动开销）
        for args in tqdm(valid_inputs, desc=f"    ConfEval ({model_name})"):
            sid, result, is_valid = _conf_eval_worker(args)
            results_list[sid] = result
            if is_valid:
                conf_valid_items.append((sid, sample_mapping[sid]))
    else:
        # 多进程模式
        try:
            # 使用 fork 而非 spawn：spawn 会重新导入模块，触发模块级代码重复执行
            # fork 直接复制父进程内存，对纯 CPU 的 RDKit/xtb 计算安全
            ctx = multiprocessing.get_context('fork')
            with ctx.Pool(effective_workers) as pool:
                for sid, result, is_valid in tqdm(
                    pool.imap_unordered(_conf_eval_worker, valid_inputs, chunksize=1),
                    total=len(valid_inputs),
                    desc=f"    ConfEval ({model_name}, {effective_workers} workers)"
                ):
                    results_list[sid] = result
                    if is_valid:
                        conf_valid_items.append((sid, sample_mapping[sid]))
        except Exception as e:
            print(f"    ⚠️ 多进程失败 ({e})，回退到单进程模式")
            for args in tqdm(valid_inputs, desc=f"    ConfEval ({model_name}, fallback)"):
                sid, result, is_valid = _conf_eval_worker(args)
                results_list[sid] = result
                if is_valid:
                    conf_valid_items.append((sid, sample_mapping[sid]))

    return results_list, conf_valid_items


def run_cond_eval_group(ref_mol, conf_valid_samples):
    """
    对一组 ConfEval 有效的样本执行条件评估。
    
    参数:
        ref_mol: 参考分子 Molecule 对象
        conf_valid_samples: list of (sample_index, sample_dict) -- 仅 conf_valid=True 的样本
    
    返回:
        cond_results: dict[sample_index] -> {cond_valid, cond_data}
        properties_df, global_attr: 原始的条件评估结果
    """
    # 构建 generated_mols 列表，同时记录映射关系
    generated_mols = []
    index_mapping = []  # generated_mols 中的位置 -> 原始 sample_index

    for sample_idx, sample in conf_valid_samples:
        try:
            rdkit_mol = create_rdkit_molecule(sample)
            if rdkit_mol is not None:
                atoms = np.array([a.GetAtomicNum() for a in rdkit_mol.GetAtoms()])
                positions = rdkit_mol.GetConformer().GetPositions()
                generated_mols.append((atoms, positions))
                index_mapping.append(sample_idx)
        except Exception:
            continue

    if len(generated_mols) == 0:
        return {}, None, None

    # 执行条件评估（启用多进程）
    cond_pipe = ConditionalEvalPipeline(
        ref_mol,
        generated_mols=generated_mols,
        condition='all',
        num_surf_points=200,
        pharm_multi_vector=False,
        solvent=None
    )
    cond_pipe.evaluate(
        num_workers=COND_NUM_WORKERS,
        num_processes=NUM_PROCESSES,
        verbose=True
    )
    properties_df, global_attr = cond_pipe.to_pandas()

    # 将 properties_df 和 global_attr 转换为每个样本的条件评估数据
    cond_results = {}

    # properties_df 通常是 Series（整组的汇总指标）
    cond_summary = {}
    if hasattr(properties_df, 'items'):
        for key, value in properties_df.items():
            cond_summary[str(key)] = serialize_value(value)
    elif hasattr(properties_df, 'to_dict'):
        cond_summary = {str(k): serialize_value(v) for k, v in properties_df.to_dict().items()}

    global_summary = {}
    if hasattr(global_attr, 'items'):
        for key, value in global_attr.items():
            global_summary[str(key)] = serialize_value(value)
    elif hasattr(global_attr, 'to_dict'):
        global_summary = {str(k): serialize_value(v) for k, v in global_attr.to_dict().items()}

    # 对每个成功传入 CondEval 的样本标记为 cond_valid=True
    for gen_idx, sample_idx in enumerate(index_mapping):
        cond_results[sample_idx] = {
            'cond_valid': True,
        }

    return cond_results, cond_summary, global_summary


# ==================== 统一评估主流程（增量模式） ====================

if not SKIP_EVALUATION:
    # 从缓存中预加载未变化模型的结果
    all_unified_results = {}
    all_cond_group_results = {}

    if cached_data is not None:
        for model_name in MODEL_FILES:
            if model_name not in models_to_eval:
                # 直接复用缓存
                all_unified_results[model_name] = cached_data['per_sample_results'][model_name]
                all_cond_group_results[model_name] = {
                    int(k): v
                    for k, v in cached_data['cond_group_results'][model_name].items()
                }
                print(f"\n📦 {model_name}: 已从缓存加载评估结果")

    print("\n" + "=" * 80)
    print("🚀 开始增量评估：仅评估路径已变化的模型")
    print("=" * 80)

    for model_name, grouped in all_model_grouped.items():
        # all_model_grouped 只包含需要重新评估的模型
        print(f"\n{'='*60}")
        print(f"🔬 模型: {model_name}")
        print(f"{'='*60}")

        model_results = []  # 每个样本一条记录
        model_cond_groups = {}  # ref_mol_idx -> {cond_summary, global_summary, ...}

        for ref_mol_idx, samples in grouped.items():
            print(f"\n  📋 参考分子 {ref_mol_idx}: {len(samples)} 个样本")

            # ------ 步骤1：并行 ConfEval ------
            global_id_offset = len(model_results)
            conf_results, conf_valid_raw = run_conf_eval_parallel(
                samples, model_name, NUM_WORKERS, NUM_PROCESSES
            )

            # 将结果整合到 model_results，并调整索引
            conf_valid_samples = []
            for i, result in enumerate(conf_results):
                global_id = global_id_offset + i
                result['structure_id'] = global_id
                result['ref_mol_index'] = ref_mol_idx
                result['local_index'] = i
                result['cond_valid'] = False
                result['both_valid'] = False
                model_results.append(result)

            # 将 conf_valid_raw 中的本地索引映射为全局索引
            for local_id, sample in conf_valid_raw:
                conf_valid_samples.append((global_id_offset + local_id, sample))

            conf_valid_count = len(conf_valid_samples)
            print(f"    ConfEval 完成: {conf_valid_count}/{len(samples)} 有效")

            # ------ 步骤2：对 conf 有效的样本组执行 CondEval ------
            if ref_mol_idx not in ref_molecules:
                print(f"    ❌ 缺少参考分子 {ref_mol_idx}，跳过条件评估")
                continue

            if conf_valid_count == 0:
                print(f"    ⚠️ 无有效构象，跳过条件评估")
                continue

            try:
                cond_results, cond_summary, global_summary = run_cond_eval_group(
                    ref_molecules[ref_mol_idx], conf_valid_samples
                )

                # 更新每个样本的 cond_valid 和 both_valid
                cond_valid_count = 0
                for global_id, is_cond_valid_info in cond_results.items():
                    model_results[global_id]['cond_valid'] = is_cond_valid_info['cond_valid']
                    model_results[global_id]['both_valid'] = (
                        model_results[global_id]['conf_valid'] and is_cond_valid_info['cond_valid']
                    )
                    if model_results[global_id]['both_valid']:
                        cond_valid_count += 1

                model_cond_groups[ref_mol_idx] = {
                    'cond_summary': cond_summary,
                    'global_summary': global_summary,
                    'num_samples': len(samples),
                    'num_conf_valid': conf_valid_count,
                    'num_both_valid': cond_valid_count,
                }

                print(f"    CondEval 完成: {cond_valid_count}/{conf_valid_count} 条件有效 (同时有效)")

            except Exception as e:
                print(f"    ❌ CondEval 评估失败: {str(e)}")
                model_cond_groups[ref_mol_idx] = {
                    'cond_summary': None,
                    'global_summary': None,
                    'num_samples': len(samples),
                    'num_conf_valid': conf_valid_count,
                    'num_both_valid': 0,
                    'error': str(e),
                }
                continue

        all_unified_results[model_name] = model_results
        all_cond_group_results[model_name] = model_cond_groups

        # 模型汇总
        total = len(model_results)
        conf_ok = sum(1 for r in model_results if r['conf_valid'])
        both_ok = sum(1 for r in model_results if r['both_valid'])
        print(f"\n  ✅ {model_name} 汇总: 总共 {total} | 构象有效 {conf_ok} | 联合有效 {both_ok}")

    # 打印数据来源汇总
    print(f"\n{'='*80}")
    print("🎉 增量评估完成!")
    print(f"{'='*80}")
    for model_name in MODEL_FILES:
        source = "🔄 重新评估" if model_name in models_to_eval else "📦 缓存复用"
        print(f"  {model_name}: {source}")

# =============================================================================
# 第四阶段：保存中间结果
# =============================================================================

# %%
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"unified_eval_results_{timestamp}.json"

save_data = {
    'metadata': {
        'timestamp': timestamp,
        'models': list(all_unified_results.keys()),
        'model_files': MODEL_FILES,
    },
    'per_sample_results': {
        model_name: results
        for model_name, results in all_unified_results.items()
    },
    'cond_group_results': {
        model_name: {
            str(ref_idx): group_data
            for ref_idx, group_data in groups.items()
        }
        for model_name, groups in all_cond_group_results.items()
    },
}

with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(save_data, f, ensure_ascii=False, indent=2)

print(f"\n💾 中间结果已保存到: {output_filename}")

# =============================================================================
# 第五阶段：统计分析（仅使用 both_valid=True 的分子）
# =============================================================================

# %%
print("\n" + "=" * 80)
print("📊 最终统计分析（仅统计构象+条件同时有效的分子）")
print("=" * 80)

# ==================== 1. 有效率对比 ====================
print("\n" + "=" * 80)
print("📈 1. 有效率对比")
print("=" * 80)

validity_table = []
for model_name, results in all_unified_results.items():
    total = len(results)
    conf_valid = sum(1 for r in results if r['conf_valid'])
    cond_valid = sum(1 for r in results if r['cond_valid'])
    both_valid = sum(1 for r in results if r['both_valid'])

    validity_table.append({
        '模型': model_name,
        '总样本': total,
        '构象有效': conf_valid,
        '构象有效率': f"{conf_valid/total*100:.1f}%",
        '条件有效': cond_valid,
        '条件有效率': f"{cond_valid/total*100:.1f}%",
        '联合有效': both_valid,
        '联合有效率': f"{both_valid/total*100:.1f}%",
    })

df_validity = pd.DataFrame(validity_table)
print(df_validity.to_string(index=False))

# ==================== 2. ConfEval 指标统计（仅 both_valid） ====================
print("\n" + "=" * 80)
print("📊 2. ConfEval 化学性质指标（仅联合有效分子）")
print("=" * 80)

conf_metrics = ['QED', 'SA_score', 'logP', 'strain_energy', 'fsp3', 'energy']
model_conf_stats = {}

for model_name, results in all_unified_results.items():
    stats = defaultdict(list)

    for r in results:
        if not r['both_valid']:
            continue  # ← 核心：仅联合有效的分子参与统计

        conf_data = r.get('conf_data', {})
        if conf_data is None:
            continue

        for metric in conf_metrics:
            value = None
            # 从 conf_data 获取
            if metric in conf_data and conf_data[metric] is not None:
                value = conf_data[metric]
            # 从顶层获取
            elif metric in r and r[metric] is not None:
                value = r[metric]
            # 尝试 post_opt 版本
            elif f"{metric}_post_opt" in conf_data and conf_data[f"{metric}_post_opt"] is not None:
                value = conf_data[f"{metric}_post_opt"]

            if value is not None and isinstance(value, (int, float)) and not np.isnan(value):
                stats[metric].append(float(value))

    model_conf_stats[model_name] = stats

# 打印统计表
for model_name in all_unified_results.keys():
    print(f"\n📌 {model_name} （联合有效分子）:")
    both_count = sum(1 for r in all_unified_results[model_name] if r['both_valid'])
    print(f"   联合有效样本数: {both_count}")

    for metric in conf_metrics:
        values = model_conf_stats[model_name][metric]
        if len(values) > 0:
            print(f"\n   {metric}:")
            print(f"      样本数: {len(values)}")
            print(f"      平均值±标准差: {np.mean(values):.4f} ± {np.std(values):.4f}")
            print(f"      范围: [{np.min(values):.4f}, {np.max(values):.4f}]")
            print(f"      四分位数: Q1={np.percentile(values, 25):.4f}, "
                  f"Q2={np.median(values):.4f}, Q3={np.percentile(values, 75):.4f}")

# ==================== 3. CondEval 条件评估指标 ====================
print("\n" + "=" * 80)
print("📊 3. CondEval 条件评估指标（按参考分子分组）")
print("=" * 80)

for model_name, groups in all_cond_group_results.items():
    print(f"\n{'='*60}")
    print(f"🔹 {model_name}")
    print(f"{'='*60}")

    for ref_idx, group_data in groups.items():
        print(f"\n  📋 参考分子 {ref_idx}:")
        print(f"     总样本: {group_data['num_samples']} | "
              f"构象有效: {group_data['num_conf_valid']} | "
              f"联合有效: {group_data['num_both_valid']}")

        if group_data.get('error'):
            print(f"     ❌ 错误: {group_data['error']}")
            continue

        cond_summary = group_data.get('cond_summary', {})
        if cond_summary:
            for key, value in cond_summary.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    key_lower = str(key).lower()
                    if any(kw in key_lower for kw in ['sims_surf', 'sims_esp', 'sims_pharm', 'rmsd']):
                        print(f"     {key}: {value:.4f}")

        global_summary = group_data.get('global_summary', {})
        if global_summary and 'rmsds' in global_summary:
            rmsd_values = global_summary['rmsds']
            if isinstance(rmsd_values, list):
                valid_rmsds = [v for v in rmsd_values if isinstance(v, (int, float)) and not np.isnan(v)]
                if valid_rmsds:
                    print(f"     RMSD (avg): {np.mean(valid_rmsds):.4f} ± {np.std(valid_rmsds):.4f}")

# ==================== 4. 模型间对比表格 ====================
print("\n" + "=" * 80)
print("🔍 4. 模型间关键指标对比（仅联合有效分子）")
print("=" * 80)

# --- 4a. ConfEval 对比 ---
print("\n📋 4a. ConfEval 指标对比（平均值±标准差）:")
print("-" * 90)
header = f"{'模型':<20}"
for metric in ['QED', 'SA_score', 'logP', 'strain_energy']:
    header += f"{metric:>16}"
print(header)
print("-" * 90)

for model_name in all_unified_results.keys():
    row = f"{model_name:<20}"
    for metric in ['QED', 'SA_score', 'logP', 'strain_energy']:
        values = model_conf_stats[model_name][metric]
        if len(values) > 0:
            cell = f"{np.mean(values):.3f}±{np.std(values):.3f}"
        else:
            cell = "N/A"
        row += f"{cell:>16}"
    print(row)

# --- 4b. CondEval 对比 ---
print(f"\n📋 4b. CondEval 指标对比:")
print("-" * 90)

# 从 cond_group_results 中提取所有模型的汇总相似度
cond_comparison = []
for model_name, groups in all_cond_group_results.items():
    all_surf, all_esp, all_pharm, all_rmsds = [], [], [], []

    for ref_idx, group_data in groups.items():
        cond_summary = group_data.get('cond_summary', {})
        if cond_summary is None:
            continue

        for key, value in cond_summary.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                key_lower = str(key).lower()
                if 'sims_surf' in key_lower:
                    all_surf.append(value)
                elif 'sims_esp' in key_lower:
                    all_esp.append(value)
                elif 'sims_pharm' in key_lower:
                    all_pharm.append(value)

        global_summary = group_data.get('global_summary', {})
        if global_summary and 'rmsds' in global_summary:
            rmsd_values = global_summary['rmsds']
            if isinstance(rmsd_values, list):
                all_rmsds.extend([v for v in rmsd_values if isinstance(v, (int, float)) and not np.isnan(v)])
            elif isinstance(rmsd_values, (int, float)) and not np.isnan(rmsd_values):
                all_rmsds.append(rmsd_values)

    # 汇总统计
    total_samples = sum(g['num_samples'] for g in groups.values())
    both_valid_total = sum(g['num_both_valid'] for g in groups.values())

    row = {
        'Model': model_name,
        'Samples': total_samples,
        'Both_Valid': both_valid_total,
        'Valid_Rate': f"{both_valid_total/total_samples*100:.1f}%" if total_samples > 0 else "N/A",
    }
    row['Surf_Sim'] = f"{np.mean(all_surf):.3f}±{np.std(all_surf):.3f}" if all_surf else "N/A"
    row['ESP_Sim'] = f"{np.mean(all_esp):.3f}±{np.std(all_esp):.3f}" if all_esp else "N/A"
    row['Pharm_Sim'] = f"{np.mean(all_pharm):.3f}±{np.std(all_pharm):.3f}" if all_pharm else "N/A"
    row['RMSD'] = f"{np.mean(all_rmsds):.3f}±{np.std(all_rmsds):.3f}" if all_rmsds else "N/A"

    cond_comparison.append(row)

df_cond_comp = pd.DataFrame(cond_comparison)
print(df_cond_comp.to_string(index=False))

# ==================== 5. 保存最终统计报告 ====================
report = {
    'summary': {
        'timestamp': timestamp,
        'models': list(all_unified_results.keys()),
        'note': '所有指标仅基于 conf_valid=True AND cond_valid=True 的联合有效分子计算',
    },
    'validity': validity_table,
    'conf_eval_statistics': {},
    'cond_eval_statistics': {},
}

for model_name in all_unified_results.keys():
    conf_stats = {}
    for metric in conf_metrics:
        values = model_conf_stats[model_name][metric]
        if values:
            conf_stats[metric] = {
                'count': len(values),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
            }
    report['conf_eval_statistics'][model_name] = conf_stats

for model_name, groups in all_cond_group_results.items():
    report['cond_eval_statistics'][model_name] = {
        str(ref_idx): group_data
        for ref_idx, group_data in groups.items()
    }

report_filename = f"unified_eval_report_{timestamp}.json"
with open(report_filename, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n💾 统计报告已保存到: {report_filename}")

print("\n" + "=" * 80)
print("✨ 统一评估流程全部完成！")
print(f"   中间结果: {output_filename}")
print(f"   统计报告: {report_filename}")
print("=" * 80)
