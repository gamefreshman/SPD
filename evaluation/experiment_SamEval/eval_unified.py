# %%
"""
统一评估脚本：一次性完成 ConfEval（构象评估）和 CondEval（条件评估），
只有同时通过两种评估的分子才参与最终指标统计。

缓存设计：
  - conf_eval_results.json: ConfEval 结果（独立保存）
  - cond_eval_results.json: CondEval 结果（独立保存）
  - 删除 cond_eval_results.json → 仅重跑 CondEval（跳过 ConfEval）
  - 删除 conf_eval_results.json → 全部重跑

使用 multiprocessing 并行加速评估。
需要通过 `python eval_unified.py` 直接运行（而非 import），
因为使用了 `if __name__ == '__main__':` 保护多进程安全。
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0,1,2"

import json
import glob
import pickle
import multiprocessing
import traceback
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
# 运行时修复：monkey-patch ConditionalEval/ConsistencyEval 的 _align_with_* 方法
# 原始代码中 float(score.numpy()) 会在 score 为多元素数组时报错:
#   "can only convert an array of size 1 to a Python scalar"
# 修复方式：用 np.float64(x).item() 替代 float(x)
# =============================================================================
from shepherd_score.evaluations.evaluate.evals import ConditionalEval, ConsistencyEval
from shepherd_score.container import MoleculePair

def _patched_align_with_surface(self, mp_ref_and_relaxed: MoleculePair) -> float:
    with torch.enable_grad():
        mp_ref_and_relaxed.align_with_surf(
            self.alpha, num_repeats=1, trans_init=False, use_jax=False
        )
    return np.float64(mp_ref_and_relaxed.sim_aligned_surf).item()

def _patched_align_with_esp(self, mp_ref_and_relaxed: MoleculePair) -> float:
    with torch.enable_grad():
        mp_ref_and_relaxed.align_with_esp(
            self.alpha, lam=self.lam, num_repeats=1, trans_init=False, use_jax=False
        )
    return np.float64(mp_ref_and_relaxed.sim_aligned_esp).item()

def _patched_align_with_pharm(self, mp_ref_and_relaxed: MoleculePair) -> float:
    with torch.enable_grad():
        mp_ref_and_relaxed.align_with_pharm(
            similarity='tanimoto', extended_points=False, only_extended=False,
            num_repeats=1, trans_init=False, use_jax=False
        )
    return np.float64(mp_ref_and_relaxed.sim_aligned_pharm).item()

# 应用 monkey-patch
for cls in (ConditionalEval, ConsistencyEval):
    cls._align_with_surface = _patched_align_with_surface
    cls._align_with_esp = _patched_align_with_esp
    cls._align_with_pharm = _patched_align_with_pharm

print("✅ 已应用 monkey-patch: _align_with_surface/esp/pharm → np.float64().item()")

# =============================================================================
# 工具函数和评估函数（模块顶层，可被子进程 import）
# =============================================================================

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

# ========== 缓存文件名（固定名称，方便管理） ==========
CONF_CACHE_FILE = 'conf_eval_results.json'
COND_CACHE_FILE = 'cond_eval_results.json'

# ========== 并行配置 ==========
_AVAILABLE_CPUS = multiprocessing.cpu_count() or 1
NUM_WORKERS = min(8, _AVAILABLE_CPUS)     # ConfEval 并行 worker 数
NUM_PROCESSES = 1                          # 每个 worker 的 xtb 进程数


# =============================================================================
# 缓存检测函数
# =============================================================================

def _get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def load_cache(cache_filename):
    """加载缓存文件，返回数据字典或 None"""
    cache_path = os.path.join(_get_script_dir(), cache_filename)
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 无法加载缓存文件 {cache_path}: {e}")
        return None


def save_cache(cache_filename, data):
    """保存缓存文件"""
    cache_path = os.path.join(_get_script_dir(), cache_filename)
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 缓存已保存: {cache_path}")


def detect_models_needing_eval(model_files, cached_data):
    """
    对比当前 MODEL_FILES 与缓存中的文件路径，
    返回需要重新评估的模型集合。
    """
    if cached_data is None:
        return set(model_files.keys())

    cached_model_files = cached_data.get('metadata', {}).get('model_files', {})

    models_to_eval = set()
    for model_name, file_path in model_files.items():
        cached_path = cached_model_files.get(model_name)
        if cached_path != file_path:
            models_to_eval.add(model_name)
            if cached_path is not None:
                print(f"  🔄 {model_name}: 路径已变化，需要重新评估")
            else:
                print(f"  🆕 {model_name}: 新增模型，需要评估")
        else:
            print(f"  ✅ {model_name}: 路径未变化，将复用缓存")

    return models_to_eval


# =============================================================================
# 评估核心函数（模块顶层，供多进程 worker 调用）
# =============================================================================

def serialize_value(value):
    """将值转换为 JSON 可序列化的格式"""
    if isinstance(value, (int, float, bool, str)):
        return value
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, pd.Series):
        return value.tolist()
    elif isinstance(value, pd.DataFrame):
        return value.to_dict(orient='list')
    elif isinstance(value, (np.integer, np.floating)):
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
        # 尝试 numpy 0-d 数组
        if hasattr(value, 'item'):
            try:
                return value.item()
            except (ValueError, TypeError):
                pass
        try:
            return str(value)
        except Exception:
            return None


def _conf_eval_worker(args):
    """
    multiprocessing worker 函数：对单个样本执行 ConfEval。
    必须是模块顶级函数才能被 pickle 序列化（spawn 模式要求）。
    """
    structure_id, model_name, atoms, positions, bonds, num_processes = args
    try:
        if isinstance(atoms, list):
            atoms = np.array(atoms)
        if isinstance(positions, list):
            positions = np.array(positions)
        if bonds is not None and isinstance(bonds, list):
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

            # 转换为 list 以确保可 pickle（spawn 模式需要序列化）
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
        # 多进程模式 - 使用 spawn 上下文（需要 __main__ 保护）
        try:
            ctx = multiprocessing.get_context('spawn')
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
        cond_summary, global_summary: 条件评估结果摘要
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

    # 执行条件评估
    # 注意：CondEval 使用 num_workers=1（单进程），避免嵌套多进程冲突
    cond_pipe = ConditionalEvalPipeline(
        ref_mol,
        generated_mols=generated_mols,
        condition='all',
        num_surf_points=200,
        pharm_multi_vector=False,
        solvent=None
    )
    cond_pipe.evaluate(
        num_workers=1,
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


# =============================================================================
# 主执行流程 —— 受 __main__ 保护，防止 spawn 子进程重复执行
# =============================================================================

def main():
    print(f"\n⚙️  并行配置: CPU核心数={_AVAILABLE_CPUS}, "
          f"ConfEval workers={NUM_WORKERS}, "
          f"xtb processes/worker={NUM_PROCESSES}")

    # =================================================================
    # 第一阶段：缓存检测
    # =================================================================

    print("\n🔍 检测缓存...")

    # 分别加载 conf 和 cond 缓存
    conf_cache = load_cache(CONF_CACHE_FILE)
    cond_cache = load_cache(COND_CACHE_FILE)

    if conf_cache:
        print(f"  ✅ ConfEval 缓存存在: {CONF_CACHE_FILE}")
    else:
        print(f"  ❌ ConfEval 缓存不存在")

    if cond_cache:
        print(f"  ✅ CondEval 缓存存在: {COND_CACHE_FILE}")
    else:
        print(f"  ❌ CondEval 缓存不存在")

    # 确定哪些模型的 ConfEval 需要重新评估
    print("\n📋 检查 ConfEval 缓存...")
    models_needing_conf = detect_models_needing_eval(MODEL_FILES, conf_cache)

    # 确定哪些模型的 CondEval 需要重新评估
    print("\n📋 检查 CondEval 缓存...")
    models_needing_cond = detect_models_needing_eval(MODEL_FILES, cond_cache)

    # 如果 conf 需要重跑，cond 也必须重跑（conf 是前置依赖）
    for model_name in models_needing_conf:
        if model_name not in models_needing_cond:
            print(f"  ⚠️ {model_name}: ConfEval 需要重跑，CondEval 也必须重跑")
            models_needing_cond.add(model_name)

    # 判断执行状态
    SKIP_CONF = len(models_needing_conf) == 0
    SKIP_COND = len(models_needing_cond) == 0

    print(f"\n📊 评估计划:")
    for model_name in MODEL_FILES:
        conf_status = "📦 缓存" if model_name not in models_needing_conf else "🔄 重跑"
        cond_status = "📦 缓存" if model_name not in models_needing_cond else "🔄 重跑"
        print(f"  {model_name}: ConfEval={conf_status}, CondEval={cond_status}")

    # =================================================================
    # 第二阶段：加载数据
    # =================================================================

    # 确定实际需要加载数据的模型（ConfEval 或 CondEval 需要重跑的）
    models_needing_data = models_needing_conf | models_needing_cond
    all_model_samples = {}
    all_model_grouped = {}

    for model_name, json_file in MODEL_FILES.items():
        if model_name not in models_needing_data:
            print(f"\n⏭️  跳过加载 {model_name}（全部使用缓存）")
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

    # =================================================================
    # 第三阶段：加载参考分子（仅 CondEval 需要时加载）
    # =================================================================

    if not SKIP_COND:
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
        print("\n⏭️  跳过参考分子加载（CondEval 全部使用缓存）")
        ref_molecules = {}

    # =================================================================
    # 第四阶段：ConfEval（完成后立即保存）
    # =================================================================

    # 从缓存预加载不需要重跑的模型结果
    all_conf_results = {}
    if conf_cache is not None:
        for model_name in MODEL_FILES:
            if model_name not in models_needing_conf:
                all_conf_results[model_name] = conf_cache['per_sample_results'].get(model_name, [])
                print(f"\n📦 {model_name}: 已从 ConfEval 缓存加载")

    if not SKIP_CONF:
        print("\n" + "=" * 80)
        print("🚀 开始 ConfEval 评估")
        print("=" * 80)

        for model_name in models_needing_conf:
            if model_name not in all_model_grouped:
                continue
            grouped = all_model_grouped[model_name]

            print(f"\n{'='*60}")
            print(f"🔬 ConfEval: {model_name}")
            print(f"{'='*60}")

            model_results = []

            for ref_mol_idx, samples in grouped.items():
                print(f"\n  📋 参考分子 {ref_mol_idx}: {len(samples)} 个样本")

                global_id_offset = len(model_results)
                conf_results, conf_valid_raw = run_conf_eval_parallel(
                    samples, model_name, NUM_WORKERS, NUM_PROCESSES
                )

                for i, result in enumerate(conf_results):
                    global_id = global_id_offset + i
                    result['structure_id'] = global_id
                    result['ref_mol_index'] = ref_mol_idx
                    result['local_index'] = i
                    model_results.append(result)

                conf_valid_count = sum(1 for r in conf_results if r and r.get('conf_valid'))
                print(f"    ConfEval 完成: {conf_valid_count}/{len(samples)} 有效")

            all_conf_results[model_name] = model_results

            total = len(model_results)
            conf_ok = sum(1 for r in model_results if r.get('conf_valid'))
            print(f"\n  ✅ {model_name} ConfEval 汇总: 总共 {total} | 构象有效 {conf_ok}")

        # 立即保存 ConfEval 结果
        conf_save_data = {
            'metadata': {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'model_files': MODEL_FILES,
            },
            'per_sample_results': all_conf_results,
        }
        save_cache(CONF_CACHE_FILE, conf_save_data)
        print("\n✅ ConfEval 完成并已保存！")
    else:
        print("\n⏭️  跳过 ConfEval（全部使用缓存）")

    # =================================================================
    # 第五阶段：CondEval（完成后立即保存）
    # =================================================================

    # 从缓存预加载不需要重跑的模型结果
    all_cond_group_results = {}
    all_cond_sample_status = {}  # model_name -> {global_id: {cond_valid: True/False}}

    if cond_cache is not None:
        for model_name in MODEL_FILES:
            if model_name not in models_needing_cond:
                all_cond_group_results[model_name] = {
                    int(k): v
                    for k, v in cond_cache.get('cond_group_results', {}).get(model_name, {}).items()
                }
                all_cond_sample_status[model_name] = cond_cache.get('per_sample_cond_status', {}).get(model_name, {})
                print(f"\n📦 {model_name}: 已从 CondEval 缓存加载")

    if not SKIP_COND:
        print("\n" + "=" * 80)
        print("🚀 开始 CondEval 评估")
        print("=" * 80)

        for model_name in models_needing_cond:
            if model_name not in all_model_grouped:
                continue
            grouped = all_model_grouped[model_name]
            model_conf_results = all_conf_results.get(model_name, [])

            print(f"\n{'='*60}")
            print(f"🔬 CondEval: {model_name}")
            print(f"{'='*60}")

            model_cond_groups = {}
            model_cond_status = {}

            for ref_mol_idx, samples in grouped.items():
                print(f"\n  📋 参考分子 {ref_mol_idx}: {len(samples)} 个样本")

                if ref_mol_idx not in ref_molecules:
                    print(f"    ❌ 缺少参考分子 {ref_mol_idx}，跳过条件评估")
                    continue

                # 从 conf 结果中找出该组中 conf_valid=True 的样本
                # 需要根据 ref_mol_index 过滤
                conf_valid_samples = []
                for r in model_conf_results:
                    if r.get('ref_mol_index') == ref_mol_idx and r.get('conf_valid'):
                        global_id = r['structure_id']
                        local_idx = r.get('local_index', 0)
                        # 从 grouped 中获取对应的样本
                        if local_idx < len(samples):
                            conf_valid_samples.append((global_id, samples[local_idx]))

                conf_valid_count = len(conf_valid_samples)
                print(f"    ConfEval 有效样本: {conf_valid_count}")

                if conf_valid_count == 0:
                    print(f"    ⚠️ 无有效构象，跳过条件评估")
                    model_cond_groups[ref_mol_idx] = {
                        'cond_summary': None,
                        'global_summary': None,
                        'num_samples': len(samples),
                        'num_conf_valid': 0,
                        'num_both_valid': 0,
                    }
                    continue

                try:
                    cond_results, cond_summary, global_summary = run_cond_eval_group(
                        ref_molecules[ref_mol_idx], conf_valid_samples
                    )

                    cond_valid_count = 0
                    for global_id, cond_info in cond_results.items():
                        model_cond_status[str(global_id)] = {
                            'cond_valid': cond_info.get('cond_valid', False)
                        }
                        if cond_info.get('cond_valid', False):
                            cond_valid_count += 1

                    model_cond_groups[ref_mol_idx] = {
                        'cond_summary': cond_summary,
                        'global_summary': global_summary,
                        'num_samples': len(samples),
                        'num_conf_valid': conf_valid_count,
                        'num_both_valid': cond_valid_count,
                    }

                    print(f"    CondEval 完成: {cond_valid_count}/{conf_valid_count} 条件有效")

                except Exception as e:
                    print(f"    ❌ CondEval 评估失败: {str(e)}")
                    traceback.print_exc()
                    model_cond_groups[ref_mol_idx] = {
                        'cond_summary': None,
                        'global_summary': None,
                        'num_samples': len(samples),
                        'num_conf_valid': conf_valid_count,
                        'num_both_valid': 0,
                        'error': str(e),
                    }
                    continue

            all_cond_group_results[model_name] = model_cond_groups
            all_cond_sample_status[model_name] = model_cond_status

            # 模型汇总
            total_both = sum(g.get('num_both_valid', 0) for g in model_cond_groups.values())
            total_conf = sum(g.get('num_conf_valid', 0) for g in model_cond_groups.values())
            print(f"\n  ✅ {model_name} CondEval 汇总: 构象有效 {total_conf} | 联合有效 {total_both}")

        # 立即保存 CondEval 结果
        cond_save_data = {
            'metadata': {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'model_files': MODEL_FILES,
            },
            'cond_group_results': {
                model_name: {
                    str(ref_idx): group_data
                    for ref_idx, group_data in groups.items()
                }
                for model_name, groups in all_cond_group_results.items()
            },
            'per_sample_cond_status': all_cond_sample_status,
        }
        save_cache(COND_CACHE_FILE, cond_save_data)
        print("\n✅ CondEval 完成并已保存！")
    else:
        print("\n⏭️  跳过 CondEval（全部使用缓存）")

    # =================================================================
    # 打印评估来源汇总
    # =================================================================

    print(f"\n{'='*80}")
    print("🎉 评估完成!")
    print(f"{'='*80}")
    for model_name in MODEL_FILES:
        conf_src = "🔄 重新评估" if model_name in models_needing_conf else "📦 缓存复用"
        cond_src = "🔄 重新评估" if model_name in models_needing_cond else "📦 缓存复用"
        print(f"  {model_name}: ConfEval={conf_src}, CondEval={cond_src}")

    # =================================================================
    # 第六阶段：合并数据并进行统计分析
    # =================================================================

    # 合并 conf 和 cond 结果，生成 unified results
    all_unified_results = {}
    for model_name in MODEL_FILES:
        model_conf = all_conf_results.get(model_name, [])
        model_cond_status = all_cond_sample_status.get(model_name, {})

        unified = []
        for r in model_conf:
            r_copy = dict(r)  # 不修改原始缓存数据
            global_id = str(r_copy.get('structure_id', ''))
            cond_info = model_cond_status.get(global_id, {})
            r_copy['cond_valid'] = cond_info.get('cond_valid', False)
            r_copy['both_valid'] = r_copy.get('conf_valid', False) and r_copy['cond_valid']
            unified.append(r_copy)

        all_unified_results[model_name] = unified

    # =================================================================
    # 统计分析
    # =================================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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
        conf_valid = sum(1 for r in results if r.get('conf_valid'))
        cond_valid = sum(1 for r in results if r.get('cond_valid'))
        both_valid = sum(1 for r in results if r.get('both_valid'))

        validity_table.append({
            '模型': model_name,
            '总样本': total,
            '构象有效': conf_valid,
            '构象有效率': f"{conf_valid/total*100:.1f}%" if total > 0 else "N/A",
            '条件有效': cond_valid,
            '条件有效率': f"{cond_valid/total*100:.1f}%" if total > 0 else "N/A",
            '联合有效': both_valid,
            '联合有效率': f"{both_valid/total*100:.1f}%" if total > 0 else "N/A",
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
            if not r.get('both_valid'):
                continue

            conf_data = r.get('conf_data', {})
            if conf_data is None:
                continue

            for metric in conf_metrics:
                value = None
                if metric in conf_data and conf_data[metric] is not None:
                    value = conf_data[metric]
                elif metric in r and r[metric] is not None:
                    value = r[metric]
                elif f"{metric}_post_opt" in conf_data and conf_data[f"{metric}_post_opt"] is not None:
                    value = conf_data[f"{metric}_post_opt"]

                if value is not None and isinstance(value, (int, float)) and not np.isnan(value):
                    stats[metric].append(float(value))

        model_conf_stats[model_name] = stats

    for model_name in all_unified_results.keys():
        print(f"\n📌 {model_name} （联合有效分子）:")
        both_count = sum(1 for r in all_unified_results[model_name] if r.get('both_valid'))
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
            print(f"     总样本: {group_data.get('num_samples', 'N/A')} | "
                  f"构象有效: {group_data.get('num_conf_valid', 'N/A')} | "
                  f"联合有效: {group_data.get('num_both_valid', 'N/A')}")

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

        total_samples = sum(g.get('num_samples', 0) for g in groups.values())
        both_valid_total = sum(g.get('num_both_valid', 0) for g in groups.values())

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
    print(f"   ConfEval 缓存: {CONF_CACHE_FILE}")
    print(f"   CondEval 缓存: {COND_CACHE_FILE}")
    print(f"   统计报告: {report_filename}")
    print("=" * 80)
    print("\n💡 提示: 如果 CondEval 出现问题，可以:")
    print(f"   1. 删除 {COND_CACHE_FILE}")
    print(f"   2. 重新运行脚本 → 将跳过 ConfEval，仅重跑 CondEval")


# =============================================================================
# 入口点：保护多进程安全
# =============================================================================
if __name__ == '__main__':
    main()
