#!/usr/bin/env python3
"""
Fragment Merging DPO训练脚本
专门用于基于条件数据（表面、药效团）的DPO微调
"""

# ==================== 系统配置 ====================
import os
import time
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
import shutil
import datetime
import pickle
import argparse
import importlib
import json
from collections import defaultdict

import multiprocessing
import numpy as np
import threading
import queue
import torch
import torch.multiprocessing
import torch_geometric
import rdkit
import rdkit.Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Lipinski
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from datetime import timedelta
from lightning_fabric.utilities.seed import seed_everything

# 项目模块
from shepherd.lightning_module import LightningModule
from shepherd.new_datasets import HeteroDataset
from shepherd.dpo_dataset import DPODataset
from shepherd.inference import inference_sample
from shepherd.extract import create_rdkit_molecule
from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores
from shepherd.shepherd_score_utils.generate_point_cloud import (
    get_atomic_vdw_radii,
    get_molecular_surface,
    get_electrostatics_given_point_charges,
)

# Shepherd Score评估模块
from shepherd_score.evaluations.evaluate import ConfEval
from shepherd_score.score.gaussian_overlap_np import get_overlap_np
from shepherd_score.score.electrostatic_scoring_np import get_overlap_esp_np
from shepherd_score.score.pharmacophore_scoring_np import get_overlap_pharm_np
from shepherd_score.score.constants import ALPHA, LAM_SCALING

# ==================== 日志配置 ====================

# ==================== 全局配置 ====================
SHARING_STRATEGY = "file_system"
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)
# torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True


def set_worker_sharing_strategy(worker_id: int) -> None:
    """DataLoader worker初始化函数"""
    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


def convert_for_json(obj):
    """递归转换numpy数组和torch张量为Python列表，用于JSON序列化"""
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_for_json(elem) for elem in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    return obj


def load_cached_marginals(cache_dir="cached_marginals"):
    """加载MOSES_aq预计算的边际分布"""
    atom_marginals_file = os.path.join(cache_dir, "MOSES_aq_atom_marginals.pt")
    bond_marginals_file = os.path.join(cache_dir, "MOSES_aq_bond_marginals.pt")
    pharm_marginals_file = os.path.join(cache_dir, "MOSES_aq_pharm_marginals.pt")

    print(f"📊 从 '{cache_dir}' 加载MOSES_aq边际分布...")
    
    atom_marginals = torch.load(atom_marginals_file, map_location=torch.device('cpu'))
    bond_marginals = torch.load(bond_marginals_file, map_location=torch.device('cpu'))
    pharm_marginals = torch.load(pharm_marginals_file, map_location=torch.device('cpu'))
    
    print(f"  - atom_marginals: {atom_marginals.shape}")
    print(f"  - bond_marginals: {bond_marginals.shape}")
    print(f"  - pharm_marginals: {pharm_marginals.shape}")

    return atom_marginals, bond_marginals, pharm_marginals




def load_fragment_merge_condition(params):
    """加载Fragment Merging条件数据
    
    返回的条件数据包含原始numpy数组的副本，避免在inference过程中被修改。
    """
    condition_path = params.get('fragment_merge_condition_path', 
                                '../data/conformers/fragment_merging/fragment_merge_condition.pickle')
    
    with open(condition_path, 'rb') as f:
        condition = pickle.load(f)
    
    print(f"✅ 加载Fragment Merging条件数据:")
    print(f"   - x2 positions: {condition['x2']['positions'].shape}")
    print(f"   - x3 positions: {condition['x3']['positions'].shape}")
    print(f"   - x3 charges: {condition['x3']['charges'].shape}")
    print(f"   - x4 types: {condition['x4']['types'].shape}")
    print(f"   - x4 positions: {condition['x4']['positions'].shape}")
    print(f"   - x4 directions: {condition['x4']['directions'].shape}")
    
    # 保存原始numpy数组的副本，用于条件评估（避免inference过程中被原地修改）
    condition['_original_x3_positions'] = np.array(condition['x3']['positions']).copy()
    condition['_original_x3_charges'] = np.array(condition['x3']['charges']).copy()
    condition['_original_x4_types'] = np.array(condition['x4']['types']).copy()
    condition['_original_x4_positions'] = np.array(condition['x4']['positions']).copy()
    condition['_original_x4_directions'] = np.array(condition['x4']['directions']).copy()
    
    # Fragment Merging使用空的molblocks_and_charges
    molblocks_and_charges = [(None, None)]
    
    return molblocks_and_charges, condition


def create_dataset(params, molblocks_and_charges, marginals):
    """创建HeteroDataset"""
    atom_marginals_x1, bond_marginals_x1, pharm_marginals_x4 = marginals
    
    dataset = HeteroDataset(
        molblocks_and_charges=molblocks_and_charges,
        noise_schedule_dict=params['noise_schedules'],
        
        # 边际分布
        atom_marginals_x1=atom_marginals_x1,
        bond_marginals_x1=bond_marginals_x1,
        pharm_marginals_x4=pharm_marginals_x4,
        
        # 数据集配置
        explicit_hydrogens=params['dataset']['explicit_hydrogens'],
        use_MMFF94_charges=params['dataset']['use_MMFF94_charges'],
        formal_charge_diffusion=False,
        
        # 模态开关
        x1=params['dataset']['compute_x1'],
        x2=params['dataset']['compute_x2'],
        x3=params['dataset']['compute_x3'],
        x4=params['dataset']['compute_x4'],
        
        # x1配置
        recenter_x1=params['dataset']['x1']['recenter'],
        add_virtual_node_x1=params['dataset']['x1']['add_virtual_node'],
        remove_noise_COM_x1=params['dataset']['x1']['remove_noise_COM'],
        atom_types_x1=params['dataset']['x1']['atom_types'],
        charge_types_x1=params['dataset']['x1']['charge_types'],
        bond_types_x1=params['dataset']['x1']['bond_types'],
        scale_atom_features_x1=params['dataset']['x1']['scale_atom_features'],
        scale_bond_features_x1=params['dataset']['x1']['scale_bond_features'],
        
        # x2配置
        independent_timesteps_x2=params['dataset']['x2']['independent_timesteps'],
        recenter_x2=params['dataset']['x2']['recenter'],
        add_virtual_node_x2=params['dataset']['x2']['add_virtual_node'],
        remove_noise_COM_x2=params['dataset']['x2']['remove_noise_COM'],
        num_points_x2=params['dataset']['x2']['num_points'],
        
        # x3配置
        independent_timesteps_x3=params['dataset']['x3']['independent_timesteps'],
        recenter_x3=params['dataset']['x3']['recenter'],
        add_virtual_node_x3=params['dataset']['x3']['add_virtual_node'],
        remove_noise_COM_x3=params['dataset']['x3']['remove_noise_COM'],
        num_points_x3=params['dataset']['x3']['num_points'],
        scale_node_features_x3=params['dataset']['x3']['scale_node_features'],
        
        # x4配置
        independent_timesteps_x4=params['dataset']['x4']['independent_timesteps'],
        recenter_x4=params['dataset']['x4']['recenter'],
        add_virtual_node_x4=params['dataset']['x4']['add_virtual_node'],
        remove_noise_COM_x4=params['dataset']['x4']['remove_noise_COM'],
        max_node_types_x4=params['dataset']['x4']['max_node_types'],
        scale_node_features_x4=params['dataset']['x4']['scale_node_features'],
        scale_vector_features_x4=params['dataset']['x4']['scale_vector_features'],
        multivectors=params['dataset']['x4']['multivectors'],
        check_accessibility=params['dataset']['x4']['check_accessibility'],
        
        probe_radius=params['dataset']['probe_radius'],
    )
    
    return dataset


def create_dpo_dataloader(params, dataset, dpo_dataset):
    """创建纯DPO DataLoader（不混合标准数据）"""
    print("🎯 创建纯DPO DataLoader...")
    
    # 纯DPO训练，只使用DPO数据集
    train_loader = torch_geometric.loader.DataLoader(
        dataset=dpo_dataset,
        num_workers=params['training']['num_workers'],
        batch_size=params['training']['batch_size'],
        shuffle=True,
        multiprocessing_context=multiprocessing.get_context("spawn") 
            if params['training']['multiprocessing_spawn'] else None,
        worker_init_fn=set_worker_sharing_strategy,
        persistent_workers=True,
    )
    
    return train_loader


def _sample_single_group(model_pl, params, fragment_merge_condition, dataset, 
                          group_id, samples_per_group, device, result_queue):
    """
    单个采样组的工作函数（在线程中执行）
    
    Args:
        model_pl: 模型（已在指定GPU上）
        params: 参数配置
        fragment_merge_condition: 条件数据
        dataset: 数据集（用于获取边际分布）
        group_id: 组ID
        samples_per_group: 每组样本数
        device: GPU设备
        result_queue: 结果队列
    """
    try:
        gpu_id = device.index if hasattr(device, 'index') else 0
        print(f"  [GPU {gpu_id}] 组 {group_id} 开始采样 {samples_per_group} 个样本...")
        
        # 使用no_grad防止梯度累积导致内存泄漏
        with torch.no_grad():
            # 获取边际分布
            atom_marginals = dataset.x1_atom_diffuser.transition_model.marginals.to(device)
            bond_marginals = dataset.x1_bond_diffuser.transition_model.marginals.to(device)
            
            # 修正虚拟节点边际概率
            if len(atom_marginals) > 0:
                atom_marginals[0] = 0.0
                atom_marginals = atom_marginals / atom_marginals.sum()
            
            # 条件数据
            surface = fragment_merge_condition['x3']['positions']
            electrostatics = fragment_merge_condition['x3']['charges']
            pharm_types = fragment_merge_condition['x4']['types']
            pharm_pos = fragment_merge_condition['x4']['positions']
            pharm_direction = fragment_merge_condition['x4']['directions']
            
            num_pharmacophores = 27
            n_atoms = params.get('sampling', {}).get('fixed_n_atoms', 70)
            
            # 准备inference参数
            inference_kwargs = {
                "batch_size": samples_per_group,
                "N_x1": n_atoms,
                "N_x4": num_pharmacophores,
                "unconditional": False,
                "prior_noise_scale": 1.0,
                "denoising_noise_scale": 1.0,
                "inject_noise_at_ts": [],
                "inject_noise_scales": [],
                "harmonize": False,
                "harmonize_ts": [],
                "harmonize_jumps": [],
                "inpaint_x2_pos": False,
                "inpaint_x3_pos": False,
                "inpaint_x3_x": False,
                "inpaint_x4_pos": True,
                "inpaint_x4_direction": True,
                "inpaint_x4_type": True,
                "stop_inpainting_at_time_x2": 0.0,
                "add_noise_to_inpainted_x2_pos": 0.0,
                "stop_inpainting_at_time_x3": 0.0,
                "add_noise_to_inpainted_x3_pos": 0.0,
                "add_noise_to_inpainted_x3_x": 0.0,
                "stop_inpainting_at_time_x4": 0.0,
                "add_noise_to_inpainted_x4_pos": 0.0,
                "add_noise_to_inpainted_x4_direction": 0.0,
                "add_noise_to_inpainted_x4_type": 0.0,
                "center_of_mass": np.zeros(3),
                "surface": surface,
                "electrostatics": electrostatics,
                "pharm_types": pharm_types,
                "pharm_pos": pharm_pos,
                "pharm_direction": pharm_direction,
                "atom_marginals": atom_marginals,
                "bond_marginals": bond_marginals,
            }
            
            # 执行采样
            with torch.cuda.device(device):
                generated_samples = inference_sample(model_pl, **inference_kwargs)
            
            # 为每个样本标记组ID
            for sample in generated_samples:
                sample['source_mol_index'] = 0  # Fragment Merging只有一个条件
                sample['group_id'] = group_id
            
            # 清理中间张量
            del atom_marginals, bond_marginals
        
        # 释放GPU缓存（在no_grad外部）
        torch.cuda.empty_cache()
        
        print(f"  [GPU {gpu_id}] ✅ 组 {group_id} 完成: {len(generated_samples)} 个样本")
        result_queue.put((group_id, generated_samples))
        
    except Exception as e:
        print(f"  [GPU {gpu_id}] ❌ 组 {group_id} 采样失败: {e}")
        import traceback
        traceback.print_exc()
        result_queue.put((group_id, []))


def sample_and_evaluate_molecules_multi_gpu(models_dict, params, molblocks_and_charges, dataset,
                                            num_groups=None, samples_per_group=4, 
                                            fragment_merge_condition=None):
    """
    多GPU并行采样和评估分子
    
    Args:
        models_dict: {gpu_id: model_pl} 字典，每个GPU一个模型
        params: 参数配置
        molblocks_and_charges: 分子数据
        dataset: 数据集
        num_groups: 采样组数（None或-1表示使用GPU数量）
        samples_per_group: 每组样本数（默认4个）
        fragment_merge_condition: Fragment Merging条件数据
    
    Returns:
        preference_pairs: 偏好对列表
    """
    num_gpus = len(models_dict)
    
    # 自动设置组数 = GPU数量（每个GPU一组，一轮完成）
    if num_groups is None or num_groups <= 0:
        num_groups = num_gpus
    
    print("\n" + "="*80)
    print(f"🧬 开始多GPU并行采样和评估")
    print(f"   GPU数量: {num_gpus}, 采样组数: {num_groups}, 每组样本: {samples_per_group}")
    print(f"   总样本数: {num_groups * samples_per_group}")
    print("="*80)
    
    # 设置所有模型为评估模式
    for gpu_id, model in models_dict.items():
        model.eval()
    
    # 创建结果队列
    result_queue = queue.Queue()
    
    gpu_ids = list(models_dict.keys())
    threads = []
    
    print(f"\n📋 任务分配 ({num_groups}组并行, 每GPU一组):")
    
    # 每个GPU一组，一轮完成所有采样
    for group_id in range(num_groups):
        gpu_id = gpu_ids[group_id % num_gpus]
        device = torch.device('cuda', gpu_id)
        model = models_dict[gpu_id]
        
        print(f"   组 {group_id} -> GPU {gpu_id}")
        
        # 创建采样线程
        t = threading.Thread(
            target=_sample_single_group,
            args=(model, params, fragment_merge_condition, dataset,
                  group_id, samples_per_group, device, result_queue)
        )
        threads.append(t)
    
    # 启动所有线程（并行执行）
    print(f"\n🚀 启动 {len(threads)} 个并行采样线程...")
    for t in threads:
        t.start()
    
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    print(f"✅ 所有采样完成")
    
    # 收集结果
    all_generated_samples = []
    results_by_group = {}
    
    while not result_queue.empty():
        group_id, samples = result_queue.get()
        results_by_group[group_id] = samples
    
    # 按组ID排序合并
    for group_id in sorted(results_by_group.keys()):
        all_generated_samples.extend(results_by_group[group_id])
    
    print(f"\n📊 采样统计:")
    print(f"   总样本数: {len(all_generated_samples)}")
    for group_id in sorted(results_by_group.keys()):
        print(f"   组 {group_id}: {len(results_by_group[group_id])} 个样本")
    
    # 保存生成的分子到JSON文件
    if len(all_generated_samples) > 0:
        try:
            output_dir = params['training'].get('output_dir', 'default_output')
            save_dir = f"jobs/{output_dir}"
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"{save_dir}/generated_mols_{timestamp}.json"
            generated_samples_for_json = convert_for_json(all_generated_samples)
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(generated_samples_for_json, f, ensure_ascii=False, indent=4)
            print(f"💾 已保存到: {json_filename}")
        except Exception as e:
            print(f"⚠️  保存失败: {e}")
    
    # 评估阶段
    print("\n" + "="*80)
    print("📈 开始评估生成的分子")
    print("="*80)
    
    preference_pairs = evaluate_and_build_pairs(
        all_generated_samples,
        [],  # reference_mols
        molblocks_and_charges,
        params,
        fragment_merge_condition=fragment_merge_condition
    )
    
    # 恢复训练模式
    for model in models_dict.values():
        model.train()
    
    return preference_pairs


def sample_and_evaluate_molecules(model_pl, params, molblocks_and_charges, dataset, 
                                   num_samples_per_mol=4, device='cuda', fragment_merge_condition=None):
    """
    单GPU采样和评估（向后兼容接口）
    """
    # 转换为多GPU接口调用
    gpu_id = device.index if hasattr(device, 'index') else 0
    models_dict = {gpu_id: model_pl}
    
    return sample_and_evaluate_molecules_multi_gpu(
        models_dict, params, molblocks_and_charges, dataset,
        num_groups=1, samples_per_group=num_samples_per_mol,
        fragment_merge_condition=fragment_merge_condition
    )


def evaluate_and_build_pairs(generated_samples, reference_mols, molblocks_and_charges, params, fragment_merge_condition=None):
    """
    评估生成的分子并构建偏好对
    
    Args:
        generated_samples: 生成的分子样本列表
        reference_mols: 参考分子列表（Fragment Merging时为None）
        molblocks_and_charges: 分子数据
        params: 参数配置
        fragment_merge_condition: Fragment Merging的条件数据（包含x2, x3, x4）
    """
    print(f"\n🔍 评估 {len(generated_samples)} 个生成样本...")
    
    # 按group_id分组（多GPU并行采样时使用group_id）
    from collections import defaultdict
    grouped_samples = defaultdict(list)
    
    for sample in generated_samples:
        # 优先使用group_id，如果没有则使用source_mol_index
        group_key = sample.get('group_id', sample.get('source_mol_index', 0))
        grouped_samples[group_key].append(sample)
    
    all_preference_pairs = []
    all_valid_molecules_across_groups = []  # 收集所有组的有效分子，用于跨组统计
    groups_with_pairs = set()  # 记录已成功构建偏好对的组
    
    # 为每组分子评估并构建偏好对
    for source_idx, samples in grouped_samples.items():
        print(f"\n📋 评估组 {source_idx}: {len(samples)} 个样本")
        
        # 1. 使用ConfEval评估每个生成的分子（或使用RDKit备选方案）
        evaluated_samples = []
        invalid_samples = []  # 单独收集无效样本
        
        for i, sample in enumerate(samples):

            # 提取原子、坐标和键信息
            atoms = sample['x1']['atoms']
            positions = sample['x1']['positions']
            bonds = sample['x1'].get('bonds', None)  # 提取键信息
            
            if isinstance(atoms, np.ndarray):
                atoms = atoms.flatten()
            
            if len(atoms) == 0:
                print(f"  🔬 样本 {i+1}: 跳过（无原子）")
                continue
            
            # 验证原子序数的有效性
            valid_atoms = (atoms > 0) & (atoms <= 118)
            num_valid = np.sum(valid_atoms)
            num_invalid = len(atoms) - num_valid
            
            if num_invalid > 0:
                if num_valid == 0:
                    print(f"  🔬 样本 {i+1}: ✗ 所有 {len(atoms)} 个原子都无效")
                    # 给无效分子赋予极低分数
                    invalid_samples.append({
                        'sample': sample,
                        'conf_scores': {
                            'qed': 0.0,
                            'logp': -10.0,
                            'strain_energy': 100.0,
                            'sa_score': 10.0,
                            'is_valid': False,
                        },
                        'atoms': atoms,
                        'positions': positions,
                        'bonds': bonds,
                    })
                    continue
            
            conf_scores = None
            
            # 尝试使用ConfEval评估（依赖xtb）
            try:
                conf_eval = ConfEval(atoms, positions, solvent='water', bonds=bonds)
                
                # 检查分子是否有效，无效则跳过
                if not conf_eval.is_valid:
                    print(f"  🔬 样本 {i+1}: ✗ 分子无效")
                    # 给无效分子赋予极低分数
                    invalid_samples.append({
                        'sample': sample,
                        'conf_scores': {
                            'qed': 0.0, 'logp': -10.0, 'strain_energy': 100.0,
                            'sa_score': 10.0, 'fsp3': 0.0, 'is_valid': False,
                        },
                        'atoms': atoms,
                        'positions': positions,
                        'bonds': bonds,
                    })
                    continue
                
                eval_df = conf_eval.to_pandas()
                
                # 提取关键指标，处理可能的None值
                # 注意：ConfEval.to_pandas() 返回的是 Series，键名是属性名
                qed_val = eval_df.get('QED', None)
                logp_val = eval_df.get('logP', None)
                strain_val = eval_df.get('strain_energy', None)
                sa_val = eval_df.get('SA_score', None)
                fsp3_val = eval_df.get('fsp3', None)
                
                # 检查关键指标是否为None或nan
                if qed_val is None or sa_val is None or (isinstance(qed_val, float) and np.isnan(qed_val)):
                    print(f"  🔬 样本 {i+1}: ✗ 指标无效 (QED={qed_val}, SA={sa_val})")
                    # 给无效分子赋予极低分数
                    invalid_samples.append({
                        'sample': sample,
                        'conf_scores': {
                            'qed': 0.0, 'logp': -10.0, 'strain_energy': 100.0,
                            'sa_score': 10.0, 'fsp3': 0.0, 'is_valid': False,
                        },
                        'atoms': atoms,
                        'positions': positions,
                        'bonds': bonds,
                    })
                    continue
                
                # 处理可能的NaN值
                def safe_float(val, default):
                    if val is None:
                        return default
                    try:
                        f_val = float(val)
                        if np.isnan(f_val) or np.isinf(f_val):
                            return default
                        return f_val
                    except:
                        return default
                
                conf_scores = {
                    'qed': safe_float(qed_val, 0.0),
                    'logp': safe_float(logp_val, 0.0),
                    'strain_energy': safe_float(strain_val, 0.0),
                    'sa_score': safe_float(sa_val, 5.0),
                    'fsp3': safe_float(fsp3_val, 0.0),
                    'is_valid': True,
                }
                # 简洁日志
                fsp3_str = f", Fsp3={fsp3_val:.2f}" if fsp3_val else ""
                print(f"  🔬 样本 {i+1}: ✓ QED={qed_val:.3f}, SA={sa_val:.2f}{fsp3_str}")
                
            except Exception as e:
                print(f"  🔬 样本 {i+1}: ✗ ConfEval失败 ({type(e).__name__}: {str(e)[:50]})")
                # 给失败分子赋予极低分数
                invalid_samples.append({
                    'sample': sample,
                    'conf_scores': {
                        'qed': 0.0, 'logp': -10.0, 'strain_energy': 100.0,
                        'sa_score': 10.0, 'fsp3': 0.0, 'is_valid': False,
                    },
                    'atoms': atoms,
                    'positions': positions,
                    'bonds': bonds,
                })
                continue
            
            if conf_scores is None:
                continue
            
            evaluated_samples.append({
                'sample': sample,
                'conf_scores': conf_scores,
                'atoms': atoms,
                'positions': positions,
                'bonds': bonds,  # 添加键信息
            })
        
        # 合并有效和无效样本，确保至少有一个有效样本作为winner
        all_evaluated = evaluated_samples + invalid_samples
        
        if len(evaluated_samples) < 1:  # 至少需要一个有效样本作为winner
            print(f"  ❌ 组 {source_idx} 偏好对构建失败: 无有效样本作为winner (有效:{len(evaluated_samples)}, 无效:{len(invalid_samples)})")
            continue
            
        if len(all_evaluated) < 2:  # 总共需要至少2个样本构建偏好对
            print(f"  ❌ 组 {source_idx} 偏好对构建失败: 总样本不足2个 ({len(all_evaluated)}/{len(samples)})")
            continue
        
        # 2. 条件评估 - 使用Shepherd Score的高斯重叠方法
        print("  🔬 开始Shepherd Score条件评估...")
        
        # 提取条件数据（使用原始numpy副本，避免inference过程中被修改的问题）
        cond_surface = fragment_merge_condition['_original_x3_positions']
        cond_esp = fragment_merge_condition['_original_x3_charges']
        cond_pharm_types = fragment_merge_condition['_original_x4_types']
        cond_pharm_pos = fragment_merge_condition['_original_x4_positions']
        cond_pharm_dir = fragment_merge_condition['_original_x4_directions']
        
        # 设置评分参数
        num_surf_points = len(cond_surface)
        alpha = ALPHA(num_surf_points)
        lam = 0.3
        lam_scaled = lam * LAM_SCALING
        
        # 对每个生成分子进行条件评估
        for i, item in enumerate(all_evaluated):
            if item['conf_scores'].get('is_valid', True) == False:
                print(f"    ⏩ 分子 {i+1}: 无效分子，跳过条件评估")
                item['cond_scores'] = {
                    'rmsd': 10.0, 'sims_surf': 0.0, 'sims_esp': 0.0, 'sims_pharm': 0.0,
                }
                continue
            
            try:
                rdkit_mol = create_rdkit_molecule(item['sample'])
                if rdkit_mol is None:
                    raise ValueError("无法创建RDKit分子")
                
                centers = rdkit_mol.GetConformer().GetPositions()
                radii = get_atomic_vdw_radii(rdkit_mol)
                
                # 生成表面点
                gen_surface = get_molecular_surface(
                    centers, radii, num_surf_points,
                    probe_radius=params['dataset']['probe_radius'],
                    num_samples_per_atom=20
                )
                
                # 提取药效团
                gen_pharm_types, gen_pharm_pos, gen_pharm_dir = get_pharmacophores(
                    rdkit_mol,
                    multi_vector=params['dataset']['x4']['multivectors'],
                    check_access=params['dataset']['x4']['check_accessibility']
                )
                
                # 计算表面电荷
                try:
                    rdkit_mol_h = rdkit.Chem.AddHs(rdkit_mol)
                    AllChem.ComputeGasteigerCharges(rdkit_mol_h)
                    partial_charges = np.array([
                        0.0 if np.isnan(c := float(a.GetProp('_GasteigerCharge'))) or np.isinf(c) else c
                        for a in rdkit_mol_h.GetAtoms()
                    ])
                except:
                    partial_charges = np.zeros(rdkit_mol.GetNumAtoms())
                
                # 计算表面静电势
                gen_esp = get_electrostatics_given_point_charges(partial_charges, centers, gen_surface)
                
                # 1. 表面相似性
                sims_surf = get_overlap_np(gen_surface, cond_surface, alpha=alpha)
                
                # 2. ESP表面相似性
                gen_esp_reshaped = gen_esp.reshape(-1, 1) if len(gen_esp.shape) == 1 else gen_esp
                cond_esp_reshaped = cond_esp.reshape(-1, 1) if len(cond_esp.shape) == 1 else cond_esp
                sims_esp = get_overlap_esp_np(
                    gen_surface, cond_surface, gen_esp_reshaped, cond_esp_reshaped,
                    alpha=alpha, lam=lam_scaled
                )
                
                # 3. 药效团相似性
                sims_pharm = 0.0
                if len(gen_pharm_types) > 0 and len(cond_pharm_types) > 0:
                    gen_pharm_types = np.array(gen_pharm_types).flatten()
                    gen_pharm_pos = np.array(gen_pharm_pos)
                    gen_pharm_dir = np.array(gen_pharm_dir)
                    try:
                        sims_pharm = get_overlap_pharm_np(
                            gen_pharm_types, cond_pharm_types,
                            gen_pharm_pos, cond_pharm_pos,
                            gen_pharm_dir, cond_pharm_dir,
                            similarity='tanimoto', extended_points=False, only_extended=False
                        )
                    except Exception as pharm_e:
                        print(f"      ⚠️ 药效团评分失败: {pharm_e}")
                
                # 处理NaN值
                sims_surf = float(sims_surf) if not np.isnan(sims_surf) else 0.0
                sims_esp = float(sims_esp) if not np.isnan(sims_esp) else 0.0
                sims_pharm = float(sims_pharm) if not np.isnan(sims_pharm) else 0.0
                
                item['cond_scores'] = {
                    'rmsd': 0.0, 'sims_surf': sims_surf, 'sims_esp': sims_esp, 'sims_pharm': sims_pharm,
                }
                print(f"    ✓ 分子 {i+1} Shepherd Score: 表面={sims_surf:.3f}, ESP={sims_esp:.3f}, 药效团={sims_pharm:.3f}")
                
            except Exception as e:
                print(f"    ⚠️ 分子 {i+1} 条件评估失败: {e}")
                item['cond_scores'] = {
                    'rmsd': 10.0, 'sims_surf': 0.0, 'sims_esp': 0.0, 'sims_pharm': 0.0,
                }
        
        # 3. 计算综合分数并构建偏好对
        for item in all_evaluated:
            conf = item['conf_scores']
            cond = item['cond_scores']
            
            # 检查分子是否有效
            if conf.get('is_valid', True) == False:
                # 无效分子赋予负无穷分
                item['total_score'] = float('-inf')
            else:
                # 综合评分（平衡Shepherd Score + 结构复杂性）
                try:
                    # 基础分：QED (0-2分)
                    total_score = conf['qed'] * 2.0
                    
                    # LogP惩罚：目标值1.5，限制最大惩罚 (最多扣1.5分)
                    logp_penalty = min(abs(conf['logp'] - 1.5), 5.0) * 0.3
                    total_score -= logp_penalty
                    
                    # 应变能惩罚：上限10 (最多扣5分)
                    total_score -= min(conf['strain_energy'], 10.0) * 0.5
                    
                    # SA Score惩罚：归一化到0-1范围 (最多扣2分)
                    sa_normalized = (conf['sa_score'] - 1.0) / 9.0  # SA通常范围1-10
                    total_score -= sa_normalized * 2.0
                    
                    # Fsp3奖励：结构复杂性 (0-1分)
                    # 较高的Fsp3表示更好的3D结构复杂性
                    total_score += conf.get('fsp3', 0.0) * 1.0
                    
                    # 表面相似性奖励 (0-1分)
                    total_score += cond['sims_surf'] * 1.0
                    
                    # 静电势相似性奖励 (0-2分，权重更高因为更重要)
                    total_score += cond['sims_esp'] * 2.0
                    
                    # 药效团相似性奖励 (0-2分)
                    total_score += cond.get('sims_pharm', 0.0) * 2.0
                    
                    
                    # RMSD惩罚：构象偏差 (最多扣2.5分)
                    # Fragment Merging时RMSD为0，不会扣分
                    total_score -= min(cond['rmsd'], 5.0) * 0.5
                    
                    # 检查是否产生了NaN
                    if np.isnan(total_score) or np.isinf(total_score):
                        print(f"    ⚠️  分数计算产生NaN/Inf - conf: {conf}, cond: {cond}")
                        total_score = -100.0  # 给一个极低但有效的分数
                    
                except Exception as e:
                    print(f"    ⚠️  分数计算异常: {e}")
                    total_score = -100.0
                
                item['total_score'] = total_score
        
        # 按分数排序（有效分子优先，无效分子排在最后）
        all_evaluated.sort(key=lambda x: x['total_score'], reverse=True)
        
        # 打印组内所有样本的得分
        valid_molecules = [item for item in all_evaluated if item['conf_scores'].get('is_valid', True)]
        print(f"  📊 组 {source_idx} 得分排名 (有效:{len(valid_molecules)}, 无效:{len(invalid_samples)}, 总计:{len(all_evaluated)})")
        for rank, item in enumerate(valid_molecules[:5]):  # 只显示前5个有效分子
            print(f"      {'🥇' if rank == 0 else ('🥈' if rank == 1 else '  ')} #{rank+1} ✓: total={item['total_score']:.3f} (QED={item['conf_scores']['qed']:.3f}, SA={item['conf_scores']['sa_score']:.2f})")
        if len(valid_molecules) > 5:
            print(f"      ... 省略 {len(valid_molecules) - 5} 个有效样本")
        
        # 收集本组的有效分子用于后续跨组统计
        for item in valid_molecules:
            item['group_id'] = source_idx
        all_valid_molecules_across_groups.extend(valid_molecules)
        
        # 尝试在组内构建偏好对（需要至少2个有效分子）
        if len(valid_molecules) >= 2:
            winner = valid_molecules[0]
            loser = valid_molecules[-1]
            winner_score = winner['total_score']
            loser_score = loser['total_score']
            score_gap = winner_score - loser_score
            min_gap = params.get('dpo', {}).get('min_score_gap', 0.3)
            
            if score_gap >= min_gap:
                winner_mol = create_rdkit_molecule(winner['sample'])
                loser_mol = create_rdkit_molecule(loser['sample'])
                
                if winner_mol is not None and loser_mol is not None:
                    pair = (
                        winner_mol,
                        loser_mol,
                        {**winner['conf_scores'], **winner['cond_scores'], 'total_score': winner['total_score']},
                        {**loser['conf_scores'], **loser['cond_scores'], 'total_score': loser['total_score']},
                    )
                    all_preference_pairs.append(pair)
                    groups_with_pairs.add(source_idx)
                    print(f"  ✅ 组 {source_idx} 偏好对构建成功: Winner={winner_score:.3f}, Loser={loser_score:.3f}, Gap={score_gap:.3f}")
                else:
                    print(f"  ❌ 组 {source_idx} 偏好对构建失败: 分子重构失败")
            else:
                print(f"  ⚠️  组 {source_idx} 组内分差不足 ({score_gap:.3f} < {min_gap})，等待跨组统计")
        else:
            print(f"  ⚠️  组 {source_idx} 有效分子不足 ({len(valid_molecules)}<2)，等待跨组统计")
    
    # === 跨组统计：如果单组构建失败，从所有有效分子中构建偏好对 ===
    if len(all_preference_pairs) < len(grouped_samples) and len(all_valid_molecules_across_groups) >= 2:
        print(f"\n🔄 跨组统计: 从 {len(all_valid_molecules_across_groups)} 个有效分子中补充偏好对")
        
        # 按分数排序所有有效分子
        all_valid_molecules_across_groups.sort(key=lambda x: x['total_score'], reverse=True)
        
        min_gap = params.get('dpo', {}).get('min_score_gap', 0.3)
        
        # 构建跨组偏好对：最高分 vs 最低分
        for i in range(min(3, len(all_valid_molecules_across_groups) - 1)):  # 最多补充3对
            winner = all_valid_molecules_across_groups[i]
            loser = all_valid_molecules_across_groups[-(i+1)]
            
            # 避免同一分子
            if winner is loser:
                continue
            
            winner_score = winner['total_score']
            loser_score = loser['total_score']
            score_gap = winner_score - loser_score
            
            if score_gap >= min_gap:
                winner_mol = create_rdkit_molecule(winner['sample'])
                loser_mol = create_rdkit_molecule(loser['sample'])
                
                if winner_mol is not None and loser_mol is not None:
                    pair = (
                        winner_mol,
                        loser_mol,
                        {**winner['conf_scores'], **winner['cond_scores'], 'total_score': winner['total_score']},
                        {**loser['conf_scores'], **loser['cond_scores'], 'total_score': loser['total_score']},
                    )
                    all_preference_pairs.append(pair)
                    w_group = winner.get('group_id', '?')
                    l_group = loser.get('group_id', '?')
                    print(f"  ✅ 跨组偏好对: 组{w_group}(Winner={winner_score:.3f}) vs 组{l_group}(Loser={loser_score:.3f}), Gap={score_gap:.3f}")
    
    print(f"\n{'='*50}")
    print(f"✅ 偏好对构建汇总: {len(all_preference_pairs)} 对 (来自 {len(grouped_samples)} 组)")
    print(f"{'='*50}")
    return all_preference_pairs


def main():
    """DPO训练主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="SPD分子生成模型 - 纯DPO训练")
    parser.add_argument("model_name", type=str, help="模型配置名称（如params_x1x3x4_dpo_finetune_nps）")
    parser.add_argument("seed", type=int, help="随机种子")
    args = parser.parse_args()
    
    # 设置随机种子
    seed_everything(seed=args.seed, workers=True)
    
    print("="*80)
    print("🚀 SPD DPO训练启动器")
    print("="*80)
    print(f"📋 配置: {args.model_name}")
    print(f"🎲 随机种子: {args.seed}")
    print("="*80)
    
    # 加载参数
    params = importlib.import_module(f'parameters.{args.model_name}').params
    
    # 确保DPO已启用
    if not params['training'].get('enable_dpo', False):
        print("⚠️  警告：参数文件中enable_dpo=False，已自动设置为True")
        params['training']['enable_dpo'] = True
    
    # 检查是否为DDP子进程
    # PyTorch Lightning在使用DDP spawn/subprocess时会设置LOCAL_RANK
    is_ddp_subprocess = os.environ.get("LOCAL_RANK") is not None
    if is_ddp_subprocess:
        print(f"🤖 检测到 DDP 子进程 (RANK {os.environ.get('LOCAL_RANK')})，将跳过预处理和首次采样...")

    # 加载Fragment Merging条件数据
    print("\n📂 加载Fragment Merging条件数据...")
    molblocks_and_charges, fragment_merge_condition = load_fragment_merge_condition(params)
    
    # 加载MOSES_aq边际分布
    print("\n📊 加载边际分布...")
    marginals = load_cached_marginals()
    
    # 创建基础数据集
    print("\n🔧 创建基础数据集...")
    dataset = create_dataset(params, molblocks_and_charges, marginals)
    print(f"✅ 基础数据集创建完成: {len(dataset)} 个样本")
    
    # 创建DPO数据集（初始为空）
    print("\n🎯 初始化DPO数据集...")
    dpo_dataset = DPODataset(
        preference_pairs=[],  # 初始为空，通过采样回调动态填充
        base_dataset=dataset,
        noise_schedule_dict=params['noise_schedules'],
        params=params,
    )
    print(f"✅ DPO数据集初始化完成（初始偏好对: {len(dpo_dataset.preference_pairs)}）")
    
    # 进行首次采样以生成初始偏好对
    # 【修改】如果是DDP子进程，跳过此步骤，防止超时
    initial_pairs = []
    if not is_ddp_subprocess:
        print("\n🔬 进行首次多GPU并行采样以生成初始偏好对...")
        
        # 检测可用GPU数量
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"   可用GPU数量: {num_gpus}")
        
        # 采样配置
        num_groups = params.get('sampling', {}).get('num_groups', 4)
        samples_per_group = params.get('sampling', {}).get('samples_per_group', 4)
        print(f"   采样配置: {num_groups} 组 × {samples_per_group} 样本/组 = {num_groups * samples_per_group} 总样本")
        
        # 在每个GPU上创建模型副本
        pretrained_path = params['training'].get('pretrained_checkpoint_path', None)
        models_dict = {}
        
        for gpu_id in range(max(1, num_gpus)):
            device = torch.device('cuda', gpu_id) if num_gpus > 0 else torch.device('cpu')
            print(f"   创建模型副本到 GPU {gpu_id}...")
            
            if pretrained_path and os.path.exists(pretrained_path):
                try:
                    model = LightningModule.load_from_checkpoint(
                        pretrained_path, params=params, strict=False
                    )
                    print(f"   ✅ GPU {gpu_id}: 加载预训练权重成功")
                except Exception as e:
                    print(f"   ❌ GPU {gpu_id}: 加载失败 - {e}")
                    model = LightningModule(params)
            else:
                model = LightningModule(params)
            
            model.to(device)
            model.model.device = device
            model.eval()
            models_dict[gpu_id] = model
        
        print(f"   ✅ 已在 {len(models_dict)} 个GPU上创建模型")
        
        # 循环采样直到生成有效的偏好对
        sample_attempt = 0
        while True:
            sample_attempt += 1
            if sample_attempt > 1:
                print(f"\n🔄 第 {sample_attempt} 次尝试多GPU并行采样...")
            
            with torch.no_grad():
                initial_pairs = sample_and_evaluate_molecules_multi_gpu(
                    models_dict,
                    params,
                    molblocks_and_charges,
                    dataset,
                    num_groups=num_groups,
                    samples_per_group=samples_per_group,
                    fragment_merge_condition=fragment_merge_condition
                )
            
            if len(initial_pairs) > 0:
                print(f"   ✅ 成功生成 {len(initial_pairs)} 个偏好对")
                break
            
            print("   ⚠️  本次采样未生成任何有效偏好对，继续尝试...")
        
        # 释放所有模型
        for model in models_dict.values():
            del model
        models_dict.clear()
        torch.cuda.empty_cache()
    else:
        print("⏩ 子进程跳过首次采样步骤")
        
        # 子进程如果没有偏好对，使用虚拟对防止报错（仅子进程）
        if len(initial_pairs) == 0 and len(dataset) > 0:
             try:
                mol_block = dataset.molblocks_and_charges[0][0]
                dummy_mol = rdkit.Chem.MolFromMolBlock(mol_block, removeHs=False)
             except:
                dummy_mol = rdkit.Chem.MolFromSmiles("C")
                rdkit.Chem.AddHs(dummy_mol)
                
             dummy_scores = {
                'qed': 0.0, 'logp': 0.0, 'strain_energy': 0.0,
                'sa_score': 0.0, 'rmsd': 0.0, 'sims_surf': 0.0,
                'sims_esp': 0.0, 'total_score': 0.0
             }
             initial_pairs = [(dummy_mol, dummy_mol, dummy_scores, dummy_scores)]

    # 移除原有的通用虚拟填充逻辑，因为主进程现在保证有数据
    if len(initial_pairs) > 0:
        print(f"   ✅ 采样准备就绪：{len(initial_pairs)} 个偏好对")
    
    # 更新DPO数据集
    dpo_dataset.preference_pairs = initial_pairs
    print(f"✅ DPO数据集更新完成（偏好对: {len(dpo_dataset.preference_pairs)}）")
    
    # 创建DataLoader
    print("\n📦 创建DataLoader...")
    train_loader = create_dpo_dataloader(params, dataset, dpo_dataset)
    
    # 设置输出目录
    output_dir = f"jobs/{params['training']['output_dir']}"
    os.makedirs("jobs/", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    print(f"✅ 输出目录: {output_dir}")
    
    # 设置回调
    print("\n⚙️  设置训练回调...")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=0,
        save_last=True,
        monitor="train_loss",
        mode="min",
        dirpath=output_dir,
        filename="best-{step:09d}",
        every_n_train_steps=params['training']['log_every_n_steps'],
    )
    
    # 自定义DPO采样回调（基于dpo_sample_and_evaluation.py）
    class DPOSamplingCallback(pl.Callback):
        def __init__(self, params, dataset, dpo_dataset, molblocks_and_charges, fragment_merge_condition=None):
            super().__init__()
            self.params = params
            self.dataset = dataset
            self.dpo_dataset = dpo_dataset  # 直接引用DPO数据集
            self.molblocks_and_charges = molblocks_and_charges
            self.fragment_merge_condition = fragment_merge_condition
            self.preference_pairs = []
            self.epoch_counter = 0  # 用于追踪采样次数
            
        def on_train_epoch_end(self, trainer, pl_module):
            """每个epoch结束时进行采样和评估，为下一个epoch准备数据
            
            注意：必须在epoch_end而非epoch_start更新数据集，否则会导致
            DistributedSampler的total_size与实际数据集大小不一致的断言错误。
            reload_dataloaders_every_n_epochs=1会在下一个epoch开始时重建DataLoader。
            """
            if trainer.current_epoch % params['training'].get('dpo_sampling_every_n_epochs', 1) != 0:
                return
            
            # 每个rank使用自己的模型在自己的GPU上采样（不创建额外模型副本，避免内存翻倍）
            rank = trainer.global_rank
            world_size = trainer.world_size if hasattr(trainer, 'world_size') else 1
            samples_per_group = params.get('sampling', {}).get('samples_per_group', 4)
            
            if rank == 0:
                self.epoch_counter += 1
                print(f"\n🔄 Epoch {trainer.current_epoch} 结束: 开始DPO采样 (第{self.epoch_counter}次)")
                print(f"   每个GPU使用现有模型采样 {samples_per_group} 个样本（避免内存翻倍）")
            
            # 每个rank在自己的GPU上采样
            device = pl_module.device
            pl_module.eval()
            
            # 为每个rank设置不同的随机种子，确保生成不同的分子
            import random
            unique_seed = trainer.current_epoch * 1000 + rank * 100 + int(time.time() % 100)
            torch.manual_seed(unique_seed)
            torch.cuda.manual_seed(unique_seed)
            np.random.seed(unique_seed)
            random.seed(unique_seed)
            
            with torch.no_grad():
                local_samples = []
                try:
                    # 使用现有模型采样（inference_sample已在文件顶部导入）
                    
                    # 获取边际分布
                    atom_marginals = self.dataset.x1_atom_diffuser.transition_model.marginals.to(device)
                    bond_marginals = self.dataset.x1_bond_diffuser.transition_model.marginals.to(device)
                    
                    if len(atom_marginals) > 0:
                        atom_marginals[0] = 0.0
                        atom_marginals = atom_marginals / atom_marginals.sum()
                    
                    # 条件数据
                    fc = self.fragment_merge_condition
                    n_atoms = params.get('sampling', {}).get('fixed_n_atoms', 70)
                    
                    inference_kwargs = {
                        "batch_size": samples_per_group,
                        "N_x1": n_atoms,
                        "N_x4": 27,
                        "unconditional": False,
                        "prior_noise_scale": 1.0,
                        "denoising_noise_scale": 1.0,
                        "inject_noise_at_ts": [],
                        "inject_noise_scales": [],
                        "harmonize": False,
                        "harmonize_ts": [],
                        "harmonize_jumps": [],
                        "inpaint_x2_pos": False,
                        "inpaint_x3_pos": False,
                        "inpaint_x3_x": False,
                        "inpaint_x4_pos": True,
                        "inpaint_x4_direction": True,
                        "inpaint_x4_type": True,
                        "stop_inpainting_at_time_x2": 0.0,
                        "add_noise_to_inpainted_x2_pos": 0.0,
                        "stop_inpainting_at_time_x3": 0.0,
                        "add_noise_to_inpainted_x3_pos": 0.0,
                        "add_noise_to_inpainted_x3_x": 0.0,
                        "stop_inpainting_at_time_x4": 0.0,
                        "add_noise_to_inpainted_x4_pos": 0.0,
                        "add_noise_to_inpainted_x4_direction": 0.0,
                        "add_noise_to_inpainted_x4_type": 0.0,
                        "center_of_mass": np.zeros(3),
                        "surface": fc['x3']['positions'],
                        "electrostatics": fc['x3']['charges'],
                        "pharm_types": fc['x4']['types'],
                        "pharm_pos": fc['x4']['positions'],
                        "pharm_direction": fc['x4']['directions'],
                        "atom_marginals": atom_marginals,
                        "bond_marginals": bond_marginals,
                    }
                    
                    local_samples = inference_sample(pl_module, **inference_kwargs)
                    for sample in local_samples:
                        sample['source_mol_index'] = 0
                        sample['group_id'] = rank
                    
                    print(f"  [Rank {rank}] ✅ 采样完成: {len(local_samples)} 个样本")
                    
                    del atom_marginals, bond_marginals
                    
                except Exception as e:
                    print(f"  [Rank {rank}] ❌ 采样失败: {e}")
                    import traceback
                    traceback.print_exc()
                    local_samples = []  # 确保即使失败也有空列表
            
            pl_module.train()
            torch.cuda.empty_cache()
            
            # 同步所有rank - 使用 try/except 防止单点故障导致全部崩溃
            try:
                if torch.distributed.is_initialized():
                    # 设置较短的超时，避免无限等待
                    torch.distributed.barrier()
            except Exception as e:
                print(f"  [Rank {rank}] ⚠️ barrier 同步失败: {e}")
            
            # 收集所有rank的采样结果到rank 0
            if torch.distributed.is_initialized() and world_size > 1:
                try:
                    # 使用pickle序列化采样结果
                    import pickle
                    local_data = pickle.dumps(local_samples)
                    local_size = torch.tensor([len(local_data)], dtype=torch.long, device=device)
                    
                    # 收集所有size
                    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
                    torch.distributed.all_gather(all_sizes, local_size)
                    
                    max_size = max(s.item() for s in all_sizes)
                    
                    # 填充到相同大小
                    padded_data = local_data + b'\x00' * (max_size - len(local_data))
                    local_tensor = torch.ByteTensor(list(padded_data)).to(device)
                    
                    # 收集所有数据
                    all_tensors = [torch.zeros(max_size, dtype=torch.uint8, device=device) for _ in range(world_size)]
                    torch.distributed.all_gather(all_tensors, local_tensor)
                    
                    # rank 0 反序列化并合并
                    if rank == 0:
                        all_samples = []
                        for i, (tensor, size) in enumerate(zip(all_tensors, all_sizes)):
                            data = bytes(tensor[:size.item()].cpu().tolist())
                            samples = pickle.loads(data)
                            all_samples.extend(samples)
                        
                        print(f"\n📊 收集到 {len(all_samples)} 个采样结果")
                        
                        # 评估并构建偏好对
                        new_pairs = evaluate_and_build_pairs(
                            all_samples,
                            [],
                            self.molblocks_and_charges,
                            self.params,
                            fragment_merge_condition=self.fragment_merge_condition
                        )
                        
                        if len(new_pairs) > 0:
                            # 只使用当前轮的偏好对，不累积历史
                            self.preference_pairs = new_pairs
                            self.dpo_dataset.preference_pairs = self.preference_pairs
                            print(f"✅ 更新DPO数据集: {len(self.preference_pairs)} 个偏好对（仅当前轮）")
                
                except Exception as e:
                    print(f"  [Rank {rank}] ⚠️ all_gather 同步失败: {e}")
                    # 如果同步失败，rank 0 使用本地样本
                    if rank == 0:
                        new_pairs = evaluate_and_build_pairs(
                            local_samples,
                            [],
                            self.molblocks_and_charges,
                            self.params,
                            fragment_merge_condition=self.fragment_merge_condition
                        )
                        if len(new_pairs) > 0:
                            # 只使用当前轮的偏好对，不累积历史
                            self.preference_pairs = new_pairs
                            self.dpo_dataset.preference_pairs = self.preference_pairs
                            print(f"✅ 使用本地样本更新DPO数据集: {len(self.preference_pairs)} 个偏好对（仅当前轮）")
            else:
                # 单GPU模式
                if rank == 0:
                    new_pairs = evaluate_and_build_pairs(
                        local_samples,
                        [],
                        self.molblocks_and_charges,
                        self.params,
                        fragment_merge_condition=self.fragment_merge_condition
                    )
                    
                    if len(new_pairs) > 0:
                        # 只使用当前轮的偏好对，不累积历史
                        self.preference_pairs = new_pairs
                        self.dpo_dataset.preference_pairs = self.preference_pairs
                        print(f"✅ 更新DPO数据集: {len(self.preference_pairs)} 个偏好对（仅当前轮）")
            
            # 最终同步
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

    sampling_callback = DPOSamplingCallback(
        params=params,
        dataset=dataset,
        dpo_dataset=dpo_dataset,
        molblocks_and_charges=molblocks_and_charges,
        fragment_merge_condition=fragment_merge_condition
    )
    
    callbacks = [checkpoint_callback, sampling_callback]
    print("✅ 回调配置完成")
    
    # 设置日志
    print("\n📝 设置日志记录...")
    csv_logger = CSVLogger(
        save_dir=output_dir,
        name='csv_logger',
    )
    
    loggers = [csv_logger]
    print("✅ 日志配置完成")
    
    # 设置训练器
    print("\n⚡ 配置PyTorch Lightning训练器...")
    cuda_available = torch.cuda.is_available()
    num_gpus_to_use = torch.cuda.device_count()
    
    print(f"   CUDA可用: {cuda_available}")
    print(f"   GPU数量: {num_gpus_to_use}")
    
    if params['training']['num_gpus'] > 1:
        from pytorch_lightning.strategies import DDPStrategy
        strategy = DDPStrategy(
            find_unused_parameters=False,  # 减少通信开销
            timeout=timedelta(hours=2),     # 增加超时时间到2小时
            gradient_as_bucket_view=True,   # 优化梯度通信
        )
    else:
        strategy = 'auto'
        
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=output_dir,
        accelerator="gpu" if (params['training']['num_gpus'] >= 1 and cuda_available) else 'cpu',
        max_epochs=10000,
        gradient_clip_val=params['training']['gradient_clip_val'],
        accumulate_grad_batches=params['training']['accumulate_grad_batches'],
        log_every_n_steps=params['training']['log_every_n_steps'],
        reload_dataloaders_every_n_epochs=1,  # 重要：每个epoch重载以获取新的偏好对
        devices=num_gpus_to_use if cuda_available else "auto",
        strategy=strategy,
        precision=32,
        detect_anomaly=True,
    )
    print("✅ 训练器配置完成")
    
    # 创建模型
    print("\n🧠 创建模型...")
    
    # 逻辑更新：使用 load_from_checkpoint 加载
    ckpt_path = f"{output_dir}/last.ckpt"
    resume_ckpt_path = None
    
    # 1. 检查是否有断点需要恢复
    if os.path.exists(ckpt_path):
        print(f"\n✅ 发现当前任务checkpoint: {ckpt_path}")
        print(f"   将继续当前任务的训练")
        resume_ckpt_path = ckpt_path
        # 恢复训练时，模型权重会被Trainer自动加载，这里只需初始化
        model_pl = LightningModule(params)
        
        # 备份当前checkpoint (仅主进程)
        if trainer.global_rank == 0:
            date = datetime.datetime.now()
            timestamp = date.strftime("%Y_%m_%d_%H_%M")
            backup_path = f"{output_dir}/last_{timestamp}.ckpt"
            try:
                shutil.copyfile(ckpt_path, backup_path)
                print(f"📦 已备份checkpoint到: {backup_path}")
            except Exception as e:
                print(f"⚠️ 备份checkpoint失败: {e}")

    # 2. 如果没有断点，检查是否有预训练模型（DPO微调）
    else:
        pretrained_path = params['training'].get('pretrained_checkpoint_path', None)
        if pretrained_path is not None:
            pretrained_ckpt_path = f"{pretrained_path}"
            if os.path.exists(pretrained_ckpt_path):
                print(f"\n🔄 DPO微调：从预训练模型加载权重: {pretrained_ckpt_path}")
                try:
                    model_pl = LightningModule.load_from_checkpoint(
                        pretrained_ckpt_path, 
                        params=params, 
                        strict=False
                    )
                    print("   ✅ 成功加载预训练权重")
                    
                    # 同步到ref_model
                    if hasattr(model_pl, 'ref_model'):
                        model_pl.ref_model.load_state_dict(model_pl.model.state_dict())
                        print(f"   ✅ 已同步权重到参考模型（ref_model）")
                        
                except Exception as e:
                    print(f"   ❌ 加载预训练权重失败: {e}")
                    print("   ⚠️ 使用随机初始化")
                    model_pl = LightningModule(params)
            else:
                print(f"\n⚠️  预训练checkpoint不存在: {pretrained_ckpt_path}")
                print("📝 从头开始DPO训练")
                model_pl = LightningModule(params)
        else:
            print("\n📝 从头开始DPO训练 (无预训练路径)")
            model_pl = LightningModule(params)

    # 统计参数量
    total_params = sum(p.numel() for p in model_pl.parameters() if p.requires_grad)
    print(f"✅ 模型创建完成")
    print(f"   可训练参数: {total_params:,}")
    
    # 开始训练
    print("\n" + "="*80)
    print("🚀 开始DPO训练...")
    print("="*80)
    
    # DDP安全检查：确保偏好对数量 >= GPU数量
    num_pairs = len(dpo_dataset.preference_pairs)
    if num_gpus_to_use > 1 and num_pairs < num_gpus_to_use:
        print(f"\n⚠️  警告: 偏好对数量({num_pairs}) < GPU数量({num_gpus_to_use})")
        print(f"   在DDP模式下，部分GPU可能没有数据")
        print(f"   建议增加采样数量或减少GPU数量")
        # 复制样本以确保每个GPU至少有一个样本
        while len(dpo_dataset.preference_pairs) < num_gpus_to_use:
            dpo_dataset.preference_pairs = dpo_dataset.preference_pairs * 2
        print(f"   已自动复制样本至 {len(dpo_dataset.preference_pairs)} 个偏好对")
    
    # 启动训练
    trainer.fit(model_pl, train_dataloaders=train_loader, ckpt_path=resume_ckpt_path)
    
    print("\n" + "="*80)
    print("🎉 训练完成！")
    print("="*80)


if __name__ == '__main__':
    main()

