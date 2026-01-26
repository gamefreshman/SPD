#!/usr/bin/env python3
"""
纯粹的DPO训练启动器
基于new_train.py的DPO训练逻辑和dpo_sample_and_evaluation.py的采样评估功能
专注于DPO微调，不涉及混合数据训练
"""

# ==================== 系统配置 ====================
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

# ==================== 导入依赖 ====================
import os
import shutil
import datetime
import pickle
import warnings
import argparse
import importlib
import multiprocessing
import json
import traceback
from functools import partial
from copy import deepcopy

import numpy as np
import torch
import torch.multiprocessing
import torch_geometric
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import rdkit
import rdkit.Chem
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from datetime import timedelta
from lightning_fabric.utilities.seed import seed_everything

# 项目模块
from shepherd.model.model import Model
from shepherd.lightning_module import LightningModule
from shepherd.new_datasets import HeteroDataset
from shepherd.dpo_dataset import DPODataset
from shepherd.inference import inference_sample
from shepherd.extract import create_rdkit_molecule
from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores
from shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates
from shepherd.shepherd_score_utils.generate_point_cloud import (
    get_atomic_vdw_radii,
    get_molecular_surface,
    get_electrostatics_given_point_charges,
)

# Shepherd Score评估模块
from shepherd_score.evaluations.evaluate import ConfEval, ConditionalEvalPipeline
from shepherd_score.container import Molecule

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


def get_bond_type_str(bond):
    """将RDKit键类型转换为字符串"""
    return str(bond.GetBondType())


def process_batch(batch, atom_types_x1, bond_types_x1, max_node_types_x4, params):
    """
    并行处理分子批次，统计特征分布
    """
    batch_atom_counts = torch.zeros(len(atom_types_x1), dtype=torch.float)
    batch_bond_counts = torch.zeros(len(bond_types_x1), dtype=torch.float)
    batch_pharm_counts = torch.zeros(max_node_types_x4, dtype=torch.float)

    for mol_block, _ in batch:
        mol = rdkit.Chem.MolFromMolBlock(mol_block, removeHs=False)
        if not mol:
            continue

        # 统计原子类型 (x1)
        if params['dataset']['compute_x1']:
            if params['dataset']['x1']['add_virtual_node']:
                batch_atom_counts[atom_types_x1.index(None)] += 1
            
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol in atom_types_x1:
                    batch_atom_counts[atom_types_x1.index(symbol)] += 1

            for bond in mol.GetBonds():
                bond_str = get_bond_type_str(bond)
                if bond_str in bond_types_x1:
                    batch_bond_counts[bond_types_x1.index(bond_str)] += 1

        # 统计药效团类型 (x4)
        if params['dataset']['compute_x4']:
            if params['dataset']['x4']['add_virtual_node']:
                batch_pharm_counts[0] += 1
            
            try:
                pharm_types, _, _ = get_pharmacophores(
                    mol, 
                    multi_vector=params['dataset']['x4']['multivectors'],
                    check_access=params['dataset']['x4']['check_accessibility']
                )
                for p_type in (pharm_types + 1):
                    if p_type < max_node_types_x4:
                        batch_pharm_counts[p_type] += 1
            except Exception:
                pass
    
    return batch_atom_counts, batch_bond_counts, batch_pharm_counts


def compute_and_cache_marginals(params, molblocks_and_charges, cache_dir="cached_marginals"):
    """
    计算或加载缓存的特征边际分布
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    dataset_name = params['data']
    atom_marginals_file = os.path.join(cache_dir, f"{dataset_name}_atom_marginals.pt")
    bond_marginals_file = os.path.join(cache_dir, f"{dataset_name}_bond_marginals.pt")
    pharm_marginals_file = os.path.join(cache_dir, f"{dataset_name}_pharm_marginals.pt")

    # 尝试加载缓存
    if (os.path.exists(atom_marginals_file) and 
        os.path.exists(bond_marginals_file) and 
        os.path.exists(pharm_marginals_file)):
        
        print(f"✅ 从 '{cache_dir}' 加载已缓存的边际分布")
        atom_marginals_x1 = torch.load(atom_marginals_file, weights_only=True)
        bond_marginals_x1 = torch.load(bond_marginals_file, weights_only=True)
        pharm_marginals_x4 = torch.load(pharm_marginals_file, weights_only=True)
        return atom_marginals_x1, bond_marginals_x1, pharm_marginals_x4

    # 如果没有缓存，进行并行计算
    print("📊 开始计算特征边际分布...")
    
    atom_types_x1 = params['dataset']['x1']['atom_types']
    bond_types_x1 = params['dataset']['x1']['bond_types']
    max_node_types_x4 = params['dataset']['x4']['max_node_types']

    total_atom_counts = torch.zeros(len(atom_types_x1), dtype=torch.float)
    total_bond_counts = torch.zeros(len(bond_types_x1), dtype=torch.float)
    total_pharm_counts = torch.zeros(max_node_types_x4, dtype=torch.float)
    
    num_processes = multiprocessing.cpu_count()
    batch_size_for_processing = 1000
    batches = [molblocks_and_charges[i:i + batch_size_for_processing] 
               for i in range(0, len(molblocks_and_charges), batch_size_for_processing)]
    
    worker_fn = partial(process_batch, 
                        atom_types_x1=atom_types_x1, 
                        bond_types_x1=bond_types_x1, 
                        max_node_types_x4=max_node_types_x4,
                        params=params)

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(worker_fn, batches, chunksize=1), 
                           total=len(batches), desc="统计特征"))

    for res in results:
        total_atom_counts += res[0]
        total_bond_counts += res[1]
        total_pharm_counts += res[2]

    # 计算边际分布
    atom_marginals_x1 = (total_atom_counts / total_atom_counts.sum() 
                         if total_atom_counts.sum() > 0 
                         else torch.ones_like(total_atom_counts) / len(total_atom_counts))
    bond_marginals_x1 = (total_bond_counts / total_bond_counts.sum() 
                         if total_bond_counts.sum() > 0 
                         else torch.ones_like(total_bond_counts) / len(total_bond_counts))
    pharm_marginals_x4 = (total_pharm_counts / total_pharm_counts.sum() 
                          if total_pharm_counts.sum() > 0 
                          else torch.ones_like(total_pharm_counts) / len(total_pharm_counts))
    
    print(f"✅ 边际分布计算完成")
    
    # 保存缓存
    torch.save(atom_marginals_x1, atom_marginals_file)
    torch.save(bond_marginals_x1, bond_marginals_file)
    torch.save(pharm_marginals_x4, pharm_marginals_file)

    return atom_marginals_x1, bond_marginals_x1, pharm_marginals_x4


def load_dataset(params):
    """加载数据集"""
    molblocks_and_charges = []
    output_file = ""
    
    if params['data'] == 'NPs':
        with open('../data/conformers/np/molblock_charges_NPs.pkl', 'rb') as f:
            molblocks_and_charges = pickle.load(f)
        print(f"✅ 加载NPs数据集: {len(molblocks_and_charges)} 个分子")
        output_file = "NPs"
    elif params['data'] == 'GDB17':
        with open('../data/conformers/gdb/example_molblock_charges.pkl', 'rb') as f:
            molblocks_and_charges = pickle.load(f)
        output_file = "GDB17"
    elif params['data'] == 'MOSES_aq':
        with open('../data/conformers/moses_aq/example_molblock_charges.pkl', 'rb') as f:
            molblocks_and_charges = pickle.load(f)
        output_file = "MOSES_aq"
    
    return molblocks_and_charges, output_file


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


def sample_and_evaluate_molecules(model_pl, params, molblocks_and_charges, dataset, 
                                   num_samples_per_mol=4, device='cuda'):
    """
    采样和评估分子，基于dpo_sample_and_evaluation.py的方式
    
    Returns:
        preference_pairs: List[(winner_data, loser_data, winner_scores, loser_scores)]
    """
    print("\n" + "="*80)
    print("🧬 开始在线采样和评估")
    print("="*80)
    
    model_pl.eval() # 切换评估模式
    # dropout 等层是关闭的
    # batch normalization 使用的是统计值
     
    # 获取边际分布（从dataset）
    atom_marginals = dataset.x1_atom_diffuser.transition_model.marginals.to(device)
    bond_marginals = dataset.x1_bond_diffuser.transition_model.marginals.to(device)
    
    # ========================================================================
    # 【修正】强制将虚拟节点（Index 0）的边际概率设为0，以避免生成无效原子
    # 这一步是为了对齐 test_sample.py 的行为，已被证实能提高生成质量
    # ========================================================================
    if len(atom_marginals) > 0:
        print(f"🔧 修正前 atom_marginals[0] = {atom_marginals[0]:.6f}")
        atom_marginals[0] = 0.0
        atom_marginals = atom_marginals / atom_marginals.sum()
        print(f"🔧 修正后 atom_marginals[0] = {atom_marginals[0]:.6f}")
    # ========================================================================

    print("atom_marginals: ", atom_marginals)
    print("bond_marginals: ", bond_marginals)
    
    # 存储所有生成的样本
    all_generated_samples = []
    all_reference_mols = []
    
    # 对每个分子进行采样
    for mol_index in range(len(molblocks_and_charges)):
        print(f"\n{'='*60}")
        print(f"🔬 处理分子 {mol_index + 1}/{len(molblocks_and_charges)}")
        print(f"{'='*60}")
        
        mol_block, charges = molblocks_and_charges[mol_index]
        mol = rdkit.Chem.MolFromMolBlock(mol_block, removeHs=False)
        charges = np.array(charges)
        
        # 保存参考分子
        all_reference_mols.append(mol)
        
        # 预处理分子坐标
        mol_coordinates = np.array(mol.GetConformer().GetPositions())
        mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis=0)
        mol = update_mol_coordinates(mol, mol_coordinates)
        
        # 提取条件特征
        centers = mol.GetConformer().GetPositions()
        radii = get_atomic_vdw_radii(mol)
        
        # 生成分子表面点云
        surface = get_molecular_surface(
            centers,
            radii,
            params['dataset']['x3']['num_points'],
            probe_radius=params['dataset']['probe_radius'],
            num_samples_per_atom=20,
        )
        
        # 提取药效团特征
        pharm_types, pharm_pos, pharm_direction = get_pharmacophores(
            mol,
            multi_vector=params['dataset']['x4']['multivectors'],
            check_access=params['dataset']['x4']['check_accessibility'],
        )
        
        # 计算表面静电势
        electrostatics = get_electrostatics_given_point_charges(
            charges, centers, surface,
        )
        
        # 采样参数
        n_atoms = params.get('sampling', {}).get('fixed_n_atoms', 70)
        num_pharmacophores = len(pharm_types)
        
        print(f"  采样配置: {num_samples_per_mol} 个样本, {n_atoms} 个原子")
        
        # 循环生成多个样本
        mol_samples = []
        for i in range(num_samples_per_mol):
            try:
                # 准备inference_sample参数字典以便记录
                inference_kwargs = {
                    "batch_size": 1,
                    "N_x1": n_atoms,
                    "N_x4": num_pharmacophores,
                    "unconditional": False,
                    
                    # 噪声控制
                    "prior_noise_scale": 1.0,
                    "denoising_noise_scale": 1.0,
                    "inject_noise_at_ts": [],
                    "inject_noise_scales": [],
                    
                    # 谐波化
                    "harmonize": False,
                    "harmonize_ts": [],
                    "harmonize_jumps": [],
                    
                    # 条件修复
                    "inpaint_x2_pos": False,
                    "inpaint_x3_pos": False,
                    "inpaint_x3_x": False,
                    "inpaint_x4_pos": True,
                    "inpaint_x4_direction": True,
                    "inpaint_x4_type": True,
                    
                    # 修复控制
                    "stop_inpainting_at_time_x2": 0.0,
                    "add_noise_to_inpainted_x2_pos": 0.0,
                    "stop_inpainting_at_time_x3": 0.0,
                    "add_noise_to_inpainted_x3_pos": 0.0,
                    "add_noise_to_inpainted_x3_x": 0.0,
                    "stop_inpainting_at_time_x4": 0.0,
                    "add_noise_to_inpainted_x4_pos": 0.0,
                    "add_noise_to_inpainted_x4_direction": 0.0,
                    "add_noise_to_inpainted_x4_type": 0.0,
                    
                    # 条件输入
                    "center_of_mass": np.zeros(3),
                    "surface": surface,
                    "electrostatics": electrostatics,
                    "pharm_types": pharm_types,
                    "pharm_pos": pharm_pos,
                    "pharm_direction": pharm_direction,
                    
                    # 边际分布
                    "atom_marginals": atom_marginals,
                    "bond_marginals": bond_marginals,
                }

                # 记录参数到JSON文件
                try:
                    debug_params_dir = "debug_inference_params"
                    os.makedirs(debug_params_dir, exist_ok=True)
                    debug_params_file = f"{debug_params_dir}/mol_{mol_index}_sample_{i}.json"
                    with open(debug_params_file, 'w') as f:
                        json.dump(convert_for_json(inference_kwargs), f, indent=4, default=str)
                    print(f"    💾 参数已记录: {debug_params_file}")
                except Exception as e:
                    print(f"    ⚠️  无法记录参数: {e}")

                # 调用inference_sample
                generated_samples = inference_sample(
                    model_pl,
                    **inference_kwargs
                )
                
                # 处理生成的样本
                if len(generated_samples) > 0:
                    sample = generated_samples[0]
                    
                    # 打印原子类型信息
                    if 'x1' in sample and 'atoms' in sample['x1']:
                        atom_types = sample['x1']['atoms']
                        unique_atoms = set(atom_types)
                        print(f"    📊 样本 {i+1}/{num_samples_per_mol} 原子类型: {sorted(unique_atoms)}")
                        if 0 in unique_atoms:
                            atom_count_0 = list(atom_types).count(0)
                            print(f"       ⚠️  包含无效原子(0): {atom_count_0} 个")
                    
                    # 设置样本所属的源分子索引，用于后续分组构建偏好对
                    sample['source_mol_index'] = mol_index
                    mol_samples.append(sample)
                else:
                    print(f"    ⚠️  样本 {i+1}/{num_samples_per_mol}: inference_sample未生成任何输出")
                    
            except Exception as e:
                print(f"    ❌ 样本 {i+1}/{num_samples_per_mol}: 采样异常 - {e}")
        
        all_generated_samples.extend(mol_samples)
        print(f"✅ 分子 {mol_index + 1} 完成: {len(mol_samples)}/{num_samples_per_mol} 个有效样本")
    
    print(f"\n📊 总采样统计: {len(all_generated_samples)} 个样本")
    
    # 保存生成的分子到JSON文件
    if len(all_generated_samples) > 0:
        print("\n💾 保存生成的分子到JSON文件...")
        try:
            # 创建保存目录
            output_dir = params['training'].get('output_dir', 'default_output')
            save_dir = f"jobs/{output_dir}"
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成文件名（带时间戳）
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = f"{save_dir}/generated_mols_{timestamp}.json"
            
            # 转换为JSON格式并保存
            generated_samples_for_json = convert_for_json(all_generated_samples)
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(generated_samples_for_json, f, ensure_ascii=False, indent=4)
            
            print(f"✅ 生成的分子已保存到: {json_filename}")
            print(f"   共保存 {len(all_generated_samples)} 个样本")
        except Exception as e:
            print(f"⚠️  保存JSON文件失败: {e}")
    
    # 评估阶段
    print("\n" + "="*80)
    print("📈 开始评估生成的分子")
    print("="*80)
    
    # 使用ConfEval和ConditionalEvalPipeline评估
    preference_pairs = evaluate_and_build_pairs(
        all_generated_samples,
        all_reference_mols,
        molblocks_and_charges,
        params
    )
    
    model_pl.train()
    
    return preference_pairs


def evaluate_and_build_pairs(generated_samples, reference_mols, molblocks_and_charges, params):
    """
    评估生成的分子并构建偏好对
    """
    print(f"\n🔍 评估 {len(generated_samples)} 个生成样本...")
    
    # 按源分子分组
    from collections import defaultdict
    grouped_samples = defaultdict(list)
    
    for sample in generated_samples:
        source_idx = sample.get('source_mol_index', 0)
        grouped_samples[source_idx].append(sample)
    
    all_preference_pairs = []
    
    # 为每组分子评估
    for source_idx, samples in grouped_samples.items():
        print(f"\n📋 评估分子组 {source_idx}: {len(samples)} 个样本")
        
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
                
                eval_df = conf_eval.to_pandas()
                
                # 提取关键指标，处理可能的None值
                # 注意：ConfEval.to_pandas() 返回的是 Series，键名是属性名
                qed_val = eval_df.get('QED', None)
                logp_val = eval_df.get('logP', None)
                strain_val = eval_df.get('strain_energy', None)
                sa_val = eval_df.get('SA_score', None)
                
                # 检查关键指标是否为None或nan
                if qed_val is None or sa_val is None or (isinstance(qed_val, float) and np.isnan(qed_val)):
                    print(f"  🔬 样本 {i+1}: ✗ 指标无效 (QED={qed_val}, SA={sa_val})")
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
                    'is_valid': True,
                }
                # 简洁日志：只打印关键指标
                print(f"  🔬 样本 {i+1}: ✓ QED={qed_val:.3f}, SA={sa_val:.2f}, Strain={strain_val:.3f}" if strain_val else f"  🔬 样本 {i+1}: ✓ QED={qed_val:.3f}, SA={sa_val:.2f}")
                
            except Exception as e:
                print(f"  🔬 样本 {i+1}: ✗ ConfEval失败 ({type(e).__name__}: {str(e)[:50]})")
                # 给失败分子赋予极低分数
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
        
        # 2. 条件评估 (相似性评分) - 每个分子单独评估
        print("  🔬 开始ConditionalEval评估...")
        
        # 检查是否有参考分子
        if reference_mols is None or source_idx >= len(reference_mols):
            print("  ⚠️  无参考分子，使用默认相似性分数")
            for item in all_evaluated:  # 包括有效和无效样本
                item['cond_scores'] = {
                    'rmsd': 10.0,
                    'sims_surf': 0.0,
                    'sims_esp': 0.0,
                }
        else:
            # 有参考分子时才进行条件评估
            try:
                from shepherd_score.evaluations.evaluate import ConditionalEvalPipeline
                from shepherd_score.container import Molecule
                
                ref_mol = reference_mols[source_idx]
                
                # 创建参考分子对象
                ref_molec = Molecule(
                    ref_mol, 
                    num_surf_points=200, 
                    probe_radius=1.2,
                    partial_charges=None, 
                    pharm_multi_vector=False
                )
                
                # 对每个分子单独进行条件评估
                for i, item in enumerate(all_evaluated):
                    # 跳过无效分子的条件评估
                    if item['conf_scores'].get('is_valid', True) == False:
                        print(f"    ⏩ 分子 {i+1}: 无效分子，跳过条件评估 (设置默认值: RMSD=10.0, 表面相似性=0.0, 静电势相似性=0.0)")
                        item['cond_scores'] = {
                            'rmsd': 10.0,
                            'sims_surf': 0.0,
                            'sims_esp': 0.0,
                        }
                        continue
                        
                    # item['sample'] 是原始字典数据，需要使用已经存储的atoms和positions
                    atoms = item['atoms']
                    positions = item['positions']
                    
                    # 为单个分子创建列表
                    single_mol_list = [(atoms, positions)]
                    
                    try:
                        # 使用GPU加速
                        with torch.enable_grad():
                            cond_pipe = ConditionalEvalPipeline(
                                ref_molec,
                                generated_mols=single_mol_list,  # 只包含一个分子
                                condition='all',
                                num_surf_points=200,
                                pharm_multi_vector=False,
                                solvent=None
                            )
                            cond_pipe.evaluate(verbose=False)
                        
                        # 获取单个分子的评估结果
                        # ConditionalEvalPipeline.to_pandas()返回：
                        # - series_global: 全局属性（pd.Series）
                        # - df_rowwise: 每个分子的属性（pd.DataFrame）
                        series_global, df_rowwise = cond_pipe.to_pandas()
                        
                        # print(f"    📊 分子 {i+1} 条件评估结果:")
                        # print(f"    全局属性 (series_global):\n{series_global}")
                        # print(f"    逐行属性 (df_rowwise):\n{df_rowwise}")
                        
                        # 提取该分子的评估指标
                        # rmsds是每个分子的属性，在DataFrame中
                        if 'rmsds' in df_rowwise.columns and len(df_rowwise) > 0:
                            rmsd = float(df_rowwise['rmsds'].iloc[0])
                        else:
                            rmsd = 10.0
                            print(f"    ⚠️  评估结果中没有RMSD数据，使用默认值10.0")
                        
                        # 获取该分子的相似性分数，处理可能的NaN值
                        # 由于只有一个分子，upper_bound就是该分子的实际值
                        def safe_cond_float(val, default):
                            try:
                                f_val = float(val)
                                if np.isnan(f_val) or np.isinf(f_val):
                                    return default
                                return f_val
                            except:
                                return default
                        
                        # 相似性分数在global attributes中（因为只有一个分子，upper_bound就是实际值）
                        sims_surf = safe_cond_float(series_global.get('sims_surf_upper_bound', 0.0), 0.0)
                        sims_esp = safe_cond_float(series_global.get('sims_esp_upper_bound', 0.0), 0.0)
                        rmsd = safe_cond_float(rmsd, 10.0)
                        
                        item['cond_scores'] = {
                            'rmsd': rmsd,
                            'sims_surf': sims_surf,
                            'sims_esp': sims_esp,
                        }
                        
                        # 打印条件评估结果
                        print(f"    ✓ 分子 {i+1} 条件评估: RMSD={rmsd:.3f}, 表面相似性={sims_surf:.3f}, 静电势相似性={sims_esp:.3f}")
                        
                    except Exception as e:
                        print(f"    ⚠️ 分子 {i+1} 条件评估失败: {e}")
                        item['cond_scores'] = {
                            'rmsd': 10.0,
                            'sims_surf': 0.0,
                            'sims_esp': 0.0,
                        }
                
                print(f"  ✓ ConditionalEval完成 ({len(all_evaluated)} 个分子)")
                
            except Exception as e:
                print(f"  ⚠️  ConditionalEval初始化失败: {e}, 使用默认相似性分数")
                for item in all_evaluated:
                    item['cond_scores'] = {
                        'rmsd': 10.0,
                        'sims_surf': 0.0,
                        'sims_esp': 0.0,
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
                # 综合评分（更平衡的Shepherd Score）
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
                    
                    # 表面相似性奖励 (0-1分)
                    total_score += cond['sims_surf'] * 1.0
                    
                    # 静电势相似性奖励 (0-2分，权重更高因为更重要)
                    total_score += cond['sims_esp'] * 2.0
                    
                    # RMSD惩罚：构象偏差 (最多扣2.5分)
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
        print(f"  📊 组 {source_idx} 得分排名 (有效:{len(evaluated_samples)}, 无效:{len(invalid_samples)}, 总计:{len(all_evaluated)})")
        for rank, item in enumerate(all_evaluated[:5]):  # 只显示前5个
            is_valid = item['conf_scores'].get('is_valid', True)
            validity = "✓" if is_valid else "✗"
            marker = "🥇" if rank == 0 else ("🥈" if rank == 1 else "  ")
            if is_valid:
                print(f"      {marker} #{rank+1} {validity}: total={item['total_score']:.3f} (QED={item['conf_scores']['qed']:.3f}, SA={item['conf_scores']['sa_score']:.2f})")
            else:
                print(f"      {marker} #{rank+1} {validity}: total=-∞ (无效分子)")
        if len(all_evaluated) > 5:
            print(f"      ... 省略 {len(all_evaluated) - 5} 个样本")
        
        # 构建偏好对：winner和loser都必须是有效分子
        # 从有效分子中选择最高分和最低分
        valid_molecules = [item for item in all_evaluated if item['conf_scores'].get('is_valid', True)]
        
        if len(valid_molecules) < 2:
            print(f"  ❌ 组 {source_idx} 偏好对构建失败: 有效分子不足2个 ({len(valid_molecules)}个)")
            continue
            
        winner = valid_molecules[0]  # 有效分子中分数最高的
        loser = valid_molecules[-1]  # 有效分子中分数最低的
        
        score_gap = winner['total_score'] - loser['total_score']
        min_gap = params.get('dpo', {}).get('min_score_gap', 0.3)
        
        if score_gap >= min_gap:
            # 将样本转换为RDKit分子对象，供DPO Dataset使用
            winner_mol = create_rdkit_molecule(winner['sample'])
            loser_mol = create_rdkit_molecule(loser['sample'])
            
            if winner_mol is not None and loser_mol is not None:
                pair = (
                    winner_mol,  # 传递RDKit对象而不是字典
                    loser_mol,   # 传递RDKit对象而不是字典
                    {**winner['conf_scores'], **winner['cond_scores'], 'total_score': winner['total_score']},
                    {**loser['conf_scores'], **loser['cond_scores'], 'total_score': loser['total_score']},
                )
                all_preference_pairs.append(pair)
                print(f"  ✅ 组 {source_idx} 偏好对构建成功: Winner={winner['total_score']:.3f}, Loser={loser['total_score']:.3f}, Gap={score_gap:.3f}")
            else:
                print(f"  ❌ 组 {source_idx} 偏好对构建失败: 分子重构失败 (Winner有效={winner_mol is not None}, Loser有效={loser_mol is not None})")
        else:
            print(f"  ❌ 组 {source_idx} 偏好对构建失败: 分差不足 ({score_gap:.3f} < {min_gap})")
    
    print(f"\n{'='*50}")
    print(f"✅ 偏好对构建汇总: {len(all_preference_pairs)}/{len(grouped_samples)} 组成功")
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

    # 加载数据集
    print("\n📂 加载数据集...")
    molblocks_and_charges, output_file = load_dataset(params)
    
    # 计算边际分布
    print("\n📊 计算特征边际分布...")
    marginals = compute_and_cache_marginals(params, molblocks_and_charges)
    
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
        print("\n🔬 进行首次采样以生成初始偏好对...")
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # 创建临时模型用于首次采样
        print("   创建模型用于首次采样...")
        
        pretrained_path = params['training'].get('pretrained_checkpoint_path', None)
        if pretrained_path is not None:
            pretrained_ckpt_path = f"jobs/{pretrained_path}"
            if os.path.exists(pretrained_ckpt_path):
                print(f"   加载预训练权重: {pretrained_ckpt_path}")
                # 使用 load_from_checkpoint 加载模型
                try:
                    temp_model_pl = LightningModule.load_from_checkpoint(
                        pretrained_ckpt_path, 
                        params=params, 
                        strict=False
                    )
                    print("   ✅ 成功加载预训练权重 (load_from_checkpoint)")
                except Exception as e:
                    print(f"   ❌ 加载失败: {e}，尝试回退到原始初始化")
                    temp_model_pl = LightningModule(params)
            else:
                print(f"   ⚠️ 预训练权重不存在: {pretrained_ckpt_path}")
                temp_model_pl = LightningModule(params)
        else:
            temp_model_pl = LightningModule(params)
        
        temp_model_pl.to(device)
        temp_model_pl.model.device = device
        temp_model_pl.eval()
        
        num_samples = params.get('sampling', {}).get('num_samples_per_molecule', 4)
        
        # 【修改】循环采样直到生成有效的偏好对
        sample_attempt = 0
        while True:
            sample_attempt += 1
            if sample_attempt > 1:
                print(f"\n🔄 第 {sample_attempt} 次尝试采样...")
            
            with torch.no_grad():
                initial_pairs = sample_and_evaluate_molecules(
                    temp_model_pl,
                    params,
                    molblocks_and_charges,
                    dataset,
                    num_samples_per_mol=num_samples,
                    device=device
                )
            
            if len(initial_pairs) > 0:
                print(f"   ✅ 成功生成 {len(initial_pairs)} 个偏好对")
                break
            
            print("   ⚠️  本次采样未生成任何有效偏好对，继续尝试...")
        
        # 释放临时模型
        del temp_model_pl
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
        def __init__(self, params, dataset, dpo_dataset, molblocks_and_charges):
            super().__init__()
            self.params = params
            self.dataset = dataset
            self.dpo_dataset = dpo_dataset  # 直接引用DPO数据集
            self.molblocks_and_charges = molblocks_and_charges
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
            
            if trainer.global_rank == 0:  # 只在主进程执行
                self.epoch_counter += 1
                print(f"\n🔄 Epoch {trainer.current_epoch} 结束: 开始DPO采样 (第{self.epoch_counter}次)")
                
                num_samples = params.get('sampling', {}).get('num_samples_per_molecule', 4)
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                
                # 调用新的采样评估函数
                new_pairs = sample_and_evaluate_molecules(
                    pl_module,
                    self.params,
                    self.molblocks_and_charges,
                    self.dataset,
                    num_samples_per_mol=num_samples,
                    device=device
                )
                
                # 更新偏好对
                if len(new_pairs) > 0:
                    # 保留所有旧数据，累积偏好对
                    self.preference_pairs = self.preference_pairs + new_pairs
                    
                    # 直接更新DPO数据集的偏好对
                    # 下一个epoch开始时，reload_dataloaders_every_n_epochs=1
                    # 会重建DataLoader和DistributedSampler，使用更新后的数据集大小
                    self.dpo_dataset.preference_pairs = self.preference_pairs
                    print(f"✅ 更新DPO数据集: {len(self.preference_pairs)} 个偏好对 (将在下一epoch生效)")

    sampling_callback = DPOSamplingCallback(
        params=params,
        dataset=dataset,
        dpo_dataset=dpo_dataset,
        molblocks_and_charges=molblocks_and_charges
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
            pretrained_ckpt_path = f"jobs/{pretrained_path}"
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

