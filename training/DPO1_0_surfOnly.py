#!/usr/bin/env python3
"""
DPO训练脚本 —— 仅优化 Surface Similarity

基于 DPO1_0_partlyFrozen.py，简化评分逻辑：
- 偏好对的 winner/loser 完全由 sims_surf_target（表面形状相似度）决定
- 不考虑 ESP、药效团、SA Score、LogP 等其他指标
- 目标：最大化生成分子与参考分子的 3D 表面形状相似度
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
import threading
import queue
from functools import partial
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def apply_freeze_strategy(model_pl, freeze_encoder=True, freeze_hetero_last_n_layers=2):
    """
    应用部分冻结策略
    
    冻结策略:
    1. Encoder 完全冻结 (各模态的独立编码器)
    2. 异构图联合编码器 (Joint Heterogeneous Encoder): 只解冻最后 n 层，其他冻结
    3. 全局信息处理 (Joint Global Processing): 参与训练
    4. Decoder (去噪器): 完全训练
    
    Args:
        model_pl: LightningModule 实例
        freeze_encoder: 是否冻结 Encoder
        freeze_hetero_last_n_layers: 异构图编码器只解冻最后 n 层 (0 表示全部冻结)
    """
    model = model_pl.model
    
    frozen_params = 0
    trainable_params = 0
    
    # ==================== 1. 冻结 Encoder (各模态的独立编码器) ====================
    encoder_modules = [
        'x1_decoder_encoder',
        'x2_decoder_encoder', 
        'x3_decoder_encoder',
        'x4_decoder_encoder',
        # 同时冻结相关的 embedding 层
        'x1_decoder_encoder_embedding',
        'x1_decoder_encoder_bond_edge_embedding',
        'x2_decoder_encoder_embedding',
        'x3_decoder_encoder_embedding',
        'x3_decoder_scalar_expansion',
        'x4_decoder_encoder_embedding',
        'x4_decoder_encoder_embedding_l1',
    ]
    
    if freeze_encoder:
        print("\n🔒 冻结 Encoder 模块:")
        for module_name in encoder_modules:
            if hasattr(model, module_name):
                module = getattr(model, module_name)
                if module is not None:
                    count = 0
                    for param in module.parameters():
                        param.requires_grad = False
                        count += param.numel()
                    frozen_params += count
                    print(f"   ✓ {module_name}: {count:,} 参数已冻结")
    
    # ==================== 2. 异构图联合编码器: 只解冻最后 n 层 ====================
    if hasattr(model, 'decoder_joint_heterogeneous_graph_encoder') and \
       model.decoder_joint_heterogeneous_graph_encoder is not None:
        
        hetero_encoder = model.decoder_joint_heterogeneous_graph_encoder
        
        if hasattr(hetero_encoder, 'blocks') and freeze_hetero_last_n_layers > 0:
            num_blocks = len(hetero_encoder.blocks)
            unfreeze_start = max(0, num_blocks - freeze_hetero_last_n_layers)
            
            print(f"\n� 异构图联合编码器 (共 {num_blocks} 层, 只解冻最后 {freeze_hetero_last_n_layers} 层):")
            
            # 冻结前面的层
            for i in range(unfreeze_start):
                block = hetero_encoder.blocks[i]
                count = 0
                for param in block.parameters():
                    param.requires_grad = False
                    count += param.numel()
                frozen_params += count
                print(f"   ✓ blocks[{i}]: {count:,} 参数已冻结")
            
            # 解冻最后 n 层
            trainable_hetero = 0
            for i in range(unfreeze_start, num_blocks):
                block = hetero_encoder.blocks[i]
                count = 0
                for param in block.parameters():
                    param.requires_grad = True
                    count += param.numel()
                trainable_hetero += count
                print(f"   ○ blocks[{i}]: {count:,} 参数可训练")
            trainable_params += trainable_hetero
            
            # norm 层也解冻（与最后几层一起训练）
            if hasattr(hetero_encoder, 'norm') and hetero_encoder.norm is not None:
                count = 0
                for param in hetero_encoder.norm.parameters():
                    param.requires_grad = True
                    count += param.numel()
                trainable_params += count
                print(f"   ○ norm: {count:,} 参数可训练")
    
    # ==================== 3. 全局信息处理: 参与训练 (不冻结) ====================
    global_modules = [
        'x1_decoder_global_timestep_embedding',
        'x2_decoder_global_timestep_embedding',
        'x3_decoder_global_timestep_embedding',
        'x4_decoder_global_timestep_embedding',
        'x1_decoder_global_l1_embedding',
        'x2_decoder_global_l1_embedding',
        'x3_decoder_global_l1_embedding',
        'x4_decoder_global_l1_embedding',
        'x1_decoder_equiformer_tensor_product',
        'x2_decoder_equiformer_tensor_product',
        'x3_decoder_equiformer_tensor_product',
        'x4_decoder_equiformer_tensor_product',
        # 局部时间步嵌入
        'x1_decoder_local_timestep_embedding',
        'x2_decoder_local_timestep_embedding',
        'x3_decoder_local_timestep_embedding',
        'x4_decoder_local_timestep_embedding',
    ]
    
    print("\n🔓 全局信息处理模块 (参与训练):")
    for module_name in global_modules:
        if hasattr(model, module_name):
            module = getattr(model, module_name)
            if module is not None:
                count = 0
                for param in module.parameters():
                    # 确保可训练
                    param.requires_grad = True
                    count += param.numel()
                trainable_params += count
                print(f"   ○ {module_name}: {count:,} 参数可训练")
    
    # ==================== 4. Decoder (去噪器): 完全训练 ====================
    decoder_modules = [
        'x1_decoder_denoiser_MLP',
        'x1_decoder_denoiser_bond_MLP',
        'x1_decoder_denoiser_E3NN',
        'x1_decoder_denoiser_EGNN',
        'x1_decoder_denoiser_bond_distance_scalar_expansion',
        'x1_denoiser_SO3_grid',
    ]
    
    print("\n🔓 Decoder 模块 (完全训练):")
    for module_name in decoder_modules:
        if hasattr(model, module_name):
            module = getattr(model, module_name)
            if module is not None:
                count = 0
                for param in module.parameters():
                    # 确保可训练
                    param.requires_grad = True
                    count += param.numel()
                trainable_params += count
                print(f"   ○ {module_name}: {count:,} 参数可训练")
    
    # ==================== 统计信息 ====================
    # 重新计算总参数量
    total_frozen = sum(p.numel() for p in model_pl.parameters() if not p.requires_grad)
    total_trainable = sum(p.numel() for p in model_pl.parameters() if p.requires_grad)
    total_params = total_frozen + total_trainable
    
    print("\n" + "="*60)
    print("📊 冻结策略统计:")
    print(f"   总参数量:     {total_params:,}")
    print(f"   可训练参数:   {total_trainable:,} ({100*total_trainable/total_params:.1f}%)")
    print(f"   冻结参数:     {total_frozen:,} ({100*total_frozen/total_params:.1f}%)")
    print("="*60)
    
    return total_trainable, total_frozen


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
    
    # DPO训练使用num_workers=0避免多进程缓存问题
    # 当preference_pairs动态更新时，多进程worker不会同步更新
    train_loader = torch_geometric.loader.DataLoader(
        dataset=dpo_dataset,
        num_workers=0,  # 避免多进程缓存导致的索引越界
        batch_size=params['training']['batch_size'],
        shuffle=True,
    )
    
    return train_loader


def _prepare_molecule_condition(mol_index, mol_block, charges, params):
    """
    并行预处理单个分子的条件特征（CPU密集型操作）
    
    Returns:
        dict: 包含分子索引、参考分子和条件特征的字典
    """
    mol = rdkit.Chem.MolFromMolBlock(mol_block, removeHs=False)
    if mol is None:
        return None
    
    charges = np.array(charges)
    
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
    
    return {
        'mol_index': mol_index,
        'mol': mol,
        'surface': surface,
        'electrostatics': electrostatics,
        'pharm_types': pharm_types,
        'pharm_pos': pharm_pos,
        'pharm_direction': pharm_direction,
        'num_pharmacophores': len(pharm_types),
    }


def _sample_single_group(model_pl, params, condition, dataset, group_id, 
                         samples_per_group, device, result_queue):
    """
    单个GPU上采样一组分子（用于多GPU并行）
    
    Args:
        model_pl: 模型（已在指定GPU上）
        params: 参数配置
        condition: 条件数据字典
        dataset: 数据集（用于获取边际分布）
        group_id: 组ID
        samples_per_group: 每组样本数
        device: GPU设备
        result_queue: 结果队列
    """
    try:
        gpu_id = device.index if hasattr(device, 'index') else 0
        print(f"  [GPU {gpu_id}] 组 {group_id} 开始采样 {samples_per_group} 个样本...")
        
        with torch.no_grad():
            # 获取边际分布
            atom_marginals = dataset.x1_atom_diffuser.transition_model.marginals.to(device)
            bond_marginals = dataset.x1_bond_diffuser.transition_model.marginals.to(device)
            
            # 修正虚拟节点边际概率
            if len(atom_marginals) > 0:
                atom_marginals[0] = 0.0
                atom_marginals = atom_marginals / atom_marginals.sum()
            
            # n_atoms = params.get('sampling', {}).get('fixed_n_atoms', 78)
            n_atoms = 78 # 直接硬编码为当前优化的原子的原子序数
            
            # 准备inference参数
            inference_kwargs = {
                "batch_size": samples_per_group,
                "N_x1": n_atoms,
                "N_x4": condition['num_pharmacophores'],
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
                "surface": condition['surface'],
                "electrostatics": condition['electrostatics'],
                "pharm_types": condition['pharm_types'],
                "pharm_pos": condition['pharm_pos'],
                "pharm_direction": condition['pharm_direction'],
                "atom_marginals": atom_marginals,
                "bond_marginals": bond_marginals,
            }
            
            # 执行采样
            with torch.cuda.device(device):
                generated_samples = inference_sample(model_pl, **inference_kwargs)
            
            # 为每个样本标记组ID和分子索引
            for sample in generated_samples:
                sample['source_mol_index'] = condition['mol_index']
                sample['group_id'] = group_id
            
            # 清理中间张量
            del atom_marginals, bond_marginals
        
        # 释放GPU缓存
        torch.cuda.empty_cache()
        
        print(f"  [GPU {gpu_id}] ✅ 组 {group_id} 完成: {len(generated_samples)} 个样本")
        result_queue.put((group_id, generated_samples, condition['mol']))
        
    except Exception as e:
        gpu_id = device.index if hasattr(device, 'index') else 0
        print(f"  [GPU {gpu_id}] ❌ 组 {group_id} 采样失败: {e}")
        traceback.print_exc()
        result_queue.put((group_id, [], None))


def sample_and_evaluate_molecules(model_pl, params, molblocks_and_charges, dataset, 
                                   num_samples_per_mol=4, num_parallel_groups=4, device='cuda'):
    """
    多GPU多组并行采样和评估
    
    并行策略：
    1. 多线程并行提取条件特征（CPU密集型）
    2. 多GPU并行采样（对同一分子进行多组并行采样，分配到不同GPU）
    
    Args:
        num_samples_per_mol: 每组采样的样本数
        num_parallel_groups: 并行采样组数（对同一分子进行多组采样以充分利用多GPU）
    
    Returns:
        preference_pairs: List[(winner_data, loser_data, winner_scores, loser_scores)]
    """
    print("\n" + "="*80)
    print("🧬 开始多GPU多分子并行采样和评估")
    print("="*80)
    
    # 检测可用GPU数量
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"   可用GPU数量: {num_gpus}")
    
    # ========================================================================
    # 阶段1：多线程并行提取条件特征（CPU密集型）
    # ========================================================================
    print(f"\n📦 阶段1: 并行提取 {len(molblocks_and_charges)} 个分子的条件特征...")
    
    num_workers = min(8, len(molblocks_and_charges))
    prepared_conditions = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for mol_index, (mol_block, charges) in enumerate(molblocks_and_charges):
            future = executor.submit(
                _prepare_molecule_condition,
                mol_index, mol_block, charges, params
            )
            futures[future] = mol_index
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="提取条件特征"):
            result = future.result()
            if result is not None:
                prepared_conditions.append(result)
    
    # 按分子索引排序
    prepared_conditions.sort(key=lambda x: x['mol_index'])
    print(f"✅ 条件特征提取完成: {len(prepared_conditions)}/{len(molblocks_and_charges)} 个分子")
    
    # ========================================================================
    # 阶段2：多GPU并行采样（对同一分子进行多组并行采样）
    # ========================================================================
    # 当分子数量少于GPU数量时，将每个分子复制为多组以充分利用GPU
    original_num_mols = len(prepared_conditions)
    if original_num_mols < num_gpus and original_num_mols > 0:
        # 对每个分子复制多组，最多 num_parallel_groups 组
        expanded_conditions = []
        for cond in prepared_conditions:
            for group_idx in range(num_parallel_groups):
                cond_copy = deepcopy(cond)
                cond_copy['group_idx'] = group_idx  # 标记组索引
                expanded_conditions.append(cond_copy)
        prepared_conditions = expanded_conditions
        print(f"\n🔄 单分子多组并行模式: 将 {original_num_mols} 个分子扩展为 {len(prepared_conditions)} 组")
    
    num_groups = len(prepared_conditions)
    print(f"\n🚀 阶段2: 多GPU并行采样")
    print(f"   GPU数量: {num_gpus}, 采样组数: {num_groups}, 每组样本数: {num_samples_per_mol}")
    print(f"   总样本数: {num_groups * num_samples_per_mol}")
    
    # 在每个GPU上创建模型副本
    model_pl.eval()
    models_dict = {}
    
    for gpu_id in range(num_gpus):
        gpu_device = torch.device('cuda', gpu_id)
        if gpu_id == 0:
            # 主GPU使用原模型
            models_dict[gpu_id] = model_pl
        else:
            # 其他GPU复制模型
            model_copy = deepcopy(model_pl)
            model_copy.to(gpu_device)
            model_copy.model.device = gpu_device
            model_copy.eval()
            models_dict[gpu_id] = model_copy
    
    print(f"   ✅ 已在 {len(models_dict)} 个GPU上准备模型")
    
    # 创建结果队列
    result_queue = queue.Queue()
    all_generated_samples = []
    all_reference_mols = []
    
    # 分批并行处理（每批最多num_gpus个分子）
    for batch_start in range(0, num_groups, num_gpus):
        batch_end = min(batch_start + num_gpus, num_groups)
        batch_conditions = prepared_conditions[batch_start:batch_end]
        
        print(f"\n📋 并行采样批次 {batch_start//num_gpus + 1}: 分子 {batch_start}-{batch_end-1}")
        
        threads = []
        for i, cond in enumerate(batch_conditions):
            gpu_id = i % num_gpus
            gpu_device = torch.device('cuda', gpu_id)
            model = models_dict[gpu_id]
            group_id = batch_start + i
            
            t = threading.Thread(
                target=_sample_single_group,
                args=(model, params, cond, dataset, group_id,
                      num_samples_per_mol, gpu_device, result_queue)
            )
            threads.append(t)
        
        # 启动所有线程
        for t in threads:
            t.start()
        
        # 等待所有线程完成
        for t in threads:
            t.join()
    
    # 收集结果
    results_by_group = {}
    while not result_queue.empty():
        group_id, samples, ref_mol = result_queue.get()
        results_by_group[group_id] = (samples, ref_mol)
    
    # 按组ID排序合并
    for group_id in sorted(results_by_group.keys()):
        samples, ref_mol = results_by_group[group_id]
        all_generated_samples.extend(samples)
        if ref_mol is not None:
            all_reference_mols.append(ref_mol)
    
    # 释放非主GPU上的模型
    for gpu_id, model in models_dict.items():
        if gpu_id != 0:
            del model
    models_dict.clear()
    torch.cuda.empty_cache()
    
    print(f"\n📊 总采样统计: {len(all_generated_samples)} 个样本")
    
    # 保存生成的分子到JSON文件
    if len(all_generated_samples) > 0:
        print("\n💾 保存生成的分子到JSON文件...")
        try:
            # 创建保存目录（从 params 中提取 base_dir）
            base_dir = params['training'].get('base_dir', 'jobs')
            output_subdir = params['training'].get('output_dir', 'default_output')
            save_dir = os.path.join(base_dir, output_subdir)
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成文件名（带时间戳）
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            json_filename = os.path.join(save_dir, f"generated_mols_{timestamp}.json")
            
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
    评估生成的分子并构建偏好对 —— 仅基于 Surface Similarity
    
    评分规则：total_score = sims_surf_target
    不考虑 ESP、药效团、SA Score、LogP 等指标
    """
    print(f"\n🔍 [SurfOnly] 评估 {len(generated_samples)} 个生成样本 (仅Surface Similarity)...")
    
    # 按采样组ID分组（而不是按源分子分组），这样每个采样组都能独立构建偏好对
    from collections import defaultdict
    grouped_samples = defaultdict(list)
    
    for sample in generated_samples:
        # 优先使用group_id分组，这样对同一分子的多组采样可以各自构建偏好对
        group_id = sample.get('group_id', sample.get('source_mol_index', 0))
        grouped_samples[group_id].append(sample)
    
    all_preference_pairs = []
    all_valid_molecules_across_groups = []  # 收集所有组的有效分子，用于跨组匹配
    groups_with_pairs = set()  # 记录已成功构建偏好对的组
    
    # 为每组分子评估
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
                        'conf_scores': {'is_valid': False},
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
                        'conf_scores': {'is_valid': False},
                        'atoms': atoms,
                        'positions': positions,
                        'bonds': bonds,
                    })
                    continue
                
                # SurfOnly模式：只需确认分子有效，不提取SA/LogP详细指标
                conf_scores = {'is_valid': True}
                print(f"  🔬 样本 {i+1}: ✓ 分子有效")
                
            except Exception as e:
                print(f"  🔬 样本 {i+1}: ✗ ConfEval失败 ({type(e).__name__}: {str(e)[:50]})")
                invalid_samples.append({
                    'sample': sample,
                    'conf_scores': {'is_valid': False},
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
        # 注意：所有组都使用第一个参考分子（因为所有组都是对同一分子的采样）
        ref_mol_idx = 0  # 始终使用第一个参考分子
        if reference_mols is None or len(reference_mols) == 0:
            print("  ⚠️  无参考分子，使用默认相似性分数")
            for item in all_evaluated:
                item['cond_scores'] = {'sims_surf_target': 0.0}
        else:
            # 有参考分子时：只计算 Surface Similarity（跳过 ESP 和药效团）
            try:
                from shepherd_score.container import Molecule
                from shepherd_score.score.gaussian_overlap_np import get_overlap_np
                from shepherd_score.score.constants import ALPHA
                from shepherd_score.evaluations.utils.convert_data import get_mol_from_atom_pos
                
                ref_mol = reference_mols[ref_mol_idx]
                
                # 创建参考分子对象（只需要表面点，不需要 ESP 和药效团）
                ref_molec = Molecule(
                    ref_mol, 
                    num_surf_points=400,
                    probe_radius=1.2,
                    partial_charges=None,
                    pharm_multi_vector=False
                )
                
                num_surf_points = 400
                alpha = ALPHA(num_surf_points)
                
                for i, item in enumerate(all_evaluated):
                    if item['conf_scores'].get('is_valid', True) == False:
                        item['cond_scores'] = {'sims_surf_target': 0.0}
                        continue
                    
                    try:
                        atoms = item['atoms']
                        positions = item['positions']
                        bonds = item.get('bonds', None)
                        
                        gen_mol, charge, xyz_block = get_mol_from_atom_pos(atoms, positions, bonds=bonds)
                        
                        if gen_mol is None:
                            item['cond_scores'] = {'sims_surf_target': 0.0}
                            continue
                        
                        gen_molec = Molecule(
                            gen_mol,
                            num_surf_points=num_surf_points,
                            probe_radius=1.2,
                            partial_charges=None,  # 不需要计算 ESP
                            pharm_multi_vector=False
                        )
                        
                        # 只计算 Surface（shape）相似度
                        sims_surf_target = 0.0
                        if (gen_molec.surf_pos is not None and ref_molec.surf_pos is not None):
                            sims_surf_target = float(get_overlap_np(
                                gen_molec.surf_pos, ref_molec.surf_pos, alpha=alpha
                            ))
                        
                        if np.isnan(sims_surf_target): sims_surf_target = 0.0
                        
                        item['cond_scores'] = {'sims_surf_target': sims_surf_target}
                        print(f"    ✓ 分子 {i+1}: Surf={sims_surf_target:.4f}")
                        
                    except Exception as e:
                        print(f"    ⚠️ 分子 {i+1} 评估失败: {e}")
                        item['cond_scores'] = {'sims_surf_target': 0.0}
                
                print(f"  ✓ Surface Similarity 评估完成 ({len(all_evaluated)} 个分子)")
                
            except Exception as e:
                print(f"  ⚠️  评估初始化失败: {e}")
                for item in all_evaluated:
                    item['cond_scores'] = {'sims_surf_target': 0.0}
        
        # 3. 评分：total_score = sims_surf_target
        for item in all_evaluated:
            conf = item['conf_scores']
            cond = item['cond_scores']
            
            if conf.get('is_valid', True) == False:
                item['total_score'] = float('-inf')
            else:
                item['total_score'] = cond['sims_surf_target']
        
        # 按 surface similarity 排序
        all_evaluated.sort(key=lambda x: x['total_score'], reverse=True)
        
        valid_molecules = [item for item in all_evaluated if item['conf_scores'].get('is_valid', True)]
        
        print(f"  📊 组 {source_idx} Surface Similarity 排名 (有效:{len(valid_molecules)}, 无效:{len(invalid_samples)})")
        for rank, item in enumerate(valid_molecules[:5]):
            marker = "🥇" if rank == 0 else ("🥈" if rank == 1 else "  ")
            print(f"      {marker} #{rank+1}: Surf={item['total_score']:.4f}")
        if len(valid_molecules) > 5:
            print(f"      ... 省略 {len(valid_molecules) - 5} 个有效样本")
        
        # 收集本组的有效分子用于后续跨组匹配
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
                        {**winner['conf_scores'], **winner['cond_scores'], 'total_score': winner_score},
                        {**loser['conf_scores'], **loser['cond_scores'], 'total_score': loser_score},
                    )
                    all_preference_pairs.append(pair)
                    groups_with_pairs.add(source_idx)
                    print(f"  ✅ 组 {source_idx} 偏好对构建成功: Winner={winner_score:.3f}, Loser={loser_score:.3f}, Gap={score_gap:.3f}")
                else:
                    print(f"  ❌ 组 {source_idx} 偏好对构建失败: 分子重构失败")
            else:
                print(f"  ⚠️  组 {source_idx} 组内分差不足 ({score_gap:.3f} < {min_gap})，等待跨组匹配")
        else:
            print(f"  ⚠️  组 {source_idx} 有效分子不足 ({len(valid_molecules)}<2)，等待跨组匹配")
    
    # === 跨组匹配：如果单组构建失败，从所有有效分子中构建偏好对 ===
    if len(all_preference_pairs) < len(grouped_samples) and len(all_valid_molecules_across_groups) >= 2:
        print(f"\n🔄 跨组匹配: 从 {len(all_valid_molecules_across_groups)} 个有效分子中补充偏好对")
        
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
                        {**winner['conf_scores'], **winner['cond_scores'], 'total_score': winner_score},
                        {**loser['conf_scores'], **loser['cond_scores'], 'total_score': loser_score},
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
    print("🚀 SPD DPO训练启动器 (SurfOnly - 仅优化Surface Similarity)")
    print("="*80)
    print(f"📋 配置: {args.model_name}")
    print(f"🎲 随机种子: {args.seed}")
    print("="*80)
    
    # 加载参数
    params = importlib.import_module(f'parameters.{args.model_name}').params
    
    # ==================== 提取目录配置 ====================
    # 基础目录（checkpoint、采样文件、日志等的根目录）
    base_dir = params['training'].get('base_dir', 'jobs/33')

    # base_dir = params['training'].get('base_dir', 'jobs/40')

    # 输出子目录
    output_subdir = params['training'].get('output_dir', 'default_output')
    # 完整输出目录
    output_dir = os.path.join(base_dir, output_subdir)
    
    print(f"\n📁 目录配置:")
    print(f"   基础目录: {base_dir}")
    print(f"   输出目录: {output_dir}")
    
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
    
    # 只使用第一个NPS分子进行训练和DPO微调
    molblocks_and_charges = molblocks_and_charges[:1]
    print(f"🎯 只使用第一个分子进行DPO微调: {len(molblocks_and_charges)} 个分子")
    
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
            # 如果是绝对路径则直接使用，否则加上 base_dir 前缀
            if os.path.isabs(pretrained_path):
                pretrained_ckpt_path = pretrained_path
            else:
                pretrained_ckpt_path = os.path.join(base_dir, pretrained_path)
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
    
    # 更新DPO数据集（使用安全方法重置索引缓存）
    dpo_dataset.update_preference_pairs(initial_pairs)
    print(f"✅ DPO数据集更新完成（偏好对: {len(dpo_dataset.preference_pairs)}）")
    
    # 创建DataLoader
    print("\n📦 创建DataLoader...")
    train_loader = create_dpo_dataloader(params, dataset, dpo_dataset)
    
    # 设置输出目录（使用前面提取的 base_dir 和 output_dir）
    os.makedirs(base_dir, exist_ok=True)
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
            self.pairs_history = []  # 存储每轮的偏好对列表
            self.max_rounds = 2  # 只保留最近两轮
            self.epoch_counter = 0  # 用于追踪采样次数
            self.round_metrics = []  # 每轮指标记录
            self.metrics_file = os.path.join(output_dir, 'dpo_round_metrics.json')  # 指标保存路径
        
        def _collect_and_save_metrics(self, pairs, epoch, train_loss=None, extra_metrics=None):
            """收集偏好对指标并保存到 JSON 文件
            
            Args:
                pairs: 偏好对列表
                epoch: 当前 epoch
                train_loss: 训练总损失
                extra_metrics: 额外的训练指标字典 (loss_dpo, implicit_acc 等)
            """
            if len(pairs) == 0:
                return
            
            # 收集所有 winner 和 loser 的指标
            metric_keys = ['sims_surf_target', 'total_score']
            
            winner_metrics = {k: [] for k in metric_keys}
            loser_metrics = {k: [] for k in metric_keys}
            
            for pair in pairs:
                # pair = (winner_mol, loser_mol, winner_scores_dict, loser_scores_dict)
                w_scores = pair[2]
                l_scores = pair[3]
                for k in metric_keys:
                    if k in w_scores:
                        val = w_scores[k]
                        if val is not None and val != float('-inf') and val != float('inf'):
                            winner_metrics[k].append(float(val))
                    if k in l_scores:
                        val = l_scores[k]
                        if val is not None and val != float('-inf') and val != float('inf'):
                            loser_metrics[k].append(float(val))
            
            # 计算平均值
            def safe_mean(vals):
                return sum(vals) / len(vals) if len(vals) > 0 else 0.0
            
            winner_avg = {k: safe_mean(winner_metrics[k]) for k in metric_keys}
            loser_avg = {k: safe_mean(loser_metrics[k]) for k in metric_keys}
            
            round_data = {
                'round': self.epoch_counter,
                'epoch': epoch,
                'num_pairs': len(pairs),
                'winner': winner_avg,
                'loser': loser_avg,
                'score_gap': winner_avg['total_score'] - loser_avg['total_score'],
                'train_loss': train_loss,
                'training_metrics': extra_metrics if extra_metrics else {},
            }
            
            # 同时记录每对的详细数据
            round_data['pairs_detail'] = []
            for pair in pairs:
                w_scores = pair[2]
                l_scores = pair[3]
                detail = {
                    'winner': {k: float(w_scores.get(k, 0.0)) for k in metric_keys if k in w_scores},
                    'loser': {k: float(l_scores.get(k, 0.0)) for k in metric_keys if k in l_scores},
                }
                round_data['pairs_detail'].append(detail)
            
            self.round_metrics.append(round_data)
            
            # 保存到 JSON
            try:
                with open(self.metrics_file, 'w', encoding='utf-8') as f:
                    json.dump(self.round_metrics, f, ensure_ascii=False, indent=2)
                print(f"📊 指标已保存到: {self.metrics_file}")
            except Exception as e:
                print(f"⚠️  保存指标文件失败: {e}")
            
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
                    # 添加新一轮的偏好对
                    self.pairs_history.append(new_pairs)
                    
                    # 只保留最近两轮
                    if len(self.pairs_history) > self.max_rounds:
                        self.pairs_history = self.pairs_history[-self.max_rounds:]
                    
                    # 合并所有保留轮次的偏好对
                    all_pairs = []
                    for round_pairs in self.pairs_history:
                        all_pairs.extend(round_pairs)
                    
                    # 直接更新DPO数据集的偏好对
                    # 下一个epoch开始时，reload_dataloaders_every_n_epochs=1
                    # 会重建DataLoader和DistributedSampler，使用更新后的数据集大小
                    self.dpo_dataset.update_preference_pairs(all_pairs)
                    print(f"✅ 更新DPO数据集: {len(all_pairs)} 个偏好对 (保留{len(self.pairs_history)}轮, 将在下一epoch生效)")
                    
                    # 收集训练损失和 DPO 相关指标
                    train_loss_dict = {}
                    # 遍历所有 callback_metrics，收集所有可用的训练指标
                    for key in ['train_loss', 'loss_dpo', 'loss_std_on_winner', 
                                'implicit_acc', 'dpo_weight', 'model_loss_diff', 'ref_loss_diff']:
                        if key in trainer.callback_metrics:
                            try:
                                val = trainer.callback_metrics[key]
                                train_loss_dict[key] = float(val.item() if hasattr(val, 'item') else val)
                            except Exception:
                                pass
                    
                    train_loss = train_loss_dict.get('train_loss', None)
                    self._collect_and_save_metrics(new_pairs, trainer.current_epoch, train_loss, train_loss_dict)

    sampling_callback = DPOSamplingCallback(
        params=params,
        dataset=dataset,
        dpo_dataset=dpo_dataset,
        molblocks_and_charges=molblocks_and_charges
    )
    
    # 记录初始偏好对的指标（round 0，训练前的采样）
    if len(initial_pairs) > 0 and not is_ddp_subprocess:
        sampling_callback._collect_and_save_metrics(initial_pairs, epoch=0, train_loss=None)
    
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
    
    # DPO微调使用单GPU模式，避免DDP通信问题
    # 多GPU采样在sample_and_evaluate_molecules中单独处理
    strategy = 'auto'
    dpo_devices = 1 if cuda_available else "auto"
    print(f"   DPO训练模式: 单GPU (devices={dpo_devices})")
        
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=output_dir,
        accelerator="gpu" if cuda_available else 'cpu',
        max_epochs=10000,
        gradient_clip_val=params['training']['gradient_clip_val'],
        accumulate_grad_batches=params['training']['accumulate_grad_batches'],
        log_every_n_steps=params['training']['log_every_n_steps'],
        reload_dataloaders_every_n_epochs=1,  # 重要：每个epoch重载以获取新的偏好对
        devices=dpo_devices,
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
            # 如果是绝对路径则直接使用，否则加上 base_dir 前缀
            if os.path.isabs(pretrained_path):
                pretrained_ckpt_path = pretrained_path
            else:
                pretrained_ckpt_path = os.path.join(base_dir, pretrained_path)
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

    # ==================== 应用部分冻结策略 ====================
    print("\n" + "="*80)
    print("🔧 应用部分冻结训练策略")
    print("="*80)
    
    # 冻结配置参数（可在 params 中配置）
    freeze_encoder = params.get('freeze_strategy', {}).get('freeze_encoder', True)
    freeze_hetero_last_n_layers = params.get('freeze_strategy', {}).get('freeze_hetero_last_n_layers', 2)
    
    trainable_params, frozen_params = apply_freeze_strategy(
        model_pl, 
        freeze_encoder=freeze_encoder,
        freeze_hetero_last_n_layers=freeze_hetero_last_n_layers
    )
    
    print(f"\n✅ 模型创建完成")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   冻结参数:   {frozen_params:,}")
    
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
            dpo_dataset.update_preference_pairs(dpo_dataset.preference_pairs * 2)
        print(f"   已自动复制样本至 {len(dpo_dataset.preference_pairs)} 个偏好对")
    
    # 启动训练
    trainer.fit(model_pl, train_dataloaders=train_loader, ckpt_path=resume_ckpt_path)
    
    print("\n" + "="*80)
    print("🎉 训练完成！")
    print("="*80)


if __name__ == '__main__':
    main()

