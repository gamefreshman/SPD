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
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
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

# ==================== 警告过滤 ====================
warnings.filterwarnings("ignore", category=UserWarning, message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*'repr' attribute.*")
warnings.filterwarnings("ignore", message=".*'frozen' attribute.*")

# ==================== 全局配置 ====================
SHARING_STRATEGY = "file_system"
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)
torch.set_float32_matmul_precision('medium')
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
    
    model_pl.eval()
    
    # 获取边际分布（从dataset）
    atom_marginals = dataset.x1_atom_diffuser.transition_model.marginals.to(device)
    bond_marginals = dataset.x1_bond_diffuser.transition_model.marginals.to(device)
    
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
                # 调用inference_sample
                generated_samples = inference_sample(
                    model_pl,
                    batch_size=1,
                    N_x1=n_atoms,
                    N_x4=num_pharmacophores,
                    unconditional=False,
                    
                    # 噪声控制
                    prior_noise_scale=1.0,
                    denoising_noise_scale=1.0,
                    inject_noise_at_ts=[],
                    inject_noise_scales=[],
                    
                    # 谐波化
                    harmonize=False,
                    harmonize_ts=[],
                    harmonize_jumps=[],
                    
                    # 条件修复
                    inpaint_x2_pos=False,
                    inpaint_x3_pos=False,
                    inpaint_x3_x=False,
                    inpaint_x4_pos=True,
                    inpaint_x4_direction=True,
                    inpaint_x4_type=True,
                    
                    # 修复控制
                    stop_inpainting_at_time_x2=0.0,
                    add_noise_to_inpainted_x2_pos=0.0,
                    stop_inpainting_at_time_x3=0.0,
                    add_noise_to_inpainted_x3_pos=0.0,
                    add_noise_to_inpainted_x3_x=0.0,
                    stop_inpainting_at_time_x4=0.0,
                    add_noise_to_inpainted_x4_pos=0.0,
                    add_noise_to_inpainted_x4_direction=0.0,
                    add_noise_to_inpainted_x4_type=0.0,
                    
                    # 条件输入
                    center_of_mass=np.zeros(3),
                    surface=surface,
                    electrostatics=electrostatics,
                    pharm_types=pharm_types,
                    pharm_pos=pharm_pos,
                    pharm_direction=pharm_direction,
                    
                    # 边际分布
                    atom_marginals=atom_marginals,
                    bond_marginals=bond_marginals,
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
                print(f"  🔬 样本 {i+1}: {len(atoms)} 个原子 (⚠️  {num_invalid} 个无效原子，原子序数范围: [{atoms.min():.0f}, {atoms.max():.0f}])")
                if num_valid == 0:
                    print(f"     ✗ 所有原子都无效，跳过此样本")
                    continue
            else:
                print(f"  🔬 样本 {i+1}: {len(atoms)} 个原子")
            
            # 检查键信息
            if bonds is not None:
                print(f"     ✓ 包含键信息: {len(bonds)} 条边")
            else:
                print(f"     ⚠️  缺少键信息")
            
            conf_scores = None
            
            # 尝试使用ConfEval评估（依赖xtb）
            try:
                conf_eval = ConfEval(atoms, positions, solvent='water', bonds=bonds)
                eval_df = conf_eval.to_pandas()
                
                # 提取关键指标
                conf_scores = {
                    'qed': float(eval_df['QEDs'].iloc[0]) if 'QEDs' in eval_df else 0.0,
                    'logp': float(eval_df['logPs'].iloc[0]) if 'logPs' in eval_df else 0.0,
                    'strain_energy': float(eval_df['strain_energies'].iloc[0]) if 'strain_energies' in eval_df else 0.0,
                    'sa_score': float(eval_df['SA_scores'].iloc[0]) if 'SA_scores' in eval_df else 5.0,
                }
                print(f"     ✓ ConfEval完成: QED={conf_scores['qed']:.3f}, LogP={conf_scores['logp']:.2f}")
                
            except Exception as e:
                print(f"     ⚠️  ConfEval失败，完整错误信息：")
                print(f"     错误类型: {type(e).__name__}")
                print(f"     错误消息: {str(e)}")
                print(f"     完整堆栈：")
                traceback.print_exc()
                print(f"     跳过此样本")
            
            if conf_scores is None:
                # 完全失败，跳过此样本
                print(f"     ✗ 所有评估方法都失败，跳过此样本")
                continue
            
            evaluated_samples.append({
                'sample': sample,
                'conf_scores': conf_scores,
                'atoms': atoms,
                'positions': positions,
            })
            
        if len(evaluated_samples) < 2:
            print(f"  ⚠️  有效样本不足2个，跳过该组")
            continue
        
        # 2. 使用ConditionalEvalPipeline计算相似性
        try:
            # 创建参考分子对象
            ref_mol = reference_mols[source_idx]
            ref_molec = Molecule(
                ref_mol,
                num_surf_points=200,
                probe_radius=1.2,
                pharm_multi_vector=False
            )
            
            # 准备生成分子列表
            generated_mols_list = [(item['atoms'], item['positions']) for item in evaluated_samples]
            
            # ConditionalEvalPipeline评估
            cond_pipe = ConditionalEvalPipeline(
                ref_molec,
                generated_mols=generated_mols_list,
                condition='all',
                num_surf_points=200,
                pharm_multi_vector=False,
                solvent=None
            )
            cond_pipe.evaluate(verbose=False)
            
            # 获取条件评估结果
            properties_series, global_attr = cond_pipe.to_pandas()
            
            print(f"  ✓ ConditionalEval完成")
            
            # 为每个样本添加条件评估分数
            for i, item in enumerate(evaluated_samples):
                # 从global_attr中提取RMSD等指标
                if hasattr(global_attr, 'iloc') and i < len(global_attr):
                    rmsd = float(global_attr['rmsds'].iloc[i]) if 'rmsds' in global_attr else 10.0
                else:
                    rmsd = 10.0
                
                # 从properties_series提取相似性指标
                sims_surf = float(properties_series.get('sims_surf_upper_bound', 0.0))
                sims_esp = float(properties_series.get('sims_esp_upper_bound', 0.0))
                
                item['cond_scores'] = {
                    'rmsd': rmsd,
                    'sims_surf': sims_surf,
                    'sims_esp': sims_esp,
                }
            
        except Exception as e:
            print(f"  ⚠️  ConditionalEval失败: {e}, 使用默认相似性分数")
            for item in evaluated_samples:
                item['cond_scores'] = {
                    'rmsd': 10.0,
                    'sims_surf': 0.0,
                    'sims_esp': 0.0,
                }
        
        # 3. 计算综合分数并构建偏好对
        for item in evaluated_samples:
            conf = item['conf_scores']
            cond = item['cond_scores']
            
            # 综合评分（类似Shepherd Score）
            total_score = conf['qed'] * 2.0
            total_score -= abs(conf['logp'] - 1.5) * 0.3
            total_score -= min(conf['strain_energy'], 10.0) * 0.5
            total_score -= conf['sa_score'] * 0.3
            total_score += cond['sims_surf'] * 1.0
            total_score += cond['sims_esp'] * 1.0
            total_score -= min(cond['rmsd'], 5.0) * 0.5
            
            item['total_score'] = total_score
        
        # 按分数排序
        evaluated_samples.sort(key=lambda x: x['total_score'], reverse=True)
        
        # 构建偏好对 (最高分 vs 最低分)
        winner = evaluated_samples[0]
        loser = evaluated_samples[-1]
        
        score_gap = winner['total_score'] - loser['total_score']
        min_gap = params.get('dpo', {}).get('min_score_gap', 0.3)
        
        if score_gap >= min_gap:
            pair = (
                winner['sample'],  # winner数据
                loser['sample'],   # loser数据
                {**winner['conf_scores'], **winner['cond_scores'], 'total_score': winner['total_score']},
                {**loser['conf_scores'], **loser['cond_scores'], 'total_score': loser['total_score']},
            )
            all_preference_pairs.append(pair)
            print(f"  ✅ 构建偏好对: Winner={winner['total_score']:.3f}, Loser={loser['total_score']:.3f}, Gap={score_gap:.3f}")
        else:
            print(f"  ⚠️  分差不足({score_gap:.3f} < {min_gap})，跳过")
    
    print(f"\n✅ 总共构建了 {len(all_preference_pairs)} 个偏好对")
    return all_preference_pairs


def handle_checkpoint_loading(params, output_dir, model_pl):
    """处理checkpoint加载（DPO模式）"""
    ckpt_path = f"{output_dir}/last.ckpt"
    pretrained_path = params['training'].get('pretrained_checkpoint_path', None)
    
    if pretrained_path is not None:
        # 从预训练模型加载权重
        pretrained_ckpt_path = f"jobs/{pretrained_path}"
        if os.path.exists(pretrained_ckpt_path):
            print(f"\n🔄 DPO微调：从预训练模型加载权重")
            print(f"   预训练checkpoint: {pretrained_ckpt_path}")
            
            try:
                checkpoint = torch.load(pretrained_ckpt_path, map_location='cpu')
                model_state_dict = checkpoint['state_dict']
                model_weights = {k: v for k, v in model_state_dict.items() if k.startswith('model.')}
                
                missing, unexpected = model_pl.model.load_state_dict(model_weights, strict=False)
                
                if hasattr(model_pl, 'ref_model'):
                    model_pl.ref_model.load_state_dict(model_weights, strict=False)
                    print(f"   ✅ 已同步权重到参考模型（ref_model）")
                
                print(f"   ✅ 成功加载预训练权重")
                
                # 检查是否继续之前的训练
                if os.path.exists(ckpt_path):
                    print(f"\n   发现当前任务的checkpoint: {ckpt_path}")
                    print(f"   将继续当前任务的训练")
                    return ckpt_path
                else:
                    print(f"\n   从预训练模型开始新的DPO微调")
                    return None
                    
            except Exception as e:
                print(f"   ❌ 加载预训练权重失败: {e}")
                return None
        else:
            print(f"\n⚠️  预训练checkpoint不存在: {pretrained_ckpt_path}")
    
    # 检查当前目录的checkpoint
    if os.path.exists(ckpt_path):
        print(f"\n✅ 发现当前checkpoint: {ckpt_path}")
        return ckpt_path
    
    print("\n📝 从头开始DPO训练")
    return None


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
    
    # 创建DPO数据集（初始为空） -- 这个数据集是否会修改sample出来的内容
    print("\n🎯 初始化DPO数据集...")
    dpo_dataset = DPODataset(
        preference_pairs=[],  # 初始为空，通过采样回调动态填充
        base_dataset=dataset,
        noise_schedule_dict=params['noise_schedules'],
        params=params,
    )
    print(f"✅ DPO数据集初始化完成（初始偏好对: {len(dpo_dataset.preference_pairs)}）")
    
    # 进行首次采样以生成初始偏好对
    print("\n🔬 进行首次采样以生成初始偏好对...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 创建临时模型用于首次采样
    print("   创建模型用于首次采样...")
    temp_model_pl = LightningModule(params)
    
    # 加载预训练权重（如果有）
    pretrained_path = params['training'].get('pretrained_checkpoint_path', None)
    if pretrained_path is not None:
        pretrained_ckpt_path = f"jobs/{pretrained_path}"
        if os.path.exists(pretrained_ckpt_path):
            print(f"   加载预训练权重: {pretrained_ckpt_path}")
            checkpoint = torch.load(pretrained_ckpt_path, map_location='cpu')
            model_state_dict = checkpoint['state_dict']
            model_weights = {k: v for k, v in model_state_dict.items() if k.startswith('model.')}
            temp_model_pl.model.load_state_dict(model_weights, strict=False)
    
    temp_model_pl.to(device)
    temp_model_pl.eval()
    
    num_samples = params.get('sampling', {}).get('num_samples_per_molecule', 4)
    
    with torch.no_grad():
        initial_pairs = sample_and_evaluate_molecules(
            temp_model_pl,
            params,
            molblocks_and_charges,
            dataset,
            num_samples_per_mol=num_samples,
            device=device
        )
    
    # 释放临时模型
    del temp_model_pl
    torch.cuda.empty_cache()
    
    if len(initial_pairs) == 0:
        print("   ⚠️  首次采样未生成任何偏好对，使用基础数据集进行训练")
        # 如果没有偏好对，至少添加一个虚拟对以允许DataLoader创建
        # 使用基础数据集的第一个样本作为虚拟偏好对
        if len(dataset) > 0:
            dummy_sample = dataset[0]
            # 创建虚拟评分（全部为0）
            dummy_scores = {
                'qed': 0.0, 'logp': 0.0, 'strain_energy': 0.0,
                'sa_score': 0.0, 'rmsd': 0.0, 'sims_surf': 0.0,
                'sims_esp': 0.0, 'total_score': 0.0
            }
            initial_pairs = [(dummy_sample, dummy_sample, dummy_scores, dummy_scores)]
            print(f"   ⚠️  使用虚拟偏好对初始化DataLoader")
    else:
        print(f"   ✅ 首次采样成功：{len(initial_pairs)} 个偏好对")
    
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
        def __init__(self, params, dataset, molblocks_and_charges):
            super().__init__()
            self.params = params
            self.dataset = dataset
            self.molblocks_and_charges = molblocks_and_charges
            self.preference_pairs = []
            self.epoch_counter = 0  # 用于追踪采样次数
            
        def on_train_epoch_start(self, trainer, pl_module):
            """每个epoch开始时进行采样和评估"""
            if trainer.current_epoch % params['training'].get('dpo_sampling_every_n_epochs', 1) != 0:
                return
            
            if trainer.global_rank == 0:  # 只在主进程执行
                self.epoch_counter += 1
                print(f"\n🔄 Epoch {trainer.current_epoch}: 开始DPO采样 (第{self.epoch_counter}次)")
                
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
                    # 保留50%旧数据
                    if len(self.preference_pairs) > 0:
                        keep_ratio = 0.5
                        n_keep = int(len(self.preference_pairs) * keep_ratio)
                        kept_pairs = self.preference_pairs[:n_keep]
                        self.preference_pairs = kept_pairs + new_pairs
                    else:
                        self.preference_pairs = new_pairs
                    
                    # 更新DPO数据集
                    if hasattr(trainer, 'train_dataloader'):
                        dataloader = trainer.train_dataloader
                        if hasattr(dataloader, 'dataset'):
                            dataloader.dataset.preference_pairs = self.preference_pairs
                            print(f"✅ 更新DPO数据集: {len(self.preference_pairs)} 个偏好对")
    
    sampling_callback = DPOSamplingCallback(
        params=params,
        dataset=dataset,
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
    
    wandb_logger = WandbLogger(
        name=f"DPO-{args.model_name}-seed_{args.seed}",
        entity="SPD_PaperParty",
        project="SPD_DPO_Training",
        save_dir=output_dir,
        log_model="all",
    )
    
    loggers = [csv_logger, wandb_logger]
    print("✅ 日志配置完成")
    
    # 设置训练器
    print("\n⚡ 配置PyTorch Lightning训练器...")
    cuda_available = torch.cuda.is_available()
    num_gpus_to_use = torch.cuda.device_count()
    
    print(f"   CUDA可用: {cuda_available}")
    print(f"   GPU数量: {num_gpus_to_use}")
    
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
        strategy=DDPStrategy(find_unused_parameters=True) 
            if (params['training']['num_gpus'] > 1 and cuda_available) else 'auto',
        precision=32,
        detect_anomaly=True,
    )
    print("✅ 训练器配置完成")
    
    # 创建模型
    print("\n🧠 创建模型...")
    model_pl = LightningModule(params)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model_pl.parameters() if p.requires_grad)
    print(f"✅ 模型创建完成")
    print(f"   可训练参数: {total_params:,}")
    
    # 设置wandb监控
    wandb_logger.watch(model_pl, log="all", log_freq=500)
    
    # 处理checkpoint加载
    print("\n💾 检查checkpoint...")
    ckpt_path = handle_checkpoint_loading(params, output_dir, model_pl)
    
    # 备份当前checkpoint
    if (ckpt_path is not None) and (trainer.global_rank == 0):
        date = datetime.datetime.now()
        timestamp = date.strftime("%Y_%m_%d_%H_%M")
        backup_path = f"{output_dir}/last_{timestamp}.ckpt"
        shutil.copyfile(ckpt_path, backup_path)
        print(f"📦 已备份checkpoint到: {backup_path}")
    
    # 开始训练
    print("\n" + "="*80)
    print("🚀 开始DPO训练...")
    print("="*80)
    print(f"📊 训练配置:")
    print(f"   - 数据集: {params['data']} ({len(molblocks_and_charges)} 个分子)")
    print(f"   - Batch size: {params['training']['batch_size']}")
    print(f"   - 学习率: {params['training']['lr']}")
    print(f"   - DPO beta: {params['training']['beta_dpo']}")
    print(f"   - DPO权重: 0.0 -> {params['training']['dpo_max_weight']} (预热{params['training']['dpo_ramp_up_epochs']}轮)")
    print(f"   - 采样比例: {params['training']['dpo_sampling_ratio']*100:.0f}%")
    print("="*80 + "\n")
    
    trainer.fit(model_pl, train_loader, ckpt_path=ckpt_path)
    
    print("\n" + "="*80)
    print("🎉 训练完成！")
    print("="*80)


if __name__ == '__main__':
    main()

