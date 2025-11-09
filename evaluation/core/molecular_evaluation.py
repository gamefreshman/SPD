#!/usr/bin/env python3
"""
分子生成与评估脚本
对3个天然产物分子各采样指定数量样本，并进行conf和cond评估
"""

import os
import json
import pickle
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict

import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# 导入必要的模块
from shepherd.lightning_module import LightningModule
from shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates
from shepherd.shepherd_score_utils.generate_point_cloud import (
    get_atomic_vdw_radii, 
    get_molecular_surface,
    get_electrostatics_given_point_charges,
)
from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores
from shepherd.inference import inference_sample
from shepherd.extract import create_rdkit_molecule

from shepherd_score.evaluations.evaluate import ConfEval, ConditionalEvalPipeline
from shepherd_score.container import Molecule

# 配置日志
log_file = Path(__file__).parent / 'evaluation.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局标志用于优雅终止
_stop_requested = False

def signal_handler(signum, frame):
    """信号处理函数"""
    global _stop_requested
    print(f"\n⚠️ 接收到中断信号 {signum}，正在尝试优雅终止...")
    print("💡 如果无响应，请在新终端中运行: kill -9 $(pgrep -f molecular_evaluation)")
    _stop_requested = True

# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class MolecularEvaluator:
    def __init__(self, config: Dict[str, Any]):
        """初始化分子评估器"""
        self.config = config
        
        # 检测可用的GPU设备 - 只使用GPU 1和2
        self.available_devices = []
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            # 只使用GPU 1和2，避开GPU 0
            target_gpus = [1, 2]  # 指定使用的GPU ID
            for gpu_id in target_gpus:
                if gpu_id < device_count:
                    self.available_devices.append(torch.device(f'cuda:{gpu_id}'))
            
            if self.available_devices:
                logger.info(f"使用指定GPU设备: {[str(d) for d in self.available_devices]}")
            else:
                # 如果指定的GPU不可用，回退到GPU 0
                self.available_devices = [torch.device('cuda:0')]
                logger.warning("指定的GPU不可用，回退到GPU 0")
        else:
            self.available_devices = [torch.device('cpu')]
            logger.info("未检测到GPU，使用CPU")
        
        self.primary_device = self.available_devices[0]
        
        # 创建输出目录
        self.output_dir = Path('data')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载模型和数据
        self.model_pl = None
        self.params = None
        self.molblocks_and_charges = None
        self.ref_molecules = {}
        
        # 边际分布
        self.atom_marginals_x1 = None
        self.bond_marginals_x1 = None
        self.pharm_marginals_x4 = None
        
        logger.info(f"初始化完成，主设备: {self.primary_device}")

    def convert_for_json(self, obj):
        """递归转换numpy数组和torch张量为Python列表"""
        if isinstance(obj, dict):
            return {k: self.convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.convert_for_json(elem) for elem in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj

    def save_raw_samples_to_spd(self, mol_index: int, samples: List[Dict[str, Any]]):
        """保存原始样本数据到data/SPD/目录"""
        if not samples:
            logger.warning(f"分子 {mol_index} 无样本数据，跳过保存")
            return
        
        # 创建SPD目录
        spd_dir = Path('data/SPD')
        spd_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取目标分子信息用于元数据
        target_mol = rdkit.Chem.MolFromMolBlock(
            self.molblocks_and_charges[mol_index][0], 
            removeHs=False
        )
        
        # 准备样本数据，包含详细元数据
        samples_data = {
            'metadata': {
                'molecule_index': mol_index,
                'natural_product_id': f'NP_{mol_index}',  # 自然产物编号
                'natural_product_name': f'Natural_Product_{mol_index}',  # 自然产物名称
                'timestamp': pd.Timestamp.now().isoformat(),
                'total_samples': len(samples),
                'target_molecule_info': {
                    'num_atoms': target_mol.GetNumAtoms(),
                    'num_bonds': target_mol.GetNumBonds(),
                    'molecular_formula': rdkit.Chem.rdMolDescriptors.CalcMolFormula(target_mol),
                    'molecular_weight': rdkit.Chem.rdMolDescriptors.CalcExactMolWt(target_mol),
                } if target_mol else None,
                'sampling_config': {
                    'n_atoms': self.config['sampling']['n_atoms'],
                    'batch_size': self.config['sampling']['batch_size'],
                    'enable_parallel': self.config['sampling']['enable_parallel'],
                    'samples_per_molecule': self.config['evaluation']['samples_per_molecule'],
                    'num_surf_points': self.config['evaluation']['num_surf_points']
                },
                'device_info': {
                    'primary_device': str(self.primary_device),
                    'available_devices': [str(d) for d in self.available_devices]
                }
            },
            'raw_samples': [
                {
                    **sample,  # 保留原始样本数据
                    'natural_product_info': {  # 添加天然产物信息
                        'natural_product_id': f'NP_{mol_index}',
                        'natural_product_name': f'Natural_Product_{mol_index}',
                        'target_molecule_index': mol_index
                    }
                } 
                for sample in samples
            ]
        }
        
        # 使用convert_for_json转换数据
        logger.info(f"🔄 转换分子 {mol_index} 的 {len(samples)} 个样本为JSON格式...")
        samples_data_json = self.convert_for_json(samples_data)
        
        # 保存到文件 - 使用更明确的天然产物标识
        output_file = spd_dir / f'NP_{mol_index}_Natural_Product_{mol_index}_raw_samples.json'
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(samples_data_json, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 天然产物 NP_{mol_index} 原始样本已保存到 {output_file}")
            logger.info(f"   📊 数据统计:")
            logger.info(f"     - 天然产物ID: NP_{mol_index}")
            logger.info(f"     - 样本数量: {len(samples)}")
            logger.info(f"     - 文件大小: {output_file.stat().st_size / (1024*1024):.2f} MB")
            if target_mol:
                logger.info(f"   🎯 天然产物分子信息:")
                logger.info(f"     - 原子数: {target_mol.GetNumAtoms()}")
                logger.info(f"     - 键数: {target_mol.GetNumBonds()}")
                logger.info(f"     - 分子式: {rdkit.Chem.rdMolDescriptors.CalcMolFormula(target_mol)}")
                logger.info(f"     - 分子量: {rdkit.Chem.rdMolDescriptors.CalcExactMolWt(target_mol):.4f}")
            logger.info(f"   📁 JSON结构: metadata + raw_samples")
            logger.info(f"   🔄 所有numpy/torch数据已转换为JSON兼容格式")
            
        except Exception as e:
            logger.error(f"❌ 保存分子 {mol_index} 原始样本失败: {str(e)}")

    def create_natural_product_mapping(self):
        """创建天然产物映射表并保存到SPD目录"""
        spd_dir = Path('data/SPD')
        spd_dir.mkdir(parents=True, exist_ok=True)
        
        mapping_data = {
            'metadata': {
                'title': '天然产物映射表',
                'description': '包含所有天然产物分子的索引和基本信息',
                'timestamp': pd.Timestamp.now().isoformat(),
                'total_natural_products': len(self.molblocks_and_charges)
            },
            'natural_products': []
        }
        
        for mol_index in range(len(self.molblocks_and_charges)):
            try:
                target_mol = rdkit.Chem.MolFromMolBlock(
                    self.molblocks_and_charges[mol_index][0], 
                    removeHs=False
                )
                
                if target_mol:
                    product_info = {
                        'molecule_index': mol_index,
                        'natural_product_id': f'NP_{mol_index}',
                        'natural_product_name': f'Natural_Product_{mol_index}',
                        'molecular_info': {
                            'num_atoms': target_mol.GetNumAtoms(),
                            'num_bonds': target_mol.GetNumBonds(),
                            'molecular_formula': rdkit.Chem.rdMolDescriptors.CalcMolFormula(target_mol),
                            'molecular_weight': rdkit.Chem.rdMolDescriptors.CalcExactMolWt(target_mol),
                            'heavy_atoms': target_mol.GetNumHeavyAtoms()
                        },
                        'sampling_files': {
                            'raw_samples_file': f'NP_{mol_index}_Natural_Product_{mol_index}_raw_samples.json',
                            'evaluation_results_file': f'molecule_{mol_index}_evaluation_results.json'
                        }
                    }
                else:
                    product_info = {
                        'molecule_index': mol_index,
                        'natural_product_id': f'NP_{mol_index}',
                        'natural_product_name': f'Natural_Product_{mol_index}',
                        'molecular_info': None,
                        'error': 'Failed to create RDKit molecule from molblock'
                    }
                
                mapping_data['natural_products'].append(product_info)
                
            except Exception as e:
                logger.warning(f"创建天然产物 {mol_index} 映射信息失败: {str(e)}")
        
        # 保存映射表
        mapping_file = spd_dir / 'natural_products_mapping.json'
        
        try:
            with open(mapping_file, 'w', encoding='utf-8') as f:
                json.dump(mapping_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 天然产物映射表已保存到 {mapping_file}")
            logger.info(f"   📋 包含 {len(mapping_data['natural_products'])} 个天然产物的详细信息")
            
        except Exception as e:
            logger.error(f"❌ 保存天然产物映射表失败: {str(e)}")

    def load_model_and_data(self):
        """加载模型和天然产物数据"""
        logger.info("加载模型...")
        
        # 加载模型
        chkpt_path = self.config['model']['checkpoint_path']
        self.model_pl = LightningModule.load_from_checkpoint(chkpt_path)
        self.params = self.model_pl.params
        self.model_pl.to(self.primary_device)
        self.model_pl.model.device = self.primary_device
        
        # 加载天然产物数据
        logger.info("加载天然产物数据...")
        with open(self.config['data']['molblocks_path'], 'rb') as f:
            self.molblocks_and_charges = pickle.load(f)
        
        logger.info(f"加载了 {len(self.molblocks_and_charges)} 个天然产物分子")
        
        # 创建参考分子对象
        self._create_reference_molecules()
        
        # 创建天然产物映射表
        logger.info("创建天然产物映射表...")
        self.create_natural_product_mapping()

    def _create_reference_molecules(self):
        """创建参考分子对象用于conditional评估"""
        logger.info("创建参考分子对象...")
        
        for mol_index in range(len(self.molblocks_and_charges)):
            try:
                mol = rdkit.Chem.MolFromMolBlock(
                    self.molblocks_and_charges[mol_index][0], 
                    removeHs=False
                )
                charges = np.array(self.molblocks_and_charges[mol_index][1])
                
                # 标准化分子坐标
                mol_coordinates = np.array(mol.GetConformer().GetPositions())
                mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis=0)
                mol = update_mol_coordinates(mol, mol_coordinates)
                
                # 创建Molecule对象
                ref_molec = Molecule(
                    mol=mol,
                    num_surf_points=self.config['evaluation']['num_surf_points'],
                    partial_charges=charges,
                    pharm_multi_vector=self.params['dataset']['x4']['multivectors'],
                    probe_radius=self.params['dataset']['probe_radius']
                )
                
                self.ref_molecules[mol_index] = ref_molec
                logger.info(f"成功创建参考分子 {mol_index}: {mol.GetNumAtoms()} 个原子")
                
            except Exception as e:
                logger.error(f"创建参考分子 {mol_index} 失败: {str(e)}")
                raise

    def compute_marginal_distributions(self):
        """计算边际分布"""
        logger.info("计算边际分布...")
        
        # 特征类型定义
        atom_types_x1 = [None, 'H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si']
        bond_types_x1 = [None, 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        max_node_types_x4 = 10
        
        # 初始化计数器
        atom_counts = torch.zeros(len(atom_types_x1), dtype=torch.float)
        bond_counts = torch.zeros(len(bond_types_x1), dtype=torch.float)
        pharm_counts = torch.zeros(max_node_types_x4, dtype=torch.float)
        
        def get_bond_type_str(bond):
            return str(bond.GetBondType())
        
        # 统计特征出现次数
        for mol_block, _ in tqdm(self.molblocks_and_charges, desc="计算边际分布"):
            mol = rdkit.Chem.MolFromMolBlock(mol_block, removeHs=False)
            if not mol:
                continue
            
            # 统计原子类型
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol in atom_types_x1:
                    atom_counts[atom_types_x1.index(symbol)] += 1
            
            # 统计键类型
            for bond in mol.GetBonds():
                bond_str = get_bond_type_str(bond)
                if bond_str in bond_types_x1:
                    bond_counts[bond_types_x1.index(bond_str)] += 1
            
            # 统计药效团类型
            try:
                pharm_types_temp, _, _ = get_pharmacophores(
                    mol, 
                    multi_vector=False,
                    check_access=False
                )
                for p_type in (pharm_types_temp + 1):
                    if p_type < max_node_types_x4:
                        pharm_counts[p_type] += 1
            except Exception:
                pass
        
        # 归一化为概率分布
        self.atom_marginals_x1 = (atom_counts / atom_counts.sum()) if atom_counts.sum() > 0 else torch.ones_like(atom_counts) / len(atom_counts)
        self.bond_marginals_x1 = (bond_counts / bond_counts.sum()) if bond_counts.sum() > 0 else torch.ones_like(bond_counts) / len(bond_counts)
        self.pharm_marginals_x4 = (pharm_counts / pharm_counts.sum()) if pharm_counts.sum() > 0 else torch.ones_like(pharm_counts) / len(pharm_counts)
        
        logger.info("边际分布计算完成")

    def compute_marginal_distributions_for_single_molecule(self, mol_index: int):
        """为单个目标分子计算其专属的边际分布"""
        logger.info(f"🧮 为分子 {mol_index} 计算专属边际分布...")
        
        # 特征类型定义
        atom_types_x1 = [None, 'H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si']
        bond_types_x1 = [None, 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
        max_node_types_x4 = 10
        
        # 初始化计数器
        atom_counts = torch.zeros(len(atom_types_x1), dtype=torch.float)
        bond_counts = torch.zeros(len(bond_types_x1), dtype=torch.float)
        pharm_counts = torch.zeros(max_node_types_x4, dtype=torch.float)
        
        def get_bond_type_str(bond):
            return str(bond.GetBondType())
        
        try:
            # 只对单个目标分子进行统计
            mol_block, _ = self.molblocks_and_charges[mol_index]
            mol = rdkit.Chem.MolFromMolBlock(mol_block, removeHs=False)
            
            if not mol:
                logger.warning(f"无法创建分子 {mol_index}，使用全局边际分布")
                return self.atom_marginals_x1, self.bond_marginals_x1, self.pharm_marginals_x4
            
            # 统计原子类型
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol in atom_types_x1:
                    atom_counts[atom_types_x1.index(symbol)] += 1
            
            # 统计键类型
            for bond in mol.GetBonds():
                bond_str = get_bond_type_str(bond)
                if bond_str in bond_types_x1:
                    bond_counts[bond_types_x1.index(bond_str)] += 1
            
            # 统计药效团类型
            try:
                pharm_types_temp, _, _ = get_pharmacophores(
                    mol, 
                    multi_vector=False,
                    check_access=False
                )
                for p_type in (pharm_types_temp + 1):
                    if p_type < max_node_types_x4:
                        pharm_counts[p_type] += 1
            except Exception:
                logger.warning(f"分子 {mol_index} 药效团计算失败")
            
            # 归一化为概率分布
            atom_marginals = (atom_counts / atom_counts.sum()) if atom_counts.sum() > 0 else torch.ones_like(atom_counts) / len(atom_counts)
            bond_marginals = (bond_counts / bond_counts.sum()) if bond_counts.sum() > 0 else torch.ones_like(bond_counts) / len(bond_counts)
            pharm_marginals = (pharm_counts / pharm_counts.sum()) if pharm_counts.sum() > 0 else torch.ones_like(pharm_counts) / len(pharm_counts)
            
            # 打印统计信息
            logger.info(f"✅ 分子 {mol_index} 边际分布统计:")
            logger.info(f"  - 原子类型数: {(atom_counts > 0).sum().item()}")
            logger.info(f"  - 键类型数: {(bond_counts > 0).sum().item()}")  
            logger.info(f"  - 药效团类型数: {(pharm_counts > 0).sum().item()}")
            
            # 显示主要原子类型及其比例
            main_atoms = []
            for i, count in enumerate(atom_counts):
                if count > 0 and i < len(atom_types_x1) and atom_types_x1[i] is not None:
                    ratio = atom_marginals[i].item()
                    main_atoms.append(f"{atom_types_x1[i]}:{count.item():.0f}({ratio:.2f})")
            logger.info(f"  - 主要原子分布: {', '.join(main_atoms[:5])}")
            
            # 显示键类型分布
            main_bonds = []
            for i, count in enumerate(bond_counts):
                if count > 0 and i < len(bond_types_x1) and bond_types_x1[i] is not None:
                    ratio = bond_marginals[i].item()
                    main_bonds.append(f"{bond_types_x1[i]}:{count.item():.0f}({ratio:.2f})")
            if main_bonds:
                logger.info(f"  - 主要键分布: {', '.join(main_bonds[:3])}")
            
            return atom_marginals, bond_marginals, pharm_marginals
            
        except Exception as e:
            logger.error(f"❌ 分子 {mol_index} 边际分布计算失败: {str(e)}")
            logger.info("回退到全局边际分布")
            return self.atom_marginals_x1, self.bond_marginals_x1, self.pharm_marginals_x4

    def sample_molecules_for_target(self, mol_index: int, num_samples: int) -> List[Dict[str, Any]]:
        """为指定目标分子采样生成分子"""
        logger.info(f"🔧 [单GPU模式] 开始为分子 {mol_index} 采样 {num_samples} 个样本...")
        logger.info(f"🔧 使用设备: {self.primary_device}")
        
        try:
            # 获取目标分子信息
            logger.info(f"🔧 加载目标分子 {mol_index} 信息...")
            mol = rdkit.Chem.MolFromMolBlock(
                self.molblocks_and_charges[mol_index][0], 
                removeHs=False
            )
            charges = np.array(self.molblocks_and_charges[mol_index][1])
            
            if mol is None:
                raise ValueError(f"无法从molblock创建分子 {mol_index}")
            
            logger.info(f"✅ 目标分子信息加载成功: {mol.GetNumAtoms()} 个原子")
        except Exception as e:
            logger.error(f"❌ 加载目标分子 {mol_index} 失败: {str(e)}")
            return []
        
        # 分子坐标标准化
        mol_coordinates = np.array(mol.GetConformer().GetPositions())
        mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis=0)
        mol = update_mol_coordinates(mol, mol_coordinates)
        
        # 条件特征提取
        centers = mol.GetConformer().GetPositions()
        radii = get_atomic_vdw_radii(mol)
        
        # 生成分子表面点云
        surface = get_molecular_surface(
            centers, 
            radii, 
            self.params['dataset']['x3']['num_points'],
            probe_radius=self.params['dataset']['probe_radius'],
            num_samples_per_atom=20,
        )
        
        # 提取药效团特征
        pharm_types, pharm_pos, pharm_direction = get_pharmacophores(
            mol,
            multi_vector=self.params['dataset']['x4']['multivectors'],
            check_access=self.params['dataset']['x4']['check_accessibility'],
        )
        
        # 计算表面静电势
        electrostatics = get_electrostatics_given_point_charges(
            charges, centers, surface,
        )
        
        # 采样参数
        n_atoms = self.config['sampling']['n_atoms']
        batch_size = self.config['sampling']['batch_size']
        num_pharmacophores = len(pharm_types)
        num_iterations = num_samples // batch_size
        
        logger.info(f"采样配置: {n_atoms}个原子, batch_size={batch_size}, {num_iterations}次迭代")
        
        # 🧮 为该分子计算专属边际分布
        mol_atom_marginals, mol_bond_marginals, mol_pharm_marginals = self.compute_marginal_distributions_for_single_molecule(mol_index)
        
        # 循环采样
        all_samples = []
        for iteration in range(num_iterations):
            logger.info(f"迭代 {iteration + 1}/{num_iterations}...")
            
            try:
                generated_samples = inference_sample(
                    self.model_pl,
                    batch_size=batch_size,
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
                    
                    # 边际分布 - 使用该分子专属的边际分布
                    atom_marginals=mol_atom_marginals,
                    bond_marginals=mol_bond_marginals,
                )
                
                # 添加分子索引信息
                for sample in generated_samples:
                    sample['source_mol_index'] = mol_index
                
                all_samples.extend(generated_samples)
                logger.info(f"完成 {len(generated_samples)} 个样本")
                
            except Exception as e:
                logger.error(f"迭代 {iteration + 1} 采样失败: {str(e)}")
                continue
        
        logger.info(f"分子 {mol_index} 采样完成: {len(all_samples)} 个样本")
        return all_samples

    def sample_molecules_parallel(self, mol_index: int, num_samples: int) -> List[Dict[str, Any]]:
        """并行采样分子（多GPU版本）"""
        print(f"🔧 DEBUG: 进入并行采样函数, mol_index={mol_index}, num_samples={num_samples}")
        logger.info(f"开始为分子 {mol_index} 并行采样 {num_samples} 个样本，使用 {len(self.available_devices)} 个GPU...")
        
        if _stop_requested:
            print("🔧 DEBUG: 收到停止信号，退出")
            logger.warning("接收到停止信号，终止采样")
            return []
        
        print(f"🔧 DEBUG: 开始获取目标分子信息...")
        
        # 获取目标分子信息
        mol = rdkit.Chem.MolFromMolBlock(
            self.molblocks_and_charges[mol_index][0], 
            removeHs=False
        )
        charges = np.array(self.molblocks_and_charges[mol_index][1])
        print(f"🔧 DEBUG: 目标分子加载完成, 原子数: {mol.GetNumAtoms() if mol else 'None'}")
        
        # 分子坐标标准化
        mol_coordinates = np.array(mol.GetConformer().GetPositions())
        mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis=0)
        mol = update_mol_coordinates(mol, mol_coordinates)
        print(f"🔧 DEBUG: 分子坐标标准化完成")
        
        # 条件特征提取
        centers = mol.GetConformer().GetPositions()
        radii = get_atomic_vdw_radii(mol)
        print(f"🔧 DEBUG: 开始生成分子表面...")
        
        # 生成分子表面点云
        surface = get_molecular_surface(
            centers, 
            radii, 
            self.params['dataset']['x3']['num_points'],
            probe_radius=self.params['dataset']['probe_radius'],
            num_samples_per_atom=20,
        )
        print(f"🔧 DEBUG: 分子表面生成完成, 点数: {len(surface)}")
        
        # 提取药效团特征
        pharm_types, pharm_pos, pharm_direction = get_pharmacophores(
            mol,
            multi_vector=self.params['dataset']['x4']['multivectors'],
            check_access=self.params['dataset']['x4']['check_accessibility'],
        )
        print(f"🔧 DEBUG: 药效团提取完成, 药效团数: {len(pharm_types)}")
        
        # 计算表面静电势
        electrostatics = get_electrostatics_given_point_charges(
            charges, centers, surface,
        )
        print(f"🔧 DEBUG: 静电势计算完成")
        
        # 采样参数
        n_atoms = self.config['sampling']['n_atoms']
        batch_size = self.config['sampling']['batch_size']
        num_pharmacophores = len(pharm_types)
        
        # 计算每个GPU的工作量
        num_gpus = len(self.available_devices)
        samples_per_gpu = num_samples // num_gpus
        remaining_samples = num_samples % num_gpus
        print(f"🔧 DEBUG: GPU工作量分配: {num_gpus}个GPU, 每GPU~{samples_per_gpu}样本")
        
        logger.info(f"并行采样配置: {num_gpus}个GPU, 每GPU~{samples_per_gpu}样本, batch_size={batch_size}")
        
        # 🧮 为该分子计算专属边际分布
        print(f"🔧 DEBUG: 开始计算分子 {mol_index} 的专属边际分布...")
        mol_atom_marginals, mol_bond_marginals, mol_pharm_marginals = self.compute_marginal_distributions_for_single_molecule(mol_index)
        print(f"🔧 DEBUG: 分子 {mol_index} 边际分布计算完成")
        
        # 创建采样任务列表
        sampling_tasks = []
        for i, device in enumerate(self.available_devices):
            gpu_samples = samples_per_gpu + (1 if i < remaining_samples else 0)
            if gpu_samples > 0:
                print(f"🔧 DEBUG: 创建GPU任务 {i}: 设备={device}, 样本数={gpu_samples}")
                sampling_tasks.append({
                    'device': device,
                    'gpu_id': i,
                    'num_samples': gpu_samples,
                    'mol_index': mol_index,
                    'n_atoms': n_atoms,
                    'batch_size': batch_size,
                    'num_pharmacophores': num_pharmacophores,
                    'surface': surface,
                    'electrostatics': electrostatics,
                    'pharm_types': pharm_types,
                    'pharm_pos': pharm_pos,
                    'pharm_direction': pharm_direction,
                    # 🧮 传递该分子专属的边际分布
                    'atom_marginals': mol_atom_marginals,
                    'bond_marginals': mol_bond_marginals,
                    'pharm_marginals': mol_pharm_marginals,
                })
        
        print(f"🔧 DEBUG: 创建了 {len(sampling_tasks)} 个采样任务")
        
        # 并行执行采样
        all_samples = []
        
        def sample_on_gpu(task):
            """在指定GPU上执行采样"""
            gpu_id = task['gpu_id']
            device = task['device']
            
            print(f"🔧 DEBUG: GPU函数被调用 - GPU {gpu_id}, 设备 {device}")
            logger.info(f"🔧 GPU {gpu_id} 开始初始化...")
            logger.info(f"  - 设备: {device}")
            logger.info(f"  - 目标样本数: {task['num_samples']}")
            logger.info(f"  - 批次大小: {task['batch_size']}")
            
            if _stop_requested:
                print(f"🔧 DEBUG: GPU {gpu_id} 收到停止信号")
                logger.warning(f"⚠️ GPU {gpu_id} 收到停止信号，退出")
                return []
                
            try:
                # 设置当前设备
                print(f"🔧 DEBUG: GPU {gpu_id} 开始设置CUDA设备...")
                logger.info(f"🔧 GPU {gpu_id} 设置CUDA设备...")
                torch.cuda.set_device(device)
                print(f"🔧 DEBUG: GPU {gpu_id} CUDA设备设置成功")
                
                # 创建该GPU上的模型副本
                logger.info(f"🔧 GPU {gpu_id} 加载模型检查点...")
                model_copy = LightningModule.load_from_checkpoint(self.config['model']['checkpoint_path'])
                
                logger.info(f"🔧 GPU {gpu_id} 将模型移至设备...")
                model_copy.to(device)
                model_copy.model.device = device
                
                logger.info(f"✅ GPU {gpu_id} 模型初始化完成")
                
                samples = []
                num_iterations = task['num_samples'] // task['batch_size']
                
                logger.info(f"GPU {task['gpu_id']} 开始采样 {task['num_samples']} 个样本...")
                
                for iteration in range(num_iterations):
                    if _stop_requested:
                        break
                        
                    try:
                        logger.info(f"🔧 GPU {gpu_id} 迭代 {iteration+1}/{num_iterations} 开始推理...")
                        logger.info(f"  - batch_size: {task['batch_size']}")
                        logger.info(f"  - N_x1 (atoms): {task['n_atoms']}")
                        logger.info(f"  - N_x4 (pharmacophores): {task['num_pharmacophores']}")
                        
                        generated_samples = inference_sample(
                            model_copy,
                            batch_size=task['batch_size'],
                            N_x1=task['n_atoms'],
                            N_x4=task['num_pharmacophores'],
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
                            surface=task['surface'],
                            electrostatics=task['electrostatics'],
                            pharm_types=task['pharm_types'],
                            pharm_pos=task['pharm_pos'],
                            pharm_direction=task['pharm_direction'],
                            
                            # 边际分布 - 使用该分子专属的边际分布
                            atom_marginals=task['atom_marginals'],
                            bond_marginals=task['bond_marginals'],
                        )
                        
                        logger.info(f"✅ GPU {gpu_id} 迭代 {iteration+1} 推理完成，生成 {len(generated_samples)} 个样本")
                        
                        # 添加分子索引信息
                        for sample in generated_samples:
                            sample['source_mol_index'] = task['mol_index']
                            sample['gpu_id'] = task['gpu_id']
                        
                        samples.extend(generated_samples)
                        logger.info(f"🔧 GPU {gpu_id} 累计样本数: {len(samples)}")
                        
                    except Exception as e:
                        logger.error(f"❌ GPU {gpu_id} 迭代 {iteration+1} 推理失败:")
                        logger.error(f"  - 异常类型: {type(e).__name__}")
                        logger.error(f"  - 异常信息: {str(e)}")
                        logger.error(f"  - 详细堆栈:", exc_info=True)
                        continue
                
                logger.info(f"GPU {task['gpu_id']} 完成采样: {len(samples)} 个样本")
                return samples
                
            except Exception as e:
                logger.error(f"❌ GPU {gpu_id} 采样任务完全失败:")
                logger.error(f"  - 异常类型: {type(e).__name__}")
                logger.error(f"  - 异常信息: {str(e)}")
                logger.error(f"  - 详细堆栈:", exc_info=True)
                return []
        
        # 使用ThreadPoolExecutor执行并行采样
        print(f"🔧 DEBUG: 准备启动ThreadPoolExecutor，任务数: {len(sampling_tasks)}")
        logger.info(f"🔧 启动 {len(sampling_tasks)} 个并行GPU任务...")
        
        try:
            print(f"🔧 DEBUG: 进入ThreadPoolExecutor...")
            with ThreadPoolExecutor(max_workers=len(sampling_tasks)) as executor:
                print(f"🔧 DEBUG: ThreadPoolExecutor创建成功")
                futures = []
                
                print(f"🔧 DEBUG: 开始提交任务...")
                for i, task in enumerate(sampling_tasks):
                    print(f"🔧 DEBUG: 提交任务 {i}")
                    future = executor.submit(sample_on_gpu, task)
                    futures.append(future)
                    print(f"🔧 DEBUG: 任务 {i} 提交成功")
                
                logger.info(f"🔧 所有GPU任务已提交到线程池")
                print(f"🔧 DEBUG: 所有 {len(futures)} 个任务已提交")
                
                # 收集结果
                print(f"🔧 DEBUG: 开始收集结果...")
                for i, future in enumerate(futures):
                    try:
                        print(f"🔧 DEBUG: 等待任务 {i} 完成...")
                        logger.info(f"🔧 等待GPU任务 {i} 完成...")
                        gpu_samples = future.result(timeout=3600)  # 1小时超时
                        print(f"🔧 DEBUG: 任务 {i} 完成，样本数: {len(gpu_samples)}")
                        logger.info(f"✅ GPU任务 {i} 完成，获得 {len(gpu_samples)} 个样本")
                        all_samples.extend(gpu_samples)
                    except Exception as e:
                        print(f"🔧 DEBUG: 任务 {i} 发生异常: {type(e).__name__}: {str(e)}")
                        logger.error(f"❌ GPU任务 {i} 异常:")
                        logger.error(f"  - 异常类型: {type(e).__name__}")  
                        logger.error(f"  - 异常信息: {str(e)}")
                        logger.error(f"  - 详细堆栈:", exc_info=True)
        except Exception as e:
            print(f"🔧 DEBUG: ThreadPoolExecutor本身出错: {type(e).__name__}: {str(e)}")
            logger.error(f"❌ ThreadPoolExecutor异常: {str(e)}", exc_info=True)
            return []
        
        logger.info(f"✅ 分子 {mol_index} 并行采样完成: 总共获得 {len(all_samples)} 个样本")
        return all_samples

    def evaluate_single_molecule(self, sample: Dict[str, Any], mol_index: int, sample_id: int) -> Dict[str, Any]:
        """评估单个分子"""
        result = {
            'sample_id': sample_id,
            'source_mol_index': mol_index,
            'rdkit_creation_success': False,
            'conf_evaluation_success': False,
            'conf_is_valid': False,
            'cond_evaluation_success': False,
            'conf_results': {},
            'cond_results': {},
            'error_messages': []
        }
        
        try:
            # 第1步: 尝试创建RDKit分子
            rdkit_mol = create_rdkit_molecule(sample)
            
            if rdkit_mol is None:
                result['error_messages'].append("RDKit分子创建失败")
                return result
            
            result['rdkit_creation_success'] = True
            
            # 提取原子和位置信息
            atoms = np.array([a.GetAtomicNum() for a in rdkit_mol.GetAtoms()])
            positions = rdkit_mol.GetConformer().GetPositions()
            
            # 第2步: ConfEval评估
            try:
                conf_eval = ConfEval(atoms, positions, solvent='water')
                result['conf_evaluation_success'] = True
                result['conf_is_valid'] = conf_eval.is_valid
                
                # 保存conf评估结果
                conf_pandas = conf_eval.to_pandas()
                result['conf_results'] = {
                    'is_valid': conf_eval.is_valid,
                    'is_valid_post_opt': conf_eval.is_valid_post_opt,
                    'is_graph_consistent': conf_eval.is_graph_consistent,
                    'strain_energy': float(conf_eval.strain_energy) if conf_eval.strain_energy is not None else None,
                    'rmsd': float(conf_eval.rmsd) if conf_eval.rmsd is not None else None,
                    'SA_score': float(conf_eval.SA_score) if conf_eval.SA_score is not None else None,
                    'QED': float(conf_eval.QED) if conf_eval.QED is not None else None,
                    'logP': float(conf_eval.logP) if conf_eval.logP is not None else None,
                    'fsp3': float(conf_eval.fsp3) if conf_eval.fsp3 is not None else None,
                    'smiles': conf_eval.smiles,
                }
                
            except Exception as e:
                result['error_messages'].append(f"ConfEval失败: {str(e)}")
                return result
            
            # 第3步: 如果conf评估分子有效，进行ConditionalEval评估
            if result['conf_is_valid']:
                try:
                    # 创建单分子列表用于ConditionalEvalPipeline
                    generated_mols = [(atoms, positions)]
                    ref_molec = self.ref_molecules[mol_index]
                    
                    cond_pipeline = ConditionalEvalPipeline(
                        ref_molec=ref_molec,
                        generated_mols=generated_mols,
                        condition='all',
                        num_surf_points=self.config['evaluation']['num_surf_points'],
                        pharm_multi_vector=self.params['dataset']['x4']['multivectors'],
                        solvent='water'
                    )
                    
                    # 执行评估
                    cond_pipeline.evaluate(verbose=False)
                    
                    # 获取结果 - 注意：to_pandas()返回(pd.Series, pd.DataFrame)
                    global_attrs, properties_df = cond_pipeline.to_pandas()
                    
                    result['cond_evaluation_success'] = True
                    
                    # 保存cond评估结果
                    if len(properties_df) > 0:
                        row = properties_df.iloc[0]  # 取第一行结果
                        
                        # 安全获取值的helper函数
                        def safe_get(obj, key, default=np.nan):
                            if hasattr(obj, 'get'):
                                return obj.get(key, default)
                            else:
                                return obj[key] if key in obj else default
                        
                        result['cond_results'] = {
                            'sim_surf_target': float(safe_get(row, 'sims_surf_target')),
                            'sim_esp_target': float(safe_get(row, 'sims_esp_target')),
                            'sim_pharm_target': float(safe_get(row, 'sims_pharm_target')),
                            'sim_surf_target_relax_optimal': float(safe_get(row, 'sims_surf_target_relax_optimal')),
                            'sim_esp_target_relax_optimal': float(safe_get(row, 'sims_esp_target_relax_optimal')),
                            'sim_pharm_target_relax_optimal': float(safe_get(row, 'sims_pharm_target_relax_optimal')),
                            'graph_similarities': float(safe_get(row, 'graph_similarities')),
                            'frac_valid': float(safe_get(global_attrs, 'frac_valid')),
                            'frac_valid_post_opt': float(safe_get(global_attrs, 'frac_valid_post_opt')),
                        }
                    
                except Exception as e:
                    result['error_messages'].append(f"ConditionalEval失败: {str(e)}")
            else:
                result['error_messages'].append("分子conf评估无效，跳过cond评估")
        
        except Exception as e:
            result['error_messages'].append(f"评估过程异常: {str(e)}")
        
        return result

    def evaluate_molecules_for_target(self, mol_index: int):
        """为指定目标分子进行完整的采样和评估流程"""
        logger.info(f"开始评估分子 {mol_index}...")
        
        # 根据配置和GPU数量决定是否使用并行采样
        use_parallel = (
            self.config.get('sampling', {}).get('enable_parallel', True) and 
            len(self.available_devices) > 1
        )
        
        logger.info(f"🔧 调试信息:")
        logger.info(f"  - enable_parallel配置: {self.config.get('sampling', {}).get('enable_parallel', 'not found')}")
        logger.info(f"  - 可用GPU数量: {len(self.available_devices)}")
        logger.info(f"  - 可用GPU列表: {[str(d) for d in self.available_devices]}")
        logger.info(f"  - 使用并行模式: {use_parallel}")
        logger.info(f"  - 目标采样数: {self.config['evaluation']['samples_per_molecule']}")
        
        try:
            if use_parallel:
                logger.info(f"🚀 启动并行采样模式: {len(self.available_devices)}个GPU")
                samples = self.sample_molecules_parallel(
                    mol_index, 
                    self.config['evaluation']['samples_per_molecule']
                )
            else:
                logger.info(f"🚀 启动单GPU采样模式: {self.primary_device}")
                samples = self.sample_molecules_for_target(
                    mol_index, 
                    self.config['evaluation']['samples_per_molecule']
                )
        except Exception as e:
            logger.error(f"❌ 采样过程中发生异常: {type(e).__name__}: {str(e)}")
            logger.error(f"❌ 异常详细信息:", exc_info=True)
            samples = []
        
        if not samples:
            logger.error(f"❌ 分子 {mol_index} 采样失败 - 无样本返回")
            return
        
        logger.info(f"✅ 分子 {mol_index} 采样成功，共获得 {len(samples)} 个样本")
        
        # 💾 保存原始样本数据到data/SPD/目录
        logger.info(f"💾 保存分子 {mol_index} 的原始样本数据...")
        self.save_raw_samples_to_spd(mol_index, samples)
        
        # 评估每个样本
        results = []
        success_counts = {
            'rdkit_success': 0,
            'conf_success': 0,
            'conf_valid': 0,
            'cond_success': 0
        }
        
        logger.info(f"开始评估 {len(samples)} 个样本...")
        for sample_id, sample in enumerate(tqdm(samples, desc=f"评估分子{mol_index}")):
            result = self.evaluate_single_molecule(sample, mol_index, sample_id)
            results.append(result)
            
            # 统计成功率
            if result['rdkit_creation_success']:
                success_counts['rdkit_success'] += 1
            if result['conf_evaluation_success']:
                success_counts['conf_success'] += 1
            if result['conf_is_valid']:
                success_counts['conf_valid'] += 1
            if result['cond_evaluation_success']:
                success_counts['cond_success'] += 1
        
        # 保存结果到JSON文件
        output_file = self.output_dir / f'molecule_{mol_index}_evaluation_results.json'
        
        # 准备输出数据
        output_data = {
            'molecule_index': mol_index,
            'total_samples': len(samples),
            'success_statistics': success_counts,
            'success_rates': {
                'rdkit_creation_rate': success_counts['rdkit_success'] / len(samples),
                'conf_evaluation_rate': success_counts['conf_success'] / len(samples),
                'conf_validity_rate': success_counts['conf_valid'] / len(samples),
                'cond_evaluation_rate': success_counts['cond_success'] / len(samples),
            },
            'evaluation_results': results
        }
        
        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_for_json(obj):
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif pd.isna(obj):
                return None
            return obj
        
        output_data = convert_for_json(output_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分子 {mol_index} 评估完成，结果已保存到 {output_file}")
        logger.info(f"成功率统计:")
        for key, rate in output_data['success_rates'].items():
            logger.info(f"  {key}: {rate:.2%}")

    def run_full_evaluation(self):
        """运行完整的评估流程"""
        logger.info("开始完整评估流程...")
        
        # 加载模型和数据
        self.load_model_and_data()
        
        # 计算边际分布
        self.compute_marginal_distributions()
        
        # 对每个天然产物分子进行评估
        for mol_index in range(len(self.molblocks_and_charges)):
            try:
                self.evaluate_molecules_for_target(mol_index)
            except Exception as e:
                logger.error(f"分子 {mol_index} 评估失败: {str(e)}")
                continue
        
        logger.info("全部评估完成!")

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
        return {
            'model': {
                'checkpoint_path': '/home1/zhh/workspace/SPD/evaluation/ckpt/last_27epoch.ckpt'
            },
            'data': {
                'molblocks_path': '/home1/zhh/workspace/SPD/data/conformers/np/molblock_charges_NPs.pkl'
            },
            'sampling': {
                'n_atoms': 70,
                'batch_size': 10,
            },
            'evaluation': {
                'samples_per_molecule': 20,
                'num_surf_points': 400,
            }
        }

def main():
    """主函数"""
    # 加载配置
    config = load_config()
    
    # 设置环境变量
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    # 创建评估器并运行
    evaluator = MolecularEvaluator(config)
    evaluator.run_full_evaluation()

if __name__ == "__main__":
    main()
