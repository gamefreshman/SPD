"""
DPO数据集：管理偏好对数据
支持共享噪声和时间步
真实实现：完全复用HeteroDataset的数据生成逻辑
"""

import random

import torch
import numpy as np
import rdkit.Chem as Chem
from torch_geometric.data import HeteroData, Dataset
from typing import List, Tuple, Optional
from copy import deepcopy

from shepherd.new_datasets import HeteroDataset


class DPODataset(Dataset):
    """
    DPO偏好对数据集
    每个样本包含：winner分子数据 + loser分子数据 + 共享的噪声和时间步
    
    真实实现：使用HeteroDataset的完整逻辑生成数据
    """
    
    def __init__(
        self,
        preference_pairs: List[Tuple],
        base_dataset,
        noise_schedule_dict,
        params,
    ):
        """
        Args:
            preference_pairs: List[(winner_mol, loser_mol, score_w, score_l)]
            base_dataset: 基础数据集，用于复用数据生成逻辑
            noise_schedule_dict: 噪声调度参数
            params: 全局参数
        """
        super().__init__()
        
        self.preference_pairs = preference_pairs
        self.base_dataset = base_dataset
        self.noise_schedule_dict = noise_schedule_dict
        self.params = params
        
        # 创建数据生成器（复用HeteroDataset的逻辑）
        self.data_generator = self._create_data_generator()
    
    def _create_data_generator(self):
        """
        创建数据生成器，复用HeteroDataset的所有逻辑
        
        Returns:
            HeteroDataset实例（用于生成数据）
        """
        # 创建一个临时的molblocks_and_charges（用于初始化）
        temp_molblocks_and_charges = [("dummy", None)]
        
        # 从base_dataset直接复制参数（更安全的方式）
        # 直接复用已初始化的base_dataset的所有配置
        if hasattr(self.base_dataset, 'x1'):
            # base_dataset已经是HeteroDataset实例，直接使用它
            # 我们只需要它的方法，不需要创建新实例
            return self.base_dataset
        else:
            # 后备方案：从params构建参数字典
            dataset_params = self.params.get('dataset', {})
            
            # 提取HeteroDataset需要的扁平化参数
            init_params = {
                'molblocks_and_charges': temp_molblocks_and_charges,
                'noise_schedule_dict': self.noise_schedule_dict,
                'atom_marginals_x1': self.base_dataset.x1_atom_diffuser.transition_model.x_marginals if hasattr(self.base_dataset, 'x1_atom_diffuser') else None,
                'bond_marginals_x1': self.base_dataset.x1_bond_diffuser.transition_model.x_marginals if hasattr(self.base_dataset, 'x1_bond_diffuser') else None,
                'pharm_marginals_x4': self.base_dataset.x4_pharm_diffuser.transition_model.x_marginals if hasattr(self.base_dataset, 'x4_pharm_diffuser') else None,
            }
            
            # 添加x1, x2, x3, x4配置
            if 'x1' in dataset_params:
                x1_params = dataset_params['x1']
                init_params.update({
                    'recenter_x1': x1_params.get('recenter', True),
                    'add_virtual_node_x1': x1_params.get('add_virtual_node', True),
                    'remove_noise_COM_x1': x1_params.get('remove_noise_COM', True),
                    'atom_types_x1': x1_params.get('atom_types'),
                    'charge_types_x1': x1_params.get('charge_types'),
                    'bond_types_x1': x1_params.get('bond_types'),
                    'scale_atom_features_x1': x1_params.get('scale_atom_features', 1.0),
                    'scale_bond_features_x1': x1_params.get('scale_bond_features', 1.0),
                })
            
            if 'x2' in dataset_params:
                x2_params = dataset_params['x2']
                init_params.update({
                    'recenter_x2': x2_params.get('recenter', False),
                    'add_virtual_node_x2': x2_params.get('add_virtual_node', True),
                    'remove_noise_COM_x2': x2_params.get('remove_noise_COM', False),
                    'num_points_x2': x2_params.get('num_points', 75),
                    'independent_timesteps_x2': x2_params.get('independent_timesteps', False),
                })
            
            if 'x3' in dataset_params:
                x3_params = dataset_params['x3']
                init_params.update({
                    'recenter_x3': x3_params.get('recenter', False),
                    'add_virtual_node_x3': x3_params.get('add_virtual_node', True),
                    'remove_noise_COM_x3': x3_params.get('remove_noise_COM', False),
                    'num_points_x3': x3_params.get('num_points', 75),
                    'independent_timesteps_x3': x3_params.get('independent_timesteps', False),
                    'scale_node_features_x3': x3_params.get('scale_node_features', 1.0),
                })
            
            if 'x4' in dataset_params:
                x4_params = dataset_params['x4']
                init_params.update({
                    'recenter_x4': x4_params.get('recenter', False),
                    'add_virtual_node_x4': x4_params.get('add_virtual_node', True),
                    'remove_noise_COM_x4': x4_params.get('remove_noise_COM', False),
                    'max_node_types_x4': x4_params.get('max_node_types', 16),
                    'scale_node_features_x4': x4_params.get('scale_node_features', 1.0),
                    'scale_vector_features_x4': x4_params.get('scale_vector_features', 1.0),
                    'independent_timesteps_x4': x4_params.get('independent_timesteps', False),
                    'multivectors': x4_params.get('multivectors', False),
                    'check_accessibility': x4_params.get('check_accessibility', False),
                })
            
            # 其他通用参数
            init_params.update({
                'explicit_hydrogens': dataset_params.get('explicit_hydrogens', True),
                'use_MMFF94_charges': dataset_params.get('use_MMFF94_charges', False),
                'probe_radius': dataset_params.get('probe_radius', 0.6),
                'x1': dataset_params.get('compute_x1', True),
                'x2': dataset_params.get('compute_x2', False),
                'x3': dataset_params.get('compute_x3', False),
                'x4': dataset_params.get('compute_x4', False),
            })
            
            # 过滤掉None值
            init_params = {k: v for k, v in init_params.items() if v is not None}
            
            generator = HeteroDataset(**init_params)
            return generator
    
    def len(self):
        return len(self.preference_pairs)
    
    def update_preference_pairs(self, new_pairs):
        """
        安全更新偏好对，同时重置torch_geometric的索引缓存
        
        Args:
            new_pairs: 新的偏好对列表
        """
        self.preference_pairs = new_pairs
        # 重置torch_geometric Dataset的索引缓存
        # 这是关键：防止DataLoader使用旧的索引范围导致IndexError
        self._indices = None
    
    def get(self, idx):
        """
        获取一个DPO训练样本
        
        Returns:
            dict: {
                'batch_type': 'dpo',
                'winner': HeteroData,
                'loser': HeteroData,
                'shared_noise': dict,
                'shared_timestep': float,
            }
        """
        winner_mol, loser_mol, score_w, score_l = self.preference_pairs[idx]
        
        # 生成共享的时间步（DPO的关键：winner和loser使用相同的时间步）
        shared_timestep = np.random.uniform(0, 1)

        # 将分子转换为HeteroData格式
        # Winner和Loser使用不同的seed，确保噪声独立（但时间步相同）
        # 使用相同seed会导致两个不同结构的分子得到相同的随机数序列，不是有意义的"共享噪声"
        winner_data = self._mol_to_hetero_data(winner_mol, shared_timestep, seed=idx * 2)
        loser_data = self._mol_to_hetero_data(loser_mol, shared_timestep, seed=idx * 2 + 1)
        
        # 构造返回的batch
        batch = {
            'batch_type': 'dpo',
            'winner': winner_data,
            'loser': loser_data,
            'shared_noise': {},
            'shared_timestep': shared_timestep,
        }
        
        return batch
    
    def _mol_to_hetero_data(self, mol, timestep, seed=None):
        """
        将RDKit分子转换为HeteroData，并添加噪声
        
        真实实现：完全复用HeteroDataset的转换逻辑
        包括DiscreteFeatureDiffusion、真实的噪声调度等
        
        Args:
            mol: RDKit分子对象
            timestep: 归一化的时间步 (0-1之间)
            seed: 随机种子，用于确保相同噪声（DPO的关键）
        
        Returns:
            HeteroData: 包含噪声的异构图数据
        """
        # 设置随机种子以确保winner和loser使用相同的噪声
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        if mol is None:
            return HeteroData()
        
        # === 1. 获取真实的噪声调度参数 ===
        t_normalized = torch.tensor([timestep], dtype=torch.float32)
        
        if hasattr(self.data_generator, 'x1_pos_diffuser'):
            noise_schedule = self.data_generator.x1_pos_diffuser.noise_schedule
            alpha_t = noise_schedule.get_alpha_bar(t_normalized).item()
            sigma_t = torch.sqrt(1 - alpha_t**2).item()
            alpha_dash_t = alpha_t
            sigma_dash_t = sigma_t
        else:
            # 后备方案
            alpha_dash_t = 1.0 - timestep
            sigma_dash_t = timestep
            alpha_t = alpha_dash_t
            sigma_t = sigma_dash_t
        
        # === 2. 准备分子坐标 ===
        # 检查分子是否有3D构象，如果没有则生成
        if mol.GetNumConformers() == 0:
            from rdkit.Chem import AllChem
            AllChem.EmbedMolecule(mol, randomSeed=seed if seed is not None else 42)
        mol_coordinates = np.array(mol.GetConformer().GetPositions())
        
        # === 3. 构建data_dict（与HeteroDataset的__getitem__完全一致）===
        data_dict = {}
        
        # === x1模态：使用真实的get_x1_data方法 ===
        if self.params['dataset'].get('compute_x1', True):
            x1_data, x1_pos, x1_virtual_node_mask = self.data_generator.get_x1_data(
                mol=mol,
                t=timestep,
                alpha_dash_t=alpha_dash_t,
                sigma_dash_t=sigma_dash_t
            )
            # 添加噪声调度参数
            x1_data['alpha_t'] = torch.tensor([alpha_t], dtype=torch.float)
            x1_data['sigma_t'] = torch.tensor([sigma_t], dtype=torch.float)
            x1_data['alpha_dash_t'] = torch.tensor([alpha_dash_t], dtype=torch.float)
            x1_data['sigma_dash_t'] = torch.tensor([sigma_dash_t], dtype=torch.float)
            data_dict['x1'] = x1_data
        
        # === x2模态：使用真实的get_x2_data方法（如果启用）===
        if self.params['dataset'].get('compute_x2', False):
            from shepherd.shepherd_score_utils.generate_point_cloud import get_atomic_vdw_radii
            atom_centers = mol_coordinates
            radii = get_atomic_vdw_radii(mol)
            num_points = self.params['dataset']['x2']['num_points']
            
            x2_data, x2_pos, x2_virtual_node_mask = self.data_generator.get_x2_data(
                radii=radii,
                atom_centers=atom_centers,
                num_points=num_points,
                recenter=self.params['dataset']['x2']['recenter'],
                add_virtual_node=self.params['dataset']['x2']['add_virtual_node'],
                remove_noise_COM=self.params['dataset']['x2']['remove_noise_COM'],
                t=timestep,
                alpha_dash_t=alpha_dash_t,
                sigma_dash_t=sigma_dash_t,
                virtual_node_pos=None
            )
            x2_data['alpha_t'] = torch.tensor([alpha_t], dtype=torch.float)
            x2_data['sigma_t'] = torch.tensor([sigma_t], dtype=torch.float)
            x2_data['alpha_dash_t'] = torch.tensor([alpha_dash_t], dtype=torch.float)
            x2_data['sigma_dash_t'] = torch.tensor([sigma_dash_t], dtype=torch.float)
            data_dict['x2'] = x2_data
        
        # === x3模态：使用真实的get_x3_data方法（如果启用）===
        if self.params['dataset'].get('compute_x3', False):
            from shepherd.shepherd_score_utils.generate_point_cloud import get_atomic_vdw_radii
            from shepherd.new_datasets import get_atomic_partial_charges
            # 使用MMFF94计算真实电荷（与预训练一致），避免全零导致x3特征退化
            try:
                charges = get_atomic_partial_charges(mol)
            except Exception:
                # MMFF94失败时使用Gasteiger电荷作为后备
                try:
                    import rdkit.Chem.AllChem
                    rdkit.Chem.AllChem.ComputeGasteigerCharges(mol)
                    charges = np.array([a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()])
                    # 处理NaN值
                    charges = np.nan_to_num(charges, nan=0.0)
                except Exception:
                    charges = np.zeros(mol.GetNumAtoms())
            charge_centers = mol_coordinates
            num_points_x3 = self.params['dataset']['x3']['num_points']
            radii_x3 = get_atomic_vdw_radii(mol)

            # x3 需要先生成表面点云（与 x2 相同步骤），再计算静电势
            # 独立于 compute_x2，不依赖 data_dict['x2']
            x3_surface_data, x3_pos, x3_virtual_node_mask = self.data_generator.get_x2_data(
                radii=radii_x3,
                atom_centers=charge_centers,
                num_points=num_points_x3,
                recenter=self.params['dataset']['x3']['recenter'],
                add_virtual_node=self.params['dataset']['x3']['add_virtual_node'],
                remove_noise_COM=self.params['dataset']['x3']['remove_noise_COM'],
                t=timestep,
                alpha_dash_t=alpha_dash_t,
                sigma_dash_t=sigma_dash_t,
                virtual_node_pos=None
            )
            # 在 x3_surface_data 基础上叠加静电势特征
            x3_data = self.data_generator.get_x3_data_electrostatics_only(
                charges=charges,
                charge_centers=charge_centers,
                data=x3_surface_data,
                pos=x3_pos,
                t=timestep,
                alpha_dash_t=alpha_dash_t,
                sigma_dash_t=sigma_dash_t
            )
            # 补充噪声调度参数
            x3_data['alpha_t'] = torch.tensor([alpha_t], dtype=torch.float)
            x3_data['sigma_t'] = torch.tensor([sigma_t], dtype=torch.float)
            x3_data['alpha_dash_t'] = torch.tensor([alpha_dash_t], dtype=torch.float)
            x3_data['sigma_dash_t'] = torch.tensor([sigma_dash_t], dtype=torch.float)
            data_dict['x3'] = x3_data
        
        # === x4模态：使用真实的get_x4_data方法（如果启用）===
        if self.params['dataset'].get('compute_x4', False):
            x4_data = self.data_generator.get_x4_data(
                mol=mol,
                recenter=self.params['dataset']['x4']['recenter'],
                add_virtual_node=self.params['dataset']['x4']['add_virtual_node'],
                remove_noise_COM=self.params['dataset']['x4']['remove_noise_COM'],
                t=timestep,
                alpha_dash_t=alpha_dash_t,
                sigma_dash_t=sigma_dash_t,
                virtual_node_pos=None
            )
            x4_data['alpha_t'] = torch.tensor([alpha_t], dtype=torch.float)
            x4_data['sigma_t'] = torch.tensor([sigma_t], dtype=torch.float)
            x4_data['alpha_dash_t'] = torch.tensor([alpha_dash_t], dtype=torch.float)
            x4_data['sigma_dash_t'] = torch.tensor([sigma_dash_t], dtype=torch.float)
            data_dict['x4'] = x4_data
        
        # === 4. 构建HeteroData对象（与HeteroDataset完全一致）===
        data = HeteroData()
        
        # x1模态数据存储
        if 'x1' in data_dict and data_dict['x1']:
            x1_data = data_dict['x1']
            node_keys = ['pos', 'pos_recentered', 'pos_forward_noised', 'pos_noise',
                'x', 'x_0', 'x_forward_noised', 'x_noise',
                'virtual_node_mask', 'com', 'com_before_centering',
                'timestep', 'alpha_t', 'sigma_t', 'alpha_dash_t', 'sigma_dash_t']
            for key in node_keys:
                if key in x1_data:
                    data['x1'][key] = x1_data[key]
            
            # 边数据存储
            if 'bond_edge_index' in x1_data:
                data['x1', 'bond', 'x1'].edge_index = x1_data['bond_edge_index']
                data['x1', 'bond', 'x1'].mask = x1_data['bond_edge_mask']
                data['x1', 'bond', 'x1'].x = x1_data['bond_edge_x']
                data['x1', 'bond', 'x1'].x_0 = x1_data['bond_edge_x_0']
                data['x1', 'bond', 'x1'].x_forward_noised = x1_data['bond_edge_x_forward_noised']
                data['x1', 'bond', 'x1'].x_noise = x1_data['bond_edge_x_noise']
        
        # x2模态数据存储
        if 'x2' in data_dict and data_dict['x2']:
            for key, value in data_dict['x2'].items():
                data['x2'][key] = value
        
        # x3模态数据存储
        if 'x3' in data_dict and data_dict['x3']:
            for key, value in data_dict['x3'].items():
                data['x3'][key] = value
        
        # x4模态数据存储
        if 'x4' in data_dict and data_dict['x4']:
            for key, value in data_dict['x4'].items():
                data['x4'][key] = value
        
        return data
    
    def update_pairs(self, new_pairs: List[Tuple]):
        """更新偏好对数据"""
        self.preference_pairs = new_pairs


class MixedBatchSampler:
    """
    混合批次采样器：交替返回标准batch和DPO batch
    """
    
    def __init__(
        self,
        standard_dataset_size: int,
        dpo_dataset_size: int,
        batch_size: int,
        dpo_ratio: float = 0.3,
        shuffle: bool = True,
    ):
        """
        Args:
            standard_dataset_size: 标准数据集大小
            dpo_dataset_size: DPO数据集大小
            batch_size: 批次大小
            dpo_ratio: DPO批次的比例（0-1之间）
            shuffle: 是否打乱
        """
        self.standard_size = standard_dataset_size
        self.dpo_size = dpo_dataset_size
        self.batch_size = batch_size
        self.dpo_ratio = dpo_ratio
        self.shuffle = shuffle
        
        # 计算每个epoch需要多少个batch
        self.n_standard_batches = standard_dataset_size // batch_size
        self.n_dpo_batches = int(self.n_standard_batches * dpo_ratio) if dpo_dataset_size > 0 else 0
        
        # 生成索引
        self.standard_indices = list(range(standard_dataset_size))
        self.dpo_indices = list(range(dpo_dataset_size))
        
        if shuffle:
            np.random.shuffle(self.standard_indices)
            np.random.shuffle(self.dpo_indices)
    
    def __iter__(self):
        """
        迭代器：交替产生标准batch和DPO batch的索引
        """
        standard_ptr = 0
        dpo_ptr = 0
        
        # 创建一个混合序列：决定每个位置是标准batch还是DPO batch
        total_batches = self.n_standard_batches + self.n_dpo_batches
        batch_types = ['standard'] * self.n_standard_batches + ['dpo'] * self.n_dpo_batches
        np.random.shuffle(batch_types)
        
        for batch_type in batch_types:
            if batch_type == 'standard':
                # 返回标准batch的索引
                end_ptr = min(standard_ptr + self.batch_size, self.standard_size)
                indices = self.standard_indices[standard_ptr:end_ptr]
                standard_ptr = end_ptr
                
                yield ('standard', indices)
                
            elif batch_type == 'dpo':
                # 返回DPO batch的索引
                if self.dpo_size > 0:
                    end_ptr = min(dpo_ptr + self.batch_size, self.dpo_size)
                    indices = self.dpo_indices[dpo_ptr:end_ptr]
                    dpo_ptr = end_ptr
                    
                    yield ('dpo', indices)
    
    def __len__(self):
        return self.n_standard_batches + self.n_dpo_batches


def collate_dpo_batch(batch_list):
    """
    自定义的DPO batch整理函数
    
    Args:
        batch_list: List of samples from DPODataset
    
    Returns:
        dict: 整理后的batch
    """
    # 如果batch中都是DPO样本
    if all(isinstance(b, dict) and b.get('batch_type') == 'dpo' for b in batch_list):
        # 分别整理winner和loser
        from torch_geometric.data import Batch
        
        winners = [b['winner'] for b in batch_list]
        losers = [b['loser'] for b in batch_list]
        
        # 使用PyG的Batch整理
        batched_winners = Batch.from_data_list(winners)
        batched_losers = Batch.from_data_list(losers)
        
        return {
            'batch_type': 'dpo',
            'winner': batched_winners,
            'loser': batched_losers,
            'shared_timestep': batch_list[0]['shared_timestep'],
        }
    
    # 如果是标准batch
    else:
        from torch_geometric.data import Batch
        return Batch.from_data_list(batch_list)


class MixedDPODataset(torch.utils.data.Dataset):
    """
    混合数据集：以可配置比例交替返回真实训练样本和 DPO 偏好对。

    解决纯 DPO 训练导致的灾难性遗忘问题——模型在 DPO 微调过程中
    需要持续接触真实分子数据以维持基础去噪能力。
    """

    def __init__(self, standard_dataset, dpo_dataset, real_data_ratio=0.5):
        """
        Args:
            standard_dataset: HeteroDataset（真实分子数据）
            dpo_dataset: DPODataset（偏好对）
            real_data_ratio: 真实数据占比（0-1），默认 0.5 即 50% 真实 + 50% DPO
        """
        self.standard_dataset = standard_dataset
        self.dpo_dataset = dpo_dataset
        self.real_data_ratio = real_data_ratio

    def __len__(self):
        return max(len(self.standard_dataset), len(self.dpo_dataset) * 2)

    def __getitem__(self, idx):
        if random.random() < self.real_data_ratio:
            real_idx = idx % len(self.standard_dataset)
            return self.standard_dataset[real_idx]
        else:
            dpo_idx = idx % len(self.dpo_dataset)
            return self.dpo_dataset[dpo_idx]


def collate_mixed_batch(batch_list):
    """
    自适应 collation：根据 batch 内样本类型自动选择合适的整理方式。

    - 全部为 DPO 样本 → collate_dpo_batch
    - 全部为标准样本 → Batch.from_data_list
    - 混合 → 按多数类型整理（丢弃少数类型，避免格式冲突）
    """
    from torch_geometric.data import Batch

    dpo_items = [b for b in batch_list if isinstance(b, dict) and b.get('batch_type') == 'dpo']
    std_items = [b for b in batch_list if not isinstance(b, dict)]

    if len(dpo_items) == len(batch_list):
        return collate_dpo_batch(batch_list)

    if len(std_items) == len(batch_list):
        return Batch.from_data_list(batch_list)

    # 混合 batch：取多数类型
    if len(dpo_items) >= len(std_items):
        return collate_dpo_batch(dpo_items)
    else:
        return Batch.from_data_list(std_items)
