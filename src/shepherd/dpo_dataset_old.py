"""
DPO数据集：管理偏好对数据
支持共享噪声和时间步
"""

import torch
import numpy as np
from torch_geometric.data import HeteroData, Dataset
from typing import List, Tuple, Optional
from copy import deepcopy


class DPODataset(Dataset):
    """
    DPO偏好对数据集
    每个样本包含：winner分子数据 + loser分子数据 + 共享的噪声和时间步
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
            base_dataset: 基础数据集，用于生成噪声数据
            noise_schedule_dict: 噪声调度参数
            params: 全局参数
        """
        super().__init__()
        
        self.preference_pairs = preference_pairs
        self.base_dataset = base_dataset
        self.noise_schedule_dict = noise_schedule_dict
        self.params = params
        
        # 用于将分子转换为数据的转换器
        # 这里简化处理，实际应该复用 HeteroDataset 的逻辑
        self.transform = None
    
    def len(self):
        return len(self.preference_pairs)
    
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
        
        # 生成共享的噪声和时间步（这是DPO的关键）
        shared_timestep = np.random.uniform(0, 1)
        
        # 将分子转换为HeteroData格式
        # 注意：这里需要应用相同的噪声
        winner_data = self._mol_to_hetero_data(winner_mol, shared_timestep, seed=idx)
        loser_data = self._mol_to_hetero_data(loser_mol, shared_timestep, seed=idx)
        
        # 构造返回的batch
        batch = {
            'batch_type': 'dpo',
            'winner': winner_data,
            'loser': loser_data,
            'shared_noise': {},  # 实际应该包含噪声信息
            'shared_timestep': shared_timestep,
        }
        
        return batch
    
    def _mol_to_hetero_data(self, mol, timestep, seed=None):
        """
        将RDKit分子转换为HeteroData，并添加噪声
        
        复用 HeteroDataset 的转换逻辑，确保winner和loser使用相同的噪声
        
        Args:
            mol: RDKit分子对象（mol block或者Mol对象）
            timestep: 归一化的时间步 (0-1之间)
            seed: 随机种子，用于确保相同噪声
        
        Returns:
            HeteroData: 包含噪声的异构图数据
        """
        # 设置随机种子以确保winner和loser使用相同的噪声（DPO的关键）
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # 将时间步转换为噪声调度参数
        # 这里需要根据base_dataset的噪声调度来获取alpha_dash_t和sigma_dash_t
        # 简化实现：使用固定的噪声参数
        # 实际应该从self.noise_schedule_dict获取
        
        # 占位符实现：直接使用base_dataset的get方法的逻辑
        # 注意：这是一个简化版本，完整版本需要复用HeteroDataset的所有逻辑
        
        data = HeteroData()
        
        # 如果mol是mol block字符串，转换为RDKit Mol对象
        if isinstance(mol, str):
            import rdkit.Chem as Chem
            mol = Chem.MolFromMolBlock(mol, removeHs=False)
        
        if mol is None:
            # 返回空数据
            return data
        
        # 提取基本信息
        num_atoms = mol.GetNumAtoms()
        
        # 获取坐标并中心化
        coords = np.array(mol.GetConformer().GetPositions())
        coords = coords - np.mean(coords, axis=0)
        
        # 添加噪声（简化版本）
        # 完整版本应该使用DiscreteFeatureDiffusion等
        noise = np.random.randn(*coords.shape) * 0.1  # 简化的噪声
        noised_coords = coords + noise
        
        # 构造基本的x1数据
        data['x1'].pos = torch.tensor(coords, dtype=torch.float32)
        data['x1'].pos_forward_noised = torch.tensor(noised_coords, dtype=torch.float32)
        data['x1'].pos_noise = torch.tensor(noise, dtype=torch.float32)
        data['x1'].timestep = torch.tensor([timestep], dtype=torch.float32)
        data['x1'].virtual_node_mask = torch.zeros(num_atoms, dtype=torch.bool)
        
        # 添加原子类型信息（简化）
        atom_types = []
        for atom in mol.GetAtoms():
            atom_types.append(atom.GetAtomicNum())
        
        # 创建one-hot编码（简化版本）
        # 实际应该使用self.params['dataset']['x1']['atom_types']
        max_atomic_num = 20
        x = torch.zeros(num_atoms, max_atomic_num)
        for i, atom_type in enumerate(atom_types):
            if atom_type < max_atomic_num:
                x[i, atom_type] = 1.0
        
        data['x1'].x = x
        data['x1'].x_0 = x.clone()
        data['x1'].x_forward_noised = x.clone()  # 简化：不对离散特征加噪
        data['x1'].x_noise = torch.zeros_like(x)
        
        # 添加噪声调度参数（简化）
        data['x1'].alpha_t = torch.tensor([1.0 - timestep], dtype=torch.float32)
        data['x1'].sigma_t = torch.tensor([timestep], dtype=torch.float32)
        data['x1'].alpha_dash_t = torch.tensor([1.0 - timestep], dtype=torch.float32)
        data['x1'].sigma_dash_t = torch.tensor([timestep], dtype=torch.float32)
        
        # 添加键信息（简化）
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])  # 无向图
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            data['x1', 'bond', 'x1'].edge_index = edge_index
            data['x1', 'bond', 'x1'].mask = torch.ones(edge_index.size(1), dtype=torch.bool)
            
            # 简化的键特征
            num_edges = edge_index.size(1)
            bond_features = torch.zeros(num_edges, 5)  # 5种键类型
            bond_features[:, 0] = 1.0  # 默认单键
            data['x1', 'bond', 'x1'].x = bond_features
            data['x1', 'bond', 'x1'].x_0 = bond_features.clone()
            data['x1', 'bond', 'x1'].x_forward_noised = bond_features.clone()
            data['x1', 'bond', 'x1'].x_noise = torch.zeros_like(bond_features)
        
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
            'shared_noise': batch_list[0]['shared_noise'],  # 简化处理
            'shared_timestep': batch_list[0]['shared_timestep'],
        }
    
    # 如果是标准batch
    else:
        from torch_geometric.data import Batch
        return Batch.from_data_list(batch_list)
