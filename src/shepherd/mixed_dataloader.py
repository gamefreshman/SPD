"""
混合DataLoader：同时提供标准训练数据和DPO偏好对数据
"""

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch_geometric.loader import DataLoader as PyGDataLoader
import multiprocessing
from typing import Optional

from shepherd.dpo_dataset import DPODataset, MixedBatchSampler, collate_dpo_batch


class MixedDataLoader:
    """
    混合数据加载器
    交替产生标准batch和DPO batch
    """
    
    def __init__(
        self,
        standard_dataset,
        dpo_dataset: Optional[DPODataset],
        batch_size: int,
        num_workers: int = 0,
        dpo_ratio: float = 0.3,
        shuffle: bool = True,
        multiprocessing_context=None,
        worker_init_fn=None,
        persistent_workers=False,
    ):
        """
        Args:
            standard_dataset: 标准训练数据集
            dpo_dataset: DPO偏好对数据集（可选）
            batch_size: 批次大小
            num_workers: 工作进程数
            dpo_ratio: DPO批次占比
            shuffle: 是否打乱
        """
        self.standard_dataset = standard_dataset
        self.dpo_dataset = dpo_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dpo_ratio = dpo_ratio
        self.shuffle = shuffle
        
        # 创建标准数据加载器
        self.standard_loader = PyGDataLoader(
            dataset=standard_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=multiprocessing_context,
            worker_init_fn=worker_init_fn,
            persistent_workers=persistent_workers,
        )
        
        # 创建DPO数据加载器（如果有）
        self.dpo_loader = None
        if dpo_dataset is not None and len(dpo_dataset) > 0:
            self.dpo_loader = PyGDataLoader(
                dataset=dpo_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers // 2 if num_workers > 1 else 0,  # 使用较少的worker
                collate_fn=collate_dpo_batch,
                multiprocessing_context=multiprocessing_context,
                worker_init_fn=worker_init_fn,
                persistent_workers=persistent_workers,
            )
    
    def __iter__(self):
        """
        迭代器：交替产生标准batch和DPO batch
        """
        standard_iter = iter(self.standard_loader)
        dpo_iter = iter(self.dpo_loader) if self.dpo_loader else None
        
        # 计算需要产生多少个batch
        n_standard = len(self.standard_loader)
        n_dpo = len(self.dpo_loader) if self.dpo_loader else 0
        
        # 创建混合序列
        if n_dpo > 0:
            # 按比例混合
            total_batches = n_standard + n_dpo
            n_dpo_to_use = min(n_dpo, int(n_standard * self.dpo_ratio))
            
            # 创建batch类型序列
            batch_types = ['standard'] * n_standard + ['dpo'] * n_dpo_to_use
            if self.shuffle:
                import random
                random.shuffle(batch_types)
            
            # 按序列产生batch
            standard_count = 0
            dpo_count = 0
            
            for batch_type in batch_types:
                try:
                    if batch_type == 'standard' and standard_count < n_standard:
                        batch = next(standard_iter)
                        standard_count += 1
                        yield batch
                    elif batch_type == 'dpo' and dpo_count < n_dpo_to_use:
                        batch = next(dpo_iter)
                        dpo_count += 1
                        yield batch
                except StopIteration:
                    # 如果某个迭代器提前结束，继续下一个
                    continue
        else:
            # 只有标准数据
            for batch in standard_iter:
                yield batch
    
    def __len__(self):
        """返回总batch数"""
        n_standard = len(self.standard_loader)
        n_dpo = len(self.dpo_loader) if self.dpo_loader else 0
        n_dpo_to_use = min(n_dpo, int(n_standard * self.dpo_ratio))
        return n_standard + n_dpo_to_use
    
    def update_dpo_dataset(self, new_preference_pairs):
        """
        更新DPO数据集
        在每个epoch开始时调用
        """
        if self.dpo_dataset is not None:
            self.dpo_dataset.update_pairs(new_preference_pairs)
            
            # 重新创建DPO loader
            if len(new_preference_pairs) > 0:
                self.dpo_loader = PyGDataLoader(
                    dataset=self.dpo_dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    num_workers=self.num_workers // 2 if self.num_workers > 1 else 0,
                    collate_fn=collate_dpo_batch,
                )


def create_mixed_dataloader(
    standard_dataset,
    dpo_dataset,
    batch_size,
    num_workers=0,
    dpo_ratio=0.3,
    shuffle=True,
    params=None,
    multiprocessing_context=None,
    worker_init_fn=None,
    persistent_workers=False,
):
    """
    工厂函数：创建混合DataLoader
    
    Args:
        standard_dataset: 标准数据集
        dpo_dataset: DPO数据集
        batch_size: 批次大小
        num_workers: 工作进程数
        dpo_ratio: DPO批次比例
        shuffle: 是否打乱
        params: 参数字典
    
    Returns:
        MixedDataLoader
    """
    return MixedDataLoader(
        standard_dataset=standard_dataset,
        dpo_dataset=dpo_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        dpo_ratio=dpo_ratio,
        shuffle=shuffle,
        multiprocessing_context=multiprocessing_context,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent_workers,
    )
