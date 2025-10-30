"""
PyTorch Lightning 回调函数
用于DPO训练的在线采样
"""

import torch
import numpy as np
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm
import time

from shepherd.dpo_utils import (
    ShepherdScorer,
    OnlineSampler,
    PreferencePairBuilder,
    DPOSamplingScheduler
)


class OnlineSamplingCallback(Callback):
    """
    在线采样回调
    在每个epoch开始前，使用当前模型生成新分子并构建偏好对
    """
    
    def __init__(self, params, dataset, molblocks_and_charges):
        """
        Args:
            params: 全局参数字典
            dataset: 训练数据集
            molblocks_and_charges: 原始分子数据
        """
        super().__init__()
        
        self.params = params
        self.dataset = dataset
        self.molblocks_and_charges = molblocks_and_charges
        
        # 初始化工具类
        self.scorer = ShepherdScorer(params)
        self.pair_builder = PreferencePairBuilder(self.scorer, params)
        self.scheduler = DPOSamplingScheduler(params)
        
        # 采样频率控制
        self.sampling_every_n_epochs = params['training'].get('dpo_sampling_every_n_epochs', 1)
        
        # 偏好对存储
        self.preference_pairs = []
        self.old_pairs_buffer = []  # 保留的旧样本
        
        # 采样统计
        self.sampling_stats = {
            'n_samples': 0,
            'n_valid_pairs': 0,
            'avg_score_winner': 0.0,
            'avg_score_loser': 0.0,
            'sampling_time': 0.0,
        }
    
    def on_train_epoch_start(self, trainer, pl_module):
        """
        在每个epoch开始时进行在线采样
        """
        # 只在DPO模式下采样
        if not self.params['training'].get('enable_dpo', False):
            return
        
        # 检查是否应该跳过第一个epoch
        skip_first_epoch = self.params['training'].get('dpo_skip_first_epoch', False)
        if trainer.current_epoch == 0 and skip_first_epoch:
            if trainer.global_rank == 0:
                print("⏭️  第一个epoch，跳过在线采样（dpo_skip_first_epoch=True）")
                print("   如需从epoch 0开始采样，请设置 'dpo_skip_first_epoch': False")
            return
        
        # 检查采样频率
        if trainer.current_epoch % self.sampling_every_n_epochs != 0:
            if trainer.global_rank == 0:
                print(f"⏭️  Epoch {trainer.current_epoch}：未到采样周期，跳过（每{self.sampling_every_n_epochs}个epoch采样一次）")
            return
        
        # ⚠️ 关键：只在rank 0上进行采样（避免DDP环境下的设备冲突）
        is_main_process = trainer.global_rank == 0
        
        if is_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {trainer.current_epoch}: 开始在线采样...")
            print(f"{'='*60}")
        
        start_time = time.time()
        new_pairs = []
        
        # 只在主进程上进行采样
        if is_main_process:
            # 1. 选择种子分子
            # 尝试从trainer获取损失信息（用于基于难度的采样）
            losses = None
            if hasattr(pl_module, 'training_losses'):
                losses = pl_module.training_losses
            
            seed_indices = self.scheduler.select_seeds(
                self.dataset,
                trainer.current_epoch,
                losses=losses
            )
            
            print(f"选择了 {len(seed_indices)} 个种子分子进行采样")
            
            # 2. 准备种子数据
            seed_mol_list = [self.molblocks_and_charges[i] for i in seed_indices]
            
            # 3. 初始化采样器（使用当前模型）
            # 在DDP环境下，确保使用cuda:0而不是分布式设备
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            
            sampler = OnlineSampler(
                model_pl=pl_module,  # 传入LightningModule
                params=self.params,
                dataset=self.dataset,  # 传入dataset以获取marginals
                device=device  # 使用cuda:0避免设备冲突
            )
            
            # 4. 批量采样
            print("开始生成分子...")
            all_generated_mols = sampler.batch_sample(seed_mol_list, show_progress=True)
            
            # 5. 构建偏好对
            print("开始构建偏好对...")
            new_pairs = self.pair_builder.batch_build_pairs(
                all_generated_mols,
                reference_mols=None,  # 如果需要参考分子，这里传入
                show_progress=True
            )
            
            # 6. 数据重用策略：保留50%旧数据
            if len(self.old_pairs_buffer) > 0:
                keep_ratio = 0.5
                n_keep = int(len(self.old_pairs_buffer) * keep_ratio)
                kept_pairs = self.old_pairs_buffer[:n_keep]
                print(f"保留了 {len(kept_pairs)} 个旧偏好对")
                
                # 合并新旧数据
                self.preference_pairs = kept_pairs + new_pairs
            else:
                self.preference_pairs = new_pairs
            
            # 7. 更新旧数据缓存
            self.old_pairs_buffer = new_pairs.copy()
            
            # 8. 更新训练器的DataLoader
            if hasattr(trainer, 'train_dataloader') and hasattr(trainer.train_dataloader, 'update_dpo_dataset'):
                trainer.train_dataloader.update_dpo_dataset(self.preference_pairs)
                print(f"已更新DataLoader，当前共有 {len(self.preference_pairs)} 个偏好对")
        
        # 9. 同步所有进程（在DDP环境下等待rank 0完成采样）
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
            if is_main_process:
                print(f"✅ 所有进程已同步")
        
        # 9. 统计信息
        elapsed_time = time.time() - start_time
        
        if len(new_pairs) > 0:
            # 从score_dict中提取total_score
            scores_winner = [pair[2]['total_score'] for pair in new_pairs]
            scores_loser = [pair[3]['total_score'] for pair in new_pairs]
            
            self.sampling_stats = {
                'n_samples': len(seed_indices),
                'n_valid_pairs': len(new_pairs),
                'avg_score_winner': np.mean(scores_winner),
                'avg_score_loser': np.mean(scores_loser),
                'score_gap': np.mean([w - l for w, l in zip(scores_winner, scores_loser)]),
                'sampling_time': elapsed_time,
                'valid_pair_ratio': len(new_pairs) / len(seed_indices),
            }
            
            # 记录到logger
            pl_module.log('dpo_sampling/n_valid_pairs', len(new_pairs))
            pl_module.log('dpo_sampling/avg_score_winner', self.sampling_stats['avg_score_winner'])
            pl_module.log('dpo_sampling/avg_score_loser', self.sampling_stats['avg_score_loser'])
            pl_module.log('dpo_sampling/score_gap', self.sampling_stats['score_gap'])
            pl_module.log('dpo_sampling/sampling_time', elapsed_time)
            pl_module.log('dpo_sampling/valid_pair_ratio', self.sampling_stats['valid_pair_ratio'])
        
        print(f"\n{'='*60}")
        print(f"采样完成！")
        print(f"  - 采样时间: {elapsed_time:.2f}秒")
        print(f"  - 生成偏好对: {len(new_pairs)}")
        print(f"  - 总偏好对: {len(self.preference_pairs)}")
        if len(new_pairs) > 0:
            print(f"  - Winner平均分: {self.sampling_stats['avg_score_winner']:.4f}")
            print(f"  - Loser平均分: {self.sampling_stats['avg_score_loser']:.4f}")
            print(f"  - 平均分差: {self.sampling_stats['score_gap']:.4f}")
        print(f"{'='*60}\n")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """
        在epoch结束时可以做一些清理或保存工作
        """
        # 可以选择保存当前的偏好对
        if self.params['training'].get('save_preference_pairs', False):
            import pickle
            save_path = f"{trainer.default_root_dir}/preference_pairs_epoch_{trainer.current_epoch}.pkl"
            with open(save_path, 'wb') as f:
                pickle.dump(self.preference_pairs, f)
            print(f"偏好对已保存到: {save_path}")


class DPOMetricsCallback(Callback):
    """
    DPO训练指标监控回调
    """
    
    def __init__(self):
        super().__init__()
        self.metrics_history = {
            'implicit_acc': [],
            'dpo_loss': [],
            'dpo_weight': [],
        }
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """记录每个batch的DPO指标"""
        # 这些指标已经在training_step中通过self.log记录了
        # 这里可以做额外的处理或可视化
        pass
    
    def on_train_epoch_end(self, trainer, pl_module):
        """在epoch结束时汇总指标"""
        # 可以打印epoch级别的统计信息
        pass
