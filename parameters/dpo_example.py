"""
DPO训练参数配置示例
在标准训练参数基础上添加DPO相关配置
"""

params = {
    'data': 'MOSES_aq',  # 或 'GDB17'
    
    # ==================== 噪声调度 ====================
    'noise_schedules': {
        'x1': 'polynomial_2',
        'x2': 'polynomial_2',
        'x3': 'polynomial_2',
        'x4': 'polynomial_2',
    },
    
    # ==================== 数据集配置 ====================
    'dataset': {
        'explicit_hydrogens': True,
        'use_MMFF94_charges': True,
        'probe_radius': 1.4,
        
        'compute_x1': True,
        'compute_x2': True,
        'compute_x3': False,  # 通常不使用x3
        'compute_x4': True,
        
        'x1': {
            'recenter': True,
            'add_virtual_node': True,
            'remove_noise_COM': True,
            'atom_types': [None, 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br'],
            'charge_types': [-1, 0, 1],
            'bond_types': ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
            'scale_atom_features': True,
            'scale_bond_features': True,
        },
        
        'x2': {
            'independent_timesteps': True,
            'recenter': True,
            'add_virtual_node': True,
            'remove_noise_COM': True,
            'num_points': 1000,
        },
        
        'x3': {
            'independent_timesteps': True,
            'recenter': True,
            'add_virtual_node': True,
            'remove_noise_COM': True,
            'num_points': 1000,
            'scale_node_features': True,
        },
        
        'x4': {
            'independent_timesteps': True,
            'recenter': True,
            'add_virtual_node': True,
            'remove_noise_COM': True,
            'max_node_types': 10,
            'scale_node_features': True,
            'scale_vector_features': True,
            'multivectors': False,
            'check_accessibility': False,
        },
    },
    
    # ==================== 训练配置 ====================
    'training': {
        # 基础训练参数
        'output_dir': 'dpo_training_run_001',
        'batch_size': 32,
        'num_workers': 8,
        'num_gpus': 1,
        'multiprocessing_spawn': True,
        
        'lr': 1e-4,
        'min_lr': 1e-6,
        'lr_steps': 1000000,
        
        'gradient_clip_val': 1.0,
        'accumulate_grad_batches': 1,
        'log_every_n_steps': 100,
        
        # 训练的模块
        'train_x1_denoising': True,
        'train_x2_denoising': True,
        'train_x3_denoising': False,
        'train_x4_denoising': True,
        
        # ==================== DPO特有参数 ====================
        'enable_dpo': True,  # 🔥 启用DPO训练
        
        # DPO核心超参数
        'beta_dpo': 0.2,  # DPO温度参数，控制模型与参考模型的偏离度（0.1-0.5）
        'dpo_ramp_up_epochs': 5,  # DPO权重预热轮数
        'dpo_max_weight': 0.5,  # DPO损失的最大权重（0.4-0.6）
        
        # 采样策略
        'dpo_sampling_ratio': 0.05,  # 每个epoch采样的数据比例（5%）
        'dpo_batch_ratio': 0.3,  # DPO batch占总batch的比例（30%）
        
        # 偏好对构建
        'dpo_min_score_gap': 0.5,  # 最小分数差距阈值
        'dpo_keep_old_ratio': 0.5,  # 保留旧偏好对的比例
        'dpo_hard_sample_threshold': 0.6,  # 困难样本的implicit_acc阈值
        
        # checkpoint和采样控制
        'dpo_skip_first_epoch': False,  # 是否跳过第一个epoch的采样（False=从epoch 0开始）
        'dpo_load_weights_only': True,  # 从旧checkpoint只加载权重不加载optimizer
        
        # 其他选项
        'save_preference_pairs': True,  # 是否保存偏好对
    },
    
    # ==================== DPO采样配置 ====================
    'sampling': {
        'timesteps': 1000,  # 采样步数
        'num_samples_per_molecule': 4,  # 每个种子分子生成4个样本
    },
    
    # ==================== DPO评分权重 ====================
    'dpo': {
        'sampling_ratio': 0.05,
        'min_score_gap': 0.5,
        
        # Shepherd Score权重配置
        'score_weights': {
            'rmsd': -1.0,
            'strain_energy': -0.5,
            'logp': 0.2,
            'mqed': 1.0,
            'validity': 5.0,
            'sims_surf': 0.8,
            'sims_esp': 0.8,
        },
        
        # 采样桶策略
        'size_buckets': {
            'small': (0, 15),
            'medium': (16, 35),
            'large': (36, 999),
        },
        'bucket_weights': {
            'small': 0.3,
            'medium': 0.4,
            'large': 0.3,
        },
    },
    
    # ==================== 模型架构配置 ====================
    'model': {
        'hidden_dim': 256,
        'num_layers': 6,
        'num_attention_heads': 8,
        
        # x1相关
        'x1_bond_diffusion': True,
        
        # 其他模型参数...
    },
}


# ==================== 使用说明 ====================
"""
使用此配置启动DPO训练：

1. 标准训练（不使用DPO）：
   python training/new_train.py standard_config 42

2. DPO训练（推荐流程）：
   
   步骤1: 先进行标准预训练
   - 设置 'enable_dpo': False
   - 训练若干epoch直到模型收敛
   
   步骤2: 切换到DPO训练
   - 设置 'enable_dpo': True
   - 从预训练checkpoint恢复
   - 继续训练
   
   命令：
   python training/new_train.py dpo_example 42

关键参数调优建议：

1. beta_dpo (0.1 - 0.5):
   - 较小值(0.1-0.2): 模型更保守，更接近参考模型
   - 较大值(0.3-0.5): 模型更激进，可能产生更多样化的分子
   
2. dpo_max_weight (0.3 - 0.6):
   - 控制DPO损失和标准损失的平衡
   - 从0.3开始，观察implicit_acc
   - 如果implicit_acc > 0.7，可以增加到0.5
   
3. dpo_sampling_ratio (0.01 - 0.1):
   - 采样比例，影响训练效率
   - 5%是一个好的起点
   - 数据集较小时可以提高到10%
   
4. dpo_batch_ratio (0.2 - 0.4):
   - DPO batch占比
   - 30%是平衡点
   
监控指标：

关键指标：
- implicit_acc: 应该 > 0.6，最好在0.7-0.8
- model_loss_diff: 应该为负值，且绝对值逐渐增大
- avg_score_winner: 应该持续上升
- score_gap: 应该稳定或增加

如果implicit_acc < 0.5:
- 降低beta_dpo
- 增加dpo_sampling_ratio
- 检查评分函数是否正确

如果implicit_acc > 0.9:
- 增加beta_dpo
- 增加min_score_gap
- 可能需要生成更难的偏好对
"""
