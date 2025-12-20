# DPO微调配置 - 基于3个天然产物分子（NPs）
# x3 implicitly includes x2

import numpy as np

# 禁用形式电荷扩散功能
diffuse_formal_charges = False # 不再扩散形式电荷，仅处理原子类型
charge_types = [0, 1, 2, -1, -2] # 保留定义但不使用
num_charge_types = 0 # 强制设为0，不包含电荷类型

diffuse_bonds = True
bond_types = [None, 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
num_bond_types = len(bond_types)

atom_types = [None, 'H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si']
num_atom_types = len(atom_types)

num_pharmacophore_types = 10 # ('Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'Halogen', 'Cation', 'Anion', 'ZnBinder') plus buffers

num_channels = 64


params = {
    'data': 'NPs',  # 使用NPs数据集（3个天然产物分子）
    
    # major architecture decisions
    
    'use_ema': False,
    'x1_bond_diffusion': diffuse_bonds,
    'x1_formal_charge_diffusion': diffuse_formal_charges, # 明确禁用形式电荷扩散
    
    # 显式扩散变量
    'explicit_diffusion_variables': ['x1', 'x4'],
    
    'exclude_variables_from_decoder_heterogeneous_graph': [], 
    # if any variables (besides x1) get recentered in the decoder/denoiser, exclude them from any heterogeneous graph (which requires a common reference frame).
    # 如果某个模态在解码器中被单独中心化了，它就脱离了公共坐标系，不能再参与到需要公共坐标系的异构图（Heterogeneous Graph）信息交互中。这里为空，表示所有模态共享一个坐标系。
    
    'training': {
        
        'train_x1_denoising': True,
        'train_x2_denoising': False,
        'train_x3_denoising': False,
        'train_x4_denoising': True,
        
        # 调整批次大小（因为只有3个分子）
        'batch_size': 3,  # 小批次，适合3个分子的微调
        'accumulate_grad_batches': 1,  # 不累积梯度
        'num_gpus': 3,  # 单GPU
        
        'lr': 0.0001,  # 较小的学习率用于微调
        'min_lr': 0.00001,
        'lr_steps': 1,
        
        'gradient_clip_val': 5.0,
        
        'num_workers': 4,  # 减少worker数量
        
        'output_dir': 'x1x3x4_dpo_finetune_nps/',  # 专门的输出目录
        
        'log_every_n_steps': 10,  # 更频繁的日志
        
        'multiprocessing_spawn': True,

        # ==================== DPO配置 ====================
        'enable_dpo': True,  # 启用DPO微调
        
        # DPO核心参数
        'beta_dpo': 0.2,  # DPO温度参数
        'dpo_ramp_up_epochs': 2,  # 较短的预热轮数
        'dpo_max_weight': 0.6,  # 较高的DPO权重（因为是微调）
        
        # 采样策略（针对小数据集调整）
        'dpo_sampling_ratio': 1.0,  # 每个epoch对所有3个分子采样（100%）
        'dpo_batch_ratio': 0.5,  # DPO batch占总batch的50%
        
        # 偏好对构建
        'dpo_min_score_gap': 0.3,  # 降低最小分数差距阈值
        'dpo_keep_old_ratio': 0.7,  # 保留更多旧偏好对
        
        # checkpoint和采样控制
        'dpo_skip_first_epoch': False,  # 从epoch 0就开始采样
        'dpo_load_weights_only': True,  # 从旧checkpoint只加载权重
        
        # 预训练模型路径（从MOSES_aq模型开始微调）
        'pretrained_checkpoint_path': 'x1x3x4_diffusion_mosesaq_20240824/last_27epoch.ckpt',
        
        # 其他选项
        'save_preference_pairs': True,  # 保存偏好对用于分析
    },
    
    # ==================== DPO采样配置 ====================
    'sampling': {
        'timesteps': 400,  # 采样步数（与训练T一致）
        'num_samples_per_molecule': 4,  # 每个种子分子生成2个样本（快速调试）
        'fixed_n_atoms': 70,  # 固定生成的原子数（与预训练模型一致）
    },
    
    # ==================== DPO评分权重 ====================
    'dpo': {
        'sampling_ratio': 1.0,
        'min_score_gap': 0.3,
    },
    
    
    'dataset': {
    
        'explicit_hydrogens': True,
        'use_MMFF94_charges': False,

        # 算分子表面（SAS）时使用的探针半径
        'probe_radius': 0.6, # for x2 and x3
    
        'compute_x1': True,
        'compute_x2': False,
        'compute_x3': True,
        'compute_x4': True,
        
        'x1': {
            'recenter': True, 
            'add_virtual_node': True,
            'remove_noise_COM': True,
            'atom_types': atom_types,
            'charge_types': charge_types,
            'bond_types': bond_types,
            'scale_atom_features': 0.25,
            'scale_bond_features': 1.0,
        },
        
        
        'x2': {
            'recenter': False,
            'add_virtual_node': False,
            'remove_noise_COM': False,
            'num_points': 75,
            'independent_timesteps': False,
        },
        
        
        'x3': {
            'independent_timesteps': False, # coupled to x1 timesteps
            
            'recenter': False, 
            'add_virtual_node': False, 
            'remove_noise_COM': False,
            'num_points': 75,
                        
            'scale_node_features': 2.0, # scaling electrostatic potential
        }, 
        
        
        'x4': {
            'independent_timesteps': False, # coupled to x1 timesteps
            'recenter': False, 
            'add_virtual_node': True, 
            'remove_noise_COM': False,
            'max_node_types': num_pharmacophore_types,
            'scale_node_features': 2.0,
            'scale_vector_features': 2.0,
            
            'multivectors': False,
            'check_accessibility': False,
        },
        
    },

    
    
    # Model Hyperparameters
    
    # for joint/global l1 embeddings (must be the same for each x1, x2, ...)
    'lmax_list': [1],
    'mmax_list': [1],
    'ffn_hidden_channels': 32,
    'grid_resolution': 16,
    
    
    'decoder_heterogeneous_graph_encoder': {
        'use': True,
        
        'num_layers': 2,
        'input_sphere_channels': num_channels,
        'sphere_channels': num_channels,
        
        'attn_hidden_channels': 24,
        'num_heads': 2,
        'attn_alpha_channels': 24,
        'attn_value_channels': 24,
        'ffn_hidden_channels': 32,
        
        'lmax_list': [1],
        'mmax_list': [1],
        'grid_resolution': 16,
        'cutoff': 5.0,
        'max_neighbors': 1000000, # essentially infinite
        
        'num_sphere_samples': 128,
        'edge_channels': 128,

    },
    
    
    
    
    'x1': {
        'decoder': {
            'input_node_channels': num_atom_types, # 仅包含原子类型，不包含电荷类型
            'node_channels': num_channels,
            'time_embedding_size': 32,
            
            'force_edges_to_virtual_nodes': True, # for both encoder and denoiser
                        
            'encoder': {
                
                'fully_connected': True, # whether to force the 3D graph to be fully connected

                'num_layers': 4,
                'input_sphere_channels': num_channels,
                'sphere_channels': num_channels,
                
                'input_bond_channels': num_bond_types,
                'edge_attr_channels': num_channels,
                
                'attn_hidden_channels': 32,
                'num_heads': 4,
                'attn_alpha_channels': 32,
                'attn_value_channels': 32,
                'ffn_hidden_channels': 64,
                
                'lmax_list': [1],
                'mmax_list': [1],
                'grid_resolution': 16,
                'cutoff': 5.0, # if fully_connected, this is still used for the Gaussian distance expansion
                'max_neighbors': 1000000, # essentially infinite
                
                'num_sphere_samples': 128,
                'edge_channels': 128,
            },
            
            'denoiser': {
                
                'output_node_channels': num_atom_types, # 仅输出原子类型，不包含电荷类型
                'output_bond_channels': num_bond_types, # must equal params['x1']['decoder']['input_bond_channels']
                
                # this is for the feature update
                'MLP_hidden_dim': 64,
                'num_MLP_hidden_layers': 2,
                
                # this is for the positional update
                'use_e3nn': True,
                'e3nn': {
                    'lmax_list': [1],
                    'mmax_list': [1],
                    'ffn_hidden_channels': 32,
                    'grid_resolution': 16,
                },
                
                'use_egnn_positions_update': True,
                'egnn': {
                    'normalize_egnn_vectors': True,
                    'distance_expansion_dim': 32,
                    'num_MLP_hidden_layers': 2,
                    'MLP_hidden_dim': 64,
                },
            
            },
            
        },
    },
     
    
    # ignored in this particular model
    'x2': {
        'decoder': {
            'input_node_channels': 2, # real or virtual node
            'node_channels': num_channels,
            'time_embedding_size': 32,
            
            'force_edges_to_virtual_nodes': False, # for both encoder and denoiser
            
            'encoder': {
                'num_layers': 2,
                'input_sphere_channels': num_channels,
                'sphere_channels': num_channels,
                
                'attn_hidden_channels': 24,
                'num_heads': 2,
                'attn_alpha_channels': 24,
                'attn_value_channels': 24,
                'ffn_hidden_channels': 32,
                
                'lmax_list': [1],
                'mmax_list': [1],
                'grid_resolution': 16,
                'cutoff': 5.0,
                'max_neighbors': 1000000, # essentially infinite
                
                'num_sphere_samples': 128,
                'edge_channels': 128,
            },
            
            'denoiser': {
                
                'output_node_channels': num_channels, # ignored
                
                'use_e3nn': True,
                'e3nn': {
                    'lmax_list': [1],
                    'mmax_list': [1],
                    'ffn_hidden_channels': 32,
                    'grid_resolution': 16,
                },
                
                'use_egnn_positions_update': False,
                'egnn': {
                    'normalize_egnn_vectors': True,
                    'distance_expansion_dim': 32,
                    'num_MLP_hidden_layers': 2,
                    'MLP_hidden_dim': 64,
                },
            
            },
            
        },
        
    },
        

    
    'x3': {
        'decoder': {
        
            'scalar_expansion_min': -10.0,
            'scalar_expansion_max': 10.0,
            'input_node_channels': num_channels,
            'node_channels': num_channels,
            'time_embedding_size': 32,
            
            'force_edges_to_virtual_nodes': False, # for both encoder and denoiser
            
            
            'encoder': {
                'num_layers': 2,
                'input_sphere_channels': num_channels,
                'sphere_channels': num_channels,
                
                'attn_hidden_channels': 24,
                'num_heads': 2,
                'attn_alpha_channels': 24,
                'attn_value_channels': 24,
                'ffn_hidden_channels': 32,
                
                'lmax_list': [1],
                'mmax_list': [1],
                'grid_resolution': 16,
                'cutoff': 5.0,
                'max_neighbors': 1000000, # essentially infinite
                
                'num_sphere_samples': 128,
                'edge_channels': 128,
            }, 
            
            
            'denoiser': {
            
                'output_node_channels': 1, # denoised coulombic potential / partial charge
                
                'MLP_hidden_dim': 64,
                'num_MLP_hidden_layers': 2,
                
                'use_e3nn': True,
                'e3nn': {
                    'lmax_list': [1],
                    'mmax_list': [1],
                    'ffn_hidden_channels': 32,
                    'grid_resolution': 16,
                },
                
                'use_egnn_positions_update': False,
                'egnn': {
                    'normalize_egnn_vectors': True,
                    'distance_expansion_dim': 32,
                    'num_MLP_hidden_layers': 2,
                    'MLP_hidden_dim': 64,
                },
            
            },
            
        },
        
    },
    
    
    'x4': {
        'decoder': {
        
            'input_node_channels': num_pharmacophore_types,
            'node_channels': num_channels,
            'time_embedding_size': 32,
            
            'force_edges_to_virtual_nodes': True, # for both encoder and denoiser
            
            'encoder': {
                'num_layers': 2,
                'input_sphere_channels': num_channels,
                'sphere_channels': num_channels,
                
                'attn_hidden_channels': 24,
                'num_heads': 2,
                'attn_alpha_channels': 24,
                'attn_value_channels': 24,
                'ffn_hidden_channels': 32,
                
                'lmax_list': [1],
                'mmax_list': [1],
                'grid_resolution': 16,
                'cutoff': 5.0,
                'max_neighbors': 1000000, # essentially infinite
                
                'num_sphere_samples': 128,
                'edge_channels': 128,
            }, 
            
            
            'denoiser': {
            
                'output_node_channels': num_pharmacophore_types, # must equal params['x4']['decoder']['input_node_channels']
                
                'MLP_hidden_dim': 64,
                'num_MLP_hidden_layers': 2,
                
                'use_e3nn': True, # ONLY RELEVANT FOR DENOISING POSITIONS; denoising directions use e3nn automatically
                'e3nn': {
                    'lmax_list': [1],
                    'mmax_list': [1],
                    'ffn_hidden_channels': 32,
                    'grid_resolution': 16,
                },
                
                'use_egnn_positions_update': False, # ONLY RELEVANT FOR DENOISING POSITIONS
                'egnn': {
                    'normalize_egnn_vectors': True,
                    'distance_expansion_dim': 32,
                    'num_MLP_hidden_layers': 2,
                    'MLP_hidden_dim': 64,
                },
            
            },
            
        },
        
    },

}


noise_schedule_dict = {}

T = 400
ts = np.arange(1, T + 1)

beta_min = 0.001 / (T//100)
beta_max = 0.35 / (T//100)
beta_ts_linear = beta_min + ts / T * (beta_max - beta_min) # variance schedule used by RFDiffusion for translations

# (slightly adjusted) cosine schedule, introduced by https://arxiv.org/pdf/2102.09672
ts_ = np.arange(0, T + 1)
s = 0.008
f_ts = np.cos(np.pi/2.1 * ((ts_/ (T+1)) + s)/(1. + s) )**2.0
f_ts = f_ts / f_ts[0]
f_ts = np.clip(f_ts, 0.0001, 0.9999)
beta_ts_cosine = (1 - f_ts[1:]/f_ts[0:-1])
beta_ts_cosine = np.clip(beta_ts_cosine, 0.0001, 0.9999)

beta_ts = 0.65*beta_ts_cosine + 0.35*beta_ts_linear

sigma_ts = beta_ts**0.5 # std deviation schedule
alpha_ts = (1. - sigma_ts**2.0)**0.5

alpha_dash_ts = np.cumprod(alpha_ts)
var_dash_ts = 1. - alpha_dash_ts**2.0
sigma_dash_ts = var_dash_ts**0.5


noise_schedule_dict['x1'] = {
    'T': T,
    'ts': ts,
    'alpha_ts': alpha_ts,
    'sigma_ts': sigma_ts,
    'alpha_dash_ts': alpha_dash_ts,
    'var_dash_ts': var_dash_ts,
    'sigma_dash_ts': sigma_dash_ts,
}

noise_schedule_dict['x2'] = {
    'T': T,
    'ts': ts,
    'alpha_ts': alpha_ts,
    'sigma_ts': sigma_ts,
    'alpha_dash_ts': alpha_dash_ts,
    'var_dash_ts': var_dash_ts,
    'sigma_dash_ts': sigma_dash_ts,
}

noise_schedule_dict['x3'] = {
    'T': T,
    'ts': ts,
    'alpha_ts': alpha_ts,
    'sigma_ts': sigma_ts,
    'alpha_dash_ts': alpha_dash_ts,
    'var_dash_ts': var_dash_ts,
    'sigma_dash_ts': sigma_dash_ts,
}

noise_schedule_dict['x4'] = {
    'T': T,
    'ts': ts,
    'alpha_ts': alpha_ts,
    'sigma_ts': sigma_ts,
    'alpha_dash_ts': alpha_dash_ts,
    'var_dash_ts': var_dash_ts,
    'sigma_dash_ts': sigma_dash_ts,
}

params['noise_schedules'] = noise_schedule_dict
