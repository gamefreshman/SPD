#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShEPhERD推理模块

本模块实现了ShEPhERD（Structured Heterogeneous Pharmacophore Embedding for Rational Drug Design）
模型的推理功能，用于生成分子结构和药效团。主要功能包括：

1. 分子结构的前向扩散过程模拟
2. 条件生成和无条件生成
3. 修复（inpainting）功能，支持部分结构约束
4. 协调化（harmonization）技术
5. 多模态分子表示的生成（原子、表面、静电势、药效团）

作者：ShEPhERD团队
版本：1.0
创建时间：2024
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 3D点云处理库
import open3d 

# ShEPhERD评分工具：点云生成相关功能
from shepherd.shepherd_score_utils.generate_point_cloud import (
    get_atom_coords,  # 获取原子坐标
    get_atomic_vdw_radii,  # 获取原子范德华半径
    get_molecular_surface,  # 生成分子表面
    get_electrostatics,  # 计算静电势
    get_electrostatics_given_point_charges,  # 基于点电荷计算静电势
)

import torch.nn.functional as F  # 用于 softmax 等操作
from shepherd.new_datasets import DiscreteFeatureDiffusion, MarginalUniformTransition   # 导入离散扩散处理器、边际均匀转移矩阵类

# 药效团处理工具
from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores

# 构象生成工具
from shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates

# RDKit化学信息学库
import rdkit
from rdkit.Chem import rdDetermineBonds  # 键序确定功能

# 数值计算和可视化
import numpy as np  # 数值计算
# PyTorch深度学习框架
import torch  # 核心PyTorch库
import torch_geometric  # 图神经网络库
from torch_geometric.nn import radius_graph  # 半径图构建
import torch_scatter  # 散射操作（聚合函数）

# 标准库
import pickle  # 序列化
from copy import deepcopy  # 深拷贝
import os  # 操作系统接口
import multiprocessing  # 多进程处理
from tqdm import tqdm  # 进度条

# PyTorch Lightning框架
import pytorch_lightning as pl  # 轻量级PyTorch训练框架
from pytorch_lightning.callbacks import ModelCheckpoint  # 模型检查点
from pytorch_lightning.loggers import CSVLogger  # CSV日志记录器

# ShEPhERD模块
from shepherd.lightning_module import LightningModule  # Lightning模块
from shepherd.datasets import HeteroDataset  # 异构数据集

# 动态导入模块
import importlib  # 动态导入功能

import numpy as np
import torch
np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)
torch.set_printoptions(profile="full")

def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """ 
    计算离散扩散模型中的后验分布 p(x_{t-1} | x_t, x_0)
    
    这个函数是离散扩散模型去噪过程的核心计算函数，用于计算给定当前状态 x_t 和预测的原始状态 x_0 时，
    前一时间步状态 x_{t-1} 的后验概率分布。
    
    数学原理：
    计算公式为 xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T，对每个可能的 x0 值进行计算
    这基于贝叶斯定理：p(x_{t-1} | x_t, x_0) ∝ p(x_t | x_{t-1}) * p(x_{t-1} | x_0)
    
    参数说明：
    X_t: 当前时间步 t 的特征状态
            - 对于节点特征：形状为 [batch_size, num_nodes, feature_dim_t]
            - 对于边特征：形状为 [batch_size, num_nodes, num_nodes, feature_dim_t]
    Qt: 单步转移矩阵，从时间步 t-1 到 t 的转移概率
        形状为 [batch_size, feature_dim_t-1, feature_dim_t]
    Qsb: 累积转移矩阵，从原始数据 x_0 到时间步 s (即 t-1) 的转移概率
            形状为 [batch_size, original_dim, feature_dim_t-1]
    Qtb: 累积转移矩阵，从原始数据 x_0 到时间步 t 的转移概率
            形状为 [batch_size, original_dim, feature_dim_t]
    
    返回值：
    后验分布概率，形状为 [batch_size, N, original_dim, feature_dim_t-1]
    其中 N 是节点数（对于节点特征）或节点数的平方（对于边特征）
    """
    X_t = X_t.float()
    Qt = Qt.float()
    Qsb = Qsb.float()
    Qtb = Qtb.float()

    Qt_T = Qt.transpose(-1, -2)                 # 转置转移矩阵：[bs, dt, d_t-1]
    left_term = X_t @ Qt_T                      # 矩阵乘法：[bs, N, d_t-1]
    left_term = left_term.unsqueeze(dim=2)      # 增加维度用于广播：[bs, N, 1, d_t-1]
    
    # 步骤3：计算分子部分 - 与累积转移矩阵 Qsb 相乘
    right_term = Qsb.unsqueeze(1)               # 增加维度用于广播：[bs, 1, d0, d_t-1]
    numerator = left_term * right_term          # 逐元素相乘（广播）：[bs, N, d0, d_t-1]
    
    # 步骤4：计算分母部分
    X_t_transposed = X_t.transpose(-1, -2)      # 转置当前状态：[bs, dt, N]
    
    # 计算 Qtb @ X_t^T，这表示从 x_0 到 x_t 的概率与当前状态的乘积
    prod = Qtb @ X_t_transposed                 # 矩阵乘法：[bs, d0, N]
    prod = prod.transpose(-1, -2)               # 转置回来：[bs, N, d0]
    denominator = prod.unsqueeze(-1)            # 增加维度用于除法：[bs, N, d0, 1]
    
    # 步骤5：数值稳定性处理
    # 避免除零错误，将零值替换为很小的正数
    denominator[denominator.abs() < 1e-6] = 1e-6
    
    # 步骤6：计算最终的后验概率分布
    # 这是贝叶斯公式的实现：posterior = likelihood * prior / evidence
    out = numerator / (denominator + 1e-8)              # [bs, N, d0, d_t-1]

    return out

# ============================================================================
# 协调化函数（Harmonization Functions）
# ============================================================================

def initial_sample_discrete_feature_noise(limit_dist_x, limit_dist_e, num_atoms):
    """ 
    从扩散过程的极限分布中采样初始噪声。
    
    此函数现在返回：
    - U_X: 节点的one-hot特征噪声 [num_atoms, num_atom_classes]
    - bond_edge_x: 边的one-hot特征噪声 [num_edges, num_bond_classes]
    - bond_edge_index: 边的连接关系 [2, num_edges]
    """
    
    # 扩展边际分布以匹配节点和边的数量
    x_limit = limit_dist_x[None, :].expand(num_atoms, -1)
    e_limit = limit_dist_e[None, None, :].expand(num_atoms, num_atoms, -1)
    
    # 从分布中采样类别索引
    U_X_indices = x_limit.flatten(end_dim=-2).multinomial(1).reshape(num_atoms)
    U_E_indices = e_limit.flatten(end_dim=-2).multinomial(1).reshape(num_atoms, num_atoms)

    # --- 步骤 1: 生成边的连接关系 (bond_edge_index) ---
    # 创建一个全连接图的上三角矩阵（不含对角线）作为基础连接关系
    # 这就是您在主代码中使用的正确逻辑
    bond_adj = np.triu(np.ones((num_atoms, num_atoms), dtype=int), k=1)
    bond_edge_index = np.stack(bond_adj.nonzero(), axis=0)
    
    # 将 NumPy 数组转换为 PyTorch 张量
    bond_edge_index = torch.from_numpy(bond_edge_index).long()

    # --- 步骤 2: 根据连接关系提取边的特征 (bond_edge_x) ---
    # 从我们采样出的完整的边类型索引矩阵 U_E_indices 中，
    # 只挑选出实际存在连接的那些边的特征。
    # bond_edge_index[0] 是起始节点列表
    # bond_edge_index[1] 是结束节点列表
    bond_edge_x_indices = U_E_indices[bond_edge_index[0], bond_edge_index[1]]

    # --- 步骤 3: 对节点和边特征进行 One-Hot 编码 ---
    U_X = F.one_hot(U_X_indices, num_classes=x_limit.shape[-1]).float()
    bond_edge_x = F.one_hot(bond_edge_x_indices, num_classes=e_limit.shape[-1]).float()
    
    return U_X, bond_edge_x

def forward_jump_parameters(t_start_idx, jump, sigma_ts):
    """
    计算前向跳跃的参数。
    
    在扩散模型中，协调化技术允许模型在不同时间步之间进行"跳跃"，
    而不是逐步进行扩散。这个函数计算跳跃所需的累积参数。
    
    参数
    ----
    t_start_idx : int
        起始时间步索引（从0开始）
    jump : int
        跳跃的时间步长度
    sigma_ts : np.ndarray
        噪声调度的标准差序列
    
    返回值
    ------
    tuple[float, float, int]
        - alpha_dash_ts_[-1]: 累积的信号保持系数
        - sigma_dash_ts_[-1]: 累积的噪声标准差
        - t_end_idx: 结束时间步索引
    
    注释
    ----
    扩散过程的数学原理：
    - alpha_t: 每步信号保持系数 = sqrt(1 - sigma_t^2)
    - alpha_dash_t: 累积信号保持系数 = prod(alpha_0 到 alpha_t)
    - sigma_dash_t: 累积噪声标准差 = sqrt(1 - alpha_dash_t^2)
    """
    t_end_idx = t_start_idx + jump  # 计算结束时间步索引
    sigma_ts_ = sigma_ts[t_start_idx : t_end_idx]  # 提取跳跃区间的标准差调度
    alpha_ts_ = (1. - sigma_ts_** 2.0)**0.5  # 计算每步的信号保持系数
    alpha_dash_ts_ = np.cumprod(alpha_ts_)  # 计算累积信号保持系数
    var_dash_ts_ = 1. - alpha_dash_ts_**2.0  # 计算累积方差
    sigma_dash_ts_ = var_dash_ts_**0.5  # 计算累积标准差
    return alpha_dash_ts_[-1], sigma_dash_ts_[-1], t_end_idx

def forward_jump(x, t_start, jump, sigma_ts, remove_COM_from_noise = False, batch = None, mask = None):
    """
    执行前向跳跃操作，在扩散过程中跳过多个时间步。
    
    这个函数实现了协调化技术的核心操作，允许从时间步t_start直接跳跃到
    t_start+jump，而不需要逐步进行扩散。这在推理过程中可以提高效率。
    
    参数
    ----
    x : torch.Tensor
        输入张量，形状为(N, D)或(N, 3)，其中N是节点数，D是特征维度
    t_start : int
        起始时间步（从1开始计数）
    jump : int
        跳跃的时间步数
    sigma_ts : np.ndarray
        噪声调度的标准差序列
    remove_COM_from_noise : bool, 可选
        是否从噪声中移除质心，默认False
    batch : torch.Tensor, 可选
        批次索引，当remove_COM_from_noise=True时必须提供
    mask : torch.Tensor, 可选
        掩码张量，指示哪些元素参与扩散，默认为全True
    
    返回值
    ------
    tuple[torch.Tensor, int]
        - x_jump: 跳跃后的张量
        - t_end: 结束时间步
    
    异常
    ----
    AssertionError
        当跳跃超过最大时间步T时抛出
    """
    assert t_start + jump <= sigma_ts.shape[0] + 1, "跳跃不能超过最大时间步T"
    
    # 如果未提供掩码，默认所有元素都参与扩散
    if mask is None:
        mask = torch.zeros(x.shape, dtype = torch.long) == 0
    
    # 转换为从0开始的索引
    t_start_idx = t_start - 1
    # 计算跳跃参数
    alpha_jump, sigma_jump, t_end_idx = forward_jump_parameters(t_start_idx, jump, sigma_ts)
    t_end = t_end_idx + 1
    
    # 生成随机噪声
    noise = torch.randn(x.shape)
    
    # 如果需要移除质心（用于坐标扩散）
    if remove_COM_from_noise:
        assert batch is not None, "移除质心时必须提供batch参数"
        assert len(x.shape) == 2, "移除质心时输入必须是2D张量"
        assert noise.shape[1] == 3, "移除质心时特征维度必须是3（坐标）"
        # 计算每个分子的质心并从噪声中减去
        noise = noise - torch_scatter.scatter_mean(noise[mask], batch[mask], dim = 0)[batch]
    
    # 对掩码外的元素不添加噪声
    noise[~mask, ...] = 0.0
    
    # 执行跳跃：x_t+jump = alpha_jump * x_t + sigma_jump * noise
    x_jump = alpha_jump * x + sigma_jump * noise
    # 保持掩码外元素不变
    x_jump[~mask] = x[~mask]
    
    return x_jump, t_end

def forward_trajectory(x, ts, alpha_ts, sigma_ts, remove_COM_from_noise = False, mask = None, deterministic = False):
    """
    模拟前向噪声轨迹，生成扩散过程中每个时间步的状态。
    
    这个函数模拟扩散模型的前向过程，从干净数据x_0开始，
    逐步添加噪声直到指定的时间步，记录整个轨迹。
    
    参数
    ----
    x : torch.Tensor
        初始干净数据，形状为(N, D)
    ts : list or np.ndarray
        时间步序列
    alpha_ts : np.ndarray
        每个时间步的信号保持系数
    sigma_ts : np.ndarray
        每个时间步的噪声标准差
    remove_COM_from_noise : bool, 可选
        是否从噪声中移除质心，默认False
    mask : torch.Tensor, 可选
        掩码张量，指示哪些元素参与扩散
    deterministic : bool, 可选
        是否使用确定性过程（不添加噪声），默认False
    
    返回值
    ------
    dict
        轨迹字典，键为时间步，值为对应时间步的状态张量
        {0: x_0, t1: x_t1, t2: x_t2, ...}
    
    注释
    ----
    前向扩散过程的数学公式：
    x_t = alpha_t * x_{t-1} + sigma_t * epsilon
    其中epsilon是标准高斯噪声
    """
    # 如果未提供掩码，默认所有元素都参与扩散
    if mask is None:
        mask = torch.ones(x.shape[0], device=x.device) == 1.0
    
    # 移除质心时，输入必须是3D坐标
    if remove_COM_from_noise:
        assert x.shape[1] == 3, "移除质心时特征维度必须是3（坐标）"
    
    # 初始化轨迹字典，t=0对应干净数据
    trajectory = {0: x}
    
    x_t = x  # 当前时间步的状态
    
    # 逐步进行前向扩散
    for t_idx, t in enumerate(ts):
        # 获取当前时间步的扩散参数
        alpha_t = alpha_ts[t_idx]  # 信号保持系数
        sigma_t = sigma_ts[t_idx]  # 噪声标准差
        
        # 生成随机噪声
        noise = torch.randn(x.shape, device=x.device)
        
        # 如果需要移除质心（用于坐标扩散）
        if remove_COM_from_noise:
            # 计算掩码内元素的质心并从噪声中减去
            noise = noise - torch.mean(noise[mask], dim = 0)
        
        # 对掩码外的元素不添加噪声
        noise[~mask, ...] = 0.0
        
        # 确定性模式下不添加噪声
        if deterministic:
            noise = 0.0
        
        # 执行前向扩散步骤：x_{t+1} = alpha_t * x_t + sigma_t * noise
        x_t_plus_1 = alpha_t * x_t + sigma_t * noise
        # 保持掩码外元素不变
        x_t_plus_1[~mask, ...] = x_t[~mask, ...]
        
        # 记录当前时间步的状态
        trajectory[t] = x_t_plus_1
        
        # 更新当前状态
        x_t = x_t_plus_1
    
    return trajectory


def inference_sample(
    model_pl,
    batch_size,
    
    N_x1,
    N_x4,
    
    unconditional,
    
    prior_noise_scale = 1.0,
    denoising_noise_scale = 1.0,
    
    inject_noise_at_ts = [],
    inject_noise_scales = [],    
    
    harmonize = False,
    harmonize_ts = [],
    harmonize_jumps = [],
    
    
    # all the below options are only relevant if unconditional is False
    
    inpaint_x2_pos = False,
    inpaint_x3_pos = False,
    inpaint_x3_x = False,
    inpaint_x4_pos = False,
    inpaint_x4_direction = False,
    inpaint_x4_type = False,
    
    stop_inpainting_at_time_x2 = 0.0,
    add_noise_to_inpainted_x2_pos = 0.0,
    
    stop_inpainting_at_time_x3 = 0.0,
    add_noise_to_inpainted_x3_pos = 0.0,
    add_noise_to_inpainted_x3_x = 0.0,
    
    stop_inpainting_at_time_x4 = 0.0,
    add_noise_to_inpainted_x4_pos = 0.0,
    add_noise_to_inpainted_x4_direction = 0.0,
    add_noise_to_inpainted_x4_type = 0.0,
    
    # these are the inpainting targets
    center_of_mass = np.zeros(3),
    surface = np.zeros((75,3)),
    electrostatics = np.zeros(75),
    pharm_types = np.zeros(5, dtype = int),
    pharm_pos = np.zeros((5,3)),
    pharm_direction = np.zeros((5,3)),

    atom_marginals = None,
    bond_marginals = None,
    
):
    """
    运行ShEPhERD推理以采样batch_size个分子。

    参数
    ----
    model_pl : PyTorch Lightning模型

    batch_size : int 单个批次采样的分子数量

    N_x1 : int 需要扩散的原子数量
    N_x4 : int 需要扩散的药效团数量
        # N_x4表示需要生成的总药效团数量,pharm_types包含已知的药效团类型
        # 当N_x4大于pharm_types长度时,表示只对前pharm_types.length个药效团进行类型约束
        # 类型约束指在生成过程中,这些药效团必须保持指定的类型(如供体、受体等)
        # 而剩余的药效团可以自由生成任意允许的类型
        # 剩余的(N_x4 - pharm_types.length)个药效团将在生成过程中自由决定其类型

    unconditional : bool 控制是否进行无条件生成

    prior_noise_scale : float (默认 = 1.0) 先验分布的噪声尺度
    denoising_noise_scale : float (默认 = 1.0) 每个去噪步骤的噪声尺度
    
    inject_noise_at_ts : list[int] (默认 = []) 注入额外噪声的时间步
    inject_noise_scales : list[int] (默认 = []) 上述时间步注入噪声的尺度
     
    harmonize : bool (默认=False) 是否使用协调化
    harmonize_ts : list[int] (默认 = []) 进行协调化的时间步
    harmonize_jumps : list[int] (默认 = []) 协调化的时间步长度

    *以下选项仅在unconditional为False时有效*
    inpaint_x2_pos : bool (默认=False) 控制修复
        注意x2是通过x3隐式建模的

    inpaint_x3_pos : bool (默认=False)
    inpaint_x3_x : bool (默认=False)

    inpaint_x4_pos : bool (默认=False)
    inpaint_x4_direction : bool (默认=False)
    inpaint_x4_type : bool (默认=False)
    
    stop_inpainting_at_time_x2 : float (默认 = 0.0) 停止修复的时间步
        t=0.0表示修复不会停止
    add_noise_to_inpainted_x2_pos : float (默认 = 0.0) 添加到修复值的噪声尺度
    
    stop_inpainting_at_time_x3 : float (默认 = 0.0)
    add_noise_to_inpainted_x3_pos : float (默认 = 0.0)
    add_noise_to_inpainted_x3_x : float (默认 = 0.0)
    
    stop_inpainting_at_time_x4 : float (默认 = 0.0)
    add_noise_to_inpainted_x4_pos : float (默认 = 0.0)
    add_noise_to_inpainted_x4_direction : float (默认 = 0.0)
    add_noise_to_inpainted_x4_type : float (默认 = 0.0)
    
    *以下是修复目标*
    center_of_mass : np.ndarray (3,) (默认 = np.zeros(3)) 如果目标分子未居中则必须提供
    surface : np.ndarray (75,3) (默认 = np.zeros((75,3)) 表面点坐标
    electrostatics : np.ndarray (75,) (默认 = np.zeros(75)) 每个表面点的静电势
    pharm_types : np.ndarray (<=N_x4,) (默认 = np.zeros(5, dtype = int)) 药效团类型
    pharm_pos : np.ndarray (<=N_x4,3) (默认 = np.zeros((5,3))) 药效团位置坐标
    pharm_direction : np.ndarray (<=N_x4,3) (默认 = np.zeros((5,3))) 药效团方向单位向量

    返回值
    -----
    generated_structures : List[Dict]
        输出字典结构如下:
        'x1': {
            'atoms': np.ndarray (N_x1,) 原子序数的整数数组
            'bonds': np.ndarray 每对原子间键类型的数组
            'positions': np.ndarray (N_x1, 3) 原子坐标
        },
        'x2': {
            'positions': np.ndarray (75, 3) 表面点坐标
        },
        'x3': {
            'charges': np.ndarray (75, 3) 表面点的静电势
            'positions': np.ndarray (75, 3) 表面点坐标
        },
        'x4': {
            'types': np.ndarray (N_x4,) 药效团类型的整数数组
            'positions': np.ndarray (N_x4, 3) 药效团坐标
            'directions': np.ndarray (N_x4, 3) 药效团单位向量
        },
    }
    """
    
    # ============================================================================
    # 初始化模型参数和设备配置
    # ============================================================================
    params = model_pl.params  # 获取模型参数配置
    device = model_pl.device  # 获取计算设备（CPU/GPU）

    T = params['noise_schedules']['x1']['ts'].max()  # 获取最大时间步T

    # 确定各模态的节点数量
    N_x2 = params['dataset']['x2']['num_points']  # 表面点数量（默认75）
    if electrostatics is not None:
        N_x3 = len(electrostatics)  # 如果提供了静电势，使用其长度
    else:
        N_x3 = params['dataset']['x3']['num_points']  # 否则使用默认配置
    
    # ============================================================================
    # 初始化离散扩散处理器（新增）
    # ============================================================================
    
    # 初始化原子特征的离散扩散处理器
    x1_atom_diffuser = DiscreteFeatureDiffusion(timesteps=T, marginals=atom_marginals)
    # 初始化键特征的离散扩散处理器
    x1_bond_diffuser = DiscreteFeatureDiffusion(timesteps=T, marginals=bond_marginals)

    # ============================================================================
    # 定义修复（Inpainting）目标
    # ============================================================================

    do_partial_inpainting = False  # 是否进行部分修复标志
    
    # 验证药效团参数的一致性
    assert len(pharm_direction) == len(pharm_pos) and len(pharm_pos) == len(pharm_types), \
        "药效团的方向、位置和类型数组长度必须一致"
    assert N_x4 >= len(pharm_pos), "生成的药效团数量不能少于提供的药效团数量"
    
    # 如果生成的药效团数量大于提供的数量，则进行部分修复
    if N_x4 > len(pharm_pos):
        do_partial_inpainting = True

    # 将所有坐标相对于提供的质心进行居中
    ### GPU CPU 数据格式转换报错

    surface = surface - center_of_mass  # 表面点居中
    pharm_pos = pharm_pos - center_of_mass  # 药效团位置居中
    
    # 为药效团位置添加小量噪声，避免重叠点（重叠会导致编码干净结构时出错）
    pharm_pos = pharm_pos + np.random.randn(*pharm_pos.shape) * 0.01
    
    # 为所有模态添加虚拟节点（用于图神经网络的全局信息聚合）
    surface = np.concatenate([np.array([[0.0, 0.0, 0.0]]), surface], axis = 0)  # 表面虚拟节点
    electrostatics = np.concatenate([np.array([0.0]), electrostatics], axis = 0)  # 静电势虚拟节点
    pharm_types = pharm_types + 1  # 药效团类型索引偏移，为虚拟节点预留0号类型
    pharm_types = np.concatenate([np.array([0]), pharm_types], axis = 0)  # 药效团类型虚拟节点
    pharm_pos = np.concatenate([np.array([[0.0, 0.0, 0.0]]), pharm_pos], axis = 0)  # 药效团位置虚拟节点
    pharm_direction = np.concatenate([np.array([[0.0, 0.0, 0.0]]), pharm_direction], axis = 0)  # 药效团方向虚拟节点
    
    # one-hot-encodings
    pharm_types_one_hot = np.zeros((pharm_types.size, params['dataset']['x4']['max_node_types']))
    pharm_types_one_hot[np.arange(pharm_types.size), pharm_types] = 1
    pharm_types = pharm_types_one_hot
    
    # ============================================================================
    # 特征缩放（根据数据集配置进行标准化）
    # ============================================================================
    electrostatics = electrostatics * params['dataset']['x3']['scale_node_features']  # 静电势特征缩放
    pharm_types = pharm_types * params['dataset']['x4']['scale_node_features']  # 药效团类型特征缩放
    pharm_direction = pharm_direction * params['dataset']['x4']['scale_vector_features']  # 药效团方向特征缩放
    
    # ============================================================================
    # 定义修复（Inpainting）目标张量
    # ============================================================================
    
    # X2模态（表面点）的修复目标
    target_inpaint_x2_pos = torch.as_tensor(surface, dtype = torch.float).to(device)  # 表面点位置
    target_inpaint_x2_mask = torch.zeros(surface.shape[0], dtype = torch.long).to(device)  # 初始化掩码
    target_inpaint_x2_mask[0] = 1  # 虚拟节点标记为1
    target_inpaint_x2_mask = target_inpaint_x2_mask == 0  # 转换为布尔掩码，True表示需要修复的节点

    # X3模态（静电势）的修复目标
    target_inpaint_x3_x = torch.as_tensor(electrostatics, dtype = torch.float).to(device)  # 静电势值
    target_inpaint_x3_pos = torch.as_tensor(surface, dtype = torch.float).to(device)  # 静电势对应的位置
    target_inpaint_x3_mask = torch.zeros(electrostatics.shape[0], dtype = torch.long).to(device)  # 初始化掩码
    target_inpaint_x3_mask[0] = 1  # 虚拟节点标记为1
    target_inpaint_x3_mask = target_inpaint_x3_mask == 0  # 转换为布尔掩码

    # X4模态（药效团）的修复目标
    target_inpaint_x4_x = torch.as_tensor(pharm_types, dtype = torch.float).to(device)  # 药效团类型
    target_inpaint_x4_pos = torch.as_tensor(pharm_pos, dtype = torch.float).to(device)  # 药效团位置
    target_inpaint_x4_direction = torch.as_tensor(pharm_direction, dtype = torch.float).to(device)  # 药效团方向
    target_inpaint_x4_mask = torch.zeros(pharm_types.shape[0], dtype = torch.long).to(device)  # 初始化掩码
    target_inpaint_x4_mask[0] = 1  # 虚拟节点标记为1
    target_inpaint_x4_mask = target_inpaint_x4_mask == 0  # 转换为布尔掩码
    
    # ============================================================================
    # 修复轨迹计算配置
    # ============================================================================
    
    # 设置各模态修复过程的确定性标志（False表示使用随机噪声）
    # 确定性参数控制修复过程中是否添加随机噪声
    # 当设置为True时:
    # - 修复过程变为确定性的,每次运行得到相同结果
    # - 不会添加随机噪声,使用纯粹的去噪过程
    # 当设置为False时:
    # - 修复过程包含随机性,每次运行结果可能不同
    # - 会在去噪过程中添加随机噪声,增加多样性
    deterministic_inpainting_x1 = False  # X1模态（分子）修复的确定性
    deterministic_inpainting_x2 = False  # X2模态（表面点）修复的确定性
    deterministic_inpainting_x3 = False  # X3模态（静电势）修复的确定性
    deterministic_inpainting_x4 = False  # X4模态（药效团）修复的确定性
    
    # ============================================================================
    # 计算各模态的前向扩散轨迹（用于修复过程）
    # ============================================================================
    
    # X2模态（表面点位置）的修复轨迹
    x2_pos_inpainting_trajectory = forward_trajectory(
        x = target_inpaint_x2_pos,  # 目标表面点位置
        ts = params['noise_schedules']['x2']['ts'],  # 时间步序列
        alpha_ts = params['noise_schedules']['x2']['alpha_ts'],  # Alpha参数序列
        sigma_ts = params['noise_schedules']['x2']['sigma_ts'],  # Sigma参数序列
        remove_COM_from_noise = False,  # 不从噪声中移除质心
        mask = target_inpaint_x2_mask,  # 修复掩码
        deterministic = deterministic_inpainting_x2,  # 确定性标志
    )
    
    # X4模态（药效团类型）的修复轨迹
    x4_x_inpainting_trajectory = forward_trajectory(
        x = target_inpaint_x4_x,  # 目标药效团类型
        ts = params['noise_schedules']['x4']['ts'],
        alpha_ts = params['noise_schedules']['x4']['alpha_ts'],
        sigma_ts = params['noise_schedules']['x4']['sigma_ts'],
        remove_COM_from_noise = False,
        mask = target_inpaint_x4_mask,
        deterministic = deterministic_inpainting_x4,
    )
    
    # X4模态（药效团位置）的修复轨迹
    x4_pos_inpainting_trajectory = forward_trajectory(
        x = target_inpaint_x4_pos,  # 目标药效团位置
        ts = params['noise_schedules']['x4']['ts'],
        alpha_ts = params['noise_schedules']['x4']['alpha_ts'],
        sigma_ts = params['noise_schedules']['x4']['sigma_ts'],
        remove_COM_from_noise = False,
        mask = target_inpaint_x4_mask,
        deterministic = deterministic_inpainting_x4,
    )
    
    # X4模态（药效团方向）的修复轨迹
    x4_direction_inpainting_trajectory = forward_trajectory(
        x = target_inpaint_x4_direction,  # 目标药效团方向
        ts = params['noise_schedules']['x4']['ts'],
        alpha_ts = params['noise_schedules']['x4']['alpha_ts'],
        sigma_ts = params['noise_schedules']['x4']['sigma_ts'],
        remove_COM_from_noise = False,
        mask = target_inpaint_x4_mask,
        deterministic = deterministic_inpainting_x4,
    )

    # ============================================================================
    # 条件覆盖选项处理
    # ============================================================================

    # 如果设置为无条件生成，则禁用所有修复选项
    if unconditional:
        inpaint_x4_pos = False      # 禁用药效团位置修复
        inpaint_x4_direction = False # 禁用药效团方向修复
        inpaint_x4_type = False     # 禁用药效团类型修复
        
    # 将停止修复的相对时间转换为绝对时间步
    stop_inpainting_at_time_x2 = int(T*stop_inpainting_at_time_x2)  # X2模态停止修复时间
    stop_inpainting_at_time_x3 = int(T*stop_inpainting_at_time_x3)  # X3模态停止修复时间
    stop_inpainting_at_time_x4 = int(T*stop_inpainting_at_time_x4)  # X4模态停止修复时间
    
    # ============================================================================
    # 在时间步 t=T 初始化状态
    # ============================================================================
    
    include_virtual_node = True  # 包含虚拟节点以提供全局上下文

    # 根据训练时的实际配置，只使用原子类型，不包含电荷类型
    num_atom_types = len(params['dataset']['x1']['atom_types'])  # 原子类型数量
    num_pharm_types = params['dataset']['x4']['max_node_types']  # 药效团类型数量
    
    # ============================================================================
    # X1模态（分子）的图结构配置
    # ============================================================================

    # pos
    virtual_node_pos_x1 = torch.tensor([[0.,0.,0.]], dtype = torch.float).to(device)  # 虚拟节点位置（原点）

    # 节点特征
    virtual_node_x_x1 = torch.zeros(num_atom_types, dtype = torch.float).to(device)  # 虚拟节点特征
    virtual_node_x_x1[0] = 1. * params['dataset']['x1']['scale_atom_features']  # 独热编码，保持不加噪
    virtual_node_x_x1 = virtual_node_x_x1[None, ...]  # 增加批次维度

    # 构建键连接的邻接矩阵（上三角矩阵，避免重复边）
    bond_adj = np.triu(1-np.diag(np.ones(N_x1, dtype = int)))  # 有向图，每个键只包含一条边
    bond_edge_index = np.stack(bond_adj.nonzero(), axis = 0)  # 不包含到虚拟节点的边
    bond_edge_index = bond_edge_index + int(include_virtual_node)  # 为虚拟节点调整索
    bond_edge_index = torch.as_tensor(bond_edge_index, dtype = torch.long).to(device)
    
    # X1模态的批次索引（包含虚拟节点）
    x1_batch = torch.cat([
        torch.ones(N_x1 + int(include_virtual_node), dtype = torch.long) * i for i in range(batch_size)
    ]).to(device)
     
    # ============================================================================
    # X2模态（表面点）的批次配置
    # ============================================================================
    
    # X2模态的批次索引（包含虚拟节点）
    x2_batch = torch.cat([
        torch.ones(N_x2 + int(include_virtual_node), dtype = torch.long) * i for i in range(batch_size)
    ]).to(device)
    
    # ============================================================================
    # X3模态（静电势）的批次配置
    # ============================================================================
    
    # X3模态的批次索引（包含虚拟节点）
    x3_batch = torch.cat([
        torch.ones(N_x3 + int(include_virtual_node), dtype = torch.long) * i for i in range(batch_size)
    ]).to(device)
    
    # ============================================================================
    # X4模态（药效团）的批次配置
    # ============================================================================
    
    # X4模态的批次索引（包含虚拟节点）
    x4_batch = torch.cat([
        torch.ones(N_x4 + int(include_virtual_node), dtype = torch.long) * i for i in range(batch_size)
    ]).to(device)
    
    # X4模态虚拟节点的位置、方向和特征
    virtual_node_direction_x4 = torch.tensor([[0.,0.,0.]], dtype = torch.float).to(device)  # 虚拟节点方向
    virtual_node_pos_x4 = torch.tensor([[0.,0.,0.]], dtype = torch.float).to(device)  # 与X1相同的虚拟节点位置
    virtual_node_x_x4 =  torch.zeros(num_pharm_types, dtype = torch.float).to(device)  # 虚拟节点特征
    virtual_node_x_x4[0] = 1. * params['dataset']['x4']['scale_node_features']  # 独热编码，保持不加噪
    virtual_node_x_x4 = virtual_node_x_x4[None, ...]  # 增加批次维度
    
    # ============================================================================
    # X1模态（分子）的初始状态生成
    # ============================================================================
    
    # 初始化存储列表
    virtual_node_mask_x1 = []  # 虚拟节点掩码列表
    pos_forward_noised_x1 = []  # 加噪位置列表
    x_forward_noised_x1 = []  # 加噪原子特征列表
    bond_edge_x_forward_noised_x1 = []  # 加噪键特征列表
    bond_edge_index_x1 = []  # 键索引列表
    num_nodes_counter = 0  # 节点计数器（用于批次处理）
    
    # 为每个批次样本生成初始状态

    for b in range(batch_size):
        # ========================================================================
        # 生成原子坐标的连续高斯噪声（带最小距离约束）
        # ========================================================================
        x1_pos_T = torch.randn(N_x1, 3, device=device) * prior_noise_scale  # t = T 时的位置
        
        # 确保原子之间有最小距离，避免边向量过于对齐导致数值不稳定
        min_distance = 0.5  # 最小距离阈值（埃）
        max_attempts = 100   # 最大尝试次数
        
        for attempt in range(max_attempts):
            # 计算所有原子对之间的距离矩阵
            distances = torch.cdist(x1_pos_T, x1_pos_T)
            # 排除对角线元素（自身距离为0），设置为大值
            distances = distances + torch.eye(N_x1, device=device) * 1000
            min_dist = torch.min(distances)  # 找到最小距离
            
            if min_dist >= min_distance:
                break  # 满足最小距离要求，退出循环
            
            # 如果距离太近，重新生成坐标
            x1_pos_T = torch.randn(N_x1, 3, device=device) * prior_noise_scale
        
        # 移除质心，确保分子居中
        x1_pos_T = x1_pos_T - torch.mean(x1_pos_T, dim = 0)
        
        # ========================================================================
        # 生成原子特征和键特征的连续高斯噪声
        # ========================================================================
        
        # 原子特征的连续高斯噪声
        x1_x_T, x1_bond_edge_x_T = initial_sample_discrete_feature_noise(atom_marginals, bond_marginals, N_x1)

        x1_bond_edge_x_T = torch.as_tensor(x1_bond_edge_x_T, dtype = torch.long).to(device)
        x1_x_T = torch.as_tensor(x1_x_T, dtype = torch.long).to(device)
        
        # ========================================================================
        # 处理虚拟节点
        # ========================================================================
        
        # 创建虚拟节点掩码
        x1_virtual_node_mask_ = torch.zeros(N_x1 + int(include_virtual_node), dtype = torch.long).to(device)
        
        if include_virtual_node:
            x1_virtual_node_mask_[0] = 1  # 第一个节点为虚拟节点
            x1_virtual_node_mask_ = x1_virtual_node_mask_ == 1  # 转换为布尔掩码
    
            # 将虚拟节点添加到位置和特征张量的开头
            x1_pos_T = torch.cat([virtual_node_pos_x1, x1_pos_T], dim = 0)
            x1_x_T = torch.cat([virtual_node_x_x1, x1_x_T], dim = 0)
        
        # ========================================================================
        # 存储当前批次的数据
        # ========================================================================
        
        pos_forward_noised_x1.append(x1_pos_T)  # 添加位置数据
        x_forward_noised_x1.append(x1_x_T)  # 添加特征数据
        virtual_node_mask_x1.append(x1_virtual_node_mask_)  # 添加虚拟节点掩码
        
        bond_edge_x_forward_noised_x1.append(x1_bond_edge_x_T)  # 添加键特征
        bond_edge_index_x1.append(bond_edge_index + num_nodes_counter)  # 调整键索引
        num_nodes_counter += N_x1 + int(include_virtual_node)  # 更新节点计数器
        
    # 拼接所有批次的X1模态数据
    pos_forward_noised_x1 = torch.cat(pos_forward_noised_x1, dim = 0)  # 位置数据
    x_forward_noised_x1 = torch.cat(x_forward_noised_x1, dim = 0)  # 原子特征数据
    virtual_node_mask_x1 = torch.cat(virtual_node_mask_x1, dim =0)  # 虚拟节点掩码
    bond_edge_x_forward_noised_x1 = torch.cat(bond_edge_x_forward_noised_x1, dim =0)  # 键特征数据
    bond_edge_index_x1 = torch.cat(bond_edge_index_x1, dim =1)  # 键索引
    
    # ============================================================================
    # X4模态（药效团）的初始状态生成
    # ============================================================================
    
    virtual_node_mask_x4 = []  # 虚拟节点掩码列表
    pos_forward_noised_x4 = []  # 加噪位置列表
    direction_forward_noised_x4 = []  # 加噪方向列表
    x_forward_noised_x4 = []  # 加噪药效团特征列表
    
    for b in range(batch_size):
        # 生成药效团位置的连续高斯噪声
        x4_pos_T = torch.randn(N_x4, 3, device=device) * prior_noise_scale  # t = T 时的位置
        # 注意：不从X4中移除质心
        
        # 生成药效团方向的连续高斯噪声
        x4_direction_T = torch.randn(N_x4, 3, device=device) * prior_noise_scale  # t = T 时的方向
        
        # 生成药效团特征的连续高斯噪声
        x4_x_T = torch.randn(N_x4, num_pharm_types).to(device)
        
        # 创建虚拟节点掩码
        x4_virtual_node_mask_ = torch.zeros(N_x4 + int(include_virtual_node), dtype = torch.long).to(device)
        if include_virtual_node:
            x4_virtual_node_mask_[0] = 1  # 第一个节点为虚拟节点
            x4_virtual_node_mask_ = x4_virtual_node_mask_ == 1  # 转换为布尔掩码
    
            # 将虚拟节点数据添加到开头
            x4_pos_T = torch.cat([virtual_node_pos_x4, x4_pos_T], dim = 0)
            x4_direction_T = torch.cat([virtual_node_direction_x4, x4_direction_T], dim = 0)
            x4_x_T = torch.cat([virtual_node_x_x4, x4_x_T], dim = 0)
        
        # 存储当前批次数据
        pos_forward_noised_x4.append(x4_pos_T)
        direction_forward_noised_x4.append(x4_direction_T)
        x_forward_noised_x4.append(x4_x_T)
        virtual_node_mask_x4.append(x4_virtual_node_mask_)
        
    # 拼接所有批次的X4模态数据
    pos_forward_noised_x4 = torch.cat(pos_forward_noised_x4, dim = 0)  # 位置数据
    direction_forward_noised_x4 = torch.cat(direction_forward_noised_x4, dim = 0)  # 方向数据
    x_forward_noised_x4 = torch.cat(x_forward_noised_x4, dim = 0)  # 药效团特征数据
    virtual_node_mask_x4 = torch.cat(virtual_node_mask_x4, dim =0)  # 虚拟节点掩码
    
    # ============================================================================
    # 变量重命名以保持一致性
    # ============================================================================
    
    # X1模态（分子）变量重命名
    x1_pos_t = pos_forward_noised_x1  # 当前时间步的原子位置
    x1_x_t = x_forward_noised_x1  # 当前时间步的原子特征
    x1_bond_edge_x_t = bond_edge_x_forward_noised_x1  # 当前时间步的键特征
    
    # X4模态（药效团）变量重命名
    x4_pos_t = pos_forward_noised_x4  # 当前时间步的药效团位置
    x4_direction_t = direction_forward_noised_x4  # 当前时间步的药效团方向
    x4_x_t = x_forward_noised_x4  # 当前时间步的药效团特征
    
    # ============================================================================
    # 计算各模态的批次节点总数
    # ============================================================================
    
    x1_batch_size_nodes = x1_pos_t.shape[0]  # X1模态总节点数（包含所有批次和虚拟节点）
    x4_batch_size_nodes = x4_pos_t.shape[0]  # X4模态总节点数
    
    # ============================================================================
    # 初始化各模态的当前时间步（从最大噪声时间开始）
    # ============================================================================
    
    x1_t = params['noise_schedules']['x1']['ts'][::-1][0]  # X1模态起始时间步（t=T）
    x2_t = params['noise_schedules']['x2']['ts'][::-1][0]  # X2模态起始时间步（t=T）
    x3_t = params['noise_schedules']['x3']['ts'][::-1][0]  # X3模态起始时间步（t=T）
    x4_t = params['noise_schedules']['x4']['ts'][::-1][0]  # X4模态起始时间步（t=T）
    
    t = x1_t
    # ============================================================================
    # 时间步一致性验证
    # ============================================================================
    
    assert x1_t == x2_t  # 确保X1和X2模态时间步一致
    assert x1_t == x3_t  # 确保X1和X3模态时间步一致
    assert x1_t == x4_t  # 确保X1和X4模态时间步一致
    
    # ============================================================================
    # 初始修复（Inpainting）条件处理
    # ============================================================================
    
    # X4模态修复（位置、方向和类型）
    if (x4_t > stop_inpainting_at_time_x4):
        if inpaint_x4_pos:
            # 药效团位置修复
            x4_pos_t_inpaint = x4_pos_inpainting_trajectory[x4_t].repeat(batch_size, 1, 1)
            if do_partial_inpainting:
                # 部分修复：只替换前N个药效团位置
                x4_pos_t = x4_pos_t.reshape(batch_size, -1, 3)
                x4_pos_t[:, :x4_pos_t_inpaint.shape[1]] = x4_pos_t_inpaint
                x4_pos_t = x4_pos_t.reshape(-1, 3)
            else:
                # 完全修复：替换所有药效团位置
                x4_pos_t = x4_pos_t_inpaint.reshape(-1,3)
        if inpaint_x4_direction:
            # 药效团方向修复
            x4_direction_t_inpaint = x4_direction_inpainting_trajectory[x4_t].repeat(batch_size, 1, 1)
            if do_partial_inpainting:
                # 部分修复：只替换前N个药效团方向
                x4_direction_t = x4_direction_t.reshape(batch_size, -1, 3)
                x4_direction_t[:, :x4_direction_t_inpaint.shape[1]] = x4_direction_t_inpaint
                x4_direction_t = x4_direction_t.reshape(-1, 3)
            else:
                # 完全修复：替换所有药效团方向
                x4_direction_t = x4_direction_t_inpaint.reshape(-1,3)
        if inpaint_x4_type:
            # 药效团类型修复
            x4_x_t_inpaint = x4_x_inpainting_trajectory[x4_t].repeat(batch_size, 1, 1)
            if do_partial_inpainting:
                # 部分修复：只替换前N个药效团类型
                x4_x_t = x4_x_t.reshape(batch_size, -1, num_pharm_types)
                x4_x_t[:, :x4_x_t_inpaint.shape[1]] = x4_x_t_inpaint
                x4_x_t = x4_x_t.reshape(-1, num_pharm_types)
            else:
                # 完全修复：替换所有药效团类型
                x4_x_t = x4_x_t_inpaint.reshape(-1, num_pharm_types)
        
    
    # ============================================================================
    # 主要去噪循环初始化
    # ============================================================================
    
    # 创建进度条（包含协调跳跃的额外步数）
    pbar = tqdm(total= T + sum(harmonize_jumps) * int(harmonize), position=0, leave=True)
    
    # ============================================================================
    # 主要去噪循环（从t=T到t=0）
    # ============================================================================
    
    while t > 0:
        
        # ============================================================================
        # 设置当前时间步输入
        # ============================================================================
        
        x1_t = t  # X1模态当前时间步
        x4_t = t  # X4模态当前时间步
        
        # ============================================================================
        # 协调跳跃处理（Harmonize Jump）
        # ============================================================================
        
        if (harmonize) and (t == harmonize_ts[0]):
            # 在指定时间步执行协调跳跃，同步各模态的噪声水平
            harmonize_ts.pop(0)  # 移除当前协调时间步
            if len(harmonize_ts) == 0:
                harmonize = False  # 所有协调步骤完成
            harmonize_jump = harmonize_jumps.pop(0)  # 获取当前跳跃步数
            
            # 获取各模态的噪声调度参数
            x1_sigma_ts = params['noise_schedules']['x1']['sigma_ts']
            x4_sigma_ts = params['noise_schedules']['x4']['sigma_ts']
            
            # X1模态协调跳跃（分子）
            x1_pos_t, x1_t_jump = forward_jump(x1_pos_t, x1_t, harmonize_jump, x1_sigma_ts, remove_COM_from_noise = True, batch = x1_batch, mask = ~virtual_node_mask_x1)
            x1_x_t, x1_t_jump = forward_jump(x1_x_t, x1_t, harmonize_jump, x1_sigma_ts, remove_COM_from_noise = False, batch = x1_batch, mask = ~virtual_node_mask_x1)
            x1_bond_edge_x_t, x1_t_jump = forward_jump(x1_bond_edge_x_t, x1_t, harmonize_jump, x1_sigma_ts, remove_COM_from_noise = False, batch = None, mask = None)
            
            # X4模态协调跳跃（药效团）
            x4_pos_t, x4_t_jump = forward_jump(x4_pos_t, x4_t, harmonize_jump, x4_sigma_ts, remove_COM_from_noise = False, batch = x4_batch, mask = ~virtual_node_mask_x4)
            x4_direction_t, x4_t_jump = forward_jump(x4_direction_t, x4_t, harmonize_jump, x4_sigma_ts, remove_COM_from_noise = False, batch = x4_batch, mask = ~virtual_node_mask_x4)
            x4_x_t, x4_t_jump = forward_jump(x4_x_t, x4_t, harmonize_jump, x4_sigma_ts, remove_COM_from_noise = False, batch = x4_batch, mask = ~virtual_node_mask_x4)
    
            # 更新所有模态的时间步
            x1_t = x1_t_jump
            x4_t = x4_t_jump
            
            # 验证时间步一致性
            assert x1_t == x2_t
            assert x1_t == x3_t
            assert x1_t == x4_t
            t = x1_t  # 更新全局时间步
        
        # ============================================================================
        # 循环内修复条件处理（每个时间步都检查）
        # ============================================================================
            
        # X4模态修复（药效团位置、方向和类型）
        if (x4_t > stop_inpainting_at_time_x4):
            if inpaint_x4_pos:
                # 药效团位置修复
                x4_pos_t_inpaint = torch.cat([x4_pos_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
                noise = torch.randn_like(x4_pos_t)
                noise[virtual_node_mask_x4] = 0.0  # 虚拟节点不添加噪声
                if do_partial_inpainting:
                    # 部分修复模式：只修复前N个药效团位置
                    x4_pos_t_inpaint = x4_pos_t_inpaint.reshape(batch_size, -1, 3)
                    noise = noise.reshape(batch_size, -1, 3)[:, :x4_pos_t_inpaint.shape[1]]

                    x4_pos_t_inpaint = x4_pos_t_inpaint + add_noise_to_inpainted_x4_pos * noise
                    x4_pos_t = x4_pos_t.reshape(batch_size, -1, 3)
                    x4_pos_t[:, :x4_pos_t_inpaint.shape[1]] = x4_pos_t_inpaint  # 替换前N个位置
                    x4_pos_t = x4_pos_t.reshape(-1, 3)
                else:
                    # 完全修复模式：替换所有药效团位置
                    x4_pos_t_inpaint = x4_pos_t_inpaint + add_noise_to_inpainted_x4_pos * noise
                    x4_pos_t = x4_pos_t_inpaint

            if inpaint_x4_direction:
                # 药效团方向修复
                x4_direction_t_inpaint = torch.cat([x4_direction_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
                noise = torch.randn_like(x4_direction_t)
                noise[virtual_node_mask_x4] = 0.0  # 虚拟节点不添加噪声
                if do_partial_inpainting:
                    # 部分修复模式：只修复前N个药效团方向
                    x4_direction_t_inpaint = x4_direction_t_inpaint.reshape(batch_size, -1, 3)
                    noise = noise.reshape(batch_size, -1, 3)[:, :x4_direction_t_inpaint.shape[1]]

                    x4_direction_t_inpaint = x4_direction_t_inpaint + add_noise_to_inpainted_x4_direction * noise
                    x4_direction_t = x4_direction_t.reshape(batch_size, -1, 3)
                    x4_direction_t[:, :x4_direction_t_inpaint.shape[1]] = x4_direction_t_inpaint  # 替换前N个方向
                    x4_direction_t = x4_direction_t.reshape(-1, 3)
                else:
                    # 完全修复模式：替换所有药效团方向
                    x4_direction_t_inpaint = x4_direction_t_inpaint + add_noise_to_inpainted_x4_direction * noise
                    x4_direction_t = x4_direction_t_inpaint
            if inpaint_x4_type:
                # 药效团类型修复
                x4_x_t_inpaint = torch.cat([x4_x_inpainting_trajectory[x4_t] for _ in range(batch_size)], dim = 0)
                noise = torch.randn_like(x4_x_t)
                noise[virtual_node_mask_x4] = 0.0  # 虚拟节点不添加噪声
                if do_partial_inpainting:
                    # 部分修复模式：只修复前N个药效团类型
                    x4_x_t_inpaint = x4_x_t_inpaint.reshape(batch_size, -1, num_pharm_types)
                    noise = noise.reshape(batch_size, -1, num_pharm_types)[:, :x4_x_t_inpaint.shape[1]]

                    x4_x_t_inpaint = x4_x_t_inpaint + add_noise_to_inpainted_x4_type * noise
                    x4_x_t = x4_x_t.reshape(batch_size, -1, num_pharm_types)
                    x4_x_t[:, :x4_x_t_inpaint.shape[1]] = x4_x_t_inpaint  # 替换前N个类型
                    x4_x_t = x4_x_t.reshape(-1, num_pharm_types)
                else:
                    # 完全修复模式：替换所有药效团类型
                    x4_x_t_inpaint = x4_x_t_inpaint + add_noise_to_inpainted_x4_type * noise
                    x4_x_t = x4_x_t_inpaint
        
        
        # ============================================================================
        # 获取各模态当前时间步的噪声调度参数
        # ============================================================================
        
        # X1模态（分子）噪声参数
        x1_t_idx = np.where(params['noise_schedules']['x1']['ts'] == x1_t)[0][0]  # 当前时间步索引
        x1_alpha_t = params['noise_schedules']['x1']['alpha_ts'][x1_t_idx]  # 累积噪声系数α_t ###
        x1_sigma_t = params['noise_schedules']['x1']['sigma_ts'][x1_t_idx]  # 噪声标准差σ_t
        x1_alpha_dash_t = params['noise_schedules']['x1']['alpha_dash_ts'][x1_t_idx]  # 修正α系数
        x1_var_dash_t = params['noise_schedules']['x1']['var_dash_ts'][x1_t_idx]  # 修正方差
        x1_sigma_dash_t = params['noise_schedules']['x1']['sigma_dash_ts'][x1_t_idx]  # 修正标准差
        
        # X1模态下一时间步噪声参数（用于去噪计算）
        if x1_t_idx > 0:
            x1_t_1 = x1_t - 1  # 下一时间步（t-1）
            x1_t_1_idx = x1_t_idx - 1  # 下一时间步索引
            x1_alpha_t_1 = params['noise_schedules']['x1']['alpha_ts'][x1_t_1_idx]  # α_{t-1}
            x1_sigma_t_1 = params['noise_schedules']['x1']['sigma_ts'][x1_t_1_idx]  # σ_{t-1}
            x1_alpha_dash_t_1 = params['noise_schedules']['x1']['alpha_dash_ts'][x1_t_1_idx]  # 修正α_{t-1}
            x1_var_dash_t_1 = params['noise_schedules']['x1']['var_dash_ts'][x1_t_1_idx]  # 修正方差_{t-1}
            x1_sigma_dash_t_1 = params['noise_schedules']['x1']['sigma_dash_ts'][x1_t_1_idx]  # 修正σ_{t-1}

        # X4模态（药效团）噪声参数
        x4_t_idx = np.where(params['noise_schedules']['x4']['ts'] == x4_t)[0][0]  # 当前时间步索引
        x4_alpha_t = params['noise_schedules']['x4']['alpha_ts'][x4_t_idx]  # 累积噪声系数α_t
        x4_sigma_t = params['noise_schedules']['x4']['sigma_ts'][x4_t_idx]  # 噪声标准差σ_t
        x4_alpha_dash_t = params['noise_schedules']['x4']['alpha_dash_ts'][x4_t_idx]  # 修正α系数
        x4_var_dash_t = params['noise_schedules']['x4']['var_dash_ts'][x4_t_idx]  # 修正方差
        x4_sigma_dash_t = params['noise_schedules']['x4']['sigma_dash_ts'][x4_t_idx]  # 修正标准差
        
        # ==================== 获取下一时间步的噪声参数 ====================
        # 为X4模态获取下一时间步(t-1)的噪声调度参数
        # 这些参数用于计算去噪过程中的噪声系数
        if x4_t_idx > 0:  # 确保不是最后一个时间步
            x4_t_1 = x4_t - 1  # 下一时间步值
            x4_t_1_idx = x4_t_idx - 1  # 下一时间步索引
            x4_alpha_t_1 = params['noise_schedules']['x4']['alpha_ts'][x4_t_1_idx]  # 下一步的alpha参数
            x4_sigma_t_1 = params['noise_schedules']['x4']['sigma_ts'][x4_t_1_idx]  # 下一步的sigma参数
            x4_alpha_dash_t_1 = params['noise_schedules']['x4']['alpha_dash_ts'][x4_t_1_idx]  # 下一步的累积alpha参数
            x4_var_dash_t_1 = params['noise_schedules']['x4']['var_dash_ts'][x4_t_1_idx]  # 下一步的累积方差参数
            x4_sigma_dash_t_1 = params['noise_schedules']['x4']['sigma_dash_ts'][x4_t_1_idx]  # 下一步的累积sigma参数
        
        
        
        # ==================== 准备当前时间步的模型输入数据 ====================
        # 为每个批次样本创建当前时间步的张量，用于模型推理
        x1_timestep = torch.tensor([x1_t] * batch_size)  # X1模态的时间步张量
        x4_timestep = torch.tensor([x4_t] * batch_size)  # X4模态的时间步张量
        
        # 为每个批次样本创建当前时间步的噪声参数张量
        # 这些参数用于模型内部的噪声调度计算
        x1_sigma_dash_t_ = torch.tensor([x1_sigma_dash_t] * batch_size, dtype = torch.float)  # X1累积sigma参数
        x1_alpha_dash_t_ = torch.tensor([x1_alpha_dash_t] * batch_size, dtype = torch.float)  # X1累积alpha参数
        
        x4_sigma_dash_t_ = torch.tensor([x4_sigma_dash_t] * batch_size, dtype = torch.float)  # X4累积sigma参数
        x4_alpha_dash_t_ = torch.tensor([x4_alpha_dash_t] * batch_size, dtype = torch.float)  # X4累积alpha参数
        
        
        
        # ==================== 构建模型输入字典 ====================
        # 获取模型所在的设备（CPU或GPU），确保数据在正确设备上
        device = model_pl.model.device
        input_dict = {}  # 初始化模型输入字典
        input_dict['device'] = device  # 设置设备信息
        input_dict['dtype'] = torch.float32  # 设置数据类型为32位浮点数
        # X1模态（分子结构）的输入数据配置
        input_dict['x1'] = {
            # 解码器使用经过前向加噪的结构数据进行去噪预测
            'decoder': {
                'pos': x1_pos_t.to(device),  # 当前时间步的原子坐标（已加噪）
                'x': x1_x_t.to(device),  # 当前时间步的原子特征（已加噪）
                'batch': x1_batch.to(device),  # 批次索引，用于区分不同分子
                
                'bond_edge_x': x1_bond_edge_x_t.to(device),  # 当前时间步的键特征（已加噪）
                'bond_edge_index': bond_edge_index_x1.to(device),  # 键的连接索引
                
                'timestep': x1_timestep.to(device),  # 当前时间步信息
                'sigma_dash_t': x1_sigma_dash_t_.to(device),  # 累积噪声标准差参数
                'alpha_dash_t': x1_alpha_dash_t_.to(device),  # 累积信号保留参数
    
                'virtual_node_mask': virtual_node_mask_x1.to(device),  # 虚拟节点掩码
                
            },
        }   

        ### x2 x3 input 处理 开始
        virtual_node_mask_x2 = []
        pos_forward_noised_x2 = []
        x_forward_noised_x2 = []
        
        for b in range(batch_size):
            x2_x_T = torch.zeros((N_x2 + int(include_virtual_node), 2))
            x2_x_T[:,0] = 1
            if include_virtual_node:
                x2_x_T[0,0] = 0
                x2_x_T[0,1] = 1
                
            x2_virtual_node_mask_ = torch.zeros(N_x2 + int(include_virtual_node), dtype = torch.long)

            pos_forward_noised_x2.append(target_inpaint_x2_pos)
            x_forward_noised_x2.append(x2_x_T)
            virtual_node_mask_x2.append(x2_virtual_node_mask_)

        pos_forward_noised_x2 = torch.cat(pos_forward_noised_x2, dim = 0)
        x_forward_noised_x2 = torch.cat(x_forward_noised_x2, dim = 0)
        virtual_node_mask_x2 = torch.cat(virtual_node_mask_x2, dim =0)

        x2_pos_t = pos_forward_noised_x2
        x2_x_t = x_forward_noised_x2
        
        x2_sigma_dash_t_ = torch.tensor([0] * batch_size, dtype = torch.float) # 不模糊 0
        x2_alpha_dash_t_ = torch.tensor([1] * batch_size, dtype = torch.float) # 很清晰 1
        
        ###
        
        pos_forward_noised_x3 = []
        x_forward_noised_x3 = []
        virtual_node_mask_x3 = []
        for b in range(batch_size):
        
            x3_virtual_node_mask_ = torch.zeros(N_x3 + int(include_virtual_node), dtype = torch.long)
            
            pos_forward_noised_x3.append(target_inpaint_x3_pos)
            x_forward_noised_x3.append(target_inpaint_x3_x)
            virtual_node_mask_x3.append(x3_virtual_node_mask_)

        pos_forward_noised_x3 = torch.cat(pos_forward_noised_x3, dim = 0)
        x_forward_noised_x3 = torch.cat(x_forward_noised_x3, dim = 0)
        virtual_node_mask_x3 = torch.cat(virtual_node_mask_x3, dim =0)

        x3_pos_t = pos_forward_noised_x3
        x3_x_t = x_forward_noised_x3

        x3_sigma_dash_t_ = torch.tensor([0] * batch_size, dtype = torch.float)
        x3_alpha_dash_t_ = torch.tensor([1] * batch_size, dtype = torch.float) 
        
        # X2模态（口袋结构）的输入数据配置
        input_dict['x2'] =  {
            # 解码器使用经过前向加噪的口袋结构数据
            'decoder': {
                'pos': x2_pos_t.to(device),  # 当前时间步的口袋原子坐标（已加噪）
                'x': x2_x_t.to(device),  # 当前时间步的口袋原子特征（已加噪）
                'batch': x2_batch.to(device),  # 批次索引，用于区分不同口袋
                
                'timestep': x1_timestep.to(device),  # 当前时间步信息 随 x1
                'sigma_dash_t': x2_sigma_dash_t_.to(device),  # 累积噪声标准差参数
                'alpha_dash_t': x2_alpha_dash_t_.to(device),  # 累积信号保留参数
                
                'virtual_node_mask': virtual_node_mask_x2.to(device),  # 虚拟节点掩码
                
            },
        }
        
        # X3模态（静电势）的输入数据配置
        input_dict['x3'] =  {
            # 解码器使用经过前向加噪的静电势数据
            'decoder': {
                'pos': x3_pos_t.to(device),  # 当前时间步的静电势点坐标（已加噪）
                'x': x3_x_t.to(device),  # 当前时间步的静电势值（已加噪）
                'batch': x3_batch.to(device),  # 批次索引，用于区分不同静电势场
                
                'timestep': x1_timestep.to(device),  # 当前时间步信息 随 x1
                'sigma_dash_t': x3_sigma_dash_t_.to(device),  # 累积噪声标准差参数
                'alpha_dash_t': x3_alpha_dash_t_.to(device),  # 累积信号保留参数
                
                'virtual_node_mask': virtual_node_mask_x3.to(device),  # 虚拟节点掩码
                
            },
        }

        ### x2 x3 input 处理 结束
        
        # X4模态（药效团）的输入数据配置
        input_dict['x4'] =  {
            # 解码器使用经过前向加噪的药效团数据
            'decoder': {
                'x': x4_x_t.to(device),  # 当前时间步的药效团类型特征（已加噪）
                'pos': x4_pos_t.to(device),  # 当前时间步的药效团位置坐标（已加噪）
                'direction': x4_direction_t.to(device),  # 当前时间步的药效团方向向量（已加噪）
                'batch': x4_batch.to(device),  # 批次索引，用于区分不同药效团
                
                'timestep': x4_timestep.to(device),  # 当前时间步信息
                'sigma_dash_t': x4_sigma_dash_t_.to(device),  # 累积噪声标准差参数
                'alpha_dash_t': x4_alpha_dash_t_.to(device),  # 累积信号保留参数
                
                'virtual_node_mask': virtual_node_mask_x4.to(device),  # 虚拟节点掩码
                
            },
        }

# =========================================================

        # ==================== 神经网络噪声预测 ====================
        # 使用训练好的扩散模型预测当前时间步的噪声
        # torch.no_grad()确保推理过程不计算梯度，节省内存和计算资源
        with torch.no_grad():
            _, output_dict = model_pl.model.forward(input_dict)  # 前向传播获取噪声预测结果
        
        # ==================== 提取X1模态的噪声预测结果 ====================
        # 从模型输出中提取X1模态（分子结构）的各项噪声预测
        # 提取并打印原子特征噪声预测
        # 从模型输出中获取原子特征噪声预测结果

        x1_x_out = output_dict['x1']['decoder']['denoiser']['x_out'].detach()  # 原子特征噪声预测

        # 提取并打印键特征噪声预测
        x1_bond_edge_x_out = output_dict['x1']['decoder']['denoiser']['bond_edge_x_out'].detach()  # 键特征噪声预测

        # 提取并打印原子坐标噪声预测
        x1_pos_out = output_dict['x1']['decoder']['denoiser']['pos_out'].detach()  # 原子坐标噪声预测
        
        # 从预测的位置噪声中移除质心（COM），保持分子的平移不变性
        x1_pos_out = x1_pos_out - torch_scatter.scatter_mean(x1_pos_out[~virtual_node_mask_x1], x1_batch[~virtual_node_mask_x1], dim = 0)[x1_batch]
        
        # 将虚拟节点的预测噪声设为零，确保虚拟节点不参与实际计算
        x1_x_out[virtual_node_mask_x1, :] = 0.0  # 虚拟节点特征噪声置零
        x1_pos_out[virtual_node_mask_x1, :] = 0.0  # 虚拟节点位置噪声置零
        
        # ==================== 提取X4模态的噪声预测结果 ====================
        # 处理X4模态（药效团）的类型、位置和方向噪声预测
        x4_x_out = output_dict['x4']['decoder']['denoiser']['x_out']  # 药效团类型噪声预测
        x4_pos_out = output_dict['x4']['decoder']['denoiser']['pos_out']  # 药效团位置噪声预测
        x4_direction_out = output_dict['x4']['decoder']['denoiser']['direction_out']  # 药效团方向噪声预测
        if x4_x_out is not None:  # 检查模型是否输出了X4的预测结果
            x4_pos_out = x4_pos_out.detach() # 注意：X4不移除质心，保持药效团的绝对位置
            x4_direction_out = x4_direction_out.detach()# 方向向量噪声预测
            x4_x_out = x4_x_out.detach()  # 类型特征噪声预测
            x4_x_out = x4_x_out.squeeze()  # 移除多余的维度
            
            # 将虚拟节点的预测噪声设为零
            x4_pos_out[virtual_node_mask_x4, :] = 0.0  # 虚拟节点位置噪声置零
            x4_direction_out[virtual_node_mask_x4, :] = 0.0  # 虚拟节点方向噪声置零
            x4_x_out[virtual_node_mask_x4] = 0.0  # 虚拟节点类型噪声置零
        
        else:
            # 如果没有预测输出，创建零张量
            x4_pos_out = torch.zeros_like(x4_pos_t)
            x4_direction_out = torch.zeros_like(x4_direction_t)
            x4_x_out = torch.zeros_like(x4_x_t)

        # ==================== 生成去噪过程中的随机噪声 - X1模态 ====================
        # 为X1模态（分子结构）生成去噪步骤中需要添加的随机噪声
        # 在去噪过程中添加随机噪声的原因:
        # 1. 防止模型陷入确定性轨迹,增加采样多样性
        # 2. 通过注入随机性来打破对称性,避免生成重复或对称的结构
        # 3. 模拟真实分子的热力学涨落,使生成的构象更加自然
        # 4. 有助于模型探索更大的构象空间,发现新颖结构
        x1_pos_epsilon = torch.randn(x1_batch_size_nodes, 3, device=device)  # 生成位置随机噪声
        # 从位置噪声中移除质心，保持分子的平移不变性
        x1_pos_epsilon = x1_pos_epsilon - torch_scatter.scatter_mean(x1_pos_epsilon[~virtual_node_mask_x1], x1_batch[~virtual_node_mask_x1], dim = 0)[x1_batch]
        x1_pos_epsilon[virtual_node_mask_x1, :] = 0.0  # 虚拟节点位置噪声置零
        
        # 计算X1模态的噪声系数，用于控制添加噪声的强度
        x1_c_t = (x1_sigma_t * x1_sigma_dash_t_1) / (x1_sigma_dash_t) if x1_t_idx > 0 else 0
        x1_c_t = x1_c_t * denoising_noise_scale  # 应用去噪噪声缩放因子
        
        # ==================== 生成去噪过程中的随机噪声 - X4模态 ====================
        # 为X4模态（药效团）生成去噪步骤中需要添加的随机噪声
        x4_pos_epsilon = torch.randn(x4_batch_size_nodes,3)  # 生成药效团位置随机噪声
        x4_pos_epsilon[virtual_node_mask_x4, :] = 0.0  # 虚拟节点位置噪声置零
        
        x4_direction_epsilon = torch.randn(x4_batch_size_nodes, 3)  # 生成药效团方向随机噪声
        x4_direction_epsilon[virtual_node_mask_x4, :] = 0.0  # 虚拟节点方向噪声置零
        
        x4_x_epsilon = torch.randn(x4_batch_size_nodes, num_pharm_types)  # 生成药效团类型随机噪声
        x4_x_epsilon[virtual_node_mask_x4, ...] = 0.0  # 虚拟节点类型噪声置零
       
        # 计算X4模态的噪声系数
        x4_c_t = (x4_sigma_t * x4_sigma_dash_t_1) / (x4_sigma_dash_t) if x4_t_idx > 0 else 0
        x4_c_t = x4_c_t * denoising_noise_scale  # 应用去噪噪声缩放因子
        
        
        # ==================== 额外噪声注入（对称性破缺） ====================
        # 在特定时间步注入额外噪声，用于破坏对称性或增加样本多样性
        x1_c_t_injected = x1_c_t  # X1模态的注入噪声系数
        x4_c_t_injected = x4_c_t  # X4模态的注入噪声系数
        if len(inject_noise_at_ts) > 0:  # 检查是否有需要注入噪声的时间步
            if t == inject_noise_at_ts[0]:  # 如果当前时间步需要注入噪声
                inject_noise_at_ts.pop(0)  # 移除已处理的时间步
                inject_noise_scale = inject_noise_scales.pop(0)  # 获取对应的噪声缩放因子
                
                # 额外噪声主要应用于位置，用于破坏对称性
                x1_c_t_injected = x1_c_t + inject_noise_scale  # 增强X1位置噪声
                x4_c_t_injected = x4_c_t + inject_noise_scale  # 增强X4位置噪声

        # 将所有变量转移到同一设备(GPU)上
        # 将numpy float64数据转换为torch tensor并移至GPU
        # 将X1模态的参数转换为张量并移至指定设备
        x1_alpha_t = torch.tensor(x1_alpha_t, dtype=torch.float32).to(device)
        x1_var_dash_t = torch.tensor(x1_var_dash_t, dtype=torch.float32).to(device) 
        x1_sigma_dash_t = torch.tensor(x1_sigma_dash_t, dtype=torch.float32).to(device)
        x1_c_t_injected = torch.tensor(x1_c_t_injected, dtype=torch.float32).to(device)
        
        # 将X4模态的参数转换为张量并移至指定设备
        x4_alpha_t = torch.tensor(x4_alpha_t, dtype=torch.float32).to(device)
        x4_var_dash_t = torch.tensor(x4_var_dash_t, dtype=torch.float32).to(device)
        x4_sigma_dash_t = torch.tensor(x4_sigma_dash_t, dtype=torch.float32).to(device)
        x4_c_t_injected = torch.tensor(x4_c_t_injected, dtype=torch.float32).to(device)
        
        x4_pos_epsilon = x4_pos_epsilon.to(device)
        x4_direction_epsilon = x4_direction_epsilon.to(device)
        x4_x_epsilon = x4_x_epsilon.to(device)

        # ==================== 反向去噪步骤 - X1模态 ====================

        x1_pos_t_1 = ((1. / x1_alpha_t) * x1_pos_t)  - ((x1_var_dash_t/(x1_alpha_t * x1_sigma_dash_t)) * x1_pos_out)  +  (x1_c_t_injected * x1_pos_epsilon)  # 原子位置去噪 连续的不修改

        # ==================== 离散特征去噪（原子类型和键类型）====================
        
        # 1. 对模型预测结果进行 softmax 归一化,将logits转换为概率分布
        # 转换网络的预测输出

        pred_x1_x = F.softmax(x1_x_out, dim=-1)  # 原子特征预测概率分布
        pred_x1_bond = F.softmax(x1_bond_edge_x_out, dim=-1)  # 键特征预测概率分布

        # 2. 计算时间步相关参数
        s_int = t - 1  # 获取前一个时间步

        # 将时间步归一化到[0,1]区间
        t_normalized = torch.tensor(t / T, dtype=torch.float32).to(device)  # 当前时间步归一化
        # 前一时间步归一化,如果是第一步则为0
        s_normalized = torch.tensor(max(0, s_int) / T, dtype=torch.float32).to(device) if s_int >= 0 else torch.tensor(0.0).to(device)

        # 3. 计算噪声调度系数
        # 获取当前时间步的累积信号保持系数
        alpha_t_bar = x1_atom_diffuser.noise_schedule.get_alpha_bar(t_normalized)
        # 获取前一时间步的累积信号保持系数,如果是第一步则为1
        alpha_s_bar = x1_atom_diffuser.noise_schedule.get_alpha_bar(s_normalized) if s_int >= 0 else torch.tensor(1.0).to(device)

        
        # 3. 计算节点转移矩阵
        Qtb_atom = x1_atom_diffuser.transition_model.get_Qt_bar(alpha_t_bar, device, True)
        Qsb_atom = x1_atom_diffuser.transition_model.get_Qt_bar(alpha_s_bar, device, True)

        
        # 3. 计算键转移矩阵
        Qtb_bond = x1_bond_diffuser.transition_model.get_Qt_bar(alpha_t_bar, device, False)
        Qsb_bond = x1_bond_diffuser.transition_model.get_Qt_bar(alpha_s_bar, device, False)

        beta_t_x = x1_atom_diffuser.noise_schedule(t_normalized)                                                    
        beta_t_x = beta_t_x.to(device)

        beta_t_e = x1_bond_diffuser.noise_schedule(t_normalized)
        beta_t_e = beta_t_e.to(device)

        Qt_x = x1_atom_diffuser.transition_model.get_Qt(beta_t_x, device, is_atom=True)
        Qt_e = x1_bond_diffuser.transition_model.get_Qt(beta_t_e, device, is_atom=False) # 形状有问题

        X_t_final_output = x1_x_out
        x1_bond_edge_final_output = x1_bond_edge_x_out
        
        # 4. 计算后验分布 p(x_{t-1} | x_t, x_0)

        # 5. 计算原子特征的后验分布
        if s_int >= 0:
            x1_x_out_batch = x1_x_out.unsqueeze(0)
            Qt_batch = Qt_x.unsqueeze(0)
            # 非最后一步，使用后验分布采样
            atom_posterior_prob = compute_batched_over0_posterior_distribution(
                x1_x_out_batch , 
                Qt_batch ,
                Qtb_atom, Qsb_atom
            )

            atom_posterior_prob = atom_posterior_prob.squeeze(0)


            # 数值保护
            weighted_X = pred_x1_x.unsqueeze(-1) * atom_posterior_prob
            unnormalized_prob_X = weighted_X.sum(dim=2)
            unnormalized_prob_X = torch.clamp(unnormalized_prob_X, min=0.0)
            unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-6
            prob_X = unnormalized_prob_X / (torch.sum(unnormalized_prob_X, dim=-1, keepdim=True) + 1e-8)
            assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-3).all()

            _, num_classes_x = prob_X.shape

            X_t = prob_X.multinomial(1)
            X_t_indices = X_t.squeeze(-1)

            X_t_final_output = F.one_hot(X_t_indices, num_classes=num_classes_x).float() # 原子特征数量
        else:
            X_t_final_output = x1_x_out

        
        # 6. 计算键特征的后验分布（类似原子特征）
        if s_int >= 0:
            x1_bond_edge_x_t_batch = x1_bond_edge_x_t.unsqueeze(0)
            Qt_batch = Qt_e.unsqueeze(0)

            bond_posterior_prob = compute_batched_over0_posterior_distribution(
                x1_bond_edge_x_t_batch, 
                Qt_batch, 
                Qtb_bond, Qsb_bond
            )
            
            bond_posterior_prob = bond_posterior_prob.squeeze(0)
            
            # 从后验分布中采样
            # 计算加权概率
            weighted_E = pred_x1_bond.unsqueeze(-1) * bond_posterior_prob


            # 计算未归一化概率
            unnormalized_prob_E = weighted_E.sum(dim=-2)


            # 处理零概率情况
            zero_mask = torch.sum(unnormalized_prob_E, dim=-1) == 0
            unnormalized_prob_E[zero_mask] = 1e-7

            # 计算归一化概率
            prob_sum = torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
            prob_E = unnormalized_prob_E / prob_sum


            # 验证概率和为1
            prob_sum_check = prob_E.sum(dim=-1)
            prob_diff = prob_sum_check - 1
            large_diff_mask = prob_diff.abs() > 1e-4
            large_diff_indices = torch.where(large_diff_mask)

            assert ((prob_sum_check - 1).abs() < 1e-3).all()

            # 采样
            _, num_classes_e = prob_E.shape

            # 检查prob_E中是否存在负数
            if torch.any(prob_E < 0):
                # 获取负数位置的索引
                neg_indices = torch.where(prob_E < 0)

                # 将负数替换为0以确保multinomial采样可以进行
                prob_E = torch.clamp(prob_E, min=0.0)
            
            E_t = prob_E.multinomial(1)
            E_t_indices = E_t.squeeze(-1)
            x1_bond_edge_final_output = F.one_hot(E_t_indices, num_classes=num_classes_e).float() # 5种键类型
        else:
            # 最后一步，直接使用预测结果
            x1_bond_edge_final_output = x1_bond_edge_x_out

        # 检查去噪后的X1结果，防止数值不稳定
        if torch.any(torch.isnan(X_t_final_output)):
            print(f"Warning: X_t_final_output contains NaN values after denoising at timestep {t}, replacing with previous state")
            
        if torch.any(torch.isnan(x1_bond_edge_final_output)):
            print(f"Warning: x1_bond_edge_final_output contains NaN values after denoising at timestep {t}, replacing with previous state")
        # ==== temp
        
        # ==================== 反向去噪步骤 - X4模态 ====================
        # X4模态（药效团）的去噪计算
        x4_pos_t_1 = ((1. / float(x4_alpha_t)) * x4_pos_t)  - ((x4_var_dash_t/(x4_alpha_t * x4_sigma_dash_t)) * x4_pos_out)  +  (x4_c_t_injected * x4_pos_epsilon)  # 药效团位置去噪
        x4_direction_t_1 = ((1. / float(x4_alpha_t)) * x4_direction_t)  - ((x4_var_dash_t/(x4_alpha_t * x4_sigma_dash_t)) * x4_direction_out)  +  (x4_c_t * x4_direction_epsilon)  # 药效团方向去噪
        x4_x_t_1 = ((1. / x4_alpha_t) * x4_x_t)  - ((x4_var_dash_t/(x4_alpha_t * x4_sigma_dash_t)) * x4_x_out)  +  (x4_c_t * x4_x_epsilon)  # 药效团类型去噪
        
        # 检查去噪后的X4结果，防止数值不稳定
        if torch.any(torch.isnan(x4_pos_t_1)):
            print(f"Warning: x4_pos_t_1 contains NaN values after denoising at timestep {t}, replacing with zeros")
            x4_pos_t_1 = torch.nan_to_num(x4_pos_t_1, nan=0.0)  # 将NaN替换为0
        if torch.any(torch.isnan(x4_direction_t_1)):
            print(f"Warning: x4_direction_t_1 contains NaN values after denoising at timestep {t}, replacing with zeros")
            x4_direction_t_1 = torch.nan_to_num(x4_direction_t_1, nan=0.0)  # 将NaN替换为0
        if torch.any(torch.isnan(x4_x_t_1)):
            print(f"Warning: x4_x_t_1 contains NaN values after denoising at timestep {t}, replacing with zeros")
            x4_x_t_1 = torch.nan_to_num(x4_x_t_1, nan=0.0)  # 将NaN替换为0
    
        
        # ==================== 重置虚拟节点状态 ====================
        # 虚拟节点在去噪过程中保持不变，确保其状态不被修改
        x1_pos_t_1[virtual_node_mask_x1] = x1_pos_t[virtual_node_mask_x1]  # 重置X1虚拟节点位置
        
        x4_pos_t_1[virtual_node_mask_x4] = x4_pos_t[virtual_node_mask_x4]  # 重置X4虚拟节点位置
        x4_direction_t_1[virtual_node_mask_x4] = x4_direction_t[virtual_node_mask_x4]  # 重置X4虚拟节点方向
        x4_x_t_1[virtual_node_mask_x4] = x4_x_t[virtual_node_mask_x4]  # 重置X4虚拟节点特征
        
        # ==================== 更新状态并迭代到下一时间步 ====================
        # 将计算得到的下一时间步状态设为当前状态
        x1_pos_t = x1_pos_t_1  # 更新X1原子位置
        x1_x_t = X_t_final_output.float()  # 更新X1原子特征 t-1 诡异啊 long
        x1_bond_edge_x_t = x1_bond_edge_final_output.float()   # 更新X1键特征 t-1
        
        x4_pos_t = x4_pos_t_1  # 更新X4药效团位置
        x4_direction_t = x4_direction_t_1  # 更新X4药效团方向
        x4_x_t = x4_x_t_1  # 更新X4药效团类型
        
        # 确保log文件夹存在
        os.makedirs('log', exist_ok=True)
        
        # 将时间步作为文件名保存到log文件夹
        with open(f'log/{t}.txt', 'a') as f:
            f.write(f'\nTime step {t}:\n\n')  # 添加换行确保新记录清晰分隔
            f.write(f'X1 atom features: {x1_x_t}\n\n')  # 原子特征
            f.write(f'X1 bond features: {x1_bond_edge_x_t}\n\n')  # 化学键特征
            f.write(f'X1 positions: {x1_pos_t}\n\n')  # 原子位置坐标
            f.write(f'X4 positions: {x4_pos_t}\n\n')  # 药效团位置
            f.write(f'X4 directions: {x4_direction_t}\n\n')  # 药效团方向
            f.write(f'X4 types: {x4_x_t}\n\n')  # 药效团类型

        # 时间步递减，向t=0迭代
        t = t - 1  # 主时间步递减
        x1_t = x1_t - 1  # X1时间步递减
        x2_t = x2_t - 1  # X2时间步递减
        x3_t = x3_t - 1  # X3时间步递减
        x4_t = x4_t - 1  # X4时间步递减
    
        pbar.update(1)  # 更新进度条
        
        # 清理CUDA内存，防止内存泄漏
        del output_dict  # 删除模型输出字典
        del input_dict  # 删除模型输入字典    
        
    pbar.close()  # 关闭进度条
    
    # ==================== X4模态最终结果处理 ====================
    # 通过最小距离匹配将连续值转换为离散的药效团类型索引
    x4_x_final = np.argmin(np.abs(x4_x_t[~virtual_node_mask_x4].cpu().numpy() - params['dataset']['x4']['scale_node_features']), axis = -1)
    x4_x_final = x4_x_final - 1  # 调整索引，补偿之前添加虚拟节点药效团类型时的偏移
    x4_pos_final = x4_pos_t[~virtual_node_mask_x4].cpu().numpy()  # 提取药效团最终位置
    
    # 处理药效团方向向量
    x4_direction_final = x4_direction_t[~virtual_node_mask_x4].cpu().numpy() / params['dataset']['x4']['scale_vector_features']  # 反缩放方向向量
    x4_direction_final_norm = np.linalg.norm(x4_direction_final, axis = 1)  # 计算方向向量的模长
    x4_direction_final[x4_direction_final_norm < 0.5] = 0.0  # 模长小于0.5的向量设为零向量（无方向性）
    # 对模长大于等于0.5的向量进行单位化
    x4_direction_final[x4_direction_final_norm >= 0.5] = x4_direction_final[x4_direction_final_norm >= 0.5] / x4_direction_final_norm[x4_direction_final_norm >= 0.5][..., None]
    
    
    # ==================== X1模态最终结果处理 ====================
    x1_pos_final = x1_pos_t[~virtual_node_mask_x1].cpu().numpy()  # 提取非虚拟节点的原子最终位置
    # 这两行代码的主要目的是将模型输出的连续值特征转换回离散的原子和键类型索引

    edge_index = bond_edge_index_x1 
    edge_mask = (~virtual_node_mask_x1[edge_index[0]]) & (~virtual_node_mask_x1[edge_index[1]])
    # 使用边掩码过滤键特征

    # x1_bond_edge_x_final = x1_bond_edge_x_t[edge_mask].long()
    x1_bond_edge_x_final = torch.argmax(x1_bond_edge_x_t[edge_mask], dim=-1).cpu().numpy()
    
    # 将原子类型索引重新映射到实际的原子序数
    atomic_number_remapping = torch.tensor([0,1,6,7,8,9,17,35,53,16,15,14])  # [None, 'H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si']
    
    # x1_x_final = x1_x_t[~virtual_node_mask_x1].long().cpu().numpy()  # 应用原子序数映射

    x1_x_final_indices = torch.argmax(x1_x_t[~virtual_node_mask_x1], dim=-1).cpu().numpy()

    # atoms_indices = np.argmax(x1_x_final, axis=1)

    x_really_finally = []
    for idx in x1_x_final_indices:
        num = atomic_number_remapping[idx].item()
        x_really_finally.append(num)
    x_really_finally = np.array(x_really_finally)
    
    # ==================== 返回生成的结构 ====================
    # 将所有模态的最终结果组织成结构化字典并返回
    generated_structures = []  # 存储生成的结构列表
    for b in range(batch_size):  # 遍历批次中的每个样本

        generated_dict = {
            'x1': {  # X1模态：分子结构
                'atoms': np.array_split(x_really_finally, batch_size)[b],  # 原子类型（原子序数）
                #'formal_charges': None, # 形式电荷（待提取）：从x1_x_t[~virtual_node_mask_x1, -len(params['dataset']['x1']['charge_types']):]中提取
                'bonds': np.array_split(x1_bond_edge_x_final, batch_size)[b],  # 键类型
                'positions': np.array_split(x1_pos_final, batch_size)[b],  # 原子位置坐标
            },
            'x2': {  # X2模态：蛋白质口袋结构
                'positions': np.array_split(surface, batch_size)[b],  # 口袋表面点位置
            },
            'x3': {  # X3模态：静电势场
                'charges': np.array_split(electrostatics, batch_size)[b],  # 静电势值
                'positions': np.array_split(pharm_pos, batch_size)[b],  # 静电势点位置
            },
            'x4': {  # X4模态：药效团特征
                'types': np.array_split(x4_x_final, batch_size)[b],  # 药效团类型
                'positions': np.array_split(x4_pos_final, batch_size)[b],  # 药效团位置
                'directions': np.array_split(x4_direction_final, batch_size)[b],  # 药效团方向向量
            },
        }
        generated_structures.append(generated_dict)  # 添加到结果列表
    
    return generated_structures  # 返回生成的多模态结构列表
