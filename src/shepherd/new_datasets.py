"""新数据集模块 - 用于分子扩散模型的数据处理和加载

本文件实现了用于分子生成扩散模型的数据集类和相关工具函数，主要功能包括：
1. 离散特征扩散处理（原子类型、键类型、药效团类型）
2. 连续特征扩散处理（分子坐标、分子表面、静电势）
3. 多模态分子数据的统一处理和加噪
4. 支持条件生成的异构图数据结构

主要组件：
- MarginalUniformTransition: 边际均匀转移矩阵类
- PredefinedNoiseScheduleDiscrete: 预定义离散噪声调度器
- DiscreteFeatureDiffusion: 离散特征扩散处理器
- HeteroDataset: 异构分子数据集类
- get_atomic_partial_charges: 原子部分电荷计算函数

作者：SPD项目组
版本：2024年版本
"""

# 3D点云处理库，用于分子表面生成和可视化
import open3d 

# 分子表面和点云生成相关工具函数
from shepherd.shepherd_score_utils.generate_point_cloud import (
    get_atom_coords,  # 获取原子坐标
    get_atomic_vdw_radii,  # 获取原子范德华半径
    get_molecular_surface,  # 生成分子表面点云
    get_electrostatics,  # 计算静电势
    get_electrostatics_given_point_charges,  # 基于给定电荷计算静电势
)

# 药效团识别和处理工具
from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores

# 分子构象生成和坐标更新工具
from shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates

# RDKit化学信息学库，用于分子处理和性质计算
import rdkit

# NumPy数值计算库，用于数组操作和数学计算
import numpy as np

# PyTorch深度学习框架，用于张量操作和神经网络
import torch

# PyTorch Geometric图神经网络库，用于图数据处理
import torch_geometric

# Python标准库，用于深拷贝对象
from copy import deepcopy

# 离散扩散相关导入
# PyTorch函数式接口，用于softmax、one-hot等操作
import torch.nn.functional as F

class MarginalUniformTransition:
    """边际均匀转移矩阵类
    
    用于离散扩散过程中的状态转移，实现从当前状态到噪声状态的概率转移矩阵。
    该类基于边际分布构建转移矩阵，支持离散特征（如原子类型、键类型）的扩散过程。
    
    属性:
        x_marginals (torch.Tensor): 节点特征的边际概率分布，形状为 [K]，其中K是类别数
        e_marginals (torch.Tensor, 可选): 边特征的边际概率分布，默认为None
        y_classes (int): 图级别标签的类别数，默认为0
    """
    
    def __init__(self, x_marginals, e_marginals = None, y_classes=0):
        """初始化边际均匀转移矩阵
        
        参数:
            x_marginals (torch.Tensor): 节点特征的边际概率分布
            e_marginals (torch.Tensor, 可选): 边特征的边际概率分布
            y_classes (int): 图级别标签的类别数
        """
        self.marginals = x_marginals 
        
    def get_Qt_bar(self, alpha_t_bar, device, is_atom=True):                
        """获取累积转移矩阵 Q_t_bar
        
        计算从时间步0到时间步t的累积转移矩阵，用于直接从原始状态采样噪声状态。
        
        参数:
            alpha_t_bar (torch.Tensor): 累积噪声系数，形状为 [batch_size] 或标量
            device (torch.device): 计算设备（CPU或GPU）
            
        返回:
            torch.Tensor: 累积转移矩阵，形状为 [batch_size, K, K] 或 [K, K]
        """

        return PredefinedNoiseScheduleDiscrete.get_Qt_bar_from_marginals(self.marginals, alpha_t_bar, device)

    def get_Qt(self, beta_t, device, is_atom=True):
        """获取单步转移矩阵 Qt
        
        计算从当前状态到噪声状态的单步转移矩阵，用于扩散过程中的状态转移。
        
        参数:
            beta_t (torch.Tensor): 噪声系数，形状为 [batch_size] 或标量
            device (torch.device): 计算设备（CPU或GPU）
            
        返回:
            torch.Tensor: 单步转移矩阵，形状为 [batch_size, K, K] 或 [K, K]
        """
        # 将输入张量移动到指定设备
        beta_t = beta_t.to(device)

        marginals = self.marginals.to(device)

        
        # 获取类别数量K
        K = marginals.size(0)

        
        # 确保beta_t具有正确的形状以支持广播
        if beta_t.dim() == 0:  # 如果是标量
            beta_t = beta_t.unsqueeze(0) 


        U = marginals.unsqueeze(0).expand(K, -1) 
        
        # 构建单位矩阵I
        I = torch.eye(K, device=device) # [K, K]
        
        # 计算单步转移矩阵Qt
        # Qt = (1 - beta_t) * I + beta_t * U
        Qt = (1.0 - beta_t) * I + beta_t * U
        return Qt

class PredefinedNoiseScheduleDiscrete:
    """预定义离散噪声调度器类
    
    为离散扩散过程提供噪声调度，管理扩散过程中的噪声系数。
    使用线性噪声调度策略，从小噪声逐渐增加到大噪声。
    
    属性:
        T (int): 总扩散时间步数
        betas (torch.Tensor): 噪声系数序列，形状为 [T]
        alphas (torch.Tensor): 信号保持系数序列，alphas = 1 - betas
        alpha_bars (torch.Tensor): 累积信号保持系数，alpha_bars[t] = ∏(alphas[0:t+1])
    """
    
    def __init__(self, timesteps):
        """初始化噪声调度器
        
        参数:
            timesteps (int): 扩散过程的总时间步数
        """
        self.T = timesteps
        # 线性噪声调度：从1e-4线性增长到0.02
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        # 信号保持系数：alpha_t = 1 - beta_t
        self.alphas = 1. - self.betas
        # 累积信号保持系数：alpha_bar_t = ∏(alpha_s) for s=0 to t
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def __call__(self, t_normalized):
        """获取归一化时间步对应的噪声系数
        
        参数:
            t_normalized (torch.Tensor): 归一化时间步，范围 [0, 1]
            
        返回:
            torch.Tensor: 对应的噪声系数 beta_t
        """
        # 将 [0,1] 安全映射到 [0, T-1]，避免 t=1.0 时产生索引 T 的越界
        # 计算缩放后的时间步,并处理numpy.int64类型
        t_scaled = t_normalized * (self.T - 1)
        if isinstance(t_scaled, np.int64):
            t_scaled = torch.tensor(t_scaled, dtype=torch.float32)

        
        t_int = torch.floor(t_scaled).long()
        t_int = torch.clamp(t_int, 0, self.T - 1)
        return self.betas[t_int]
    
    def get_alpha_bar(self, t_normalized):
        """获取归一化时间步对应的累积信号保持系数
        
        参数:
            t_normalized (torch.Tensor): 归一化时间步，范围 [0, 1]
            
        返回:
            torch.Tensor: 对应的累积信号保持系数 alpha_bar_t
        """
        # 与 __call__ 保持一致的安全映射
        t_scaled = t_normalized * (self.T - 1)
        t_int = torch.floor(t_scaled).long()
        t_int = torch.clamp(t_int, 0, self.T - 1)
        return self.alpha_bars[t_int]
        
    @staticmethod
    def get_Qt_bar_from_marginals(mu_x, alpha_bar_t, device):
        """
        从边际分布计算累积转移矩阵Q_t_bar
        
        参数:
            mu_x: 边际分布向量
            alpha_bar_t: 累积信号保持系数
            device: 计算设备
            
        返回:
            Qt_bar: 累积转移矩阵
        """
        # 将输入张量移动到指定设备
        mu_x = mu_x.to(device)
        alpha_bar_t = alpha_bar_t.to(device)
        # 获取类别数量K
        K = mu_x.size(0)
        
        # 重塑张量以支持广播机制
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1)  # 形状变为(batch_size, 1, 1)
        mu_x = mu_x.view(1, 1, K)  # 形状变为(1, 1, K)

        # 计算累积转移矩阵Qt_bar
        # Qt_bar = α_t * I + (1-α_t) * μ
        # 其中I为单位矩阵，μ为边际分布
        Qt_bar = alpha_bar_t * torch.eye(K, device=device).unsqueeze(0) + (1. - alpha_bar_t) * mu_x
        
        return Qt_bar

# 离散特征扩散的简单实现
class DiscreteFeatureDiffusion:
    """离散特征扩散处理器类
    
    用于对one-hot编码的离散特征（如原子类型、键类型、药效团类型）应用扩散噪声。
    该类实现了离散扩散过程，通过转移矩阵将原始特征逐步转换为噪声特征。
    
    属性:
        T (int): 扩散过程的总时间步数
        noise_schedule (PredefinedNoiseScheduleDiscrete): 噪声调度器实例
        transition_model (MarginalUniformTransition): 转移矩阵模型
    """
    def __init__(self, timesteps, marginals):
        """初始化离散特征扩散处理器
        
        参数:
            timesteps (int): 扩散过程的总时间步数 (T)
            marginals (torch.Tensor): 一维张量，包含每个类别的边际概率分布
        """
        self.T = timesteps
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(timesteps=self.T)
        
        # 确保边际概率和为1，进行归一化处理
        marginals = marginals.float()
        feature_marginals = marginals / torch.sum(marginals)
        
        print(f"Initialized DiscreteFeatureDiffusion with marginals: {feature_marginals}")
        
        # 只需要特征(X/E)的转移模型，不需要属性(y)的转移模型
        self.transition_model = MarginalUniformTransition(x_marginals=feature_marginals, e_marginals=None, y_classes=0)

    def get_params_for_t(self, t_int, device):
        """获取给定整数时间步t的扩散参数
        
        参数:
            t_int (torch.Tensor): 整数时间步
            device (torch.device): 计算设备
            
        返回:
            dict: 包含beta_t、alpha_s_bar、alpha_t_bar的参数字典
        """
        # 如果输入是多元素张量，取第一个元素
        if t_int.numel() > 1:
            t_int = t_int[0]

        # 将时间步裁剪到有效范围[1, T]，与上游设计保持一致
        t_int = torch.clamp(t_int, 1, self.T)

        # 将整数时间步转换为归一化浮点数时间步
        t_float = t_int.float() / self.T
        s_int = t_int - 1  # 前一个时间步
        s_float = torch.clamp(s_int.float(), min=-1.0) / self.T
        
        # 获取当前时间步的噪声系数
        beta_t = self.noise_schedule(t_normalized=t_float)
        # 处理边界情况：当s < 0时，alpha_s_bar = 1
        if (s_int < 0).any():
             alpha_s_bar = torch.ones_like(beta_t)
        else:
            alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        # 获取当前时间步的累积信号保持系数
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)
        
        return {
            'beta_t': beta_t,
            'alpha_s_bar': alpha_s_bar,
            'alpha_t_bar': alpha_t_bar
        }

    def apply_noise(self, features_0, t_int, device):
        """对原始特征应用扩散噪声
        
        根据给定的时间步t，从原始特征x_0采样得到噪声特征x_t。
        使用转移矩阵和多项式分布进行采样。

        参数:
            features_0 (torch.Tensor): 原始的one-hot编码特征，形状为 [N, C]
            t_int (int): 单个整数时间步
            device (torch.device): 计算设备
        
        返回:
            torch.Tensor: 加噪后的one-hot特征，形状为 [N, C]
        """
        # 获取转移参数
        t_int_tensor = torch.tensor([t_int], device=device)
        params = self.get_params_for_t(t_int_tensor, device)
        alpha_t_bar = params['alpha_t_bar']
        
        # 获取累积转移矩阵 Q_t_bar
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=device)
        
        # 计算在时间步t的概率分布：p(x_t | x_0) = x_0 * Q_t_bar
        prob_t = torch.matmul(features_0, Qtb.squeeze(0)) # [N, C]
        
        # ========================= 鲁棒性修改开始 =========================
        # 对概率进行防御性处理，防止 torch.multinomial 函数崩溃

        # 1. 确保概率值非负（防止浮点数精度问题导致的负值）
        prob_t = torch.clamp(prob_t, min=0)
        
        # 2. 计算每行的概率和，用于后续归一化
        row_sums = prob_t.sum(dim=-1, keepdim=True)
        
        # 3. 处理数值不稳定情况：
        #    - 对于行和接近零的行，使用均匀分布作为后备方案
        #    - 对于正常行，进行归一化确保概率和为1
        uniform_fallback = torch.ones_like(prob_t) / prob_t.shape[-1]
        prob_t = torch.where(row_sums > 1e-6, prob_t / row_sums, uniform_fallback)
        
        # ========================= 鲁棒性修改结束 ===========================
        
        # 从分类分布中采样得到加噪特征的类别索引
        sampled_indices_t = torch.multinomial(prob_t, num_samples=1).squeeze(-1) # [N]
        # 将采样得到的类别索引转换为one-hot编码
        features_t = F.one_hot(sampled_indices_t, num_classes=features_0.shape[1]).float() # [N, C]
        
        return features_t

# 在大多数情况下不会使用此函数，因为我们使用xTB电荷而不是MMFF电荷
def get_atomic_partial_charges(mol: rdkit.Chem.Mol) -> np.ndarray:
    """获取给定分子的原子部分电荷
    
    计算分子中每个原子的部分电荷，优先使用MMFF力场方法，
    如果MMFF不可用则回退到Gasteiger方法。
    假设输入分子已经包含优化的构象。

    参数:
        mol (rdkit.Chem.Mol): RDKit分子对象，需包含优化的几何构象
    
    返回:
        np.ndarray: 形状为(N,)的数组，包含分子中每个原子的部分电荷
        
    异常:
        ValueError: 当输入分子对象没有嵌入构象时抛出
    """
    
    # 检查分子是否包含构象信息
    try:
        mol.GetConformer()
    except ValueError as e:
        raise ValueError(f"提供的rdkit.Chem.Mol对象没有嵌入构象信息。", e)
    
    # 尝试使用MMFF力场计算部分电荷
    molec_props = rdkit.Chem.AllChem.MMFFGetMoleculeProperties(mol)
    if molec_props:
        # 电荷单位为电子单位(e)
        charges = np.array([molec_props.GetMMFFPartialCharge(i) for i, _ in enumerate(mol.GetAtoms())])
    else:
        # 如果MMFF不可用，使用Gasteiger方法作为后备
        print("输入分子无法使用MMFF电荷计算，回退到Gasteiger电荷方法。")
        rdkit.Chem.AllChem.ComputeGasteigerCharges(mol)
        charges=np.array([a.GetDoubleProp('_GasteigerCharge') for a in mol.GetAtoms()])
    
    return charges


class HeteroDataset(torch_geometric.data.Dataset):
    """异构分子数据集类
    
    用于处理多模态分子数据的PyTorch Geometric数据集类。
    支持四种不同的模态数据：
    - x1: 分子图结构（原子坐标、原子类型、键类型）
    - x2: 分子表面点云数据
    - x3: 静电势数据
    - x4: 药效团数据
    
    该类实现了扩散模型训练所需的数据预处理、噪声添加和批处理功能。
    """
    def __init__(self, 
                 
            molblocks_and_charges, 
            
            noise_schedule_dict,
            
            # 离散扩散参数
            atom_marginals_x1,
            bond_marginals_x1,
            pharm_marginals_x4,

            explicit_hydrogens = True,
            use_MMFF94_charges = False,
            formal_charge_diffusion = False,
            
            x1 = True,
            x2 = True,
            x3 = True,
            x4 = True,
                
            recenter_x1 = True, 
            add_virtual_node_x1 = True, 
            remove_noise_COM_x1 = True,
            atom_types_x1 = [None, 'H', 'C', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'Si'],
            charge_types_x1 = [0,1,2,-1,-2],
            bond_types_x1 = [None, 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'],
            scale_atom_features_x1 = 1.0,
            scale_bond_features_x1 = 1.0,

            
            
            independent_timesteps_x2 = False,
            recenter_x2 = False, # 希望x2的中心是虚拟节点（其位置是x1的中心）
            add_virtual_node_x2 = True,
            remove_noise_COM_x2 = False,
            num_points_x2 = 75,

            
            independent_timesteps_x3 = False,
            recenter_x3 = False,
            add_virtual_node_x3 = True,
            remove_noise_COM_x3 = False,
            num_points_x3 = 75,
            scale_node_features_x3 = 1.0,
            
                 
            independent_timesteps_x4 = False,
            recenter_x4 = False, 
            add_virtual_node_x4 = True, # 必须为True，处理分子没有药效团的边界情况
            remove_noise_COM_x4 = False,
            max_node_types_x4 = 16, # 药效团类型数量（可以设置得比数据集中实际表示的更大）
            scale_node_features_x4 = 1.0,
            scale_vector_features_x4 = 1.0,
            multivectors = False,
            check_accessibility = False,
            
            probe_radius = 0.6,                 
        ):
        """初始化异构多模态分子数据集
        
        该数据集支持四种模态的分子表示，用于多模态分子扩散模型的训练：
        - x1: 分子图模态（原子坐标、原子类型、键信息）
        - x2: 分子表面模态（表面点云）
        - x3: 静电势模态（表面静电势分布）
        - x4: 药效团模态（药效团类型和位置）
        
        参数:
            molblocks_and_charges (list): 分子块和电荷信息的列表，每个元素包含分子结构数据
            noise_schedule_dict (dict): 噪声调度字典，包含各模态的时间步信息和扩散参数
            
            # 离散扩散相关参数
            atom_marginals_x1 (torch.Tensor): x1模态原子类型的边际分布，用于离散扩散过程
            bond_marginals_x1 (torch.Tensor): x1模态键类型的边际分布，用于离散扩散过程  
            pharm_marginals_x4 (torch.Tensor): x4模态药效团类型的边际分布，用于离散扩散过程
            
            # 通用参数
            explicit_hydrogens (bool): 是否显式包含氢原子，当前实现要求为True
            use_MMFF94_charges (bool): 是否使用MMFF94电荷计算方法，默认False
            formal_charge_diffusion (bool): 是否对原子形式电荷进行扩散处理，默认False
            
            # 模态开关
            x1 (bool): 是否启用x1模态（分子图），默认True
            x2 (bool): 是否启用x2模态（分子表面），默认True
            x3 (bool): 是否启用x3模态（静电势），默认True
            x4 (bool): 是否启用x4模态（药效团），默认True
            
            # x1模态参数（分子图结构）
            recenter_x1 (bool): 是否重新中心化x1坐标到质心，默认True
            add_virtual_node_x1 (bool): 是否添加虚拟节点作为全局信息聚合点，默认True
            remove_noise_COM_x1 (bool): 是否移除噪声的质心偏移，默认True
            atom_types_x1 (list): 支持的原子类型列表，None表示虚拟节点类型
            charge_types_x1 (list): 支持的原子电荷类型列表
            bond_types_x1 (list): 支持的化学键类型列表，None表示非键连接
            scale_atom_features_x1 (float): 原子特征的缩放因子，默认1.0
            scale_bond_features_x1 (float): 键特征的缩放因子，默认1.0
            
            # x2模态参数（分子表面）
            independent_timesteps_x2 (bool): 是否使用独立于x1的时间步采样，默认False
            recenter_x2 (bool): 是否重新中心化x2坐标，默认False（希望x2的中心是虚拟节点）
            add_virtual_node_x2 (bool): 是否添加虚拟节点，默认True
            remove_noise_COM_x2 (bool): 是否移除噪声质心，默认False
            num_points_x2 (int): x2模态表面点云的点数，默认75
            
            # x3模态参数（静电势）
            independent_timesteps_x3 (bool): 是否使用独立于x1的时间步采样，默认False
            recenter_x3 (bool): 是否重新中心化x3坐标，默认False
            add_virtual_node_x3 (bool): 是否添加虚拟节点，默认True
            remove_noise_COM_x3 (bool): 是否移除噪声质心，默认False
            num_points_x3 (int): x3模态静电势点云的点数，默认75
            scale_node_features_x3 (float): x3节点特征（静电势值）的缩放因子，默认1.0
            
            # x4模态参数（药效团）
            independent_timesteps_x4 (bool): 是否使用独立于x1的时间步采样，默认False
            recenter_x4 (bool): 是否重新中心化x4坐标，默认False
            add_virtual_node_x4 (bool): 是否添加虚拟节点，必须为True以处理无药效团分子
            remove_noise_COM_x4 (bool): 是否移除噪声质心，默认False
            max_node_types_x4 (int): 最大药效团类型数量，可设置得比实际数据更大，默认16
            scale_node_features_x4 (float): x4节点特征的缩放因子，默认1.0
            scale_vector_features_x4 (float): x4向量特征的缩放因子，默认1.0
            multivectors (bool): 是否使用多向量表示药效团方向，默认False
            check_accessibility (bool): 是否检查药效团的可达性，默认False
            
            # 其他参数
            probe_radius (float): 分子表面生成时的探针半径，默认0.6Å
            multivectors (bool): 是否使用多向量，默认False
            check_accessibility (bool): 是否检查可达性，默认False
            
            # 其他参数
            probe_radius (float): 探针半径，用于表面生成，默认0.6
        """
        
        # 调用父类构造函数，初始化PyTorch Geometric数据集
        super().__init__()

        # 存储分子数据和基本配置
        self.molblocks_and_charges = molblocks_and_charges  # 分子块和电荷信息列表
        self.length = len(molblocks_and_charges)  # 数据集大小
        self.use_MMFF94_charges = use_MMFF94_charges  # 是否使用MMFF94电荷
        
        # 噪声调度配置
        self.noise_schedule_dict = noise_schedule_dict
        
        # 氢原子处理配置
        self.explicit_hydrogens = explicit_hydrogens
        assert self.explicit_hydrogens == True  # 当前实现要求显式包含氢原子
        
        # 形式电荷扩散配置
        self.formal_charge_diffusion = formal_charge_diffusion
        
        # 各模态开关配置
        self.x1 = x1  # 分子图模态
        self.x2 = x2  # 分子表面模态
        self.x3 = x3  # 静电势模态
        self.x4 = x4  # 药效团模态
        
        # x1模态（分子图）配置参数
        self.recenter_x1 = recenter_x1  # 是否重新中心化坐标
        self.add_virtual_node_x1 = add_virtual_node_x1  # 是否添加虚拟节点
        self.remove_noise_COM_x1 = remove_noise_COM_x1  # 是否移除噪声质心
        self.atom_types_x1 = atom_types_x1  # 支持的原子类型列表
        self.charge_types_x1 = charge_types_x1  # 支持的电荷类型列表
        self.bond_types_x1 = bond_types_x1  # 支持的键类型列表
        self.scale_atom_features_x1 = scale_atom_features_x1  # 原子特征缩放因子
        self.scale_bond_features_x1 = scale_bond_features_x1  # 键特征缩放因子
        
        # x2模态（分子表面）配置参数
        self.recenter_x2 = recenter_x2  # 是否重新中心化坐标
        self.add_virtual_node_x2 = add_virtual_node_x2  # 是否添加虚拟节点
        self.remove_noise_COM_x2 = remove_noise_COM_x2  # 是否移除噪声质心
        self.num_points_x2 = num_points_x2  # 表面点云的点数
        self.independent_timesteps_x2 = independent_timesteps_x2  # 是否使用独立时间步
        
        # x3模态（静电势）配置参数
        self.recenter_x3 = recenter_x3  # 是否重新中心化坐标
        self.add_virtual_node_x3 = add_virtual_node_x3  # 是否添加虚拟节点
        self.remove_noise_COM_x3 = remove_noise_COM_x3  # 是否移除噪声质心
        self.num_points_x3 = num_points_x3  # 静电势点云的点数
        self.independent_timesteps_x3 = independent_timesteps_x3  # 是否使用独立时间步
        self.scale_node_features_x3 = scale_node_features_x3  # 节点特征缩放因子

        # x4模态（药效团）配置参数
        self.independent_timesteps_x4 = independent_timesteps_x4  # 是否使用独立时间步
        self.recenter_x4 = recenter_x4  # 是否重新中心化坐标
        self.add_virtual_node_x4 = add_virtual_node_x4  # 是否添加虚拟节点
        self.remove_noise_COM_x4 = remove_noise_COM_x4  # 是否移除噪声质心
        self.max_node_types_x4 = max_node_types_x4  # 最大药效团类型数
        self.scale_node_features_x4 = scale_node_features_x4  # 节点特征缩放因子
        self.scale_vector_features_x4 = scale_vector_features_x4  # 向量特征缩放因子
        self.multivectors = multivectors  # 是否使用多向量表示
        self.check_accessibility = check_accessibility  # 是否检查药效团可达性
        
        # 其他配置参数
        self.probe_radius = probe_radius  # 分子表面生成的探针半径
        self.scale_electrostatics = self.scale_node_features_x3  # 静电势缩放因子（别名）
    
        # 初始化离散扩散处理器
        T = len(noise_schedule_dict['x1']['ts'])  # 获取扩散时间步总数
        
        # 为x1模态初始化原子和键的离散扩散器
        if self.x1:
            self.x1_atom_diffuser = DiscreteFeatureDiffusion(timesteps=T, marginals=atom_marginals_x1)
            self.x1_bond_diffuser = DiscreteFeatureDiffusion(timesteps=T, marginals=bond_marginals_x1)
        
        # 为x4模态初始化药效团的离散扩散器
        if self.x4:
            self.x4_pharm_diffuser = DiscreteFeatureDiffusion(timesteps=T, marginals=pharm_marginals_x4)
    
    def get_x1_data(self, mol, t, alpha_dash_t, sigma_dash_t):
        """获取x1模态（分子图）的训练数据
        
        从RDKit分子对象提取分子图信息，包括原子坐标、原子类型、键信息等，
        并应用扩散噪声生成训练样本。该方法对位置坐标和离散特征使用相同的噪声调度。
        
        参数:
            mol (rdkit.Chem.Mol): RDKit分子对象，包含分子结构和构象信息
            t (int): 当前扩散时间步
            alpha_dash_t (float): 当前时间步的信号保持系数
            sigma_dash_t (float): 当前时间步的噪声强度系数
            
        返回:
            dict: 包含x1模态所有必要数据的字典
        """
        # 位置坐标和原子类型/特征使用相同的噪声调度
        
        data = {}  # 初始化数据字典
        data['timestep'] = torch.tensor([t], dtype=torch.long)  # 存储当前时间步

        # 提取原子类型信息，将原子符号映射为索引
        # 例如：'C' -> 2, 'N' -> 3, 'O' -> 4 等，索引0保留给虚拟节点（None）
        atom_types = [self.atom_types_x1.index(a.GetSymbol()) for a in mol.GetAtoms()]
        
        # 如果启用形式电荷扩散，提取并映射形式电荷信息
        # 形式电荷是原子在分子中的理论电荷状态，影响化学反应性
        if self.formal_charge_diffusion:
            formal_charges = [int(a.GetFormalCharge()) for a in mol.GetAtoms()]  # 获取每个原子的形式电荷
            formal_charge_map = {c:self.charge_types_x1.index(c) for c in self.charge_types_x1}  # 创建电荷到索引的映射
            formal_charges_mapped = [formal_charge_map[f] for f in formal_charges]  # 将电荷值映射为索引
            
        # 获取分子的三维坐标信息
        # 这些坐标来自分子的构象，代表原子在三维空间中的位置
        pos = np.array(mol.GetConformer().GetPositions())
        num_atoms = len(pos)  # 原子总数，用于后续数组初始化
        
        # 构建键的邻接矩阵和边索引
        # 在分子图中，我们需要表示原子之间的连接关系（化学键）
        bond_adj = 1-np.diag(np.ones(num_atoms, dtype = int))  # 创建全连接矩阵，对角线为0（原子不与自己相连）
        bond_adj = np.triu(bond_adj)  # 取上三角矩阵，避免重复边（无向图转有向图表示）
        bond_edge_index = np.stack(bond_adj.nonzero(), axis = 0)  # 获取边索引，形状为[2, num_edges]
        
        # 构建键类型映射字典和提取键类型信息
        # 键类型包括：None（非键连接）、SINGLE（单键）、DOUBLE（双键）、TRIPLE（三键）、AROMATIC（芳香键）
        bond_types_dict = {b:self.bond_types_x1.index(b) for b in self.bond_types_x1}
        max_bond_types_x1 = len(bond_types_dict)  # 最大键类型数，用于one-hot编码维度
        bond_types = []  # 存储每条边的键类型索引
        
        # 遍历所有边，确定键类型
        # 注意：这里包括真实的化学键和虚拟的非键连接
        for b in range(bond_edge_index.shape[1]):
            idx_1 = int(bond_edge_index[0, b])  # 边的起始原子索引
            idx_2 = int(bond_edge_index[1, b])  # 边的终止原子索引
            bond = mol.GetBondBetweenAtoms(idx_1, idx_2)  # 获取两原子间的键对象
            if bond is None:
                bond_types.append(bond_types_dict[None])  # 非键连接边类型，索引为0
            else:
                bond_type = bond_types_dict[str(bond.GetBondType())]  # 获取真实化学键的类型索引
                bond_types.append(bond_type)
        
        # 创建真实键的掩码（True表示真实的化学键，False表示非键连接）
        # 这个掩码在训练时用于区分真实键和虚拟连接，只对真实键计算损失
        data['bond_edge_mask'] = torch.from_numpy((np.array(bond_types) != 0).copy()).bool()
        
        # 计算和存储质心信息
        COM_before_centering = pos.mean(0)[None, ...]  # 中心化前的质心坐标
        data['com_before_centering'] = torch.from_numpy(COM_before_centering.copy()).float()
        pos_recentered = pos - pos.mean(0)  # 中心化后的坐标
        
        # 根据配置决定是否使用中心化坐标
        if self.recenter_x1:
            pos = pos_recentered
        
        COM = pos.mean(0)[None, ...]  # 当前坐标的质心
        data['com'] = torch.from_numpy(COM.copy()).float()

        # 初始化虚拟节点掩码
        # 掩码长度为原子数加上可能的虚拟节点数（0或1）
        virtual_node_mask = np.zeros(pos.shape[0] + int(self.add_virtual_node_x1))
        
        # 如果启用虚拟节点，进行相应的数据调整
        # 虚拟节点的作用：1) 作为全局信息聚合点 2) 改善长程依赖建模
        if self.add_virtual_node_x1:
            assert self.atom_types_x1[0] == None  # 确保第一个原子类型为None（虚拟节点类型）
            atom_types.insert(0, 0)  # 在原子类型列表开头插入虚拟节点类型（索引0）
            bond_edge_index = bond_edge_index + 1  # 边索引加1，为虚拟节点留出索引0
            virtual_node_pos = COM  # 虚拟节点位置设为质心，便于与所有原子交互
            pos = np.concatenate([virtual_node_pos, pos], axis = 0)  # 将虚拟节点位置（非零质心）添加到坐标开头
            pos_recentered = np.concatenate([virtual_node_pos * 0.0, pos_recentered], axis = 0)  # 将虚拟节点在中心化坐标中设为零点
            virtual_node_mask[0] = 1  # 标记第一个节点（索引0）为虚拟节点
        
        virtual_node_mask = virtual_node_mask == 1  # 转换为布尔掩码，True表示虚拟节点
        num_nodes = num_atoms + int(self.add_virtual_node_x1)  # 计算总节点数（原子数+虚拟节点数）
        
        # 存储边索引、坐标和虚拟节点掩码信息
        data['bond_edge_index'] = torch.from_numpy(bond_edge_index.copy()).long()
        data['pos'] = torch.from_numpy(pos.copy()).float()
        data['pos_recentered'] = torch.from_numpy(pos_recentered.copy()).float()
        data['virtual_node_mask'] = torch.from_numpy(virtual_node_mask.copy()).bool()
        
        # 为非噪声结构创建原子类型和形式电荷的（缩放）one-hot编码
        # one-hot编码将离散的原子类型转换为神经网络可处理的向量表示
        x = np.zeros((num_nodes, len(self.atom_types_x1)))  # 初始化原子特征矩阵，形状为[节点数, 原子类型数]
        x[np.arange(num_nodes), atom_types] = 1  # 设置one-hot编码，每行只有一个位置为1
        x = x * self.scale_atom_features_x1  # 应用缩放因子，调整特征值范围以改善训练稳定性
        
        # 如果启用形式电荷扩散，添加形式电荷特征
        # 形式电荷信息对于准确建模分子的电子结构和反应性至关重要
        if self.formal_charge_diffusion:
            x_formal_charges = np.zeros((len(formal_charges_mapped), len(self.charge_types_x1)))  # 初始化形式电荷特征矩阵
            x_formal_charges[np.arange(len(formal_charges_mapped)), formal_charges_mapped] = 1  # 设置形式电荷的one-hot编码
            x_formal_charges = x_formal_charges * self.scale_atom_features_x1  # 应用缩放因子，保持与原子特征相同的尺度
            
            if self.add_virtual_node_x1:
                # 虚拟节点的形式电荷one-hot特征全为零
                # 虚拟节点不代表真实原子，因此没有形式电荷概念
                x_formal_charges = np.concatenate((np.zeros(len(self.charge_types_x1), dtype = x_formal_charges.dtype)[None, ...], x_formal_charges), axis = 0)
            
            x = np.concatenate((x, x_formal_charges), axis = 1)  # 拼接原子类型和形式电荷特征，形成完整的节点特征
        
        data['x'] = torch.from_numpy(x.copy()).float()  # 存储节点特征到数据字典
        
        # 为非噪声结构创建键类型的（缩放）one-hot编码
        # 注意：这不包含到虚拟节点的任何边，虚拟节点的连接在后续单独处理
        bond_edge_x = np.zeros((bond_edge_index.shape[1], max_bond_types_x1))  # 初始化键特征矩阵，形状为[边数, 键类型数]
        bond_edge_x[np.arange(len(bond_types)), bond_types] = 1  # 设置键类型的one-hot编码，每行只有一个位置为1
        bond_edge_x = bond_edge_x * self.scale_bond_features_x1  # 应用缩放因子，调整键特征的数值范围
        data['bond_edge_x'] = torch.from_numpy(bond_edge_x.copy()).float()  # 存储键特征到数据字典
        
        # ========================= 扩散噪声应用开始 =========================
        # 该部分实现了多模态扩散模型的前向噪声过程，包括连续特征（坐标）和离散特征（原子/键类型）的噪声添加
        
        # --- 1. 保存 t=0 时刻的干净特征 ---
        # 这是关键修改！损失函数需要这些 t=0 的真实标签用于训练目标
        # 在扩散模型中，模型需要预测从噪声数据恢复到干净数据的过程
        data['x_0'] = data['x'].clone()  # 保存原始原子特征（one-hot编码）
        data['bond_edge_x_0'] = data['bond_edge_x'].clone()  # 保存原始键特征（one-hot编码）

        # --- 2. 处理连续特征（坐标）的加噪 ---
        # 对分子的三维坐标应用高斯噪声，遵循扩散过程的前向方程
        pos_noise = np.random.randn(*pos.shape)  # 生成标准高斯噪声，形状与坐标矩阵相同
        pos_noise[virtual_node_mask] = 0.0  # 虚拟节点不添加噪声，保持其作为参考点的作用
        
        # 如果启用质心噪声移除，确保噪声的质心为零
        # 这样可以保持分子的整体位置稳定，避免分子在空间中漂移
        if self.remove_noise_COM_x1:
            pos_noise[~virtual_node_mask] = pos_noise[~virtual_node_mask] - np.mean(pos_noise[~virtual_node_mask], axis = 0)
        
        data['pos_noise'] = torch.from_numpy(pos_noise.copy()).float()  # 存储位置噪声用于调试和分析
        
        # 应用前向扩散过程：x_t = α_t * x_0 + σ_t * ε
        # 这是扩散模型的核心方程，其中α_t控制信号保持程度，σ_t控制噪声强度
        pos_forward_noised = alpha_dash_t * pos  +  sigma_dash_t * pos_noise
        pos_forward_noised[virtual_node_mask] = pos[virtual_node_mask]  # 虚拟节点位置保持不变
        data['pos_forward_noised'] = torch.from_numpy(pos_forward_noised.copy()).float()  # 存储加噪后的位置

        # --- 3. 处理离散特征（原子和键类型）的加噪 ---
        # 离散特征使用专门的离散扩散器，而不是简单的高斯噪声
        # 离散扩散通过转移矩阵来模拟类别之间的转换过程
        device = data['x'].device  # 获取张量所在的计算设备
        t_tensor = torch.tensor([t], device=device)  # 创建时间步张量，用于离散扩散器

        # 对原子类型应用离散扩散噪声（仅对非虚拟节点操作）
        # 虚拟节点的类型保持不变，因为它们不代表真实的原子
        x_clean_no_vn = data['x_0'][~virtual_node_mask]  # 提取非虚拟节点的干净原子特征
        x_noised_no_vn = self.x1_atom_diffuser.apply_noise(x_clean_no_vn, t_tensor, device)  # 应用离散扩散噪声
        
        x_forward_noised = data['x_0'].clone()  # 从干净的 t=0 数据开始
        x_forward_noised[~virtual_node_mask] = x_noised_no_vn  # 更新非虚拟节点的特征
        data['x_forward_noised'] = x_forward_noised  # 存储加噪后的原子特征
        
        # 对键类型应用离散扩散噪声
        # 键类型的扩散独立于原子类型，使用单独的转移矩阵
        bond_edge_x_clean = data['bond_edge_x_0']  # 获取干净的键特征
        bond_edge_x_noised = self.x1_bond_diffuser.apply_noise(bond_edge_x_clean, t_tensor, device)  # 应用离散扩散噪声
        data['bond_edge_x_forward_noised'] = bond_edge_x_noised  # 存储加噪后的键特征

        # 旧的噪声字段不再需要，设为零以保持向后兼容性
        # 这些字段在早期版本中用于存储高斯噪声，现在已被离散扩散器取代
        data['x_noise'] = torch.zeros_like(data['x'])  # 原子特征噪声（已弃用）
        data['bond_edge_x_noise'] = torch.zeros_like(data['bond_edge_x'])  # 键特征噪声（已弃用）

        # ========================= 扩散噪声应用结束 =========================

        return data, pos, virtual_node_mask
    
    
    def get_x2_data(self, radii, atom_centers, num_points, recenter, add_virtual_node, remove_noise_COM, t, alpha_dash_t, sigma_dash_t, virtual_node_pos = None):
        """获取x2模态（分子表面）的训练数据
        
        生成分子表面点云数据，用于条件生成。该方法计算分子的可溶剂可达表面，
        并生成表面点云用作扩散模型的条件输入。
        
        参数:
            radii (np.ndarray): 原子半径数组
            atom_centers (np.ndarray): 原子中心坐标数组
            num_points (int): 生成的表面点数量
            recenter (bool): 是否重新中心化坐标
            add_virtual_node (bool): 是否添加虚拟节点
            remove_noise_COM (bool): 是否移除噪声质心（此处未使用）
            t (int): 当前扩散时间步
            alpha_dash_t (float): 当前时间步的信号保持系数（此处未使用）
            sigma_dash_t (float): 当前时间步的噪声强度系数（此处未使用）
            virtual_node_pos (np.ndarray, optional): 虚拟节点位置，默认为None
            
        返回:
            dict: 包含x2模态所有必要数据的字典
        """
        
        data = {}  # 初始化数据字典
        data['timestep'] = torch.tensor([t], dtype=torch.long)  # 存储当前时间步
        
        # 生成分子表面点云
        pos = get_molecular_surface(
            atom_centers,
            radii,
            num_points=num_points,
            probe_radius = self.probe_radius,  # 探针半径，用于计算可溶剂可达表面
            num_samples_per_atom = 20,  # 每个原子的采样点数
        )
        
        # 处理虚拟节点的添加
        if add_virtual_node:
            # 计算虚拟节点位置（如果未提供，则使用表面点的质心）
            if virtual_node_pos is None:
                virtual_node_pos = pos.mean(0, keepdims=True)
            
            # 在位置数组开头添加虚拟节点
            pos = np.concatenate([virtual_node_pos, pos], axis=0)
            
            # 创建虚拟节点掩码，标记第一个点为虚拟节点
            virtual_node_mask = np.zeros(pos.shape[0], dtype=bool)
            virtual_node_mask[0] = True
        else:
            # 即使没有虚拟节点，也创建全False的布尔掩码，保证下游处理一致性
            virtual_node_mask = np.zeros(pos.shape[0], dtype=bool)
        
        # 计算和存储中心化前的质心
        COM_before_centering = pos.mean(0)[None, :]
        data['com_before_centering'] = torch.from_numpy(COM_before_centering.copy()).float()
    
        # 执行坐标中心化处理
        pos_recentered = pos - pos.mean(0)  # 计算中心化后的坐标
        if recenter:
            pos = pos_recentered  # 根据配置决定是否使用中心化坐标
    
        # 计算和存储当前坐标的质心
        COM = pos.mean(0)[None, :]
        data['com'] = torch.from_numpy(COM.copy()).float()
        
        # 存储位置坐标信息
        data['pos'] = torch.from_numpy(pos.copy()).float()
        data['pos_recentered'] = torch.from_numpy(pos_recentered.copy()).float()        
        
        # 创建节点类型的one-hot编码（区分真实节点和虚拟节点）
        x = np.zeros((pos.shape[0], 2))  # 2维特征：[虚拟节点, 真实节点]
        if add_virtual_node:
            x[0, 0] = 1  # 第一个节点标记为虚拟节点
            x[1:, 1] = 1  # 其余节点标记为真实表面点
        else:
            x[:, 1] = 1  # 所有节点都标记为真实表面点
        
        data['x'] = torch.from_numpy(x.copy()).float()  # 存储节点特征
        
        # 存储虚拟节点掩码信息
        if virtual_node_mask is not None:
            data['virtual_node_mask'] = torch.from_numpy(virtual_node_mask)
        else:
            data['virtual_node_mask'] = None
    
        # x2模态作为条件输入，不添加噪声
        data['x_forward_noised'] = data['x'].clone() 
    
        data['pos_noise'] = torch.zeros_like(data['pos'])
        
        data['pos_forward_noised'] = data['pos'].clone()
        
        return data, pos, virtual_node_mask
    
    
    def get_x3_data_electrostatics_only(self, charges, charge_centers, data, pos, t, alpha_dash_t, sigma_dash_t):
        """获取x3模态（静电势）的训练数据
        
        计算分子表面各点的静电势值，用作条件生成的输入。该方法基于原子的部分电荷
        和表面点坐标，计算每个表面点处的静电势，作为扩散模型的条件信息。
        
        参数:
            charges (np.ndarray): 原子的部分电荷数组
            charge_centers (np.ndarray): 原子中心坐标数组
            data (dict): 从get_x2_data继承的数据字典
            pos (np.ndarray): 表面点坐标数组
            t (int): 当前扩散时间步（此处未使用）
            alpha_dash_t (float): 当前时间步的信号保持系数（此处未使用）
            sigma_dash_t (float): 当前时间步的噪声强度系数（此处未使用）
            
        返回:
            dict: 更新后的数据字典，包含静电势信息
        """
        
        # 计算表面各点的静电势值
        x = get_electrostatics_given_point_charges(charges, charge_centers, pos)
        # 应用缩放因子，标准化静电势值
        x = x * self.scale_node_features_x3
                
        # 存储静电势特征
        data['x'] = torch.from_numpy(x.copy()).float()
        
        # x3模态作为条件输入，不添加噪声
        data['x_noise'] = torch.zeros_like(data['x'])  # 噪声为零
        data['x_forward_noised'] = data['x'].clone()  # 前向传播时保持原值

        # 保留从get_x2_data继承的虚拟节点掩码，不进行覆盖
        # 这确保了x2和x3模态之间虚拟节点信息的一致性
        
        return data
    

    def get_x4_data(self, mol, recenter, add_virtual_node, remove_noise_COM, t, alpha_dash_t, sigma_dash_t, virtual_node_pos = None):
        """获取x4模态（药效团）的训练数据
        
        提取分子的药效团特征，包括药效团类型、位置和方向信息。该方法处理药效团的
        离散特征（类型）和连续特征（坐标、方向），并应用相应的扩散噪声。
        
        参数:
            mol (rdkit.Mol): RDKit分子对象
            recenter (bool): 是否重新中心化坐标
            add_virtual_node (bool): 是否添加虚拟节点（必须为True）
            remove_noise_COM (bool): 是否移除噪声的质心
            t (int): 当前扩散时间步
            alpha_dash_t (float): 当前时间步的信号保持系数
            sigma_dash_t (float): 当前时间步的噪声强度系数
            virtual_node_pos (np.ndarray, optional): 虚拟节点位置，默认为None
            
        返回:
            dict: 包含x4模态所有必要数据的字典
        """
        
        # 药效团模态必须包含虚拟节点，以处理无药效团分子的情况
        assert add_virtual_node

        data = {}  # 初始化数据字典
        data['timestep'] = torch.tensor([t], dtype=torch.long)  # 存储当前时间步
        
        # 1. --- 获取基础药效团信息 ---
        # 提取分子的药效团特征：类型、位置和方向
        pharm_types, pos, direction = get_pharmacophores(
            mol, 
            multi_vector = self.multivectors,  # 是否使用多向量表示
            check_access=self.check_accessibility,  # 是否检查可及性
        )
        # 为虚拟节点预留类型0，将真实药效团类型向后偏移
        pharm_types = pharm_types + 1 
        # 添加微小随机扰动，避免药效团位置完全重合
        pos = pos + np.random.randn(*pos.shape) * 0.05
        
        # 2. --- 统一处理虚拟节点和空分子的情况 ---
        # 处理无药效团分子的特殊情况
        if pharm_types.shape[0] == 0:
            pharm_types = np.array([])  # 保持为空数组，后续统一处理
            pos = np.empty((0, 3))  # 空的位置数组
            direction = np.empty((0, 3))  # 空的方向数组
            
        # 在数组开头添加虚拟节点（类型为0）
        pharm_types = np.concatenate([np.array([0]), pharm_types], axis=0)
        
        # 计算虚拟节点的位置（如果未提供）
        if virtual_node_pos is None:
            if pos.shape[0] > 0:  # 如果存在真实药效团
                virtual_node_pos = pos.mean(0, keepdims=True)  # 使用药效团质心
            else:  # 如果只有虚拟节点
                virtual_node_pos = np.zeros((1, 3))  # 放置在原点

        # 将虚拟节点添加到位置和方向数组的开头
        pos = np.concatenate([virtual_node_pos, pos], axis=0)
        direction = np.concatenate([np.zeros((1, 3)), direction], axis=0)  # 虚拟节点方向为零
        
        # 3. --- 计算坐标、掩码和t=0特征 ---
        # 计算中心化前的质心
        COM_before_centering = pos.mean(0, keepdims=True)
        pos_recentered = pos - COM_before_centering  # 中心化坐标
        if recenter:
            pos = pos_recentered  # 根据配置决定是否使用中心化坐标
        
        # 计算当前质心
        COM = pos.mean(0, keepdims=True)
        # 创建虚拟节点掩码，标记第一个节点为虚拟节点
        virtual_node_mask = np.zeros(pos.shape[0], dtype=bool)
        virtual_node_mask[0] = True

        # 创建药效团类型的one-hot编码（t=0时的干净特征）
        x = np.zeros((pharm_types.size, self.max_node_types_x4))
        x[np.arange(pharm_types.size), pharm_types] = 1  # one-hot编码
        data['x'] = torch.from_numpy(x.copy()).float() * self.scale_node_features_x4
        data['x_0'] = data['x'].clone()  # 保存干净的特征用于损失计算

        # 存储其他t=0时的连续特征
        data['pos'] = torch.from_numpy(pos.copy()).float()  # 药效团位置
        data['direction'] = torch.from_numpy(direction.copy()).float() * self.scale_vector_features_x4  # 药效团方向
        data['virtual_node_mask'] = torch.from_numpy(virtual_node_mask)  # 虚拟节点掩码
        
        # 4. --- 对所有特征进行扩散噪声处理 ---
        # 连续特征噪声处理（坐标和方向）
        pos_noise = np.random.randn(*pos.shape)  # 生成位置噪声
        pos_noise[virtual_node_mask] = 0.0  # 虚拟节点不添加噪声
        if remove_noise_COM:
            # 移除真实节点噪声的质心，保持分子整体位置稳定
            pos_noise[~virtual_node_mask] = pos_noise[~virtual_node_mask] - pos_noise[~virtual_node_mask].mean(0)
        
        direction_noise = np.random.randn(*direction.shape)  # 生成方向噪声
        direction_noise[virtual_node_mask] = 0.0  # 虚拟节点不添加噪声

        # 存储生成的噪声向量
        data['pos_noise'] = torch.from_numpy(pos_noise.copy()).float()  # 位置噪声
        data['direction_noise'] = torch.from_numpy(direction_noise.copy()).float()  # 方向噪声

        # 将噪声应用到连续特征上
        data['pos_forward_noised'] = data['pos'] + data['pos_noise'] * sigma_dash_t  # 加噪后的位置
        data['direction_forward_noised'] = data['direction'] + data['direction_noise'] * sigma_dash_t  # 加噪后的方向
        
        # 离散特征（药效团类型）的噪声处理
        device = data['x'].device  # 获取张量设备
        t_tensor = torch.tensor([t], device=device).long()  # 时间步张量
        
        # 只对非虚拟节点的药效团类型应用离散扩散噪声
        x_clean_no_vn = data['x_0'][~virtual_node_mask]  # 提取真实药效团的干净特征
        
        if x_clean_no_vn.shape[0] > 0:  # 确保存在真实药效团节点才进行加噪
            # 使用离散扩散器对药效团类型添加噪声
            x_noised_no_vn = self.x4_pharm_diffuser.apply_noise(x_clean_no_vn, t_tensor, device)
            # 创建完整的加噪特征矩阵
            x_forward_noised = data['x_0'].clone()
            x_forward_noised[~virtual_node_mask] = x_noised_no_vn  # 只更新真实节点的特征
        else:  # 如果只有虚拟节点，则保持原始特征不变
            x_forward_noised = data['x_0'].clone()
            
        data['x_forward_noised'] = x_forward_noised  # 存储加噪后的药效团特征
        # 创建占位符噪声张量（新版损失函数不再需要显式噪声）
        data['x_noise'] = torch.zeros_like(data['x'])

        return data
    
    def __getitem__(self, k):
        """获取数据集中第k个样本的训练数据
        
        这是数据集类的核心方法，负责处理单个分子样本，生成所有模态的训练数据。
        该方法会根据配置生成x1（原子图）、x2（分子表面）、x3（静电势）、x4（药效团）
        四个模态的数据，并应用相应的扩散噪声。
        
        参数:
            k (int): 样本索引
            
        返回:
            torch_geometric.data.HeteroData: 包含所有模态数据的异构图数据对象
        """
        
        # 从预处理数据中获取分子块和电荷信息
        mol_block = self.molblocks_and_charges[k][0]  # 分子的MOL格式数据
        charges = np.array(self.molblocks_and_charges[k][1])  # 预计算的原子电荷（如xTB计算结果）
        
        # 从MOL块创建RDKit分子对象，保留氢原子
        mol = rdkit.Chem.MolFromMolBlock(mol_block, removeHs = False)
        # 获取分子中每个原子的原子序数
        atomic_numbers = np.array([int(a.GetAtomicNum()) for a in mol.GetAtoms()])
        
        # 确保使用显式氢原子表示（如果要使用隐式氢，需要调整x2、x3、x4的计算方式）
        assert self.explicit_hydrogens
        
        # 获取分子坐标并进行中心化处理
        mol_coordinates = np.array(mol.GetConformer().GetPositions())
        mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis = 0)  # 中心化到原点

        # 更新分子对象的坐标
        mol = update_mol_coordinates(mol, mol_coordinates)

        # 获取每个原子的范德华半径（用于表面计算）
        radii = get_atomic_vdw_radii(mol)

        # 根据配置选择电荷计算方法
        if self.use_MMFF94_charges:
            charges = get_atomic_partial_charges(mol)  # 使用MMFF94方法重新计算电荷
        
        # 初始化数据字典，存储所有模态的数据
        data_dict = {
            'molecule_id': torch.tensor([k], dtype=torch.long),  # 分子ID
            'x1': {},  # 原子图模态数据
            'x2': {},  # 分子表面模态数据
            'x3': {},  # 静电势模态数据
            'x4': {},  # 药效团模态数据
        }
        
        
        # === x1模态（原子图）数据生成 ===
        if self.x1:
            # 获取x1模态的噪声调度时间序列
            ts = self.noise_schedule_dict['x1']['ts']
            
            # 使用分层时间步采样策略，而非均匀采样
            # 这种策略有助于模型更好地学习不同噪声水平的去噪过程
            T = ts.shape[0]  # 总时间步数

            # 将时间步分为三个区间，采用不同的采样概率
            ts_end = ts[0:int(T*0.125)]      # 结束阶段：0到12.5%（如T=400时为0-50）
            ts_middle = ts[int(T*0.125):int(T*0.625)]  # 中间阶段：12.5%到62.5%（如T=400时为50-250）
            ts_start = ts[int(T*0.625):]     # 开始阶段：62.5%到100%（如T=400时为250-400）
            
            # 随机决定从哪个时间段采样
            ts_prob = np.random.uniform(0,1)

            if ts_prob < 0.075:
                t = np.random.choice(ts_end)    # 7.5%概率采样结束阶段（低噪声）
            elif ts_prob < (0.075 + 0.75):
                t = np.random.choice(ts_middle) # 75%概率采样中间阶段（中等噪声）
            else:
                t = np.random.choice(ts_start)  # 17.5%概率采样开始阶段（高噪声）
            
            # 保存x1模态的时间步信息，供其他模态使用
            ts_x1 = ts
            t_x1 = t
            t_idx = np.where(ts == t)[0][0]  # 获取时间步在序列中的索引

            # 根据时间步获取对应的噪声调度参数
            alpha_t = self.noise_schedule_dict['x1']['alpha_ts'][t_idx]        # 信号保持系数
            sigma_t = self.noise_schedule_dict['x1']['sigma_ts'][t_idx]        # 噪声强度系数
            alpha_dash_t = self.noise_schedule_dict['x1']['alpha_dash_ts'][t_idx]  # 累积信号系数
            var_dash_t = self.noise_schedule_dict['x1']['var_dash_ts'][t_idx]      # 累积方差
            sigma_dash_t = self.noise_schedule_dict['x1']['sigma_dash_ts'][t_idx]  # 累积噪声强度
            
            # 生成x1模态的训练数据
            x1_data, x1_pos, x1_virtual_node_mask = self.get_x1_data(mol, t, alpha_dash_t, sigma_dash_t)
            
            # 将噪声调度参数添加到数据中
            x1_data['alpha_t'] = torch.tensor([alpha_t], dtype=torch.float)
            x1_data['sigma_t'] = torch.tensor([sigma_t], dtype=torch.float)
            x1_data['alpha_dash_t'] = torch.tensor([alpha_dash_t], dtype=torch.float)
            x1_data['sigma_dash_t'] = torch.tensor([sigma_dash_t], dtype=torch.float)
                        
            data_dict['x1'] = x1_data  # 存储x1模态数据
        
        
        # === x2模态（分子表面）数据生成 ===
        if self.x2:
            
            # 根据配置决定x2模态的时间步采样策略
            if self.independent_timesteps_x2:
                # 独立时间步：x2模态使用自己的噪声调度
                ts = self.noise_schedule_dict['x2']['ts']
                
                # 使用与x1相同的分层时间步采样策略
                T = ts.shape[0]  # 总时间步数
                ts_end = ts[0:int(T*0.125)]      # 结束阶段：0到12.5%（如T=400时为0-50）
                ts_middle = ts[int(T*0.125):int(T*0.625)]  # 中间阶段：12.5%到62.5%（如T=400时为50-250）
                ts_start = ts[int(T*0.625):]     # 开始阶段：62.5%到100%（如T=400时为250-400）
                
                # 随机决定从哪个时间段采样
                ts_prob = np.random.uniform(0,1)
                if ts_prob < 0.075:
                    t = np.random.choice(ts_end)    # 7.5%概率采样结束阶段（低噪声）
                elif ts_prob < (0.075 + 0.75):
                    t = np.random.choice(ts_middle) # 75%概率采样中间阶段（中等噪声）
                else:
                    t = np.random.choice(ts_start)  # 17.5%概率采样开始阶段（高噪声）
                
            else:
                # 共享时间步：x2模态与x1模态使用相同的时间步
                assert self.x1 == True  # 确保x1模态已启用
                # 验证x1和x2的时间序列一致
                assert (self.noise_schedule_dict['x2']['ts'] == self.noise_schedule_dict['x1']['ts']).all()
                ts = ts_x1  # 使用x1的时间序列
                t = t_x1    # 使用x1的时间步
                
            # 保存x2模态的时间步信息
            ts_x2 = ts
            t_x2 = t
            t_idx = np.where(ts == t)[0][0]  # 获取时间步在序列中的索引
            
            # 根据时间步获取对应的噪声调度参数
            alpha_t = self.noise_schedule_dict['x2']['alpha_ts'][t_idx]        # 信号保持系数
            sigma_t = self.noise_schedule_dict['x2']['sigma_ts'][t_idx]        # 噪声强度系数
            alpha_dash_t = self.noise_schedule_dict['x2']['alpha_dash_ts'][t_idx]  # 累积信号系数
            var_dash_t = self.noise_schedule_dict['x2']['var_dash_ts'][t_idx]      # 累积方差
            sigma_dash_t = self.noise_schedule_dict['x2']['sigma_dash_ts'][t_idx]  # 累积噪声强度
            
            # 根据是否启用x1模态来确定原子中心坐标
            if self.x1:
                # 如果启用x1模态，使用x1的原子位置（排除虚拟节点）
                atom_centers = x1_pos[~x1_virtual_node_mask,:]
                # 如果需要添加虚拟节点且不重新中心化，则虚拟节点位置为原子中心的平均值
                virtual_node_pos = atom_centers.mean(0)[None, ...] if ((self.add_virtual_node_x2) and (self.recenter_x2 == False)) else None
            else:
                # 如果未启用x1模态，使用原始分子坐标
                atom_centers = mol_coordinates
                # 虚拟节点位置将在get_x2_data中重新设置为x2的质心（而非分子坐标的质心）
                virtual_node_pos = None
            
            # 生成x2模态的训练数据（分子表面点云）
            x2_data, x2_pos, x2_virtual_node_mask = self.get_x2_data(
                radii,                    # 原子范德华半径
                atom_centers,             # 原子中心坐标
                self.num_points_x2,       # 表面点数量
                self.recenter_x2,         # 是否重新中心化
                self.add_virtual_node_x2, # 是否添加虚拟节点
                self.remove_noise_COM_x2, # 是否移除噪声质心
                t, alpha_dash_t, sigma_dash_t,  # 时间步和噪声参数
                virtual_node_pos = virtual_node_pos,  # 虚拟节点位置
            )
            
            # 将噪声调度参数添加到数据中
            x2_data['alpha_t'] = torch.tensor([alpha_t], dtype=torch.float)
            x2_data['sigma_t'] = torch.tensor([sigma_t], dtype=torch.float)
            x2_data['alpha_dash_t'] = torch.tensor([alpha_dash_t], dtype=torch.float)
            x2_data['sigma_dash_t'] = torch.tensor([sigma_dash_t], dtype=torch.float)
                        
            data_dict['x2'] = x2_data  # 存储x2模态数据
        
        
        
        # === x3模态（静电势）数据生成 ===
        if self.x3:
            # 根据配置决定x3模态的时间步采样策略
            if self.independent_timesteps_x3:
                # 独立时间步：x3模态使用自己的噪声调度
                ts = self.noise_schedule_dict['x3']['ts']
                
                # 使用与x1相同的分层时间步采样策略
                T = ts.shape[0]  # 总时间步数
                ts_end = ts[0:int(T*0.125)]      # 结束阶段：0到12.5%（如T=400时为0-50）
                ts_middle = ts[int(T*0.125):int(T*0.625)]  # 中间阶段：12.5%到62.5%（如T=400时为50-250）
                ts_start = ts[int(T*0.625):]     # 开始阶段：62.5%到100%（如T=400时为250-400）
                
                # 随机决定从哪个时间段采样
                ts_prob = np.random.uniform(0,1)
                if ts_prob < 0.075:
                    t = np.random.choice(ts_end)    # 7.5%概率采样结束阶段（低噪声）
                elif ts_prob < (0.075 + 0.75):
                    t = np.random.choice(ts_middle) # 75%概率采样中间阶段（中等噪声）
                else:
                    t = np.random.choice(ts_start)  # 17.5%概率采样开始阶段（高噪声）
                
            else:
                # 共享时间步：x3模态与x1模态使用相同的时间步
                assert self.x1 == True  # 确保x1模态已启用
                
                # 验证x1和x3的时间序列一致
                assert (self.noise_schedule_dict['x3']['ts'] == self.noise_schedule_dict['x1']['ts']).all()
                ts = ts_x1  # 使用x1的时间序列
                t = t_x1    # 使用x1的时间步
            
            # 保存x3模态的时间步信息
            ts_x3 = ts
            t_x3 = t
            t_idx = np.where(ts == t)[0][0]  # 获取时间步在序列中的索引
            
            # 根据时间步获取对应的噪声调度参数
            alpha_t = self.noise_schedule_dict['x3']['alpha_ts'][t_idx]        # 信号保持系数
            sigma_t = self.noise_schedule_dict['x3']['sigma_ts'][t_idx]        # 噪声强度系数
            alpha_dash_t = self.noise_schedule_dict['x3']['alpha_dash_ts'][t_idx]  # 累积信号系数
            var_dash_t = self.noise_schedule_dict['x3']['var_dash_ts'][t_idx]      # 累积方差
            sigma_dash_t = self.noise_schedule_dict['x3']['sigma_dash_ts'][t_idx]  # 累积噪声强度
            
            # 根据是否启用x1模态来确定原子中心坐标
            if self.x1:
                # 如果启用x1模态，使用x1的原子位置（排除虚拟节点）
                atom_centers = x1_pos[~x1_virtual_node_mask,:]
                # 如果需要添加虚拟节点且不重新中心化，则虚拟节点位置为原子中心的平均值
                virtual_node_pos = atom_centers.mean(0)[None, ...] if ((self.add_virtual_node_x3) and (self.recenter_x3 == False)) else None  
            else:
                # 如果未启用x1模态，使用原始分子坐标（可能需要在分配给电荷中心前进行中心化）
                atom_centers = mol_coordinates
                # 虚拟节点位置将在get_x3_data中重新设置为x3的质心（而非分子坐标的质心）
                virtual_node_pos = None
            
            # x3模态使用与x2相同的表面点云生成方法
            # 首先生成表面点云的基础数据结构
            x3_data, x3_pos, x3_virtual_node_mask = self.get_x2_data(
                radii,                    # 原子范德华半径
                atom_centers,             # 原子中心坐标
                self.num_points_x3,       # 表面点数量
                self.recenter_x3,         # 是否重新中心化
                self.add_virtual_node_x3, # 是否添加虚拟节点
                self.remove_noise_COM_x3, # 是否移除噪声质心
                t, alpha_dash_t, sigma_dash_t,  # 时间步和噪声参数
                virtual_node_pos = virtual_node_pos,  # 虚拟节点位置
            )
            
            # 在表面点云基础上计算静电势特征
            x3_data = self.get_x3_data_electrostatics_only(
                charges,                  # 原子部分电荷
                atom_centers,             # 原子中心坐标（用于静电势计算）
                x3_data,                  # 表面点云数据
                x3_pos,                   # 表面点位置
                t, alpha_dash_t, sigma_dash_t,  # 时间步和噪声参数
            )
            
            # 注意：这里不再需要重新设置virtual_node_mask，
            # 因为get_x3_data_electrostatics_only不会覆盖它
            # virtual_node_mask已由get_x2_data正确设置为tensor格式
            
            # 将噪声调度参数添加到数据中
            x3_data['alpha_t'] = torch.tensor([alpha_t], dtype=torch.float)
            x3_data['sigma_t'] = torch.tensor([sigma_t], dtype=torch.float)
            x3_data['alpha_dash_t'] = torch.tensor([alpha_dash_t], dtype=torch.float)
            x3_data['sigma_dash_t'] = torch.tensor([sigma_dash_t], dtype=torch.float)
                        
            data_dict['x3'] = x3_data  # 存储x3模态数据

        
        # === x4模态（药效团）数据生成 ===
        if self.x4:
            
            # 根据配置决定x4模态的时间步采样策略
            if self.independent_timesteps_x4:
                # 独立时间步：x4模态使用自己的噪声调度
                ts = self.noise_schedule_dict['x4']['ts']
                
                # 使用与x1相同的分层时间步采样策略
                T = ts.shape[0]  # 总时间步数
                ts_end = ts[0:int(T*0.125)]      # 结束阶段：0到12.5%（如T=400时为0-50）
                ts_middle = ts[int(T*0.125):int(T*0.625)]  # 中间阶段：12.5%到62.5%（如T=400时为50-250）
                ts_start = ts[int(T*0.625):]     # 开始阶段：62.5%到100%（如T=400时为250-400）
                
                # 随机决定从哪个时间段采样
                ts_prob = np.random.uniform(0,1)
                if ts_prob < 0.075:
                    t = np.random.choice(ts_end)    # 7.5%概率采样结束阶段（低噪声）
                elif ts_prob < (0.075 + 0.75):
                    t = np.random.choice(ts_middle) # 75%概率采样中间阶段（中等噪声）
                else:
                    t = np.random.choice(ts_start)  # 17.5%概率采样开始阶段（高噪声）
                
            else:
                # 共享时间步：x4模态与x1模态使用相同的时间步
                assert self.x1 == True  # 确保x1模态已启用
                # 验证x1和x4的时间序列一致
                assert (self.noise_schedule_dict['x4']['ts'] == self.noise_schedule_dict['x1']['ts']).all()
                ts = ts_x1  # 使用x1的时间序列
                t = t_x1    # 使用x1的时间步
                
            # 保存x4模态的时间步信息
            ts_x4 = ts
            t_x4 = t
            t_idx = np.where(ts == t)[0][0]  # 获取时间步在序列中的索引
            
            # 根据时间步获取对应的噪声调度参数
            alpha_t = self.noise_schedule_dict['x4']['alpha_ts'][t_idx]        # 信号保持系数
            sigma_t = self.noise_schedule_dict['x4']['sigma_ts'][t_idx]        # 噪声强度系数
            alpha_dash_t = self.noise_schedule_dict['x4']['alpha_dash_ts'][t_idx]  # 累积信号系数
            var_dash_t = self.noise_schedule_dict['x4']['var_dash_ts'][t_idx]      # 累积方差
            sigma_dash_t = self.noise_schedule_dict['x4']['sigma_dash_ts'][t_idx]  # 累积噪声强度
            
            # 根据是否启用x1模态来确定原子中心坐标
            if self.x1:
                # 如果启用x1模态，使用x1的原子位置（排除虚拟节点）
                atom_centers = x1_pos[~x1_virtual_node_mask,:]
                # 如果需要添加虚拟节点且不重新中心化，则虚拟节点位置为原子中心的平均值
                virtual_node_pos = atom_centers.mean(0)[None, ...] if ((self.add_virtual_node_x4) and (self.recenter_x4 == False)) else None
            else:
                # 如果未启用x1模态，使用原始分子坐标
                atom_centers = mol_coordinates
                # 虚拟节点位置将在get_x4_data中重新设置为x4的质心（而非分子坐标的质心）
                virtual_node_pos = None
            
            # 生成x4模态的训练数据（药效团特征）
            x4_data = self.get_x4_data(
                mol,                      # RDKit分子对象
                self.recenter_x4,         # 是否重新中心化
                self.add_virtual_node_x4, # 是否添加虚拟节点
                self.remove_noise_COM_x4, # 是否移除噪声质心
                t,                        # 当前时间步
                alpha_dash_t,             # 累积信号系数
                sigma_dash_t,             # 累积噪声强度
                virtual_node_pos,         # 虚拟节点位置
            )
            
            # 将噪声调度参数添加到数据中
            x4_data['alpha_t'] = torch.tensor([alpha_t], dtype=torch.float)
            x4_data['sigma_t'] = torch.tensor([sigma_t], dtype=torch.float)
            x4_data['alpha_dash_t'] = torch.tensor([alpha_dash_t], dtype=torch.float)
            x4_data['sigma_dash_t'] = torch.tensor([sigma_dash_t], dtype=torch.float)
                        
            data_dict['x4'] = x4_data  # 存储x4模态数据
        
        
        # === 构建HeteroData对象 ===
        # 创建PyTorch Geometric的异构图数据对象，用于存储多模态数据
        data = torch_geometric.data.HeteroData()
        
        # 添加分子ID信息（如果存在）
        if 'molecule_id' in data_dict:
            data.molecule_id = data_dict['molecule_id']

        # === x1模态数据存储（原子图） ===
        if 'x1' in data_dict and data_dict['x1']:
            x1_data = data_dict['x1']
            
            # 定义节点相关的数据键
            node_keys = ['pos', 'pos_recentered', 'pos_forward_noised', 'pos_noise',  # 位置相关
                'x', 'x_0', 'x_forward_noised', 'x_noise',                            # 原子类型相关
                'virtual_node_mask', 'com', 'com_before_centering',                   # 掩码和质心
                'timestep', 'alpha_t', 'sigma_t', 'alpha_dash_t', 'sigma_dash_t']    # 时间步和噪声参数

            # 将节点数据存储到x1模态中
            for key in node_keys:
                if key in x1_data:
                    data['x1'][key] = x1_data[key]

            # === x1模态边数据存储（化学键） ===
            # 将所有边相关的属性存入异构图的边类型 ('x1', 'bond', 'x1')
            edge_keys = ['bond_edge_index', 'bond_edge_mask',
                         'bond_edge_x', 'bond_edge_x_0', 'bond_edge_x_forward_noised', 'bond_edge_x_noise']
            
            data['x1', 'bond', 'x1'].edge_index = x1_data['bond_edge_index']           # 边索引
            data['x1', 'bond', 'x1'].mask = x1_data['bond_edge_mask']                   # 边掩码
            data['x1', 'bond', 'x1'].x = x1_data['bond_edge_x']                         # 原始边特征
            data['x1', 'bond', 'x1'].x_0 = x1_data['bond_edge_x_0']                     # 去噪目标边特征
            data['x1', 'bond', 'x1'].x_forward_noised = x1_data['bond_edge_x_forward_noised']  # 前向加噪边特征
            data['x1', 'bond', 'x1'].x_noise = x1_data['bond_edge_x_noise']             # 边噪声

        # === x2模态数据存储（分子表面点云） ===
        if 'x2' in data_dict and data_dict['x2']:
            # 直接复制所有x2模态的数据
            for key, value in data_dict['x2'].items():
                data['x2'][key] = value
                
        # === x3模态数据存储（静电势特征） ===
        if 'x3' in data_dict and data_dict['x3']:
            # 直接复制所有x3模态的数据
            for key, value in data_dict['x3'].items():
                data['x3'][key] = value

        # === x4模态数据存储（药效团特征） ===
        if 'x4' in data_dict and data_dict['x4']:
            x4_data = data_dict['x4']
            # 直接复制所有x4模态的数据
            for key, value in x4_data.items():
                data['x4'][key] = value

        return data  # 返回构建完成的异构图数据对象
    
    
    # === 兼容性方法 ===
    # 为不同版本的PyTorch Geometric提供兼容性支持
    def __len__(self): 
        """返回数据集长度"""
        return self.length
        
    def len(self): 
        """返回数据集长度（兼容性方法）"""
        return self.__len__()
        
    def getitem(self, k): 
        """获取数据项（兼容性方法）"""
        return self.__getitem__(k)
        
    def get(self, k): 
        """获取数据项（兼容性方法）"""
        return self.__getitem__(k)
