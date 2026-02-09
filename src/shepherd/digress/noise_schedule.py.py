# -*- coding: utf-8 -*-
"""
噪声调度模块

本文件实现了扩散模型中的噪声调度功能，包括预定义的噪声调度和边际均匀转移。
主要功能包括：
1. 连续扩散模型的噪声调度 (PredefinedNoiseSchedule)
2. 离散扩散模型的噪声调度 (PredefinedNoiseScheduleDiscrete)
3. 边际均匀转移矩阵计算 (MarginalUniformTransition)

噪声调度控制扩散过程中噪声添加的强度和时间步长，
是扩散模型训练和采样的核心组件。

作者：SPD项目组
创建时间：2024年
"""

# 深度学习框架导入
import torch  # PyTorch深度学习框架
from torch.nn import functional as F  # PyTorch函数式接口
import numpy as np  # 数值计算库
import math  # 数学函数库

# 项目内部工具导入
from .utils import PlaceHolder  # 数据占位符工具类


# ==================== 噪声调度函数定义 ====================

def cosine_beta_schedule(timesteps, s=0.008):
    """
    余弦噪声调度函数（连续扩散模型）
    
    基于余弦函数的噪声调度策略，提供平滑的噪声增长曲线。
    该调度在扩散过程的早期保持较低的噪声水平，
    在后期快速增加噪声强度。
    
    参数:
        timesteps (int): 扩散时间步数
        s (float): 偏移参数，用于调整调度曲线，默认为0.008
        
    返回:
        np.ndarray: alpha^2值序列，形状为 (timesteps + 1,)
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """
    余弦噪声调度函数（离散扩散模型）
    
    为离散扩散模型提供余弦噪声调度，与连续版本类似但
    针对离散状态转换进行了优化。
    
    参数:
        timesteps (int): 扩散时间步数
        s (float): 偏移参数，用于调整调度曲线，默认为0.008
        
    返回:
        np.ndarray: beta值序列，形状为 (timesteps,)
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


def polynomial_beta_schedule(timesteps, exponent=2):
    """
    多项式噪声调度函数
    
    基于多项式函数的噪声调度策略，通过调整指数参数
    可以控制噪声增长的速度和形状。
    
    参数:
        timesteps (int): 扩散时间步数
        exponent (float): 多项式指数，控制噪声增长曲线
        
    返回:
        np.ndarray: alpha^2值序列，形状为 (timesteps + 1,)
    """
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = (1 - (x / timesteps) ** exponent)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)

class PredefinedNoiseSchedule(torch.nn.Module):
    """
    预定义噪声调度类（连续扩散模型）
    
    该类实现了连续扩散模型的噪声调度功能，为预定义（非学习）的噪声调度创建查找数组。
    通过计算alpha和sigma参数来控制扩散过程中的噪声强度。
    
    参数:
        noise_schedule (str): 噪声调度类型，支持 'cosine', 'custom'
        timesteps (int): 扩散过程的时间步数
    
    属性:
        timesteps (int): 时间步数
        gamma (torch.nn.Parameter): gamma参数，控制噪声强度的对数比值
    """

    def __init__(self, noise_schedule, timesteps):
        """
        初始化噪声调度器
        
        参数:
            noise_schedule (str): 噪声调度策略名称
            timesteps (int): 扩散时间步数
        """
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps  # 保存时间步数

        # 根据不同的噪声调度策略计算alpha^2值
        if noise_schedule == 'cosine':
            alphas2 = cosine_beta_schedule(timesteps)  # 余弦调度
        elif noise_schedule == 'custom':
            raise NotImplementedError()  # 自定义调度暂未实现
        else:
            raise ValueError(noise_schedule)  # 不支持的调度类型

        # print('alphas2', alphas2)  # 调试输出：alpha^2值

        sigmas2 = 1 - alphas2  # 计算sigma^2 = 1 - alpha^2

        # 计算对数值，用于数值稳定性
        log_alphas2 = np.log(alphas2)  # log(alpha^2)
        log_sigmas2 = np.log(sigmas2)  # log(sigma^2)

        # 计算gamma = log(sigma^2/alpha^2) = log(sigma^2) - log(alpha^2)
        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2     # 形状: (timesteps + 1, )

        # print('gamma', -log_alphas2_to_sigmas2)  # 调试输出：gamma参数

        # 将gamma参数注册为不可训练的模型参数
        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        """
        前向传播：根据时间步获取对应的gamma值
        
        参数:
            t (torch.Tensor): 时间步索引张量
            
        返回:
            torch.Tensor: 对应时间步的gamma值
        """
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]  # 返回指定时间步的gamma参数



class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """
    预定义噪声调度类（离散扩散模型）
    
    该类实现了离散扩散模型的噪声调度功能，通过beta调度来控制
    每个时间步的噪声添加强度。与连续版本不同，这里使用离散的
    beta参数和累积alpha参数。
    
    参数:
        timesteps (int): 扩散过程的时间步数
    
    属性:
        timesteps (int): 时间步数
        betas (torch.Tensor): beta参数序列，控制每步噪声强度
        alphas (torch.Tensor): alpha参数序列，alpha = 1 - beta
        alphas_bar (torch.Tensor): 累积alpha参数，alpha_bar_t = ∏(alpha_s) for s≤t
    """

    def __init__(self, timesteps):
        """
        初始化离散噪声调度器
        
        参数:
            timesteps (int): 扩散时间步数
        """
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps  # 保存时间步数

        # 使用余弦调度计算beta值序列
        betas = cosine_beta_schedule_discrete(timesteps)

        # 将beta参数注册为缓冲区（不参与梯度计算但会保存到模型状态）
        self.register_buffer('betas', torch.from_numpy(betas).float())

        # 计算alpha参数：alpha_t = 1 - beta_t，限制在合理范围内
        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        # 计算累积alpha参数：alpha_bar_t = ∏_{s=1}^t alpha_s
        # 使用对数空间计算以提高数值稳定性
        log_alpha = torch.log(self.alphas)  # 计算log(alpha)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)  # 累积求和得到log(alpha_bar)
        self.alphas_bar = torch.exp(log_alpha_bar)  # 指数化得到alpha_bar
        # print(f"[Noise schedule: {noise_schedule}] alpha_bar:", self.alphas_bar)  # 调试输出

    def forward(self, t_normalized=None, t_int=None):
        """
        前向传播：根据时间步获取对应的beta值
        
        参数:
            t_normalized (torch.Tensor, optional): 归一化时间步 [0,1]
            t_int (torch.Tensor, optional): 整数时间步 [0, timesteps]
            
        返回:
            torch.Tensor: 对应时间步的beta值
            
        注意:
            t_normalized 和 t_int 必须且只能提供一个
        """
        # 确保只提供一个时间步参数
        assert int(t_normalized is None) + int(t_int is None) == 1
        
        # 如果提供的是归一化时间步，转换为整数时间步
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
            
        return self.betas[t_int.long()]  # 返回对应时间步的beta值

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        """
        获取累积alpha参数（alpha_bar）
        
        参数:
            t_normalized (torch.Tensor, optional): 归一化时间步 [0,1]
            t_int (torch.Tensor, optional): 整数时间步 [0, timesteps]
            
        返回:
            torch.Tensor: 对应时间步的alpha_bar值
            
        注意:
            t_normalized 和 t_int 必须且只能提供一个
        """
        # 确保只提供一个时间步参数
        assert int(t_normalized is None) + int(t_int is None) == 1
        
        # 如果提供的是归一化时间步，转换为整数时间步
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
            
        return self.alphas_bar.to(t_int.device)[t_int.long()]  # 返回对应时间步的alpha_bar值


class MarginalUniformTransition:
    """
    边际均匀转移类
    
    该类实现了离散扩散模型中的边际均匀转移机制。在扩散过程中，
    离散状态（如原子类型、边类型、图标签）会按照特定的转移矩阵进行状态转换，
    最终收敛到各自的边际分布。
    
    转移矩阵的形式为：
    Qt = (1 - beta_t) * I + beta_t * U
    其中 I 是单位矩阵，U 是均匀转移矩阵，beta_t 是噪声强度
    
    参数:
        x_marginals (torch.Tensor): 节点特征的边际分布
        e_marginals (torch.Tensor): 边特征的边际分布
        y_classes (int): 图标签的类别数，默认为2
    
    属性:
        X_classes (int): 节点特征类别数
        E_classes (int): 边特征类别数
        y_classes (int): 图标签类别数
        x_marginals (torch.Tensor): 节点特征边际分布
        e_marginals (torch.Tensor): 边特征边际分布
        u_x (torch.Tensor): 节点特征均匀转移矩阵
        u_e (torch.Tensor): 边特征均匀转移矩阵
        u_y (torch.Tensor): 图标签均匀转移矩阵
    """
    
    def __init__(self, x_marginals, e_marginals, y_classes=2):
        """
        初始化边际均匀转移
        
        参数:
            x_marginals (torch.Tensor): 节点特征的边际分布
            e_marginals (torch.Tensor): 边特征的边际分布
            y_classes (int): 图标签的类别数
        """
        self.X_classes = len(x_marginals)  # 节点特征类别数
        self.E_classes = len(e_marginals)  # 边特征类别数
        self.y_classes = y_classes  # 图标签类别数
        self.x_marginals = x_marginals  # 保存节点特征边际分布
        self.e_marginals = e_marginals  # 保存边特征边际分布

        # 构建节点特征的均匀转移矩阵：每行都是边际分布
        self.u_x = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        # 构建边特征的均匀转移矩阵：每行都是边际分布
        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        # 构建图标签的均匀转移矩阵：每行都是均匀分布
        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes  # 归一化为均匀分布

    def get_Qt(self, beta_t, device):
        """
        获取单步转移矩阵 Qt
        
        计算从时间步 t-1 到 t 的一步转移矩阵：
        Qt = (1 - beta_t) * I + beta_t * U
        其中 I 是单位矩阵，U 是均匀转移矩阵
        
        参数:
            beta_t (torch.Tensor): 噪声强度参数，形状为 (bs,)
            device (torch.device): 计算设备
            
        返回:
            PlaceHolder: 包含转移矩阵的占位符对象
                - X: 节点特征转移矩阵，形状为 (bs, dx, dx)
                - E: 边特征转移矩阵，形状为 (bs, de, de)
                - y: 图标签转移矩阵，形状为 (bs, dy, dy)
        """
        # 为广播添加维度：(bs,) -> (bs, 1)
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        # 计算各类型的单步转移矩阵
        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)

        return PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t, device):
        """
        获取累积转移矩阵 Qt_bar
        
        计算从时间步 0 到 t 的累积转移矩阵：
        Qt_bar = ∏_{s=1}^t Qs
        
        对于边际均匀转移，有闭式解：
        Qt_bar = alpha_bar_t * I + (1 - alpha_bar_t) * U
        其中 alpha_bar_t = ∏_{s=1}^t (1 - beta_s)
        
        参数:
            alpha_bar_t (torch.Tensor): 累积alpha参数，形状为 (bs,)
            device (torch.device): 计算设备
            
        返回:
            PlaceHolder: 包含累积转移矩阵的占位符对象
                - X: 节点特征累积转移矩阵，形状为 (bs, dx, dx)
                - E: 边特征累积转移矩阵，形状为 (bs, de, de)
                - y: 图标签累积转移矩阵，形状为 (bs, dy, dy)
        """
        # 为广播添加维度：(bs,) -> (bs, 1)
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        # 计算各类型的累积转移矩阵
        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y

        return PlaceHolder(X=q_x, E=q_e, y=q_y)

