# -*- coding: utf-8 -*-
"""
离散扩散模型实现文件

本文件实现了用于分子图生成的离散扩散模型，基于PyTorch Lightning框架。
主要功能包括：
1. 离散扩散过程的前向和反向采样
2. 分子图的噪声添加和去噪
3. 训练、验证和测试流程
4. 分子生成和可视化

作者：SPD项目组
创建时间：2024年
"""

import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 神经网络函数库
import pytorch_lightning as pl  # PyTorch Lightning训练框架
import time  # 时间处理模块
import wandb  # 实验跟踪和可视化工具
import os  # 操作系统接口模块

# 模型相关导入
from models.transformer_model import GraphTransformer  # 图Transformer模型
from diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete,\
    MarginalUniformTransition  # 离散扩散的噪声调度器
from src.diffusion import diffusion_utils  # 扩散过程工具函数
from metrics.train_metrics import TrainLossDiscrete  # 训练损失计算
from metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL  # 评估指标
from src import utils  # 通用工具函数


class DiscreteDenoisingDiffusion(pl.LightningModule):
    """
    离散去噪扩散模型类
    
    基于PyTorch Lightning实现的离散扩散模型，用于分子图的生成任务。
    该模型通过逐步添加和去除噪声来学习数据分布，实现高质量的分子图生成。
    
    参数:
        cfg: 配置对象，包含模型和训练的所有超参数
        dataset_infos: 数据集信息对象，包含输入输出维度、节点分布等
        train_metrics: 训练指标计算器
        sampling_metrics: 采样指标计算器
        visualization_tools: 可视化工具
        extra_features: 额外特征提取器
        domain_features: 领域特征提取器
    """
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        # 从数据集信息中提取维度信息
        input_dims = dataset_infos.input_dims  # 输入特征维度字典
        output_dims = dataset_infos.output_dims  # 输出特征维度字典
        nodes_dist = dataset_infos.nodes_dist  # 节点数量分布

        # 基本配置参数
        self.cfg = cfg  # 保存配置对象
        self.name = cfg.general.name  # 模型名称
        self.model_dtype = torch.float32  # 模型数据类型
        self.T = cfg.model.diffusion_steps  # 扩散步数，控制噪声添加的时间步长度

        # 输入特征维度设置
        self.Xdim = input_dims['X']  # 节点特征输入维度（原子类型等）
        self.Edim = input_dims['E']  # 边特征输入维度（键类型等）
        self.ydim = input_dims['y']  # 全局特征输入维度
        
        # 输出特征维度设置
        self.Xdim_output = output_dims['X']  # 节点特征输出维度（去噪后的原子类型数）
        self.Edim_output = output_dims['E']  # 边特征输出维度（去噪后的键类型数）
        self.ydim_output = output_dims['y']  # 全局特征输出维度
        
        # 数据分布信息
        self.node_dist = nodes_dist  # 节点数量的概率分布，用于采样时确定分子大小
        self.dataset_info = dataset_infos  # 保存完整的数据集信息

        # 损失函数和评估指标初始化
        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)  # 训练阶段的离散损失函数

        # 验证阶段的评估指标
        self.val_nll = NLL()  # 负对数似然损失
        self.val_X_kl = SumExceptBatchKL()  # 节点特征的KL散度
        self.val_E_kl = SumExceptBatchKL()  # 边特征的KL散度
        self.val_X_logp = SumExceptBatchMetric()  # 节点特征的对数概率
        self.val_E_logp = SumExceptBatchMetric()  # 边特征的对数概率

        # 测试阶段的评估指标
        self.test_nll = NLL()  # 负对数似然损失
        self.test_X_kl = SumExceptBatchKL()  # 节点特征的KL散度
        self.test_E_kl = SumExceptBatchKL()  # 边特征的KL散度
        self.test_X_logp = SumExceptBatchMetric()  # 节点特征的对数概率
        self.test_E_logp = SumExceptBatchMetric()  # 边特征的对数概率

        # 外部传入的评估工具和特征处理器
        self.train_metrics = train_metrics  # 训练阶段的指标计算器
        self.sampling_metrics = sampling_metrics  # 采样阶段的指标计算器

        self.visualization_tools = visualization_tools  # 可视化工具集
        self.extra_features = extra_features  # 额外特征处理器
        self.domain_features = domain_features  # 领域特定特征处理器

        # 图变换器模型初始化
        self.model = GraphTransformer(n_layers=cfg.model.n_layers,  # 图变换器层数
                                      input_dims=input_dims,  # 输入特征维度
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,  # 隐藏层MLP维度
                                      hidden_dims=cfg.model.hidden_dims,  # 隐藏层维度
                                      output_dims=output_dims,  # 输出特征维度
                                      act_fn_in=nn.ReLU(),  # 输入激活函数
                                      act_fn_out=nn.ReLU())  # 输出激活函数

        # 噪声调度器初始化，控制扩散过程中的噪声强度
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        # 转移模型初始化：控制扩散过程中的状态转移
        if cfg.model.transition == 'uniform':
            # 均匀转移模型：所有类别的转移概率相等
            self.transition_model = DiscreteUniformTransition(x_classes=self.Xdim_output, e_classes=self.Edim_output,
                                                              y_classes=self.ydim_output)
            # 设置均匀分布的极限分布
            x_limit = torch.ones(self.Xdim_output) / self.Xdim_output  # 节点类别均匀分布
            e_limit = torch.ones(self.Edim_output) / self.Edim_output  # 边类别均匀分布
            y_limit = torch.ones(self.ydim_output) / self.ydim_output  # 全局特征均匀分布
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)
        elif cfg.model.transition == 'marginal':
            # 边际转移模型：基于数据集中类别的真实分布进行转移
            node_types = self.dataset_info.node_types.float()  # 获取节点类型统计
            x_marginals = node_types / torch.sum(node_types)  # 计算节点类型的边际分布

            edge_types = self.dataset_info.edge_types.float()  # 获取边类型统计
            e_marginals = edge_types / torch.sum(edge_types)  # 计算边类型的边际分布
            print(f"类别的边际分布: 节点 {x_marginals}, 边 {e_marginals}")
            # 使用边际分布初始化转移模型
            self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                              y_classes=self.ydim_output)
            # 设置基于真实数据分布的极限分布
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                                y=torch.ones(self.ydim_output) / self.ydim_output)

        # 保存超参数（忽略不需要保存的指标对象）
        self.save_hyperparameters(ignore=['train_metrics', 'sampling_metrics'])
        
        # 训练过程控制变量初始化
        self.start_epoch_time = None  # 记录每个epoch开始时间
        self.train_iterations = None  # 训练迭代次数计数器
        self.val_iterations = None  # 验证迭代次数计数器
        self.log_every_steps = cfg.general.log_every_steps  # 日志记录频率
        self.number_chain_steps = cfg.general.number_chain_steps  # 采样链步数
        self.best_val_nll = 1e8  # 记录最佳验证负对数似然值
        self.val_counter = 0  # 验证计数器

    def training_step(self, data, i):
        """
        训练步骤：执行单次前向传播和损失计算
        
        Args:
            data: 批次数据，包含图结构信息
            i: 当前步骤索引
            
        Returns:
            loss: 计算得到的训练损失
        """
        # 检查批次数据是否包含边，如果没有边则跳过
        if data.edge_index.numel() == 0:
            self.print("发现没有边的批次数据，跳过处理。")
            return
            
        # 将稀疏图数据转换为密集表示
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)  # 应用节点掩码
        X, E = dense_data.X, dense_data.E  # 提取节点特征和边特征
        
        # 对干净数据添加噪声，模拟扩散过程
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        
        # 计算额外特征数据
        extra_data = self.compute_extra_data(noisy_data)
        
        # 前向传播：预测去噪后的数据
        pred = self.forward(noisy_data, extra_data, node_mask)
        
        # 计算训练损失
        loss = self.train_loss(masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
                               true_X=X, true_E=E, true_y=data.y,
                               log=i % self.log_every_steps == 0)  # 根据日志频率决定是否记录

        # 计算并记录训练指标
        self.train_metrics(masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
                           log=i % self.log_every_steps == 0)

        return {'loss': loss}  # 返回损失值供PyTorch Lightning使用

    def configure_optimizers(self):
        """
        配置优化器：设置AdamW优化器及其参数
        
        Returns:
            torch.optim.AdamW: 配置好的AdamW优化器
        """
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        """
        训练开始时的初始化操作
        """
        # 获取训练数据加载器的迭代次数
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print("输入特征维度:", self.Xdim, self.Edim, self.ydim)
        # 仅在主进程中设置wandb（避免多进程重复初始化）
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        """
        每个训练epoch开始时的操作
        """
        self.print("开始训练epoch...")
        self.start_epoch_time = time.time()  # 记录epoch开始时间
        self.train_loss.reset()  # 重置训练损失累计器
        self.train_metrics.reset()  # 重置训练指标累计器

    def on_train_epoch_end(self) -> None:
        """
        每个训练epoch结束时的操作：记录指标和打印日志
        """
        # 获取并记录epoch级别的损失指标
        to_log = self.train_loss.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
                      f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
                      f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
                      f" -- {time.time() - self.start_epoch_time:.1f}s ")
        
        # 获取并记录epoch级别的训练指标
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}")
        
        # 打印GPU内存使用情况（如果可用）
        if torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        else:
            print("CUDA不可用，跳过内存摘要。")

    def on_validation_epoch_start(self) -> None:
        """
        每个验证epoch开始时的操作：重置所有验证指标
        """
        self.val_nll.reset()  # 重置验证负对数似然
        self.val_X_kl.reset()  # 重置节点特征KL散度
        self.val_E_kl.reset()  # 重置边特征KL散度
        self.val_X_logp.reset()  # 重置节点特征对数概率
        self.val_E_logp.reset()  # 重置边特征对数概率
        self.sampling_metrics.reset()  # 重置采样指标

    def validation_step(self, data, i):
        """
        验证步骤：计算验证损失和指标
        
        Args:
            data: 验证批次数据
            i: 当前步骤索引
            
        Returns:
            dict: 包含验证损失的字典
        """
        # 转换为密集表示并应用掩码
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        
        # 添加噪声并进行前向传播
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        
        # 计算验证损失
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y,  node_mask, test=False)
        return {'loss': nll}

    def on_validation_epoch_end(self) -> None:
        """
        每个验证epoch结束时的操作：计算并记录验证指标，执行采样
        """
        # 计算所有验证指标（KL散度乘以时间步数T进行标准化）
        metrics = [self.val_nll.compute(), self.val_X_kl.compute() * self.T, self.val_E_kl.compute() * self.T,
                   self.val_X_logp.compute(), self.val_E_logp.compute()]
        
        # 如果wandb可用，记录验证指标
        if wandb.run:
            wandb.log({"val/epoch_NLL": metrics[0],
                       "val/X_kl": metrics[1],
                       "val/E_kl": metrics[2],
                       "val/X_logp": metrics[3],
                       "val/E_logp": metrics[4]}, commit=False)

        # 打印验证指标
        self.print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type KL {metrics[1] :.2f} -- ",
                   f"Val Edge type KL: {metrics[2] :.2f}")

        # 使用Lightning默认日志记录器记录验证NLL，以便检查点回调监控
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)

        # 更新最佳验证损失
        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        self.print('验证损失: %.4f \t 最佳验证损失:  %.4f\n' % (val_nll, self.best_val_nll))

        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples = []

            ident = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                samples.extend(self.sample_batch(batch_id=ident, batch_size=to_generate, num_nodes=None,
                                                 save_final=to_save,
                                                 keep_chain=chains_save,
                                                 number_chain_steps=self.number_chain_steps))
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save
            self.print("Computing sampling metrics...")
            self.sampling_metrics.forward(samples, self.name, self.current_epoch, val_counter=-1, test=False,
                                          local_rank=self.local_rank)
            self.print(f'Done. Sampling took {time.time() - start:.2f} seconds\n')
            print("Validation epoch end ends...")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        nll = self.compute_val_loss(pred, noisy_data, dense_data.X, dense_data.E, data.y, node_mask, test=True)
        return {'loss': nll}

    def on_test_epoch_end(self) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_kl.compute(), self.test_E_kl.compute(),
                   self.test_X_logp.compute(), self.test_E_logp.compute()]
        if wandb.run:
            wandb.log({"test/epoch_NLL": metrics[0],
                       "test/X_kl": metrics[1],
                       "test/E_kl": metrics[2],
                       "test/X_logp": metrics[3],
                       "test/E_logp": metrics[4]}, commit=False)

        self.print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type KL {metrics[1] :.2f} -- ",
                   f"Test Edge type KL: {metrics[2] :.2f}")

        test_nll = metrics[0]
        if wandb.run:
            wandb.log({"test/epoch_NLL": test_nll}, commit=False)

        self.print(f'Test loss: {test_nll :.4f}')

        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        id = 0
        while samples_left_to_generate > 0:
            self.print(f'Samples left to generate: {samples_left_to_generate}/'
                       f'{self.cfg.general.final_model_samples_to_generate}', end='', flush=True)
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps))
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save
        self.print("Saving the generated graphs")
        filename = f'generated_samples1.txt'
        for i in range(2, 10):
            if os.path.exists(filename):
                filename = f'generated_samples{i}.txt'
            else:
                break
        with open(filename, 'w') as f:
            for item in samples:
                f.write(f"N={item[0].shape[0]}\n")
                atoms = item[0].tolist()
                f.write("X: \n")
                for at in atoms:
                    f.write(f"{at} ")
                f.write("\n")
                f.write("E: \n")
                for bond_list in item[1]:
                    for bond in bond_list:
                        f.write(f"{bond} ")
                    f.write("\n")
                f.write("\n")
        self.print("Generated graphs Saved. Computing sampling metrics...")
        self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True, local_rank=self.local_rank)
        self.print("Done testing.")


    def kl_prior(self, X, E, node_mask):
        """
        计算q(z1 | x)和先验p(z1)之间的KL散度
        
        虽然在实际损失中这部分通常可以忽略，但计算它有助于检测噪声调度中的错误。
        
        Args:
            X: 节点特征 (bs, n, dx)
            E: 边特征 (bs, n, n, de)
            node_mask: 节点掩码 (bs, n)
            
        Returns:
            torch.Tensor: KL散度值
        """
        # 计算最后的alpha值，alpha_T
        ones = torch.ones((X.size(0), 1), device=X.device)
        Ts = self.T * ones  # 时间步设为最大值T
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs, 1)

        # 获取累积转移矩阵
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)

        # 计算转移概率
        probX = X @ Qtb.X  # 节点特征转移概率 (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # 边特征转移概率 (bs, n, n, de_out)
        assert probX.shape == X.shape

        bs, n, _ = probX.shape

        # 扩展极限分布到批次和节点维度
        limit_X = self.limit_dist.X[None, None, :].expand(bs, n, -1).type_as(probX)
        limit_E = self.limit_dist.E[None, None, None, :].expand(bs, n, n, -1).type_as(probE)

        # 确保被掩码的行不会对损失产生贡献
        limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(true_X=limit_X.clone(),
                                                                                      true_E=limit_E.clone(),
                                                                                      pred_X=probX,
                                                                                      pred_E=probE,
                                                                                      node_mask=node_mask)

        # 计算KL散度
        kl_distance_X = F.kl_div(input=probX.log(), target=limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(input=probE.log(), target=limit_dist_E, reduction='none')

        # 返回节点和边特征的总KL散度
        return diffusion_utils.sum_except_batch(kl_distance_X) + \
               diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        pred_probs_X = F.softmax(pred.X, dim=-1)
        pred_probs_E = F.softmax(pred.E, dim=-1)
        pred_probs_y = F.softmax(pred.y, dim=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        # Compute distributions to compare with KL
        bs, n, d = X.shape
        prob_true = diffusion_utils.posterior_distributions(X=X, E=E, y=y, X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_true.E = prob_true.E.reshape((bs, n, n, -1))
        prob_pred = diffusion_utils.posterior_distributions(X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
                                                            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'],
                                                            y_t=noisy_data['y_t'], Qt=Qt, Qsb=Qsb, Qtb=Qtb)
        prob_pred.E = prob_pred.E.reshape((bs, n, n, -1))

        # Reshape and filter masked rows
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(true_X=prob_true.X,
                                                                                                true_E=prob_true.E,
                                                                                                pred_X=prob_pred.X,
                                                                                                pred_E=prob_pred.E,
                                                                                                node_mask=node_mask)
        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true.X, torch.log(prob_pred.X))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true.E, torch.log(prob_pred.E))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, node_mask):
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = X @ Q0.X  # (bs, n, dx_out)
        probE0 = E @ Q0.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled0 = diffusion_utils.sample_discrete_features(probX=probX0, probE=probE0, node_mask=node_mask)

        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).float()
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).float()
        y0 = sampled0.y
        assert (X.shape == X0.shape) and (E.shape == E0.shape)

        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)

        # Predictions
        noisy_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y, 'node_mask': node_mask,
                      't': torch.zeros(X0.shape[0], 1).type_as(y0)}
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        probX0 = F.softmax(pred0.X, dim=-1)
        probE0 = F.softmax(pred0.E, dim=-1)
        proby0 = F.softmax(pred0.y, dim=-1)

        # Set masked rows to arbitrary values that don't contribute to loss
        probX0[~node_mask] = torch.ones(self.Xdim_output).type_as(probX0)
        probE0[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))] = torch.ones(self.Edim_output).type_as(probE0)

        diag_mask = torch.eye(probE0.size(1)).type_as(probE0).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(probE0.size(0), -1, -1)
        probE0[diag_mask] = torch.ones(self.Edim_output).type_as(probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    def apply_noise(self, X, E, y, node_mask):
        """
        对数据应用噪声的核心方法
        
        该方法实现了离散扩散过程中的前向噪声添加过程，将原始数据转换为带噪声的数据。
        
        Args:
            X: 节点特征 (bs, n, dx)
            E: 边特征 (bs, n, n, de)
            y: 全局特征 (bs, dy)
            node_mask: 节点掩码 (bs, n)
            
        Returns:
            dict: 包含噪声数据和相关参数的字典
        """
        # 采样时间步t
        # 在评估时，t=0的损失会单独计算
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1  # 前一个时间步

        # 将整数时间步归一化到[0,1]区间
        t_float = t_int / self.T
        s_float = s_int / self.T

        # 计算噪声调度参数，用于去噪和损失计算
        beta_t = self.noise_schedule(t_normalized=t_float)                         # 当前时间步的噪声强度 (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # 前一时间步的累积alpha (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # 当前时间步的累积alpha (bs, 1)

        # 获取累积转移矩阵
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        # 验证转移矩阵的概率性质（每行和为1）
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # 计算转移概率
        probX = X @ Qtb.X  # 节点特征转移概率 (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # 边特征转移概率 (bs, n, n, de_out)

        # 根据转移概率采样噪声特征
        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        # 将采样结果转换为one-hot编码
        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        # 创建带噪声的数据占位符并应用掩码
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        # 返回包含所有噪声数据和参数的字典
        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        """
        计算变分下界的估计器
        
        该方法实现了ELBO（Evidence Lower BOund）的计算，包含四个主要组成部分：
        1. 节点数量的对数概率
        2. 先验KL散度
        3. 扩散损失
        4. 重构损失
        
        Args:
            pred: 模型预测结果 (batch_size, n, total_features)
            noisy_data: 噪声数据字典
            X: 节点特征 (bs, n, dx)
            E: 边特征 (bs, n, n, de)
            y: 全局特征 (bs, dy)
            node_mask: 节点掩码 (bs, n)
            test: 是否为测试模式
            
        Returns:
            torch.Tensor: 负对数似然损失值
        """
        t = noisy_data['t']

        # 1. 计算节点数量的对数概率
        N = node_mask.sum(1).long()  # 每个图的节点数量
        log_pN = self.node_dist.log_prob(N)  # 节点数量分布的对数概率

        # 2. 计算q(z_T | x)和p(z_T)之间的KL散度，理论上应该接近零
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. 计算扩散损失（所有时间步的损失）
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. 计算重构损失
        # 计算L0项：-log p(X, E, y | z_0) = 重构损失
        prob0 = self.reconstruction_logp(t, X, E, node_mask)

        # 计算重构项的对数概率
        loss_term_0 = self.val_X_logp(X * prob0.X.log()) + self.val_E_logp(E * prob0.E.log())

        # 组合所有损失项
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # 更新NLL指标对象并返回批次平均NLL
        nll = (self.test_nll if test else self.val_nll)(nlls)        # 在批次上取平均

        # 记录各项损失到wandb
        if wandb.run:
            wandb.log({"先验KL散度": kl_prior.mean(),
                       "扩散损失项": loss_all_t.mean(),
                       "节点数量对数概率": log_pN.mean(),
                       "重构损失项": loss_term_0,
                       '批次测试NLL' if test else '验证NLL': nll}, commit=False)
        return nll

    def forward(self, noisy_data, extra_data, node_mask):
        """
        模型前向传播方法
        
        将噪声数据和额外特征拼接后输入到图Transformer模型中进行预测。
        
        Args:
            noisy_data: 包含噪声数据的字典
            extra_data: 额外特征数据
            node_mask: 节点掩码
            
        Returns:
            模型预测结果
        """
        # 拼接噪声节点特征和额外节点特征
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        # 拼接噪声边特征和额外边特征
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        # 拼接噪声全局特征和额外全局特征
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        # 输入到图Transformer模型进行预测
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y



        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        if self.visualization_tools is not None:
            self.print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)       # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())
                self.print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
            self.print('\nVisualizing molecules...')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
           if last_step, return the graph prediction as well"""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)               # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)               # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
