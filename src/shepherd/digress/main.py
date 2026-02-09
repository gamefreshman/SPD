# -*- coding: utf-8 -*-
"""
分子图扩散模型训练主程序

本文件是分子图生成扩散模型的主要训练入口，支持多种数据集和模型类型。
主要功能包括：
1. 配置管理和模型初始化
2. 数据集加载和预处理
3. 训练、验证和测试流程控制
4. 模型检查点管理
5. 指标计算和可视化

支持的数据集：
- 分子数据集：QM9, GuacaMol, MOSES
- 图数据集：SBM, Comm20, Planar

支持的模型类型：
- 离散扩散模型 (DiscreteDenoisingDiffusion)
- 连续扩散模型 (LiftedDenoisingDiffusion)

作者：SPD项目组
创建时间：2024年
"""

# 系统和工具库导入
import graph_tool as gt  # 图处理工具库
import os  # 操作系统接口
import pathlib  # 路径处理工具
import warnings  # 警告处理

# PyTorch相关导入
import torch  # PyTorch深度学习框架
torch.cuda.empty_cache()  # 清空CUDA缓存
import hydra  # 配置管理框架
from omegaconf import DictConfig  # 配置对象类型
from pytorch_lightning import Trainer  # PyTorch Lightning训练器
from pytorch_lightning.callbacks import ModelCheckpoint  # 模型检查点回调
from pytorch_lightning.utilities.warnings import PossibleUserWarning  # Lightning警告类型

# 项目内部模块导入
from src import utils  # 通用工具函数
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics  # 抽象训练指标

# 扩散模型导入
from diffusion_model import LiftedDenoisingDiffusion  # 连续扩散模型
from diffusion_model_discrete import DiscreteDenoisingDiffusion  # 离散扩散模型
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures  # 额外特征处理

# 忽略PyTorch Lightning的可能用户警告
warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    """
    恢复运行配置（测试模式）
    
    加载之前的配置而不允许更新键值，主要用于测试模式。
    从检查点文件中加载模型和配置，并更新相关设置。
    
    参数:
        cfg: 当前配置对象
        model_kwargs: 模型关键字参数字典
    
    返回:
        tuple: (更新后的配置对象, 加载的模型对象)
    """
    saved_cfg = cfg.copy()  # 保存当前配置的副本
    name = cfg.general.name + '_resume'  # 生成恢复运行的名称
    resume = cfg.general.test_only  # 获取测试检查点路径
    # 根据模型类型加载相应的扩散模型
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg  # 使用模型中保存的配置
    cfg.general.test_only = resume  # 设置测试检查点路径
    cfg.general.name = name  # 更新运行名称
    cfg = utils.update_config_with_new_keys(cfg, saved_cfg)  # 用新配置更新键值
    return cfg, model


def get_resume_adaptive(cfg, model_kwargs):
    """
    自适应恢复运行配置（继续训练模式）
    
    加载之前的配置但允许进行一些修改，主要用于恢复训练。
    从检查点文件中加载模型，并用当前配置覆盖原有配置的某些部分。
    
    参数:
        cfg: 当前配置对象
        model_kwargs: 模型关键字参数字典
    
    返回:
        tuple: (更新后的配置对象, 加载的模型对象)
    """
    saved_cfg = cfg.copy()  # 保存当前配置的副本
    # 获取当前文件路径以确定基础路径
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]  # 提取根目录路径

    resume_path = os.path.join(root_dir, cfg.general.resume)  # 构建完整的恢复路径

    # 根据模型类型加载相应的扩散模型
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion.load_from_checkpoint(resume_path, **model_kwargs)
    new_cfg = model.cfg  # 获取模型中保存的配置

    # 用当前配置覆盖模型配置的所有参数
    for category in cfg:
        for arg in cfg[category]:
            new_cfg[category][arg] = cfg[category][arg]

    new_cfg.general.resume = resume_path  # 设置恢复路径
    new_cfg.general.name = new_cfg.general.name + '_resume'  # 更新运行名称

    new_cfg = utils.update_config_with_new_keys(new_cfg, saved_cfg)  # 用新配置更新键值
    return new_cfg, model



@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    """
    主训练函数
    
    根据配置文件初始化数据集、模型、指标和可视化工具，
    然后执行训练、验证或测试流程。
    
    参数:
        cfg: Hydra配置对象，包含所有训练参数
    """
    dataset_config = cfg["dataset"]  # 获取数据集配置

    # ========== 图数据集处理 ==========
    if dataset_config["name"] in ['sbm', 'comm20', 'planar']:
        # 导入图数据集相关的模块和指标
        from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
        from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
        from analysis.visualization import NonMolecularVisualization

        # 初始化图数据模块
        datamodule = SpectreGraphDataModule(cfg)
        
        # 根据数据集类型选择相应的采样指标
        if dataset_config['name'] == 'sbm':
            sampling_metrics = SBMSamplingMetrics(datamodule)  # 随机块模型采样指标
        elif dataset_config['name'] == 'comm20':
            sampling_metrics = Comm20SamplingMetrics(datamodule)  # 社区网络采样指标
        else:
            sampling_metrics = PlanarSamplingMetrics(datamodule)  # 平面图采样指标

        # 初始化数据集信息和训练指标
        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
        visualization_tools = NonMolecularVisualization()  # 非分子可视化工具

        # 配置额外特征处理器
        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        else:
            extra_features = DummyExtraFeatures()  # 虚拟特征处理器
        domain_features = DummyExtraFeatures()  # 领域特征处理器

        # 计算输入输出维度
        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        # 构建模型参数字典
        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}

    # ========== 分子数据集处理 ==========
    elif dataset_config["name"] in ['qm9', 'guacamol', 'moses']:
        # 导入分子数据集相关的指标和可视化工具
        from metrics.molecular_metrics import TrainMolecularMetrics, SamplingMolecularMetrics
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from diffusion.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        # 根据数据集名称初始化相应的分子数据模块
        if dataset_config["name"] == 'qm9':
            # QM9量子化学数据集
            from datasets import qm9_dataset
            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            # 获取训练集SMILES字符串用于新颖性评估
            train_smiles = qm9_dataset.get_train_smiles(cfg=cfg, train_dataloader=datamodule.train_dataloader(),
                                                        dataset_infos=dataset_infos, evaluate_dataset=False)
        elif dataset_config['name'] == 'guacamol':
            # GuacaMol分子生成基准数据集
            from datasets import guacamol_dataset
            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
            train_smiles = None  # GuacaMol不需要训练SMILES

        elif dataset_config.name == 'moses':
            # MOSES分子生成数据集
            from datasets import moses_dataset
            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            train_smiles = None  # MOSES不需要训练SMILES
        else:
            raise ValueError("数据集未实现")

        # 配置分子特征处理器
        if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
            extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)  # 分子领域特征
        else:
            extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

        # 计算输入输出维度
        dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                                domain_features=domain_features)

        # 根据模型类型选择训练指标
        if cfg.model.type == 'discrete':
            train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)  # 离散分子训练指标
        else:
            train_metrics = TrainMolecularMetrics(dataset_infos)  # 连续分子训练指标

        # 初始化采样指标（训练期间不评估新颖性）
        sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
        visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

        # 构建模型参数字典
        model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                        'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                        'extra_features': extra_features, 'domain_features': domain_features}
    else:
        raise NotImplementedError("未知数据集 {}".format(cfg["dataset"]))

    # ========== 模型恢复和初始化 ==========
    if cfg.general.test_only:
        # 测试模式：完全加载之前的配置
        cfg, _ = get_resume(cfg, model_kwargs)
        os.chdir(cfg.general.test_only.split('checkpoints')[0])
    elif cfg.general.resume is not None:
        # 恢复训练模式：可以覆盖之前配置的某些部分
        cfg, _ = get_resume_adaptive(cfg, model_kwargs)
        os.chdir(cfg.general.resume.split('checkpoints')[0])

    # 创建必要的文件夹
    utils.create_folders(cfg)

    # 根据模型类型初始化扩散模型
    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)  # 离散扩散模型
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)  # 连续扩散模型

    # ========== 回调函数配置 ==========
    callbacks = []
    if cfg.train.save_model:
        # 配置模型检查点保存
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}",
                                              filename='{epoch}',
                                              monitor='val/epoch_NLL',  # 监控验证集负对数似然
                                              save_top_k=5,  # 保存最好的5个模型
                                              mode='min',  # 最小化监控指标
                                              every_n_epochs=1)
        # 保存最新的检查点
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{cfg.general.name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    # 配置指数移动平均（EMA）回调
    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    # ========== 训练器配置 ==========
    name = cfg.general.name
    if name == 'debug':
        print("[警告]: 运行名称为'debug' -- 将使用快速开发运行模式。")

    # 检查GPU可用性
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    
    # 初始化PyTorch Lightning训练器
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,  # 梯度裁剪值
                      strategy="ddp_find_unused_parameters_true",  # 分布式训练策略，用于加载旧检查点
                      accelerator='gpu' if use_gpu else 'cpu',  # 加速器类型
                      devices=cfg.general.gpus if use_gpu else 1,  # 设备数量
                      max_epochs=cfg.train.n_epochs,  # 最大训练轮数
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,  # 验证频率
                      fast_dev_run=cfg.general.name == 'debug',  # 调试模式快速运行
                      enable_progress_bar=False,  # 禁用进度条
                      callbacks=callbacks,  # 回调函数列表
                      log_every_n_steps=50 if name != 'debug' else 1,  # 日志记录频率
                      logger = [])  # 禁用默认日志记录器

    # ========== 训练/测试执行 ==========
    if not cfg.general.test_only:
        # 训练模式：执行训练和测试
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
        # 非调试和测试模式下执行最终测试
        if cfg.general.name not in ['debug', 'test']:
            trainer.test(model, datamodule=datamodule)
    else:
        # 测试模式：仅执行测试
        # 首先评估指定的测试检查点
        trainer.test(model, datamodule=datamodule, ckpt_path=cfg.general.test_only)
        
        # 如果配置了评估所有检查点，则遍历目录中的所有检查点文件
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            print("检查点目录:", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if '.ckpt' in file:  # 检查点文件
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue  # 跳过已经测试过的检查点
                    print("加载检查点", ckpt_path)
                    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


# ========== 程序入口点 ==========
if __name__ == '__main__':
    """
    程序主入口
    
    当脚本直接运行时，调用main函数开始训练或测试流程。
    Hydra会自动处理配置文件的加载和命令行参数的解析。
    """
    main()  # 启动主训练函数
