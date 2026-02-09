#!/usr/bin/env python3
"""
SPD (Shepherd) 分子生成模型训练脚本
支持标准扩散训练和DPO（Direct Preference Optimization）微调
"""

# ==================== 系统配置 ====================
import resource
# 提高文件描述符限制，防止多进程数据加载时文件句柄耗尽
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

# ==================== 导入依赖 ====================
# 标准库
import os
import shutil
import datetime
import pickle
import warnings
import argparse
import importlib
import multiprocessing
from functools import partial
from copy import deepcopy

# 第三方库
import numpy as np
import torch
import torch.multiprocessing
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import rdkit
import rdkit.Chem
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from lightning_fabric.utilities.seed import seed_everything

# 项目模块
from shepherd.model.model import Model
from shepherd.lightning_module import LightningModule
from shepherd.new_datasets import HeteroDataset
from shepherd.dpo_dataset import DPODataset
from shepherd.mixed_dataloader import create_mixed_dataloader
from shepherd.callbacks import OnlineSamplingCallback, DPOMetricsCallback
from shepherd.dpo_utils import OnlineSampler, ShepherdScorer
from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores

# ==================== 警告过滤 ====================
warnings.filterwarnings("ignore", category=UserWarning, message=".*UnsupportedFieldAttributeWarning.*")
warnings.filterwarnings("ignore", message=".*'repr' attribute.*")
warnings.filterwarnings("ignore", message=".*'frozen' attribute.*")

# ==================== 全局配置 ====================
# PyTorch多进程共享策略
SHARING_STRATEGY = "file_system"
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)

# 性能优化
torch.set_float32_matmul_precision('medium')  # 利用Tensor Cores
torch.backends.cudnn.benchmark = True  # 优化cudnn性能


# ==================== 辅助函数 ====================
def set_worker_sharing_strategy(worker_id: int) -> None:
    """DataLoader worker初始化函数"""
    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


def get_bond_type_str(bond):
    """将RDKit键类型转换为字符串"""
    return str(bond.GetBondType())


def process_batch(batch, atom_types_x1, bond_types_x1, max_node_types_x4, params):
    """
    并行处理分子批次，统计特征分布
    
    Args:
        batch: 分子数据批次
        atom_types_x1: 原子类型列表
        bond_types_x1: 键类型列表
        max_node_types_x4: 最大药效团类型数
        params: 参数字典
        
    Returns:
        tuple: (原子计数, 键计数, 药效团计数)
    """
    # 初始化计数器
    batch_atom_counts = torch.zeros(len(atom_types_x1), dtype=torch.float)
    batch_bond_counts = torch.zeros(len(bond_types_x1), dtype=torch.float)
    batch_pharm_counts = torch.zeros(max_node_types_x4, dtype=torch.float)

    # 处理每个分子
    for mol_block, _ in batch:
        mol = rdkit.Chem.MolFromMolBlock(mol_block, removeHs=False)
        if not mol:
            continue

        # 统计原子类型 (x1)
        if params['dataset']['compute_x1']:
            if params['dataset']['x1']['add_virtual_node']:
                batch_atom_counts[atom_types_x1.index(None)] += 1
            
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol in atom_types_x1:
                    batch_atom_counts[atom_types_x1.index(symbol)] += 1

            for bond in mol.GetBonds():
                bond_str = get_bond_type_str(bond)
                if bond_str in bond_types_x1:
                    batch_bond_counts[bond_types_x1.index(bond_str)] += 1

        # 统计药效团类型 (x4)
        if params['dataset']['compute_x4']:
            if params['dataset']['x4']['add_virtual_node']:
                batch_pharm_counts[0] += 1
            
            try:
                pharm_types, _, _ = get_pharmacophores(
                    mol, 
                    multi_vector=params['dataset']['x4']['multivectors'],
                    check_access=params['dataset']['x4']['check_accessibility']
                )
                for p_type in (pharm_types + 1):
                    if p_type < max_node_types_x4:
                        batch_pharm_counts[p_type] += 1
            except Exception:
                pass  # 忽略无法处理的分子
    
    return batch_atom_counts, batch_bond_counts, batch_pharm_counts


def compute_and_cache_marginals(params, molblocks_and_charges, cache_dir="cached_marginals"):
    """
    计算或加载缓存的特征边际分布
    
    Args:
        params: 参数字典
        molblocks_and_charges: 分子数据列表
        cache_dir: 缓存目录
        
    Returns:
        tuple: (原子边际分布, 键边际分布, 药效团边际分布)
    """
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    # 定义缓存文件路径
    dataset_name = params['data']
    atom_marginals_file = os.path.join(cache_dir, f"{dataset_name}_atom_marginals.pt")
    bond_marginals_file = os.path.join(cache_dir, f"{dataset_name}_bond_marginals.pt")
    pharm_marginals_file = os.path.join(cache_dir, f"{dataset_name}_pharm_marginals.pt")

    # 尝试加载缓存
    if (os.path.exists(atom_marginals_file) and 
        os.path.exists(bond_marginals_file) and 
        os.path.exists(pharm_marginals_file)):
        
        print(f"--- 从 '{cache_dir}' 加载已缓存的边际分布 ---")
        atom_marginals_x1 = torch.load(atom_marginals_file, weights_only=True)
        bond_marginals_x1 = torch.load(bond_marginals_file, weights_only=True)
        pharm_marginals_x4 = torch.load(pharm_marginals_file, weights_only=True)
        print("--- 边际分布加载完毕 ---\n")
        return atom_marginals_x1, bond_marginals_x1, pharm_marginals_x4

    # 如果没有缓存，进行并行计算
    print("--- 未找到缓存，开始并行计算特征边际分布 ---")
    
    atom_types_x1 = params['dataset']['x1']['atom_types']
    bond_types_x1 = params['dataset']['x1']['bond_types']
    max_node_types_x4 = params['dataset']['x4']['max_node_types']

    # 初始化总计数器
    total_atom_counts = torch.zeros(len(atom_types_x1), dtype=torch.float)
    total_bond_counts = torch.zeros(len(bond_types_x1), dtype=torch.float)
    total_pharm_counts = torch.zeros(max_node_types_x4, dtype=torch.float)
    
    # 设置并行计算参数
    num_processes = multiprocessing.cpu_count()
    batch_size_for_processing = 1000
    batches = [molblocks_and_charges[i:i + batch_size_for_processing] 
               for i in range(0, len(molblocks_and_charges), batch_size_for_processing)]
    
    # 创建偏函数
    worker_fn = partial(process_batch, 
                        atom_types_x1=atom_types_x1, 
                        bond_types_x1=bond_types_x1, 
                        max_node_types_x4=max_node_types_x4,
                        params=params)

    # 并行处理
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(worker_fn, batches, chunksize=1), 
                           total=len(batches), desc="并行统计特征"))

    # 汇总结果
    for res in results:
        total_atom_counts += res[0]
        total_bond_counts += res[1]
        total_pharm_counts += res[2]

    # 计算边际分布
    atom_marginals_x1 = (total_atom_counts / total_atom_counts.sum() 
                         if total_atom_counts.sum() > 0 
                         else torch.ones_like(total_atom_counts) / len(total_atom_counts))
    bond_marginals_x1 = (total_bond_counts / total_bond_counts.sum() 
                         if total_bond_counts.sum() > 0 
                         else torch.ones_like(total_bond_counts) / len(total_bond_counts))
    pharm_marginals_x4 = (total_pharm_counts / total_pharm_counts.sum() 
                          if total_pharm_counts.sum() > 0 
                          else torch.ones_like(total_pharm_counts) / len(total_pharm_counts))
    
    print("\n--- 边际分布计算完毕 ---")
    print(f"Atom Marginals (x1): {atom_marginals_x1}")
    print(f"Bond Marginals (x1): {bond_marginals_x1}")
    print(f"Pharmacophore Marginals (x4): {pharm_marginals_x4}")
    
    # 保存结果
    print(f"--- 将计算结果缓存到 '{cache_dir}' ---")
    torch.save(atom_marginals_x1, atom_marginals_file)
    torch.save(bond_marginals_x1, bond_marginals_file)
    torch.save(pharm_marginals_x4, pharm_marginals_file)
    print("---------------------------------------\n")

    return atom_marginals_x1, bond_marginals_x1, pharm_marginals_x4


def load_dataset(params):
    """
    根据参数加载数据集
    
    Args:
        params: 参数字典
        
    Returns:
        tuple: (分子数据列表, 输出文件名)
    """
    molblocks_and_charges = []
    output_file = ""
    
    if params['data'] == 'GDB17':
        # 示例数据
        with open('../data/conformers/gdb/example_molblock_charges.pkl', 'rb') as f:
            molblocks_and_charges = pickle.load(f)
        output_file = "GDB17"
        
    elif params['data'] == 'MOSES_aq':
        # 示例数据
        with open('../data/conformers/moses_aq/example_molblock_charges.pkl', 'rb') as f:
            molblocks_and_charges = pickle.load(f)
        output_file = "MOSES_aq"
    
    elif params['data'] == 'NPs':
        # NPs数据集 - 用于DPO微调（3个天然产物分子）
        with open('../data/conformers/np/molblock_charges_NPs.pkl', 'rb') as f:
            molblocks_and_charges = pickle.load(f)
        print(f"\n✅ 加载NPs数据集: {len(molblocks_and_charges)} 个分子（用于DPO微调）")
        output_file = "NPs"
    
    return molblocks_and_charges, output_file


def create_dataset(params, molblocks_and_charges, marginals):
    """
    创建HeteroDataset
    
    Args:
        params: 参数字典
        molblocks_and_charges: 分子数据
        marginals: 边际分布元组
        
    Returns:
        HeteroDataset: 数据集对象
    """
    atom_marginals_x1, bond_marginals_x1, pharm_marginals_x4 = marginals
    
    dataset = HeteroDataset(
        molblocks_and_charges=molblocks_and_charges,
        noise_schedule_dict=params['noise_schedules'],
        
        # 边际分布
        atom_marginals_x1=atom_marginals_x1,
        bond_marginals_x1=bond_marginals_x1,
        pharm_marginals_x4=pharm_marginals_x4,
        
        # 数据集配置
        explicit_hydrogens=params['dataset']['explicit_hydrogens'],
        use_MMFF94_charges=params['dataset']['use_MMFF94_charges'],
        formal_charge_diffusion=False,  # 不进行形式电荷的扩散
        
        # 模态开关
        x1=params['dataset']['compute_x1'],
        x2=params['dataset']['compute_x2'],
        x3=params['dataset']['compute_x3'],
        x4=params['dataset']['compute_x4'],
        
        # x1配置
        recenter_x1=params['dataset']['x1']['recenter'],
        add_virtual_node_x1=params['dataset']['x1']['add_virtual_node'],
        remove_noise_COM_x1=params['dataset']['x1']['remove_noise_COM'],
        atom_types_x1=params['dataset']['x1']['atom_types'],
        charge_types_x1=params['dataset']['x1']['charge_types'],
        bond_types_x1=params['dataset']['x1']['bond_types'],
        scale_atom_features_x1=params['dataset']['x1']['scale_atom_features'],
        scale_bond_features_x1=params['dataset']['x1']['scale_bond_features'],
        
        # x2配置
        independent_timesteps_x2=params['dataset']['x2']['independent_timesteps'],
        recenter_x2=params['dataset']['x2']['recenter'],
        add_virtual_node_x2=params['dataset']['x2']['add_virtual_node'],
        remove_noise_COM_x2=params['dataset']['x2']['remove_noise_COM'],
        num_points_x2=params['dataset']['x2']['num_points'],
        
        # x3配置
        independent_timesteps_x3=params['dataset']['x3']['independent_timesteps'],
        recenter_x3=params['dataset']['x3']['recenter'],
        add_virtual_node_x3=params['dataset']['x3']['add_virtual_node'],
        remove_noise_COM_x3=params['dataset']['x3']['remove_noise_COM'],
        num_points_x3=params['dataset']['x3']['num_points'],
        scale_node_features_x3=params['dataset']['x3']['scale_node_features'],
        
        # x4配置
        independent_timesteps_x4=params['dataset']['x4']['independent_timesteps'],
        recenter_x4=params['dataset']['x4']['recenter'],
        add_virtual_node_x4=params['dataset']['x4']['add_virtual_node'],
        remove_noise_COM_x4=params['dataset']['x4']['remove_noise_COM'],
        max_node_types_x4=params['dataset']['x4']['max_node_types'],
        scale_node_features_x4=params['dataset']['x4']['scale_node_features'],
        scale_vector_features_x4=params['dataset']['x4']['scale_vector_features'],
        multivectors=params['dataset']['x4']['multivectors'],
        check_accessibility=params['dataset']['x4']['check_accessibility'],
        
        probe_radius=params['dataset']['probe_radius'],
    )
    
    return dataset


def create_dataloader(params, dataset, dpo_dataset=None):
    """
    创建DataLoader（标准或混合模式）
    
    Args:
        params: 参数字典
        dataset: 标准数据集
        dpo_dataset: DPO数据集（可选）
        
    Returns:
        DataLoader: 数据加载器
    """
    if params['training'].get('enable_dpo', False) and dpo_dataset is not None:
        print("创建混合DataLoader（标准 + DPO）...")
        train_loader = create_mixed_dataloader(
            standard_dataset=dataset,
            dpo_dataset=dpo_dataset,
            batch_size=params['training']['batch_size'],
            num_workers=params['training']['num_workers'],
            dpo_ratio=params['training'].get('dpo_batch_ratio', 0.3),
            shuffle=True,
            params=params,
            multiprocessing_context=multiprocessing.get_context("spawn") 
                if params['training']['multiprocessing_spawn'] else None,
            worker_init_fn=set_worker_sharing_strategy,
            persistent_workers=True,
        )
    else:
        print("创建标准DataLoader...")
        train_loader = torch_geometric.loader.DataLoader(
            dataset=dataset,
            num_workers=params['training']['num_workers'],
            batch_size=params['training']['batch_size'],
            shuffle=True,
            multiprocessing_context=multiprocessing.get_context("spawn") 
                if params['training']['multiprocessing_spawn'] else None,
            worker_init_fn=set_worker_sharing_strategy,
            persistent_workers=True,
        )
    
    return train_loader


def setup_callbacks(params, dataset, molblocks_and_charges, output_dir, args):
    """
    设置训练回调
    
    Args:
        params: 参数字典
        dataset: 数据集
        molblocks_and_charges: 分子数据
        output_dir: 输出目录
        args: 命令行参数
        
    Returns:
        list: 回调列表
    """
    # 基础回调
    checkpoint_callback = ModelCheckpoint(
        save_top_k=0,
        save_last=True,
        monitor="train_loss",
        mode="min",
        dirpath=output_dir,
        filename="best-{step:09d}",
        every_n_train_steps=params['training']['log_every_n_steps'],
    )
    
    callbacks = [checkpoint_callback]
    
    # DPO回调
    if params['training'].get('enable_dpo', False):
        print("\n添加DPO在线采样回调...")
        sampling_callback = OnlineSamplingCallback(
            params=params,
            dataset=dataset,
            molblocks_and_charges=molblocks_and_charges
        )
        dpo_metrics_callback = DPOMetricsCallback()
        callbacks.extend([sampling_callback, dpo_metrics_callback])
        print("DPO回调已添加")
    
    return callbacks


def setup_loggers(output_dir, args, params):
    """
    设置日志记录器
    
    Args:
        output_dir: 输出目录
        args: 命令行参数
        params: 参数字典
        
    Returns:
        list: 日志记录器列表
    """
    csv_logger = CSVLogger(
        save_dir=output_dir,
        name='csv_logger',
    )
    
    wandb_logger = WandbLogger(
        name=f"{args.model_name}-seed_{args.seed}-bs_{params['training']['batch_size']}",
        entity="SPD_PaperParty",
        project="SPD_Molecule_Generation",
        save_dir=output_dir,
        log_model="all",
    )
    
    return [csv_logger, wandb_logger]


def handle_checkpoint_loading(params, output_dir, model_pl):
    """
    处理checkpoint加载逻辑
    
    Args:
        params: 参数字典
        output_dir: 输出目录
        model_pl: PyTorch Lightning模型
        
    Returns:
        str or None: checkpoint路径
    """
    resume_from_checkpoint = True
    ckpt_path = f"{output_dir}/last.ckpt"
    
    if not params['training'].get('enable_dpo', False):
        # 标准模式
        return ckpt_path if (os.path.exists(ckpt_path) and resume_from_checkpoint) else None
    
    # DPO模式的复杂加载逻辑
    pretrained_path = params['training'].get('pretrained_checkpoint_path', None)
    
    if pretrained_path is not None:
        # 优先使用预训练模型
        pretrained_ckpt_path = f"jobs/{pretrained_path}"
        if os.path.exists(pretrained_ckpt_path):
            print(f"\n🔄 DPO微调：从预训练模型加载权重")
            print(f"   预训练checkpoint: {pretrained_ckpt_path}")
            
            try:
                checkpoint = torch.load(pretrained_ckpt_path, map_location='cpu')
                model_state_dict = checkpoint['state_dict']
                model_weights = {k: v for k, v in model_state_dict.items() if k.startswith('model.')}
                
                missing, unexpected = model_pl.model.load_state_dict(model_weights, strict=False)
                
                if hasattr(model_pl, 'ref_model'):
                    model_pl.ref_model.load_state_dict(model_weights, strict=False)
                    print(f"   ✅ 已同步权重到参考模型（ref_model）")
                
                print(f"   ✅ 成功加载预训练权重")
                if len(missing) > 0:
                    print(f"   ⚠️  缺失的键: {len(missing)} 个")
                if len(unexpected) > 0:
                    print(f"   ⚠️  未预期的键: {len(unexpected)} 个")
                
                # 检查是否继续之前的训练
                if os.path.exists(ckpt_path) and resume_from_checkpoint:
                    print(f"\n   发现当前任务的checkpoint: {ckpt_path}")
                    print(f"   将继续当前任务的训练")
                    return ckpt_path
                else:
                    print(f"\n   未找到当前任务的checkpoint，将从预训练模型开始新的DPO微调")
                    return None
                    
            except Exception as e:
                print(f"   ❌ 加载预训练权重失败: {e}")
                print(f"   将从头开始训练")
                return None
        else:
            print(f"\n⚠️  预训练checkpoint不存在: {pretrained_ckpt_path}")
    
    # 检查当前目录的checkpoint
    if os.path.exists(ckpt_path) and resume_from_checkpoint:
        load_weights_only = params['training'].get('dpo_load_weights_only', True)
        if load_weights_only:
            print("\n✅ DPO模式：加载当前checkpoint的模型权重")
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                model_state_dict = checkpoint['state_dict']
                model_weights = {k: v for k, v in model_state_dict.items() if k.startswith('model.')}
                model_pl.model.load_state_dict(model_weights, strict=False)
                if hasattr(model_pl, 'ref_model'):
                    model_pl.ref_model.load_state_dict(model_weights, strict=False)
                print(f"   已从 {ckpt_path} 加载模型权重")
                return None  # 只加载权重，不恢复训练状态
            except Exception as e:
                print(f"   ⚠️ 加载权重失败: {e}")
                return None
    
    print("\n📝 DPO模式：未找到任何checkpoint，从头开始训练")
    return None


# ==================== 主函数 ====================
def main():
    """主训练函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="SPD分子生成模型训练")
    parser.add_argument("model_name", type=str, help="模型配置名称")
    parser.add_argument("seed", type=int, help="随机种子")
    args = parser.parse_args()
    
    # 设置随机种子
    seed_everything(seed=args.seed, workers=True)
    
    # 加载参数
    params = importlib.import_module(f'parameters.{args.model_name}').params
    
    # 加载数据集
    molblocks_and_charges, output_file = load_dataset(params)
    
    # 计算边际分布
    print("计算特征边际分布...")
    marginals = compute_and_cache_marginals(params, molblocks_and_charges)
    print("✅ 计算完边际分布！")
    
    # 创建数据集
    dataset = create_dataset(params, molblocks_and_charges, marginals)
    print("✅ 配置普通数据集完成")
    
    # 创建DPO数据集（如果启用）
    dpo_dataset = None
    if params['training'].get('enable_dpo', False):
        print("\n初始化DPO数据集...")
        dpo_dataset = DPODataset(
            preference_pairs=[],
            base_dataset=dataset,
            noise_schedule_dict=params['noise_schedules'],
            params=params,
        )
    
    # 创建DataLoader
    train_loader = create_dataloader(params, dataset, dpo_dataset)
    
    # 设置输出目录
    output_dir = f"jobs/{params['training']['output_dir']}"
    os.makedirs("jobs/", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置回调和日志
    callbacks = setup_callbacks(params, dataset, molblocks_and_charges, output_dir, args)
    loggers = setup_loggers(output_dir, args, params)
    
    # 设置训练器
    cuda_available = torch.cuda.is_available()
    num_gpus_to_use = torch.cuda.device_count()
    
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=loggers,
        default_root_dir=output_dir,
        accelerator="gpu" if (params['training']['num_gpus'] >= 1 and cuda_available) else 'cpu',
        max_epochs=10000,
        gradient_clip_val=params['training']['gradient_clip_val'],
        accumulate_grad_batches=params['training']['accumulate_grad_batches'],
        log_every_n_steps=params['training']['log_every_n_steps'],
        reload_dataloaders_every_n_epochs=1,
        devices=num_gpus_to_use if cuda_available else "auto",
        strategy=DDPStrategy(find_unused_parameters=True) 
            if ((params['training']['num_gpus'] > 1 and cuda_available) or 
                params['training'].get('enable_dpo', False)) else 'auto',
        precision=32,
        detect_anomaly=True,
    )
    
    # 创建模型
    model_pl = LightningModule(params)
    
    # 设置wandb监控
    loggers[1].watch(model_pl, log="all", log_freq=500)
    
    print(f"模型参数总数: {sum(p.numel() for p in model_pl.parameters() if p.requires_grad)}")
    
    # 处理checkpoint加载
    ckpt_path = handle_checkpoint_loading(params, output_dir, model_pl)
    
    # 备份当前checkpoint
    if (ckpt_path is not None) and (trainer.global_rank == 0):
        date = datetime.datetime.now()
        timestamp = date.strftime("%Y_%m_%d_%H_%M")
        shutil.copyfile(ckpt_path, f"{output_dir}/last_{timestamp}.ckpt")
    
    # 开始训练
    print('开始训练...')
    trainer.fit(model_pl, train_loader, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()
