import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

#  **功能**: 提高系统允许打开的文件数量上限。
#  **讲解**: 
# 在数据加载时，尤其是使用多个工作进程 (`num_workers > 0`) 时，每个进程都可能打开数据文件。
# 操作系统对单个进程能打开的文件描述符数量有限制。
# 这几行代码获取当前的限制 (`getrlimit`)，然后将其下限提高到 2048 (`setrlimit`)，以防止因打开文件过多而导致程序崩溃。

import rdkit
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch_geometric
from torch_geometric.nn import radius_graph
import torch_scatter

import pickle
from copy import deepcopy
import os
import shutil
import datetime
import multiprocessing
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from torch_geometric.data import HeteroData

from shepherd.model.model import Model
from shepherd.lightning_module import LightningModule

# 数据集类，负责加载和预处理分子数据。
# from shepherd.datasets import HeteroDataset  # OLD
from shepherd.new_datasets import HeteroDataset # NEW (adjust path as needed)

from lightning_fabric.utilities.seed import seed_everything # 这么酷的名字

import importlib

import os
import torch
import rdkit
from tqdm import tqdm
import multiprocessing
from functools import partial

# --- 并行计算的工作函数 ---
def process_batch(batch, atom_types_x1, bond_types_x1, max_node_types_x4, params):
    """
    处理一小批分子数据，并返回该批次中各种特征的计数结果。
    这是并行化的核心部分。
    """
    # 初始化当前批次的计数器
    batch_atom_counts = torch.zeros(len(atom_types_x1), dtype=torch.float)
    batch_bond_counts = torch.zeros(len(bond_types_x1), dtype=torch.float)
    batch_pharm_counts = torch.zeros(max_node_types_x4, dtype=torch.float)

    # 辅助函数：将 RDKit 键类型对象映射到参数中使用的字符串表示形式
    def get_bond_type_str(bond):
        return str(bond.GetBondType())

    # 导入 get_pharmacophores 函数
    from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores

    # 迭代处理批次中的每个分子
    for mol_block, _ in batch:
        mol = rdkit.Chem.MolFromMolBlock(mol_block, removeHs=False)
        if not mol:
            continue

        # --- 1. 统计原子和键的类型 (for x1) ---
        if params['dataset']['compute_x1']:
            if params['dataset']['x1']['add_virtual_node']:
                batch_atom_counts[atom_types_x1.index(None)] += 1
            
            for atom in mol.GetAtoms(): # 遍历rdkit分子中的每个原子
                symbol = atom.GetSymbol()
                if symbol in atom_types_x1:
                    batch_atom_counts[atom_types_x1.index(symbol)] += 1

            for bond in mol.GetBonds():
                bond_str = get_bond_type_str(bond)
                if bond_str in bond_types_x1:
                    batch_bond_counts[bond_types_x1.index(bond_str)] += 1

        # --- 2. 统计药效团类型 (for x4) ---
        if params['dataset']['compute_x4']:
            if params['dataset']['x4']['add_virtual_node']:
                batch_pharm_counts[0] += 1
            
            try: # 返回三个值（类型、位置和方向）
                pharm_types, _, _ = get_pharmacophores(
                    mol, 
                    multi_vector=params['dataset']['x4']['multivectors'],  # 是否使用多向量
                    check_access=params['dataset']['x4']['check_accessibility'] # 是否检查 accessibility
                )
                for p_type in (pharm_types + 1):
                    if p_type < max_node_types_x4:
                        batch_pharm_counts[p_type] += 1
            except Exception:
                # 忽略无法处理的分子
                pass
    
    return batch_atom_counts, batch_bond_counts, batch_pharm_counts

# --- 主函数，用于缓存、加载和并行计算 (修正版) ---
def compute_and_cache_marginals(params, molblocks_and_charges, cache_dir="cached_marginals"):
    """
    计算或加载缓存的特征边际分布。
    如果缓存文件存在，则直接加载；否则，并行计算特征，保存结果，然后返回。
    """
    # 确保缓存目录存在
    os.makedirs(cache_dir, exist_ok=True)
    
    # 定义缓存文件路径
    dataset_name = params['data']
    atom_marginals_file = os.path.join(cache_dir, f"{dataset_name}_atom_marginals.pt")
    bond_marginals_file = os.path.join(cache_dir, f"{dataset_name}_bond_marginals.pt")
    pharm_marginals_file = os.path.join(cache_dir, f"{dataset_name}_pharm_marginals.pt")

    # --- 1. 检查并加载缓存 ---
    if os.path.exists(atom_marginals_file) and \
       os.path.exists(bond_marginals_file) and \
       os.path.exists(pharm_marginals_file):
        
        print(f"--- 从 '{cache_dir}' 加载已缓存的边际分布 ---")
        atom_marginals_x1 = torch.load(atom_marginals_file, weights_only=True)
        bond_marginals_x1 = torch.load(bond_marginals_file, weights_only=True)
        pharm_marginals_x4 = torch.load(pharm_marginals_file, weights_only=True)
        print("--- 边际分布加载完毕 ---\n")
        return atom_marginals_x1, bond_marginals_x1, pharm_marginals_x4

    # --- 2. 如果没有缓存，则进行并行计算 ---
    print("--- 未找到缓存，开始并行计算特征边际分布 ---")
    
    atom_types_x1 = params['dataset']['x1']['atom_types']
    bond_types_x1 = params['dataset']['x1']['bond_types']
    max_node_types_x4 = params['dataset']['x4']['max_node_types']

    # 初始化总计数器
    total_atom_counts = torch.zeros(len(atom_types_x1), dtype=torch.float)
    total_bond_counts = torch.zeros(len(bond_types_x1), dtype=torch.float)
    total_pharm_counts = torch.zeros(max_node_types_x4, dtype=torch.float)
    
    # 设置并行计算参数
    num_processes = multiprocessing.cpu_count()  # 使用所有可用的CPU核心
    
    # 【修正点 1】定义一个合理的批次大小，并创建 batches 列表
    # 这个值决定了每个“工作包裹”的大小
    batch_size_for_processing = 1000   # 一直写死吗
    batches = [molblocks_and_charges[i:i + batch_size_for_processing] for i in range(0, len(molblocks_and_charges), batch_size_for_processing)]
    
    # 创建一个偏函数 (partial function) 来固定 process_batch 的参数
    worker_fn = partial(process_batch, 
                        atom_types_x1=atom_types_x1, 
                        bond_types_x1=bond_types_x1, 
                        max_node_types_x4=max_node_types_x4,
                        params=params)

    # 【修正点 2】使用进程池并行处理我们创建好的 batches
    with multiprocessing.Pool(processes=num_processes) as pool:
        # 使用 tqdm 显示进度条。imap 现在迭代的是 batches 列表。
        # chunksize=1 表示每次给一个工作进程分配一个 batch，这对于已经分好块的任务是高效的。
        results = list(tqdm(pool.imap(worker_fn, batches, chunksize=1), total=len(batches), desc="并行统计特征"))

    # 汇总所有进程的结果
    for res in results:
        total_atom_counts += res[0]
        total_bond_counts += res[1]
        total_pharm_counts += res[2]

    # --- 3. 计算边际分布 ---
    atom_marginals_x1 = (total_atom_counts / total_atom_counts.sum()) if total_atom_counts.sum() > 0 else torch.ones_like(total_atom_counts) / len(total_atom_counts)
    bond_marginals_x1 = (total_bond_counts / total_bond_counts.sum()) if total_bond_counts.sum() > 0 else torch.ones_like(total_bond_counts) / len(total_bond_counts)
    pharm_marginals_x4 = (total_pharm_counts / total_pharm_counts.sum()) if total_pharm_counts.sum() > 0 else torch.ones_like(total_pharm_counts) / len(total_pharm_counts)
    
    print("\n--- 边际分布 计算完毕 ---")
    print(f"Atom Marginals (x1): {atom_marginals_x1}")
    print(f"Bond Marginals (x1): {bond_marginals_x1}")
    print(f"Pharmacophore Marginals (x4): {pharm_marginals_x4}")
    
    # --- 4. 保存结果以备后用 ---
    print(f"--- 将计算结果缓存到 '{cache_dir}' ---")
    torch.save(atom_marginals_x1, atom_marginals_file)
    torch.save(bond_marginals_x1, bond_marginals_file)
    torch.save(pharm_marginals_x4, pharm_marginals_file)
    print("---------------------------------------\n")

    return atom_marginals_x1, bond_marginals_x1, pharm_marginals_x4

# PyTorch 的 DataLoader 在使用多进程时，需要在主进程和工作进程之间共享数据。
# 默认的共享策略是 "file_descriptor"，它有时会因为文件描述符耗尽而出错（与上面的 resource 设置相关）。
# "file_system" 是一种更稳定、更通用的策略，它通过在共享内存中创建文件来实现数据共享。

sharing_strategy = "file_system"
torch.multiprocessing.set_sharing_strategy(sharing_strategy)

torch.set_float32_matmul_precision('medium')  # 利用Tensor Cores
torch.backends.cudnn.benchmark = True  # 优化cudnn性能

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(sharing_strategy)

if __name__ == '__main__':
    """
    This repository includes only a small subset of the training data so that the repository is self-contained.
    After downloading the full training datasets (see README), change the corresponding lines of code below.
    """
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    
    # workers=True 确保 DataLoader 的工作进程也是可复现的。
    seed_everything(seed = args.seed, workers = True)

    # 它根据命令行传入的 model_name 动态地导入位于 parameters/ 目录下的对应 Python 文件（例如 parameters/my_model_config.py），并从中获取名为 params 的字典。
    params = importlib.import_module(f'parameters.{args.model_name}').params
    
    # CHANGE ME ONCE FULL DATASETS ARE DOWNLOADED
    if params['data'] == 'GDB17':
        # sample data
        molblocks_and_charges = []
        with open(f'../data/conformers/gdb/example_molblock_charges.pkl', 'rb') as f:
            molblocks_and_charges = pickle.load(f)

        output_file = "GDB17"

        """
        # full dataset
        molblocks_and_charges = []
        for i in [0,1,2]:
            with open(f'conformers/gdb/molblock_charges_{i}.pkl', 'rb') as f:
                molblocks_and_charges_ = pickle.load(f) 
            molblocks_and_charges += molblocks_and_charges_
        
        # removing randomly-chosen test-set molecules prior to training
        test_indices = np.load('conformers/gdb/random_split_test_indices.npy')
        for index in tqdm(sorted(test_indices)[::-1]): # removing from end of list
            if index < len(molblocks_and_charges):
                molblocks_and_charges.pop(index)
        """
    
    # CHANGE ME ONCE FULL DATASETS ARE DOWNLOADED
    if params['data'] == 'MOSES_aq':

        # full dataset
        molblocks_and_charges = []
        for i in [0,1,2,3,4]:
            with open(f'../data/molblock_charges_{i}.pkl', 'rb') as f:
                molblocks_and_charges_ = pickle.load(f) 
            molblocks_and_charges += molblocks_and_charges_

        # sample data data/molblock_charges_0.pkl
        # molblocks_and_charges = []
        # with open(f'../data/conformers/moses_aq/example_molblock_charges.pkl', 'rb') as f:
        #     molblocks_and_charges = pickle.load(f)

        output_file = "MOSES_aq"

    #  pharmacophore 用于计算分子的药效团特征
    atom_marginals_x1, bond_marginals_x1, pharm_marginals_x4 = compute_and_cache_marginals(
        params=params, 
        molblocks_and_charges=molblocks_and_charges
    )

    dataset = HeteroDataset(
        molblocks_and_charges = molblocks_and_charges, 
        
        noise_schedule_dict = params['noise_schedules'],

        atom_marginals_x1=atom_marginals_x1,
        bond_marginals_x1=bond_marginals_x1,
        pharm_marginals_x4=pharm_marginals_x4,
        
        explicit_hydrogens = params['dataset']['explicit_hydrogens'],
        use_MMFF94_charges = params['dataset']['use_MMFF94_charges'],
        
        # formal_charge_diffusion = params['x1_formal_charge_diffusion'], # 不进行形式电荷的扩散
        formal_charge_diffusion = False ,

        x1 = params['dataset']['compute_x1'],
        x2 = params['dataset']['compute_x2'],
        x3 = params['dataset']['compute_x3'],
        x4 = params['dataset']['compute_x4'],
        
        recenter_x1 = params['dataset']['x1']['recenter'], 
        add_virtual_node_x1 = params['dataset']['x1']['add_virtual_node'],
        remove_noise_COM_x1 = params['dataset']['x1']['remove_noise_COM'],
        atom_types_x1 = params['dataset']['x1']['atom_types'],
        charge_types_x1 = params['dataset']['x1']['charge_types'],
        bond_types_x1 = params['dataset']['x1']['bond_types'],
        scale_atom_features_x1 = params['dataset']['x1']['scale_atom_features'],
        scale_bond_features_x1 = params['dataset']['x1']['scale_bond_features'],

        independent_timesteps_x2 = params['dataset']['x2']['independent_timesteps'],
        recenter_x2 = params['dataset']['x2']['recenter'],
        add_virtual_node_x2 = params['dataset']['x2']['add_virtual_node'],
        remove_noise_COM_x2 = params['dataset']['x2']['remove_noise_COM'],
        num_points_x2 = params['dataset']['x2']['num_points'],
        
        independent_timesteps_x3 = params['dataset']['x3']['independent_timesteps'],
        recenter_x3 = params['dataset']['x3']['recenter'],
        add_virtual_node_x3 = params['dataset']['x3']['add_virtual_node'],
        remove_noise_COM_x3 = params['dataset']['x3']['remove_noise_COM'],
        num_points_x3 = params['dataset']['x3']['num_points'],
        scale_node_features_x3 = params['dataset']['x3']['scale_node_features'],        
        
        independent_timesteps_x4 = params['dataset']['x4']['independent_timesteps'],
        recenter_x4 = params['dataset']['x4']['recenter'],
        add_virtual_node_x4 = params['dataset']['x4']['add_virtual_node'],
        remove_noise_COM_x4 = params['dataset']['x4']['remove_noise_COM'],
        max_node_types_x4 = params['dataset']['x4']['max_node_types'],
        scale_node_features_x4 = params['dataset']['x4']['scale_node_features'],
        scale_vector_features_x4 = params['dataset']['x4']['scale_vector_features'],
        multivectors = params['dataset']['x4']['multivectors'],
        check_accessibility = params['dataset']['x4']['check_accessibility'],
        
        probe_radius = params['dataset']['probe_radius'], # for x2 and x3
        
    )
    
    # debug : 非并行  multiprocessing提供的是进程启动方式，spawn安全，不共享内存  ['training']['multiprocessing_spawn']启动选择
    if params['training']['multiprocessing_spawn']:
        train_loader = torch_geometric.loader.DataLoader(
            dataset = dataset,
            num_workers = params['training']['num_workers'],     # 多进程       
            batch_size = params['training']['batch_size'],
            shuffle = True,
            multiprocessing_context = multiprocessing.get_context("spawn"),
            worker_init_fn=set_worker_sharing_strategy,
            persistent_workers=True,  # 添加这一行 减少冷启动开销
        )
    else:
        train_loader = torch_geometric.loader.DataLoader(
            dataset = dataset,
            num_workers = params['training']['num_workers'],
            batch_size = params['training']['batch_size'],
            shuffle = True,
            worker_init_fn=set_worker_sharing_strategy,
            persistent_workers=True,  # 添加这一行
        )
    
    
    output_dir = f"jobs/{params['training']['output_dir']}"
    try: os.mkdir(f"jobs/")
    except: pass
    try: os.mkdir(output_dir)
    except: pass
    
    checkpoint_callback = ModelCheckpoint( # 保存训练过程中模型
        save_top_k = 0,
        save_last = True,
        monitor="train_loss",
        mode="min",
        dirpath = output_dir,
        filename="best-{step:09d}",
        every_n_train_steps = params['training']['log_every_n_steps'],
    )
    csv_logger = CSVLogger( # 
        save_dir = output_dir,
        name = 'csv_logger',
    )
    wandb_logger = WandbLogger(
        name=f"{args.model_name}-seed_{args.seed}-bs_{params['training']['batch_size']}",
        entity="SPD_PaperParty",
        project="SPD_Molecule_Generation",
        save_dir=output_dir,
        log_model="all", 
    )
    
    gradient_clip_val = params['training']['gradient_clip_val'] # 梯度裁剪阈值
    accumulate_grad_batches = params['training']['accumulate_grad_batches'] # 累积多少个batch才更新
    
    from pytorch_lightning.strategies.ddp import DDPStrategy
    
    cuda_available = torch.cuda.is_available()
    num_gpus_to_use = torch.cuda.device_count()
    
    trainer = pl.Trainer(
        callbacks = [checkpoint_callback],
        logger = [csv_logger, wandb_logger], # 可以挂多个
        
        default_root_dir = output_dir,
        accelerator = "gpu" if (params['training']['num_gpus'] >= 1 and cuda_available) else 'cpu', 
        
        max_epochs = 10000,
        
        gradient_clip_val = gradient_clip_val, # 在 optimizer.step() 前自动应用裁剪，防止梯度爆炸
        accumulate_grad_batches = accumulate_grad_batches, # 梯度累积步数
        
        log_every_n_steps = params['training']['log_every_n_steps'], # 临时调整    多少step输出一次
            
        reload_dataloaders_every_n_epochs = 1, # re-shuffle training data after each epoch
        
        devices = num_gpus_to_use  if cuda_available else "auto", # 
        
        strategy = DDPStrategy(find_unused_parameters=True) if (params['training']['num_gpus'] > 1 and cuda_available) else 'auto',
        # DDP：单机多卡分布式训练  find_unused_parameters=True增加开销
        precision = 32, 
        
        detect_anomaly = True, # 出现异常时，会抛出带栈信息的错误  调试使用
    )
    
    model_pl = LightningModule(params)

    wandb_logger.watch(model_pl, log="all", log_freq=500) # wandb 开始跟踪模型参数和梯度


    print(sum(p.numel() for p in model_pl.parameters() if p.requires_grad))
    
    resume_from_checkpoint = True

    ckpt_path = f"{output_dir}/last.ckpt"
    ckpt_path = ckpt_path if (os.path.exists(ckpt_path) & resume_from_checkpoint) else None
    
    # avoid overwriting previous "last.ckpt"    trainer.global_rank保证只有主进程进行拷贝
    if (ckpt_path is not None) and (trainer.global_rank == 0):
        date = datetime.datetime.now()
        timestamp = str(date.year) + '_' + str(date.month).zfill(2) + '_' + str(date.day).zfill(2) + '_' + str(date.hour).zfill(2) + '_' + str(date.minute).zfill(2)
        shutil.copyfile(ckpt_path, f"{output_dir}/last_{timestamp}.ckpt")
    
    
    print('beginning to train...')
    trainer.fit(model_pl, train_loader, ckpt_path = ckpt_path)
