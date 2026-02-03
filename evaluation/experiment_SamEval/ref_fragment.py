
import open3d 
from shepherd.shepherd_score_utils.generate_point_cloud import (
    get_atom_coords, 
    get_atomic_vdw_radii, 
    get_molecular_surface,
    get_electrostatics,
    get_electrostatics_given_point_charges,
)
from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores
from shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates

print('importing rdkit')
import rdkit
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

import numpy as np
import matplotlib.pyplot as plt

print('importing torch')
import torch
import torch_geometric
from torch_geometric.nn import radius_graph
import torch_scatter

import pickle
from copy import deepcopy
import os
import multiprocessing
from tqdm import tqdm


print('importing lightning')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from shepherd.lightning_module import LightningModule
from shepherd.datasets import HeteroDataset

import importlib

from shepherd.inference import *
from shepherd.extract import create_rdkit_molecule

chkpt = '../data/shepherd_chkpts/x1x3x4_diffusion_mosesaq_20240824_submission.ckpt' # checkpoint used for evaluations in preprint

import json
import gc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# 设置multiprocessing启动方法为spawn（CUDA要求）
multiprocessing.set_start_method('spawn', force=True)

# 检测可用GPU数量
num_gpus = torch.cuda.device_count()
print(f"检测到 {num_gpus} 个GPU")

# 配置参数
CONDITION_FILE = '../data/conformers/fragment_merging/fragment_merge_condition.pickle'
N_ATOMS_LIST = list(range(50, 90))  # 50-89，共40个梯度
SAMPLES_PER_N_ATOMS = 2  # 每个梯度采样2个分子
BATCH_SIZE = 2  # batch size

# 工作函数：在指定GPU上运行采样
def run_sampling_on_gpu(gpu_id, tasks, condition_file):
    """
    在指定GPU上运行采样任务
    tasks: list of (n_atoms, num_samples) 元组
    """
    import torch
    import numpy as np
    import json
    import os
    import pickle
    from shepherd.lightning_module import LightningModule
    from shepherd.inference import inference_sample
    from shepherd.extract import create_rdkit_molecule
    import gc
    
    chkpt = '../data/shepherd_chkpts/x1x3x4_diffusion_mosesaq_20240824_submission.ckpt'
    
    # 设置当前进程使用的GPU
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    print(f"GPU {gpu_id}: 加载模型...")
    model_pl = LightningModule.load_from_checkpoint(chkpt)
    params = model_pl.params
    model_pl.to(device)
    model_pl.model.device = device
    model_pl.eval()
    
    # 加载条件数据
    with open(condition_file, 'rb') as f:
        condition_data = pickle.load(f)
    
    surface = condition_data['x3']['positions']
    electrostatics = condition_data['x3']['charges']
    pharm_types = condition_data['x4']['types']
    pharm_pos = condition_data['x4']['positions']
    pharm_direction = condition_data['x4']['directions']
    num_pharmacophores = len(pharm_types)
    
    # 按n_atoms分组的结果
    all_task_results = {}
    
    MAX_RETRIES = 5  # 最大重试次数
    
    for task_idx, (n_atoms, num_samples) in enumerate(tasks):
        print(f"GPU {gpu_id}: 处理任务 {task_idx + 1}/{len(tasks)}, n_atoms={n_atoms}, 采样数={num_samples}", flush=True)
        
        task_results = []
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # 添加重试机制处理偶发的几何断言错误
        for retry in range(MAX_RETRIES):
            try:
                with torch.no_grad():
                    generated_samples = inference_sample(
                        model_pl,
                        batch_size=num_samples,
                        N_x1=n_atoms,
                        N_x4=num_pharmacophores,
                        unconditional=False,
                        prior_noise_scale=1.0,
                        denoising_noise_scale=1.0,
                        inject_noise_at_ts=[],
                        inject_noise_scales=[],    
                        harmonize=False,
                        harmonize_ts=[],
                        harmonize_jumps=[],
                        inpaint_x2_pos=False,
                        inpaint_x3_pos=True,
                        inpaint_x3_x=True,
                        inpaint_x4_pos=True,
                        inpaint_x4_direction=True,
                        inpaint_x4_type=True,
                        stop_inpainting_at_time_x2=0.0,
                        add_noise_to_inpainted_x2_pos=0.0,
                        stop_inpainting_at_time_x3=0.0,
                        add_noise_to_inpainted_x3_pos=0.0,
                        add_noise_to_inpainted_x3_x=0.0,
                        stop_inpainting_at_time_x4=0.0,
                        add_noise_to_inpainted_x4_pos=0.0,
                        add_noise_to_inpainted_x4_direction=0.0,
                        add_noise_to_inpainted_x4_type=0.0,
                        center_of_mass=np.zeros(3),
                        surface=surface,
                        electrostatics=electrostatics,
                        pharm_types=pharm_types,
                        pharm_pos=pharm_pos,
                        pharm_direction=pharm_direction,
                        return_trajectories=False,
                        verbose=False,
                    )
                break  # 成功则跳出重试循环
            except AssertionError as e:
                print(f"GPU {gpu_id}: n_atoms={n_atoms} 遇到几何断言错误，重试 {retry + 1}/{MAX_RETRIES}", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                if retry == MAX_RETRIES - 1:
                    print(f"GPU {gpu_id}: n_atoms={n_atoms} 达到最大重试次数，跳过此任务", flush=True)
                    generated_samples = []
        
        # 转换样本
        for sample in generated_samples:
            sample_data = {
                'n_atoms': n_atoms,
                'atoms': sample['x1']['atoms'],
                'positions': sample['x1']['positions'],
                'sample_serializable': {
                    'n_atoms': n_atoms,
                    'x1': {
                        'atoms': sample['x1']['atoms'].tolist(),
                        'positions': sample['x1']['positions'].tolist(),
                        'bonds': sample['x1']['bonds'].tolist() if 'bonds' in sample['x1'] else None,
                    },
                    'x2': {'positions': sample['x2']['positions'].tolist()},
                    'x3': {
                        'charges': sample['x3']['charges'].tolist(),
                        'positions': sample['x3']['positions'].tolist(),
                    },
                    'x4': {
                        'types': sample['x4']['types'].tolist(),
                        'positions': sample['x4']['positions'].tolist(),
                        'directions': sample['x4']['directions'].tolist(),
                    }
                }
            }
            task_results.append(sample_data)
        
        all_task_results[n_atoms] = task_results
        print(f"GPU {gpu_id}: n_atoms={n_atoms} 完成，生成 {len(task_results)} 个样本", flush=True)
        
        del generated_samples
        gc.collect()
        torch.cuda.empty_cache()
    
    total_samples = sum(len(v) for v in all_task_results.values())
    print(f"GPU {gpu_id}: 完成所有任务，共生成 {total_samples} 个样本", flush=True)
    return all_task_results


def main():
    print(f"\n{'='*60}")
    print(f"Fragment Merge 条件采样 - 多GPU版")
    print(f"  条件文件: {CONDITION_FILE}")
    print(f"  原子数量范围: {N_ATOMS_LIST[0]}-{N_ATOMS_LIST[-1]} (共{len(N_ATOMS_LIST)}个梯度)")
    print(f"  每个梯度采样: {SAMPLES_PER_N_ATOMS} 个分子")
    print(f"  总计采样: {len(N_ATOMS_LIST) * SAMPLES_PER_N_ATOMS} 个分子")
    print(f"  GPU数量: {num_gpus}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"{'='*60}\n")
    
    # 构建任务列表：每个n_atoms是一个任务
    all_tasks = [(n_atoms, SAMPLES_PER_N_ATOMS) for n_atoms in N_ATOMS_LIST]
    
    # 将任务均匀分配到GPU（交替分配以平衡负载）
    tasks_per_gpu = [[] for _ in range(num_gpus)]
    for i, task in enumerate(all_tasks):
        gpu_id = i % num_gpus
        tasks_per_gpu[gpu_id].append(task)
    
    print(f"任务分配:")
    for gpu_id, tasks in enumerate(tasks_per_gpu):
        n_atoms_list = [t[0] for t in tasks]
        total_samples = sum(t[1] for t in tasks)
        print(f"  GPU {gpu_id}: {len(tasks)} 个n_atoms任务, 共 {total_samples} 个样本")
        print(f"           n_atoms: {n_atoms_list}")
    
    # 使用多进程并行执行
    results_by_n_atoms = {n_atoms: [] for n_atoms in N_ATOMS_LIST}
    
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for gpu_id, task_list in enumerate(tasks_per_gpu):
            if task_list:
                future = executor.submit(
                    run_sampling_on_gpu,
                    gpu_id, task_list, CONDITION_FILE
                )
                futures.append(future)
        
        # 收集结果
        for future in as_completed(futures):
            try:
                gpu_results = future.result()
                for n_atoms, samples in gpu_results.items():
                    results_by_n_atoms[n_atoms].extend(samples)
            except Exception as e:
                print(f"错误: {e}")
                import traceback
                traceback.print_exc()
    
    # 统计结果
    total_samples = sum(len(v) for v in results_by_n_atoms.values())
    print(f"\n所有GPU完成采样，共 {total_samples} 个样本")
    
    # 保存结果
    os.makedirs('data', exist_ok=True)
    
    # 保存JSON
    all_serializable = {}
    for n_atoms in N_ATOMS_LIST:
        samples = results_by_n_atoms[n_atoms]
        serializable_samples = [s['sample_serializable'] for s in samples]
        all_serializable[f'n_atoms_{n_atoms}'] = {
            'n_atoms': n_atoms,
            'num_samples': len(serializable_samples),
            'samples': serializable_samples
        }
    
    json_path = 'data/fragment_merge_samples.json'
    with open(json_path, 'w') as f:
        json.dump({
            'condition_file': CONDITION_FILE,
            'n_atoms_range': [N_ATOMS_LIST[0], N_ATOMS_LIST[-1]],
            'samples_per_n_atoms': SAMPLES_PER_N_ATOMS,
            'total_samples': total_samples,
            'results': all_serializable
        }, f, indent=2)
    print(f"\nJSON已保存到: {json_path}")
    
    print(f"\n{'='*60}")
    print(f"采样完成！")
    print(f"  原子数量范围: {N_ATOMS_LIST[0]}-{N_ATOMS_LIST[-1]}")
    print(f"  每个梯度: {SAMPLES_PER_N_ATOMS} 个分子")
    print(f"  总共生成: {total_samples} 个分子")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()