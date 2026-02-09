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

# 原子数量列表：25个不同的原子数量
N_ATOMS_LIST = [36, 40, 44, 48, 49,
                50, 51, 52, 56, 60, 
                64, 68, 70, 72, 76, 
                77, 78, 79, 80, 81, 
                83, 84, 85, 86, 87]
SAMPLES_PER_N_ATOMS = 100  # 每个原子数量生成的分子数


def count_existing_samples(molecule_index, n_atoms):
    """
    检测增量保存文件中已有的样本数量，用于断点续传。
    
    Returns:
        int: 已保存的样本数量
    """
    json_path = f'data/incremental/mol{molecule_index}_n{n_atoms}_samples.jsonl'
    if not os.path.exists(json_path):
        return 0
    try:
        with open(json_path, 'r') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def load_samples_from_incremental(molecule_index, n_atoms):
    """
    从增量保存文件中加载已保存的样本数据。
    
    Returns:
        list: 样本数据列表，每个元素包含 sample_serializable 格式的数据
    """
    json_path = f'data/incremental/mol{molecule_index}_n{n_atoms}_samples.jsonl'
    if not os.path.exists(json_path):
        return []
    
    samples = []
    try:
        with open(json_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    sample_serializable = json.loads(line)
                    # 重建完整的 sample_data 格式
                    sample_data = {
                        'n_atoms': n_atoms,
                        'atoms': np.array(sample_serializable['x1']['atoms']),
                        'positions': np.array(sample_serializable['x1']['positions']),
                        'sample_serializable': sample_serializable
                    }
                    samples.append(sample_data)
    except Exception as e:
        print(f"警告: 读取增量文件 {json_path} 时出错: {e}")
        return []
    
    return samples

# 工作函数：在指定GPU上运行采样（处理多个n_atoms任务）
def run_sampling_on_gpu(gpu_id, tasks, molecule_index, molblock, charges_list, 
                        params_dict, batch_size):
    """
    在指定GPU上运行采样任务
    tasks: list of (n_atoms, num_batches) 元组
    """
    import torch
    import numpy as np
    import json
    import os
    from shepherd.lightning_module import LightningModule
    from shepherd.inference import inference_sample
    from shepherd.extract import create_rdkit_molecule
    from shepherd.shepherd_score_utils.generate_point_cloud import (
        get_atomic_vdw_radii, get_molecular_surface, get_electrostatics_given_point_charges
    )
    from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores
    from shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates
    import rdkit
    from rdkit import Chem
    import gc
    
    # 确保增量保存目录存在
    os.makedirs('data/incremental', exist_ok=True)
    
    # 设置当前进程使用的GPU
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    print(f"GPU {gpu_id}: 加载模型...")
    model_pl = LightningModule.load_from_checkpoint(chkpt)
    params = model_pl.params
    model_pl.to(device)
    model_pl.model.device = device
    model_pl.eval()
    
    # 准备分子数据
    mol = rdkit.Chem.MolFromMolBlock(molblock, removeHs=False)
    charges = np.array(charges_list)
    
    mol_coordinates = np.array(mol.GetConformer().GetPositions())
    mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis=0)
    mol = update_mol_coordinates(mol, mol_coordinates)
    
    # 计算条件目标（只需计算一次）
    centers = mol.GetConformer().GetPositions()
    radii = get_atomic_vdw_radii(mol)
    surface = get_molecular_surface(
        centers, radii, 
        params_dict['dataset']['x3']['num_points'], 
        probe_radius=params_dict['dataset']['probe_radius'],
        num_samples_per_atom=20,
    )
    
    pharm_types, pharm_pos, pharm_direction = get_pharmacophores(
        mol,
        multi_vector=params_dict['dataset']['x4']['multivectors'],
        check_access=params_dict['dataset']['x4']['check_accessibility'],
    )
    
    electrostatics = get_electrostatics_given_point_charges(
        charges, centers, surface,
    )
    
    num_pharmacophores = len(pharm_types)
    
    # 按n_atoms分组的结果
    all_task_results = {}
    
    # 处理分配给此GPU的所有任务
    for task_idx, (n_atoms, num_batches, target_samples) in enumerate(tasks):
        # ===== 断点续传：检测已完成的样本数 =====
        existing_count = count_existing_samples(molecule_index, n_atoms)
        remaining_samples = max(0, target_samples - existing_count)
        
        if remaining_samples == 0:
            print(f"GPU {gpu_id}: ⏭️ n_atoms={n_atoms} 已完成 ({existing_count}/{target_samples})，跳过")
            all_task_results[n_atoms] = []  # 标记为空，结果已在文件中
            continue
        
        # 计算需要继续的批次数
        remaining_batches = (remaining_samples + batch_size - 1) // batch_size
        start_batch_idx = num_batches - remaining_batches
        
        print(f"GPU {gpu_id}: 处理任务 {task_idx + 1}/{len(tasks)}, n_atoms={n_atoms}")
        print(f"         📊 已有 {existing_count}/{target_samples} 样本，需续采 {remaining_samples} 个")
        
        task_results = []
        for batch_idx in range(start_batch_idx, num_batches):
            current_batch_num = batch_idx - start_batch_idx + 1
            print(f"GPU {gpu_id}: n_atoms={n_atoms}, 批次 {current_batch_num}/{remaining_batches}")
            
            gc.collect()
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                generated_samples = inference_sample(
                    model_pl,
                    batch_size=batch_size,
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
            
            # 转换样本
            batch_samples = []
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
                batch_samples.append(sample_data)
                task_results.append(sample_data)
            
            # 每个batch完成后立即保存JSON
            json_path = f'data/incremental/mol{molecule_index}_n{n_atoms}_samples.jsonl'
            with open(json_path, 'a') as f:
                for sample in batch_samples:
                    f.write(json.dumps(sample['sample_serializable']) + '\n')
            
            print(f"GPU {gpu_id}: n_atoms={n_atoms}, 批次 {batch_idx + 1}/{num_batches} 已保存")
            
            del generated_samples, batch_samples
            gc.collect()
            torch.cuda.empty_cache()
        
        all_task_results[n_atoms] = task_results
        print(f"GPU {gpu_id}: n_atoms={n_atoms} 完成，生成 {len(task_results)} 个样本")
    
    total_samples = sum(len(v) for v in all_task_results.values())
    print(f"GPU {gpu_id}: 完成所有任务，共生成 {total_samples} 个样本")
    return all_task_results

# 主函数：使用多GPU并行处理
if __name__ == '__main__':
    # ==================== 配置参数 ====================
    # 采样参数 - 根据GPU内存调整batch_size
    # 24GB GPU: batch_size=2-3
    # 140GB GPU: batch_size=15-25
    BATCH_SIZE = 2  # 在24GB GPU上使用小batch_size避免OOM
    
    print(f"\n{'='*100}")
    print(f"采样配置:")
    print(f"  GPU数量: {num_gpus}")
    print(f"  原子数量列表: {N_ATOMS_LIST} (共{len(N_ATOMS_LIST)}种)")
    print(f"  每种原子数量生成: {SAMPLES_PER_N_ATOMS} 个分子")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  总计每个自然分子生成: {len(N_ATOMS_LIST) * SAMPLES_PER_N_ATOMS} 个样本")
    print(f"  总计生成: {3 * len(N_ATOMS_LIST) * SAMPLES_PER_N_ATOMS} 个样本")
    print(f"{'='*100}\n")
    
    # 加载所有三种自然分子
    with open('../data/conformers/np/molblock_charges_NPs.pkl', 'rb') as f:
        molblocks_and_charges = pickle.load(f)
    
    # 加载模型参数（仅用于获取配置，在主进程中）
    print("加载模型配置...")
    model_pl_temp = LightningModule.load_from_checkpoint(chkpt)
    params_dict = model_pl_temp.params
    del model_pl_temp
    gc.collect()
    
    # 计算每个n_atoms需要的批次数
    num_batches_per_n_atoms = (SAMPLES_PER_N_ATOMS + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"每个n_atoms需要 {num_batches_per_n_atoms} 个批次 (每批 {BATCH_SIZE} 个样本)")
    
    # 存储所有结果
    all_results = {}
    
    # 对三种自然分子循环
    for mol_index in range(3):  # 0, 1, 2
        print(f"\n{'='*100}")
        print(f"处理自然分子 {mol_index + 1}/3 (index={mol_index})")
        print(f"{'='*100}\n")
        
        # 提取目标分子数据
        molblock = molblocks_and_charges[mol_index][0]
        charges_list = molblocks_and_charges[mol_index][1]
        
        # 显示分子（在主进程中）
        mol = rdkit.Chem.MolFromMolBlock(molblock, removeHs=False)
        print(f"分子 {mol_index}: 原子数={mol.GetNumAtoms()}")
        
        # 构建任务列表：每个n_atoms是一个任务
        # 任务格式: (n_atoms, num_batches, target_samples)
        all_tasks = [(n_atoms, num_batches_per_n_atoms, SAMPLES_PER_N_ATOMS) for n_atoms in N_ATOMS_LIST]
        
        # 将任务均匀分配到GPU（交替分配以平衡负载）
        tasks_per_gpu = [[] for _ in range(num_gpus)]
        for i, task in enumerate(all_tasks):
            gpu_id = i % num_gpus
            tasks_per_gpu[gpu_id].append(task)
        
        print(f"任务分配:")
        for gpu_id, tasks in enumerate(tasks_per_gpu):
            n_atoms_list = [t[0] for t in tasks]
            total_batches = sum(t[1] for t in tasks)
            print(f"  GPU {gpu_id}: {len(tasks)} 个n_atoms任务, 共 {total_batches} 批次")
            print(f"           n_atoms: {n_atoms_list}")
        
        # 使用多进程并行执行
        results_by_n_atoms = {n_atoms: [] for n_atoms in N_ATOMS_LIST}
        
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id, task_list in enumerate(tasks_per_gpu):
                if task_list:  # 只提交非空任务列表
                    future = executor.submit(
                        run_sampling_on_gpu,
                        gpu_id, task_list, mol_index, molblock, charges_list,
                        params_dict, BATCH_SIZE
                    )
                    futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    gpu_results = future.result()  # dict: {n_atoms: [samples]}
                    for n_atoms, samples in gpu_results.items():
                        results_by_n_atoms[n_atoms].extend(samples)
                except Exception as e:
                    print(f"错误: {e}")
                    import traceback
                    traceback.print_exc()
        
        # 从增量文件中加载已保存的样本（用于断点续传场景）
        print(f"\n从增量文件加载已保存的样本...")
        for n_atoms in N_ATOMS_LIST:
            if len(results_by_n_atoms[n_atoms]) == 0:
                # 如果 GPU 没有返回结果，尝试从增量文件加载
                loaded_samples = load_samples_from_incremental(mol_index, n_atoms)
                if loaded_samples:
                    results_by_n_atoms[n_atoms] = loaded_samples
                    print(f"  n_atoms={n_atoms}: 从增量文件加载了 {len(loaded_samples)} 个样本")
        
        # 统计结果
        total_samples = sum(len(v) for v in results_by_n_atoms.values())
        print(f"\n所有GPU完成采样，共 {total_samples} 个样本")
        
        # 保存结果 - 按n_atoms分组
        os.makedirs('data', exist_ok=True)
        
        molecule_results = {}
        for n_atoms in N_ATOMS_LIST:
            samples = results_by_n_atoms[n_atoms]
            # 截取到SAMPLES_PER_N_ATOMS个（可能多生成了一些）
            samples = samples[:SAMPLES_PER_N_ATOMS]
            
            serializable_samples = []
            for sample_data in samples:
                serializable_samples.append(sample_data['sample_serializable'])
            
            molecule_results[f'n_atoms_{n_atoms}'] = {
                'n_atoms': n_atoms,
                'num_samples': len(serializable_samples),
                'samples': serializable_samples
            }
            print(f"  n_atoms={n_atoms}: {len(serializable_samples)} 个样本")
        
        # 保存当前分子的所有样本到JSON
        all_results[f'molecule_{mol_index}'] = molecule_results
        
        # 保存当前分子的JSON（增量保存，避免丢失）
        mol_output_file = f'data/molecule_{mol_index}_all_samples.json'
        with open(mol_output_file, 'w') as f:
            json.dump(molecule_results, f, indent=2)
        print(f"  JSON已保存到: {mol_output_file}")
        
        print(f"\n分子 {mol_index} 完成！共生成 {total_samples} 个样本\n")
        
        # 清理内存
        gc.collect()
    
    # 保存所有结果到JSON文件
    output_file = 'data/generated_samples_all_molecules.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 统计总数
    total_all = 0
    for mol_key, mol_data in all_results.items():
        for n_key, n_data in mol_data.items():
            total_all += n_data['num_samples']
    
    print(f"\n{'='*100}")
    print(f"所有采样完成！")
    print(f"总共生成了 {total_all} 个样本")
    print(f"  3 个自然分子 × {len(N_ATOMS_LIST)} 种原子数量 × {SAMPLES_PER_N_ATOMS} 个样本")
    print(f"结果已保存到: {output_file}")
    print(f"{'='*100}")