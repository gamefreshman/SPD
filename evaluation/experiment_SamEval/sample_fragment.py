# 核心库导入
import os
import json
import pickle
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# 科学计算库
import numpy as np
import torch
import rdkit
from tqdm import tqdm

# Shepherd相关模块
from shepherd.lightning_module import LightningModule
from shepherd.inference import inference_sample
from shepherd.extract import create_rdkit_molecule
from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores

# 检测可用GPU
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"🖥️  检测到 {num_gpus} 张GPU")
    gpu_devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name}, {gpu_total_memory:.1f} GB")
else:
    print("⚠️  未检测到GPU，将使用CPU")
    num_gpus = 0
    gpu_devices = [torch.device('cpu')]

# ==================== 配置参数 ====================
chkpt = '/home1/zhh/workspace/SPD/evaluation/core_data/data/3/DPO/last.ckpt'
CONDITION_FILE = '/home1/zhh/workspace/SPD/data/conformers/fragment_merging/fragment_merge_condition.pickle'
MARGINALS_DATA_FILE = '/home1/zhh/workspace/SPD/data/conformers/np/molblock_charges_NPs.pkl'
N_ATOMS_LIST = list(range(50, 90))  # 50-89，共40个梯度
SAMPLES_PER_N_ATOMS = 2  # 每个梯度采样2个分子
UNIFORM_BATCH_SIZE = 2

def get_optimal_batch_size(gpu_id):
    """返回统一的batch size"""
    return UNIFORM_BATCH_SIZE

# 加载模型获取params用于边际分布计算
print("加载模型配置...")
model_pl = LightningModule.load_from_checkpoint(chkpt)
params = model_pl.params

# 将模型复制到所有GPU
models_dict = {}
for i, device in enumerate(gpu_devices):
    model_copy = LightningModule.load_from_checkpoint(chkpt)
    model_copy.to(device)
    model_copy.model.device = device
    model_copy.eval()
    models_dict[i] = model_copy
    print(f"✅ 模型已加载到 {device}")

# 加载边际分布计算用的数据
with open(MARGINALS_DATA_FILE, 'rb') as f:
    molblocks_and_charges = pickle.load(f)
    print(f"加载边际分布数据: {len(molblocks_and_charges)} 个分子")

# 加载条件数据
with open(CONDITION_FILE, 'rb') as f:
    condition_data = pickle.load(f)
    print(f"加载条件数据: {CONDITION_FILE}")

surface = condition_data['x3']['positions']
electrostatics = condition_data['x3']['charges']
pharm_types = condition_data['x4']['types']
pharm_pos = condition_data['x4']['positions']
pharm_direction = condition_data['x4']['directions']
num_pharmacophores = len(pharm_types)
print(f"  - 药效团数量: {num_pharmacophores}")

# ==================== 辅助函数定义 ====================
def get_bond_type_str(bond):
    return str(bond.GetBondType())

def process_batch(batch, atom_types_x1, bond_types_x1, max_node_types_x4, params):
    """并行处理分子批次，统计特征分布"""
    batch_atom_counts = torch.zeros(len(atom_types_x1), dtype=torch.float)
    batch_bond_counts = torch.zeros(len(bond_types_x1), dtype=torch.float)
    batch_pharm_counts = torch.zeros(max_node_types_x4, dtype=torch.float)

    for mol_block, _ in batch:
        mol = rdkit.Chem.MolFromMolBlock(mol_block, removeHs=False)
        if not mol:
            continue

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

        if params['dataset']['compute_x4']:
            if params['dataset']['x4']['add_virtual_node']:
                batch_pharm_counts[0] += 1
            
            try:
                p_types, _, _ = get_pharmacophores(
                    mol, 
                    multi_vector=params['dataset']['x4']['multivectors'],
                    check_access=params['dataset']['x4']['check_accessibility']
                )
                for p_type in (p_types + 1):
                    if p_type < max_node_types_x4:
                        batch_pharm_counts[p_type] += 1
            except Exception:
                pass
    
    return batch_atom_counts, batch_bond_counts, batch_pharm_counts

# ==================== 计算边际分布 ====================
print("📊 开始计算特征边际分布...")

atom_types_x1 = params['dataset']['x1']['atom_types']
bond_types_x1 = params['dataset']['x1']['bond_types']
max_node_types_x4 = params['dataset']['x4']['max_node_types']

total_atom_counts = torch.zeros(len(atom_types_x1), dtype=torch.float)
total_bond_counts = torch.zeros(len(bond_types_x1), dtype=torch.float)
total_pharm_counts = torch.zeros(max_node_types_x4, dtype=torch.float)

num_processes = multiprocessing.cpu_count()
batch_size_for_processing = 1000
batches = [molblocks_and_charges[i:i + batch_size_for_processing] 
           for i in range(0, len(molblocks_and_charges), batch_size_for_processing)]

worker_fn = partial(process_batch, 
                    atom_types_x1=atom_types_x1, 
                    bond_types_x1=bond_types_x1, 
                    max_node_types_x4=max_node_types_x4,
                    params=params)

with multiprocessing.Pool(processes=num_processes) as pool:
    results = list(tqdm(pool.imap(worker_fn, batches, chunksize=1), 
                       total=len(batches), desc="统计特征"))

for res in results:
    total_atom_counts += res[0]
    total_bond_counts += res[1]
    total_pharm_counts += res[2]

atom_marginals_x1 = (total_atom_counts / total_atom_counts.sum() 
                     if total_atom_counts.sum() > 0 
                     else torch.ones_like(total_atom_counts) / len(total_atom_counts))
bond_marginals_x1 = (total_bond_counts / total_bond_counts.sum() 
                     if total_bond_counts.sum() > 0 
                     else torch.ones_like(total_bond_counts) / len(total_bond_counts))

print(f"✅ 边际分布计算完成")

if len(atom_marginals_x1) > 0:
    atom_marginals_x1[0] = 0.0
    atom_marginals_x1 = atom_marginals_x1 / atom_marginals_x1.sum()

marginals = (atom_marginals_x1, bond_marginals_x1)

# ==================== 采样函数 ====================
def generate_samples_for_n_atoms(model, device, n_atoms, num_samples, marginals, max_retries=5):
    """为指定原子数生成样本"""
    
    gc.collect()
    torch.cuda.empty_cache()
    
    for retry in range(max_retries):
        try:
            with torch.no_grad():
                generated_samples = inference_sample(
                    model,
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
                    # 边际分布
                    atom_marginals=marginals[0].to(device),
                    bond_marginals=marginals[1].to(device),
                )
            return generated_samples
        except AssertionError as e:
            gc.collect()
            torch.cuda.empty_cache()
            if retry == max_retries - 1:
                return []
    return []

# ==================== 主采样流程 ====================
print(f"\n{'='*60}")
print(f"Fragment Merge 条件采样")
print(f"  条件文件: {CONDITION_FILE}")
print(f"  原子数量范围: {N_ATOMS_LIST[0]}-{N_ATOMS_LIST[-1]} (共{len(N_ATOMS_LIST)}个梯度)")
print(f"  每个梯度采样: {SAMPLES_PER_N_ATOMS} 个分子")
print(f"  总计采样: {len(N_ATOMS_LIST) * SAMPLES_PER_N_ATOMS} 个分子")
print(f"  GPU数量: {num_gpus}")
print(f"  Batch size: {UNIFORM_BATCH_SIZE}")
print(f"{'='*60}\n")

# 结果存储
all_results = {n_atoms: [] for n_atoms in N_ATOMS_LIST}

if num_gpus > 1:
    import queue
    import threading
    
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # 构建任务列表
    for n_atoms in N_ATOMS_LIST:
        task_queue.put((n_atoms, SAMPLES_PER_N_ATOMS))
    
    total_tasks = len(N_ATOMS_LIST)
    progress_lock = threading.Lock()
    progress_count = [0]
    
    def gpu_worker(gpu_id, model, marginals):
        device = gpu_devices[gpu_id]
        while True:
            try:
                task = task_queue.get(timeout=1)
                if task is None:
                    break
                
                n_atoms, num_samples = task
                samples = generate_samples_for_n_atoms(model, device, n_atoms, num_samples, marginals)
                
                # 转换样本
                task_results = []
                for sample in samples:
                    sample_data = {
                        'n_atoms': n_atoms,
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
                
                result_queue.put((n_atoms, task_results))
                
                with progress_lock:
                    progress_count[0] += 1
                print(f"  ✅ n_atoms={n_atoms} 完成，生成 {len(task_results)} 个样本")
                
            except queue.Empty:
                continue
            except Exception as e:
                pass
    
    # 启动工作线程
    workers = []
    for gpu_id in range(num_gpus):
        worker = threading.Thread(target=gpu_worker, args=(gpu_id, models_dict[gpu_id], marginals))
        worker.start()
        workers.append(worker)
        print(f"  🚀 GPU {gpu_id} 工作线程已启动")
    
    # 添加结束信号
    for _ in range(num_gpus):
        task_queue.put(None)
    
    # 等待完成
    for worker in workers:
        worker.join()
    
    # 收集结果
    while not result_queue.empty():
        n_atoms, samples = result_queue.get()
        all_results[n_atoms].extend(samples)

else:
    # 单GPU处理
    print("  使用单GPU顺序处理")
    device = gpu_devices[0]
    model = models_dict[0]
    
    for n_atoms in N_ATOMS_LIST:
        samples = generate_samples_for_n_atoms(model, device, n_atoms, SAMPLES_PER_N_ATOMS, marginals)
        
        for sample in samples:
            sample_data = {
                'n_atoms': n_atoms,
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
            all_results[n_atoms].append(sample_data)
        
        print(f"  ✅ n_atoms={n_atoms} 完成，生成 {len(samples)} 个样本")

# 统计结果
total_samples = sum(len(v) for v in all_results.values())
print(f"\n{'='*60}")
print(f"🎉 采样完成！")
print(f"  原子数量范围: {N_ATOMS_LIST[0]}-{N_ATOMS_LIST[-1]}")
print(f"  每个梯度: {SAMPLES_PER_N_ATOMS} 个分子")
print(f"  总共生成: {total_samples} 个分子")
print(f"{'='*60}")

# 保存结果
os.makedirs('data', exist_ok=True)

all_serializable = {}
for n_atoms in N_ATOMS_LIST:
    samples = all_results[n_atoms]
    serializable_samples = [s['sample_serializable'] for s in samples]
    all_serializable[f'n_atoms_{n_atoms}'] = {
        'n_atoms': n_atoms,
        'num_samples': len(serializable_samples),
        'samples': serializable_samples
    }

json_path = f'data/fragment_merge_samples_{os.path.basename(chkpt)}.json'
with open(json_path, 'w') as f:
    json.dump({
        'condition_file': CONDITION_FILE,
        'n_atoms_range': [N_ATOMS_LIST[0], N_ATOMS_LIST[-1]],
        'samples_per_n_atoms': SAMPLES_PER_N_ATOMS,
        'total_samples': total_samples,
        'results': all_serializable
    }, f, indent=2)
print(f"\n💾 JSON已保存到: {json_path}")
