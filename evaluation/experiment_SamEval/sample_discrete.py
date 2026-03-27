# 核心库导入
import os
import json
import pickle
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

# 科学计算库
import numpy as np
import torch
import rdkit
from tqdm import tqdm

# Shepherd相关模块
from shepherd.lightning_module import LightningModule
from shepherd.inference import inference_sample
from shepherd.extract import create_rdkit_molecule_from_mol
from shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates
from shepherd.shepherd_score_utils.generate_point_cloud import (
    get_atomic_vdw_radii, 
    get_molecular_surface,
    get_electrostatics_given_point_charges,
)
from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores

# 设置CUDA环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"

# 检测可用GPU
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"🖥️  检测到 {num_gpus} 张GPU")
    gpu_devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
    
    # 获取GPU信息和内存状态
    gpu_memory_info = []
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        
        # 获取当前可用内存
        torch.cuda.set_device(i)
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        free = gpu_total_memory - reserved
        
        gpu_memory_info.append({
            'device': i,
            'name': gpu_name,
            'total': gpu_total_memory,
            'free': free,
            'allocated': allocated
        })
        
        print(f"  GPU {i}: {gpu_name}")
        print(f"    - 总内存: {gpu_total_memory:.1f} GB")
        print(f"    - 可用内存: {free:.1f} GB")
        print(f"    - 已分配: {allocated:.1f} GB")
else:
    print("⚠️  未检测到GPU，将使用CPU")
    num_gpus = 0
    gpu_devices = [torch.device('cpu')]
    gpu_memory_info = []

# 根据GPU内存动态调整批量大小
# 固定的最优batch size配置（经测试验证）
GPU_BATCH_SIZES = {
    0: 2,   # GPU 0 (4090)
    1: 2,  # GPU 1 (3090)
    2: 2,  # GPU 2 (3090)
}

def get_optimal_batch_size(gpu_id):
    """返回固定的最优batch size"""
    return GPU_BATCH_SIZES.get(gpu_id, 2)

# 加载模型
chkpt = '/home1/zhh/workspace/SPD/evaluation/ckpt/last-33epoch.ckpt'

# 主设备用于加载模型
main_device = gpu_devices[0] if gpu_devices else torch.device('cpu')
model_pl = LightningModule.load_from_checkpoint(chkpt)
params = model_pl.params

# 将模型复制到所有GPU（用于多GPU并行推理）
models_dict = {}
for i, device in enumerate(gpu_devices):
    model_copy = LightningModule.load_from_checkpoint(chkpt)
    model_copy.to(device)
    model_copy.model.device = device
    model_copy.eval()  # 设置为评估模式
    models_dict[i] = model_copy
    print(f"✅ 模型已加载到 {device}")


with open('/home1/zhh/workspace/SPD/data/conformers/np/molblock_charges_NPs.pkl', 'rb') as f:
    # 从pkl文件中读取molblock和charges数据
    molblocks_and_charges = pickle.load(f)
    # 打印数据长度以确认实际包含的分子数量
    print(f"加载的数据包含 {len(molblocks_and_charges)} 个分子")

# ==================== 修改：处理所有天然产物分子 ====================
# 将处理所有天然产物分子

print("\n将对所有天然产物分子进行采样：")
for idx in range(len(molblocks_and_charges)):
    mol = rdkit.Chem.MolFromMolBlock(molblocks_and_charges[idx][0], removeHs=False)
    if mol:
        print(f"  - 分子 {idx}: {mol.GetNumAtoms()} 个原子")


# ==================== 辅助函数定义 ====================
def get_bond_type_str(bond):
    return str(bond.GetBondType())

def process_batch(batch, atom_types_x1, bond_types_x1, max_node_types_x4, params):
    """
    并行处理分子批次，统计特征分布（与dpo_trainer.py保持一致）
    """
    batch_atom_counts = torch.zeros(len(atom_types_x1), dtype=torch.float)
    batch_bond_counts = torch.zeros(len(bond_types_x1), dtype=torch.float)
    batch_pharm_counts = torch.zeros(max_node_types_x4, dtype=torch.float)

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
                pass
    
    return batch_atom_counts, batch_bond_counts, batch_pharm_counts

# ==================== 计算边际分布（与dpo_trainer.py保持一致） ====================
# 必须在采样前计算，为扩散模型提供先验分布

print("📊 开始计算特征边际分布...")

# 特征类型定义（从params获取）
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
                       total=len(batches), desc="统计特征"))

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

print(f"✅ 边际分布计算完成")
print(f"  - Atom Marginals: {atom_marginals_x1.shape}")
print(f"  - Bond Marginals: {bond_marginals_x1.shape}")
print(f"  - Pharmacophore Marginals: {pharm_marginals_x4.shape}")

print(f"  - Atom Marginals: {atom_marginals_x1}")
print(f"  - Bond Marginals: {bond_marginals_x1}")
print(f"  - Pharmacophore Marginals: {pharm_marginals_x4}")

if len(atom_marginals_x1) > 0:
    print(f"🔧 修正前 atom_marginals[0] = {atom_marginals_x1[0]:.6f}")
    atom_marginals_x1[0] = 0.0
    atom_marginals_x1 = atom_marginals_x1 / atom_marginals_x1.sum()
    print(f"🔧 修正后 atom_marginals[0] = {atom_marginals_x1[0]:.6f}")

# ==================== 分子预处理函数（可并行） ====================
def preprocess_molecule(args):
    """
    预处理单个分子，提取所有必要的特征
    Args:
        args: 包含(mol_index, molblock_charges, params)的元组
    Returns:
        分子索引和预处理后的特征字典
    """
    mol_index, (mol_block, charges), params = args
    
    try:
        # 从molblock创建RDKit分子对象
        mol = rdkit.Chem.MolFromMolBlock(mol_block, removeHs=False)
        if not mol:
            return mol_index, None
        
        charges = np.array(charges)
        
        # 分子坐标标准化
        mol_coordinates = np.array(mol.GetConformer().GetPositions())
        mol_coordinates = mol_coordinates - np.mean(mol_coordinates, axis=0)
        mol = update_mol_coordinates(mol, mol_coordinates)
        
        # 条件特征提取
        centers = mol.GetConformer().GetPositions()
        radii = get_atomic_vdw_radii(mol)
        
        # 生成分子表面点云
        surface = get_molecular_surface(
            centers, 
            radii, 
            params['dataset']['x3']['num_points'],
            probe_radius=params['dataset']['probe_radius'],
            num_samples_per_atom=20,
        )
        
        # 提取药效团特征
        pharm_types, pharm_pos, pharm_direction = get_pharmacophores(
            mol,
            multi_vector=params['dataset']['x4']['multivectors'],
            check_access=params['dataset']['x4']['check_accessibility'],
        )
        
        # 计算表面静电势
        electrostatics = get_electrostatics_given_point_charges(
            charges, centers, surface,
        )
        
        return mol_index, {
            'mol': mol,
            'n_atoms': mol.GetNumAtoms(),
            'surface': surface,
            'electrostatics': electrostatics,
            'pharm_types': pharm_types,
            'pharm_pos': pharm_pos,
            'pharm_direction': pharm_direction,
            'num_pharmacophores': len(pharm_types)
        }
    except Exception as e:
        print(f"分子 {mol_index} 预处理失败: {e}")
        return mol_index, None

# ==================== 采样配置 ====================
TOTAL_SAMPLES_PER_MOL = 20  # 每个分子生成的样本数

# ==================== 主进程中执行模型推理（支持多GPU） ====================
def generate_samples_batch(mol_index, mol_features, model_pl, device, marginals, params, gpu_id, batch_size, num_samples):
    """
    在指定GPU上为单个分子生成指定数量的样本（用于细粒度并行）
    """
    if mol_features is None:
        return mol_index, []
    
    n_atoms = 70
    mol_samples = []
    
    # 计算需要的迭代次数
    num_iterations = (num_samples + batch_size - 1) // batch_size
    
    for iteration in range(num_iterations):
        current_batch_size = min(batch_size, num_samples - iteration * batch_size)
        try:
            with torch.cuda.device(gpu_id):
                generated_samples = inference_sample(
                    model_pl,
                    batch_size=current_batch_size,
                    N_x1=n_atoms,
                    N_x4=mol_features['num_pharmacophores'],
                    unconditional=False,
                    
                    # 噪声控制
                    prior_noise_scale=1.0,
                    denoising_noise_scale=1.0,
                    inject_noise_at_ts=[],
                    inject_noise_scales=[],
                    
                    # 谐波化
                    harmonize=False,
                    harmonize_ts=[],
                    harmonize_jumps=[],
                    
                    # 条件修复
                    inpaint_x2_pos=False,
                    inpaint_x3_pos=False,
                    inpaint_x3_x=False,
                    inpaint_x4_pos=True,
                    inpaint_x4_direction=True,
                    inpaint_x4_type=True,
                    
                    # 修复控制
                    stop_inpainting_at_time_x2=0.0,
                    add_noise_to_inpainted_x2_pos=0.0,
                    stop_inpainting_at_time_x3=0.0,
                    add_noise_to_inpainted_x3_pos=0.0,
                    add_noise_to_inpainted_x3_x=0.0,
                    stop_inpainting_at_time_x4=0.0,
                    add_noise_to_inpainted_x4_pos=0.0,
                    add_noise_to_inpainted_x4_direction=0.0,
                    add_noise_to_inpainted_x4_type=0.0,
                    
                    # 条件输入
                    center_of_mass=np.zeros(3),
                    surface=mol_features['surface'],
                    electrostatics=mol_features['electrostatics'],
                    pharm_types=mol_features['pharm_types'],
                    pharm_pos=mol_features['pharm_pos'],
                    pharm_direction=mol_features['pharm_direction'],
                    
                    # 边际分布
                    atom_marginals=marginals[0].to(device),
                    bond_marginals=marginals[1].to(device),
                    pharm_marginals=marginals[2].to(device) if len(marginals) > 2 and marginals[2] is not None else None,
                )
            
            for sample in generated_samples:
                sample['source_mol_index'] = mol_index
            mol_samples.extend(generated_samples)
            
        except Exception as e:
            print(f"[GPU {gpu_id}] 分子 {mol_index} 批次 {iteration} 采样失败: {e}")
            continue
    
    return mol_index, mol_samples

def generate_samples_for_molecule(mol_index, mol_features, model_pl, device, marginals, params, gpu_id=0):
    """
    在指定GPU上为单个分子生成所有样本（兼容旧接口）
    """
    batch_size = get_optimal_batch_size(gpu_id)
    return generate_samples_batch(mol_index, mol_features, model_pl, device, marginals, params, gpu_id, batch_size, TOTAL_SAMPLES_PER_MOL)

# ==================== 多GPU并行推理任务 ====================
def gpu_inference_worker(task_queue, result_queue, gpu_id, model, marginals, params):
    """
    GPU工作线程，从任务队列获取任务并执行推理
    """
    device = gpu_devices[gpu_id]
    while True:
        try:
            # 获取任务
            task = task_queue.get(timeout=1)
            if task is None:  # 结束信号
                break
                
            mol_index, mol_features = task
            
            # 执行推理
            mol_index, samples = generate_samples_for_molecule(
                mol_index, mol_features, model, device, marginals, params, gpu_id
            )
            
            # 返回结果
            result_queue.put((mol_index, samples))
            
            print(f"  [GPU {gpu_id}] ✅ 分子 {mol_index + 1} 完成: 生成 {len(samples)} 个样本")
            
        except queue.Empty:
            # 队列为空时继续等待
            continue
        except Exception as e:
            print(f"[GPU {gpu_id}] 错误: {e}")
            # 出错时也要返回空结果，避免结果丢失
            if 'mol_index' in locals():
                result_queue.put((mol_index, []))

# ==================== 对每个分子循环采样（优化版本） ====================

# 辅助函数：转换数据为JSON格式
def convert_for_json(obj):
    """递归转换numpy数组和torch张量为Python列表"""
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_for_json(elem) for elem in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    return obj

# 存储所有分子的所有样本
all_generated_samples = []

print(f"🚀 开始处理 {len(molblocks_and_charges)} 个分子")
print(f"{'='*60}")

# Step 1: 预处理所有分子（避免多进程与CUDA冲突）
print("📋 Step 1: 预处理分子特征...")

# 预处理配置
# 注意：由于CUDA已初始化，fork子进程会导致死锁，这里使用串行预处理
# 只有3个分子，串行预处理非常快
use_parallel_preprocessing = False  # 禁用多进程（避免CUDA fork问题）

mol_features_dict = {}

if use_parallel_preprocessing:
    # 使用多进程并行预处理（纯CPU操作，不涉及CUDA）
    num_workers = min(multiprocessing.cpu_count(), len(molblocks_and_charges))
    print(f"  使用 {num_workers} 个进程并行预处理...")
    
    preprocess_args = [
        (mol_index, molblocks_and_charges[mol_index], params)
        for mol_index in range(len(molblocks_and_charges))
    ]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_mol = {executor.submit(preprocess_molecule, args): args[0] 
                        for args in preprocess_args}
        
        for future in tqdm(as_completed(future_to_mol), total=len(future_to_mol), 
                          desc="预处理分子"):
            mol_index = future_to_mol[future]
            try:
                idx, features = future.result()
                mol_features_dict[idx] = features
                if features:
                    print(f"  ✓ 分子 {idx}: {features['n_atoms']} 个原子, "
                         f"{features['num_pharmacophores']} 个药效团")
            except Exception as e:
                print(f"  ✗ 分子 {mol_index} 预处理失败: {e}")
                mol_features_dict[mol_index] = None
else:
    # 串行预处理
    print("  使用串行预处理...")
    for mol_index in tqdm(range(len(molblocks_and_charges)), desc="预处理分子"):
        args = (mol_index, molblocks_and_charges[mol_index], params)
        try:
            idx, features = preprocess_molecule(args)
            mol_features_dict[idx] = features
            if features:
                print(f"  ✓ 分子 {idx}: {features['n_atoms']} 个原子, "
                     f"{features['num_pharmacophores']} 个药效团")
        except Exception as e:
            print(f"  ✗ 分子 {mol_index} 预处理失败: {e}")
            mol_features_dict[mol_index] = None

print(f"\n✅ 预处理完成: {sum(1 for f in mol_features_dict.values() if f)} 个分子成功")

# Step 2: 使用多GPU并行执行模型推理
print(f"\n{'='*60}")
print("🔬 Step 2: 使用多GPU并行推理生成样本...")
print(f"  使用 {num_gpus} 张GPU进行并行推理")

# 准备边际分布
marginals = (atom_marginals_x1, bond_marginals_x1, pharm_marginals_x4)

if num_gpus > 1:
    import queue
    import threading
    
    # 创建任务队列和结果队列
    task_queue = queue.Queue()
    result_queue = queue.Queue()
    
    # ==================== 基于batch_size的任务拆分 ====================
    # 根据GPU batch_size来分配子任务，使任务量与GPU处理能力成比例
    # 总batch_size用于计算任务比例
    total_batch_capacity = sum(GPU_BATCH_SIZES.get(i, 8) for i in range(num_gpus))
    print(f"  📊 GPU配置: {[f'GPU{i}={GPU_BATCH_SIZES.get(i,8)}' for i in range(num_gpus)]}")
    print(f"  📊 总批处理能力: {total_batch_capacity}")
    
    # 每个子任务的样本数 - 使用较小的值以便更均匀分配
    samples_per_subtask = 20  # 每个子任务生成的样本数
    
    total_subtasks = 0
    for mol_index in sorted(mol_features_dict.keys()):
        if mol_features_dict[mol_index] is not None:
            # 将200样本拆成多个子任务
            num_subtasks = (TOTAL_SAMPLES_PER_MOL + samples_per_subtask - 1) // samples_per_subtask
            for subtask_id in range(num_subtasks):
                samples_this_subtask = min(samples_per_subtask, 
                                           TOTAL_SAMPLES_PER_MOL - subtask_id * samples_per_subtask)
                # 任务格式: (mol_index, mol_features, subtask_id, num_samples)
                task_queue.put((mol_index, mol_features_dict[mol_index], subtask_id, samples_this_subtask))
                total_subtasks += 1
    
    valid_mol_count = sum(1 for f in mol_features_dict.values() if f is not None)
    print(f"  📋 任务拆分: {valid_mol_count} 个分子 → {total_subtasks} 个子任务")
    print(f"  🔄 每个子任务生成 {samples_per_subtask} 个样本，可在不同GPU上并行")
    
    # 添加进度追踪
    progress_lock = threading.Lock()
    progress_count = [0]
    
    def update_progress():
        with progress_lock:
            progress_count[0] += 1
            pct = progress_count[0] / total_subtasks * 100
            print(f"  📊 进度: {progress_count[0]}/{total_subtasks} 子任务完成 ({pct:.1f}%)")
    
    # GPU工作线程 - 处理细粒度子任务
    def gpu_inference_worker_fine_grained(task_queue, result_queue, gpu_id, model, marginals, params):
        """细粒度并行的GPU工作线程"""
        device = gpu_devices[gpu_id]
        batch_size = get_optimal_batch_size(gpu_id)
        processed = 0
        
        while True:
            try:
                task = task_queue.get(timeout=1)
                if task is None:
                    break
                
                mol_index, mol_features, subtask_id, num_samples = task
                
                # 生成这个子任务的样本
                _, samples = generate_samples_batch(
                    mol_index, mol_features, model, device, marginals, params, 
                    gpu_id, batch_size, num_samples
                )
                
                result_queue.put((mol_index, subtask_id, samples))
                processed += 1
                update_progress()
                
                print(f"    [GPU {gpu_id}] ✅ 分子{mol_index}_子任务{subtask_id} | "
                      f"样本数: {len(samples)} | 本GPU已处理: {processed}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"    [GPU {gpu_id}] ❌ 错误: {e}")
                import traceback
                traceback.print_exc()
    
    # 启动GPU工作线程
    workers = []
    for gpu_id in range(num_gpus):
        worker = threading.Thread(
            target=gpu_inference_worker_fine_grained,
            args=(task_queue, result_queue, gpu_id, models_dict[gpu_id], marginals, params)
        )
        worker.start()
        workers.append(worker)
        print(f"  🚀 GPU {gpu_id} 工作线程已启动 (batch_size: {get_optimal_batch_size(gpu_id)})")
    
    # 添加结束信号
    for _ in range(num_gpus):
        task_queue.put(None)
    
    # 等待所有工作线程完成
    for worker in workers:
        worker.join()
    
    # 收集并合并结果
    results = {}  # {mol_index: [samples]}
    while not result_queue.empty():
        mol_index, subtask_id, samples = result_queue.get()
        if mol_index not in results:
            results[mol_index] = []
        results[mol_index].extend(samples)
    
    # 按原始顺序整理结果
    for mol_index in sorted(results.keys()):
        all_generated_samples.extend(results[mol_index])
    
    # 显示统计
    print(f"\n  📈 GPU并行统计:")
    for i in range(num_gpus):
        print(f"    GPU {i}: batch_size={get_optimal_batch_size(i)}, 显存={gpu_memory_info[i]['free']:.1f}GB")
        
else:
    # 单GPU或CPU处理
    print("  ⚠️  使用单设备顺序处理")
    
    for mol_index in sorted(mol_features_dict.keys()):
        mol_features = mol_features_dict[mol_index]
        if mol_features is None:
            print(f"  ⚠️  分子 {mol_index} 无有效特征，跳过")
            continue
        
        print(f"\n{'='*50}")
        print(f"🧪 处理分子 {mol_index + 1}/{len(molblocks_and_charges)}")
        print(f"  - 原子数: 70")
        print(f"  - 药效团数: {mol_features['num_pharmacophores']}")
        print(f"  - 总样本数: 20")
        
        # 生成样本
        mol_index, mol_samples = generate_samples_for_molecule(
            mol_index, mol_features, models_dict[0], main_device, marginals, params
        )
        
        all_generated_samples.extend(mol_samples)
        print(f"  ✅ 完成: 生成 {len(mol_samples)} 个样本")

print(f"\n{'='*60}")
print(f"🎉 所有采样完成!")
print(f"{'='*60}")
print(f"总样本数: {len(all_generated_samples)}")
print(f"每个分子采样 500 个样本")
# 统计每个分子的样本数
for mol_idx in sorted(set(s['source_mol_index'] for s in all_generated_samples)):
    count = sum(1 for s in all_generated_samples if s['source_mol_index'] == mol_idx)
    print(f"  - 分子 {mol_idx}: {count} 个样本")

# 保存结果
generated_samples_for_json = convert_for_json(all_generated_samples)
with open(f'output_all_mols_{os.path.basename(chkpt)}.json', 'w', encoding='utf-8') as f:
    json.dump(generated_samples_for_json, f, ensure_ascii=False, indent=4)
print(f"\n💾 数据已保存到 output_all_mols_{os.path.basename(chkpt)}.json")
