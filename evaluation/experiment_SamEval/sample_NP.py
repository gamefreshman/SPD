# 核心库导入
import os
import json
import pickle
import multiprocessing
import gc
from datetime import datetime
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

# 科学计算库
import numpy as np
import torch
import rdkit
from tqdm import tqdm

# 设置 multiprocessing 启动方式为 spawn（CUDA 要求）
# 必须在导入 torch 后、初始化 CUDA 前设置
multiprocessing.set_start_method('spawn', force=True)

# Shepherd相关模块（仅用于获取配置，不加载模型）
from shepherd.lightning_module import LightningModule
from shepherd.shepherd_score_utils.conformer_generation import update_mol_coordinates
from shepherd.shepherd_score_utils.generate_point_cloud import (
    get_atomic_vdw_radii, 
    get_molecular_surface,
    get_electrostatics_given_point_charges,
)
from shepherd.shepherd_score_utils.pharm_utils.pharmacophore import get_pharmacophores

# 检测可用GPU数量（不初始化 CUDA 上下文）
num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"🖥️  检测到 {num_gpus} 张GPU")

if num_gpus > 0:
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {gpu_name}, {gpu_total_memory:.1f} GB")
else:
    print("⚠️  未检测到GPU，将使用CPU")

# 统一的batch size配置
BATCH_SIZE = 2

# 模型 checkpoint 路径
# chkpt = '/home1/zhh/workspace/SPD/training/jobs/33/x1x3x4_dpo_finetune_nps/last.ckpt'
chkpt = '/home1/zhh/workspace/SPD/training/jobs/33/x1x3x4_dpo_finetune_nps/epoch-epoch=009.ckpt'
CONDITION_SOURCE_FILE = '/home1/zhh/workspace/SPD/data/conformers/np/molblock_charges_NPs.pkl'
CONDITION_SEMANTICS = 'x2_x3_fixed__x4_noised_condition__optimize_x1'
CONDITION_FLAG_SUMMARY = {
    'inpaint_x2_pos': False,
    'inpaint_x3_pos': False,
    'inpaint_x3_x': False,
    'inpaint_x4_pos': True,
    'inpaint_x4_direction': True,
    'inpaint_x4_type': True,
    'stop_inpainting_at_time_x2': 0.0,
    'add_noise_to_inpainted_x2_pos': 0.0,
    'stop_inpainting_at_time_x3': 0.0,
    'add_noise_to_inpainted_x3_pos': 0.0,
    'add_noise_to_inpainted_x3_x': 0.0,
    'stop_inpainting_at_time_x4': 0.0,
    'add_noise_to_inpainted_x4_pos': 0.0,
    'add_noise_to_inpainted_x4_direction': 0.0,
    'add_noise_to_inpainted_x4_type': 0.0,
}


def assert_condition_semantics():
    assert CONDITION_FLAG_SUMMARY['inpaint_x2_pos'] is False
    assert CONDITION_FLAG_SUMMARY['inpaint_x3_pos'] is False
    assert CONDITION_FLAG_SUMMARY['inpaint_x3_x'] is False
    assert CONDITION_FLAG_SUMMARY['inpaint_x4_pos'] is True
    assert CONDITION_FLAG_SUMMARY['inpaint_x4_direction'] is True
    assert CONDITION_FLAG_SUMMARY['inpaint_x4_type'] is True
    print(
        f"🧭 采样语义: {CONDITION_SEMANTICS} | "
        "x2/x3 固定条件, x4 带噪条件, 目标仅生成 x1"
    )


def write_sidecar_metadata(sample_json_path):
    sidecar_path = f"{sample_json_path}.meta.json"
    payload = {
        'sample_json_path': os.path.abspath(sample_json_path),
        'checkpoint_path': chkpt,
        'source_script': os.path.abspath(__file__),
        'condition_semantics': CONDITION_SEMANTICS,
        'sampling_flags': CONDITION_FLAG_SUMMARY,
        'condition_source_file': CONDITION_SOURCE_FILE,
        'written_at': datetime.now().isoformat(timespec='seconds'),
    }
    with open(sidecar_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"📝 sidecar metadata 已保存到: {sidecar_path}")

# 仅加载配置（不加载模型到 GPU，避免 fork 问题）
print("加载模型配置...")
model_pl_temp = LightningModule.load_from_checkpoint(chkpt, map_location='cpu')
params = model_pl_temp.params
del model_pl_temp
gc.collect()

assert_condition_semantics()

with open(CONDITION_SOURCE_FILE, 'rb') as f:
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
# 原子数量列表：25个不同的原子数量
N_ATOMS_LIST = [36, 40, 44, 48, 49, 50, 51, 52, 56, 60, 
                64, 68, 70, 72, 76, 77, 78, 79, 80, 81, 
                83, 84, 85, 86, 87]
SAMPLES_PER_N_ATOMS = 60  # 每个原子数量生成的分子数 (25×60=1500/mol, 3mol×1500=4500 total)


# ==================== 断点续传函数 ====================
def count_existing_samples(molecule_index, n_atoms):
    """
    检测增量保存文件中已有的样本数量，用于断点续传。
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
                    samples.append(sample_serializable)
    except Exception as e:
        print(f"警告: 读取增量文件 {json_path} 时出错: {e}")
        return []
    
    return samples

# ==================== GPU Worker 函数（多进程模式） ====================
def run_sampling_on_gpu(gpu_id, tasks, molecule_index, mol_features_serializable, 
                        params_dict, batch_size, atom_marginals_list, bond_marginals_list):
    """
    在指定GPU上运行采样任务（多进程 worker）
    
    Args:
        gpu_id: GPU 编号
        tasks: list of (n_atoms, num_batches, target_samples) 元组
        molecule_index: 分子索引
        mol_features_serializable: 可序列化的分子特征
        params_dict: 模型参数
        batch_size: 批次大小
        atom_marginals_list: 原子边际分布 (list)
        bond_marginals_list: 键边际分布 (list)
    
    Returns:
        dict: {n_atoms: [samples]}
    """
    import torch
    import numpy as np
    import json
    import os
    import gc
    from shepherd.lightning_module import LightningModule
    from shepherd.inference import inference_sample
    
    # 确保增量保存目录存在
    os.makedirs('data/incremental', exist_ok=True)
    
    # 设置当前进程使用的GPU
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    print(f"GPU {gpu_id}: 加载模型...")
    model_pl = LightningModule.load_from_checkpoint(chkpt)
    model_pl.to(device)
    model_pl.model.device = device
    model_pl.eval()
    
    # 转换边际分布为 tensor
    atom_marginals = torch.tensor(atom_marginals_list, dtype=torch.float).to(device)
    bond_marginals = torch.tensor(bond_marginals_list, dtype=torch.float).to(device)
    
    # 恢复分子特征
    mol_features = {
        'surface': np.array(mol_features_serializable['surface']),
        'electrostatics': np.array(mol_features_serializable['electrostatics']),
        'pharm_types': np.array(mol_features_serializable['pharm_types']),
        'pharm_pos': np.array(mol_features_serializable['pharm_pos']),
        'pharm_direction': np.array(mol_features_serializable['pharm_direction']),
        'num_pharmacophores': mol_features_serializable['num_pharmacophores'],
    }
    
    # 按n_atoms分组的结果
    all_task_results = {}
    
    # 处理分配给此GPU的所有任务
    for task_idx, (n_atoms, num_batches, target_samples) in enumerate(tasks):
        # 断点续传：检测已完成的样本数
        existing_count = count_existing_samples(molecule_index, n_atoms)
        remaining_samples = max(0, target_samples - existing_count)
        
        if remaining_samples == 0:
            print(f"GPU {gpu_id}: ⏭️ n_atoms={n_atoms} 已完成 ({existing_count}/{target_samples})，跳过")
            all_task_results[n_atoms] = []
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
                    N_x4=mol_features['num_pharmacophores'],
                    unconditional=False,
                    prior_noise_scale=1.0,
                    denoising_noise_scale=1.0,
                    inject_noise_at_ts=[],
                    inject_noise_scales=[],
                    harmonize=False,
                    harmonize_ts=[],
                    harmonize_jumps=[],
                    **CONDITION_FLAG_SUMMARY,
                    center_of_mass=np.zeros(3),
                    surface=mol_features['surface'],
                    electrostatics=mol_features['electrostatics'],
                    pharm_types=mol_features['pharm_types'],
                    pharm_pos=mol_features['pharm_pos'],
                    pharm_direction=mol_features['pharm_direction'],
                    atom_marginals=atom_marginals,
                    bond_marginals=bond_marginals,
                )
            
            # 转换样本为可序列化格式
            batch_samples = []
            for sample in generated_samples:
                sample_serializable = {
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
                batch_samples.append(sample_serializable)
                task_results.append(sample_serializable)
            
            # 每个batch完成后立即保存（增量保存）
            json_path = f'data/incremental/mol{molecule_index}_n{n_atoms}_samples.jsonl'
            with open(json_path, 'a') as f:
                for sample in batch_samples:
                    f.write(json.dumps(sample) + '\n')
            
            print(f"GPU {gpu_id}: n_atoms={n_atoms}, 批次 {current_batch_num}/{remaining_batches} 已保存")
            
            del generated_samples, batch_samples
            gc.collect()
            torch.cuda.empty_cache()
        
        all_task_results[n_atoms] = task_results
        print(f"GPU {gpu_id}: n_atoms={n_atoms} 完成，生成 {len(task_results)} 个样本")
    
    total_samples = sum(len(v) for v in all_task_results.values())
    print(f"GPU {gpu_id}: 完成所有任务，共生成 {total_samples} 个样本")
    return all_task_results

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


# ==================== 主程序入口 ====================
if __name__ == '__main__':
    # ==================== 计算边际分布（与dpo_trainer.py保持一致） ====================
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
    
    if len(atom_marginals_x1) > 0:
        print(f"🔧 修正前 atom_marginals[0] = {atom_marginals_x1[0]:.6f}")
        atom_marginals_x1[0] = 0.0
        atom_marginals_x1 = atom_marginals_x1 / atom_marginals_x1.sum()
        print(f"🔧 修正后 atom_marginals[0] = {atom_marginals_x1[0]:.6f}")

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

    # Step 2: 使用多GPU并行执行模型推理（多进程模式）
    print(f"\n{'='*100}")
    print("🔬 Step 2: 使用多GPU并行推理生成样本...")
    print(f"  使用 {num_gpus} 张GPU进行并行推理（多进程模式）")
    
    # 转换边际分布为 list（用于跨进程传输）
    atom_marginals_list = atom_marginals_x1.tolist()
    bond_marginals_list = bond_marginals_x1.tolist()
    
    print(f"  📋 采样配置:")
    print(f"    - 原子数量列表: {N_ATOMS_LIST} (共{len(N_ATOMS_LIST)}种)")
    print(f"    - 每种原子数量生成: {SAMPLES_PER_N_ATOMS} 个样本")
    print(f"    - Batch size: {BATCH_SIZE}")
    print(f"    - 每个分子总计: {len(N_ATOMS_LIST) * SAMPLES_PER_N_ATOMS} 个样本")
    print(f"{'='*100}\n")
    
    # 结果存储
    all_results = {}
    
    # 确保目录存在
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/incremental', exist_ok=True)
    
    # 计算每个n_atoms需要的批次数
    num_batches_per_n_atoms = (SAMPLES_PER_N_ATOMS + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"每个n_atoms需要 {num_batches_per_n_atoms} 个批次 (每批 {BATCH_SIZE} 个样本)")
    
    # 对每个分子循环处理
    for mol_index in sorted(mol_features_dict.keys()):
        mol_features = mol_features_dict[mol_index]
        if mol_features is None:
            print(f"  ⚠️  分子 {mol_index} 无有效特征，跳过")
            continue
        
        print(f"\n{'='*100}")
        print(f"处理分子 {mol_index + 1}/{len(molblocks_and_charges)} (index={mol_index})")
        print(f"{'='*100}\n")
        
        # 将分子特征转换为可序列化格式（用于跨进程传输）
        mol_features_serializable = {
            'surface': mol_features['surface'].tolist() if isinstance(mol_features['surface'], np.ndarray) else mol_features['surface'],
            'electrostatics': mol_features['electrostatics'].tolist() if isinstance(mol_features['electrostatics'], np.ndarray) else mol_features['electrostatics'],
            'pharm_types': mol_features['pharm_types'].tolist() if isinstance(mol_features['pharm_types'], np.ndarray) else mol_features['pharm_types'],
            'pharm_pos': mol_features['pharm_pos'].tolist() if isinstance(mol_features['pharm_pos'], np.ndarray) else mol_features['pharm_pos'],
            'pharm_direction': mol_features['pharm_direction'].tolist() if isinstance(mol_features['pharm_direction'], np.ndarray) else mol_features['pharm_direction'],
            'num_pharmacophores': mol_features['num_pharmacophores'],
        }
        
        # 构建任务列表：每个n_atoms是一个任务
        # 任务格式: (n_atoms, num_batches, target_samples)
        all_tasks = [(n_atoms, num_batches_per_n_atoms, SAMPLES_PER_N_ATOMS) for n_atoms in N_ATOMS_LIST]
        
        # 将任务均匀分配到GPU（交替分配 Round-Robin 实现负载均衡）
        tasks_per_gpu = [[] for _ in range(max(num_gpus, 1))]
        for i, task in enumerate(all_tasks):
            gpu_id = i % max(num_gpus, 1)
            tasks_per_gpu[gpu_id].append(task)
        
        print(f"任务分配:")
        for gpu_id, tasks in enumerate(tasks_per_gpu):
            n_atoms_list = [t[0] for t in tasks]
            total_batches = sum(t[1] for t in tasks)
            print(f"  GPU {gpu_id}: {len(tasks)} 个n_atoms任务, 共 {total_batches} 批次")
            print(f"           n_atoms: {n_atoms_list}")
        
        # 使用多进程并行执行
        results_by_n_atoms = {n_atoms: [] for n_atoms in N_ATOMS_LIST}
        
        if num_gpus > 0:
            with ProcessPoolExecutor(max_workers=num_gpus) as executor:
                futures = []
                for gpu_id, task_list in enumerate(tasks_per_gpu):
                    if task_list:
                        future = executor.submit(
                            run_sampling_on_gpu,
                            gpu_id, task_list, mol_index, mol_features_serializable,
                            params, BATCH_SIZE, atom_marginals_list, bond_marginals_list
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
        else:
            # 单GPU/CPU处理
            print("  使用单进程处理...")
            gpu_results = run_sampling_on_gpu(
                0, all_tasks, mol_index, mol_features_serializable,
                params, BATCH_SIZE, atom_marginals_list, bond_marginals_list
            )
            for n_atoms, samples in gpu_results.items():
                results_by_n_atoms[n_atoms].extend(samples)
        
        # 从增量文件中加载已保存的样本（用于断点续传场景）
        print(f"\n从增量文件加载已保存的样本...")
        for n_atoms in N_ATOMS_LIST:
            if len(results_by_n_atoms[n_atoms]) == 0:
                loaded_samples = load_samples_from_incremental(mol_index, n_atoms)
                if loaded_samples:
                    results_by_n_atoms[n_atoms] = loaded_samples
                    print(f"  n_atoms={n_atoms}: 从增量文件加载了 {len(loaded_samples)} 个样本")
        
        # 统计结果
        total_samples = sum(len(v) for v in results_by_n_atoms.values())
        print(f"\n所有GPU完成采样，共 {total_samples} 个样本")
        
        # 保存结果
        molecule_results = {}
        for n_atoms in N_ATOMS_LIST:
            samples = results_by_n_atoms[n_atoms]
            samples = samples[:SAMPLES_PER_N_ATOMS]  # 截取到目标数量
            
            molecule_results[f'n_atoms_{n_atoms}'] = {
                'n_atoms': n_atoms,
                'num_samples': len(samples),
                'samples': samples
            }
            print(f"  n_atoms={n_atoms}: {len(samples)} 个样本")
        
        # 保存当前分子的JSON
        all_results[mol_index] = molecule_results
        
        mol_output_file = f'data/molecule_{mol_index}_all_samples.json'
        with open(mol_output_file, 'w') as f:
            json.dump(molecule_results, f, indent=2)
        print(f"  JSON已保存到: {mol_output_file}")
        
        print(f"\n分子 {mol_index} 完成！共生成 {total_samples} 个样本\n")
        
        gc.collect()
    
    # 保存汇总JSON
    all_results_json = {}
    for mol_index in sorted(all_results.keys()):
        all_results_json[f'molecule_{mol_index}'] = all_results[mol_index]
    
    output_file = f'data/generated_samples_all_molecules_{os.path.basename(chkpt)}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results_json, f, indent=2)
    write_sidecar_metadata(output_file)
    
    # 统计总数
    total_all = 0
    for mol_key, mol_data in all_results_json.items():
        for n_key, n_data in mol_data.items():
            total_all += n_data['num_samples']
    
    print(f"\n{'='*100}")
    print(f"所有采样完成！")
    print(f"总共生成了 {total_all} 个样本")
    print(f"  {len(molblocks_and_charges)} 个分子 × {len(N_ATOMS_LIST)} 种原子数量 × {SAMPLES_PER_N_ATOMS} 个样本")
    print(f"结果已保存到: {output_file}")
    print(f"{'='*100}")
