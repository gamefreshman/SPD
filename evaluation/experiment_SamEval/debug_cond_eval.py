"""
CondEval 调试脚本：用 1 个样本逐步诊断 ConditionalEval 的每个阶段。

运行方式: python debug_cond_eval.py

诊断步骤：
  1. 加载参考分子
  2. 加载 1 个样本，运行独立 ConfEval
  3. 逐步运行 ConditionalEval.__init__ 的每个阶段
  4. 对比 CondEval Pipeline 结果
"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "0,1,2"

import json
import pickle
import traceback
import numpy as np
import rdkit
from rdkit import Chem

from lightning_fabric.utilities.seed import seed_everything
seed_everything(0)

from shepherd.extract import create_rdkit_molecule

from shepherd_score.container import Molecule, MoleculePair
from shepherd_score.conformer_generation import embed_conformer_from_smiles
from shepherd_score.evaluations.evaluate import (
    ConfEval,
    ConditionalEvalPipeline,
)
from shepherd_score.evaluations.evaluate.evals import ConditionalEval

# ========== 路径配置 ==========
MODEL_FILE = '/home1/zhh/workspace/SPD/evaluation/core_data/data/1/DIS/33/generated_samples_all_molecules_last_33epoch.ckpt.json'
REF_MOL_PKL = '/home1/zhh/workspace/SPD/data/conformers/np/molblock_charges_NPs.pkl'

# ========== 工具函数 ==========
def convert_sample_format(sample):
    modal_keys = ['x1', 'x2', 'x3', 'x4']
    for modal_key in modal_keys:
        if modal_key in sample and isinstance(sample[modal_key], dict):
            for data_key in sample[modal_key]:
                if isinstance(sample[modal_key][data_key], list):
                    sample[modal_key][data_key] = np.array(sample[modal_key][data_key])
    return sample


def load_first_n_samples(data, mol_idx=0, n=3):
    """从指定分子组中加载前 n 个样本"""
    mol_key = f'molecule_{mol_idx}'
    if mol_key not in data:
        print(f"❌ 找不到 {mol_key}")
        return []
    
    mol_data = data[mol_key]
    samples = []
    for n_atoms_key in mol_data.keys():
        for sample in mol_data[n_atoms_key].get('samples', []):
            sample = convert_sample_format(sample)
            sample['ref_mol_index'] = mol_idx
            samples.append(sample)
            if len(samples) >= n:
                return samples
    return samples


def print_separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ========== 主调试流程 ==========
def main():
    # ===== 步骤 1: 加载参考分子 =====
    print_separator("步骤 1: 加载参考分子")
    
    with open(REF_MOL_PKL, 'rb') as f:
        molblocks_and_charges = pickle.load(f)
    
    mol_idx = 0
    mol = Chem.MolFromMolBlock(molblocks_and_charges[mol_idx][0], removeHs=False)
    print(f"  参考分子 {mol_idx}: {mol.GetNumAtoms()} 个原子")
    
    ref_molec = Molecule(
        mol,
        num_surf_points=200,
        probe_radius=1.2,
        pharm_multi_vector=False
    )
    print(f"  ✅ Molecule 对象创建成功")
    print(f"     surf_pos: {'✅' if ref_molec.surf_pos is not None else '❌ None'} "
          f"(shape={ref_molec.surf_pos.shape if ref_molec.surf_pos is not None else 'N/A'})")
    print(f"     surf_esp: {'✅' if ref_molec.surf_esp is not None else '❌ None'}")
    print(f"     pharm_ancs: {'✅' if ref_molec.pharm_ancs is not None else '❌ None'}")
    print(f"     pharm_types: {'✅' if ref_molec.pharm_types is not None else '❌ None'}")
    
    # ===== 步骤 2: 加载样本 =====
    print_separator("步骤 2: 加载样本")
    
    with open(MODEL_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = load_first_n_samples(data, mol_idx=0, n=3)
    print(f"  加载了 {len(samples)} 个样本")
    
    # ===== 步骤 3: 独立 ConfEval =====
    print_separator("步骤 3: 独立 ConfEval（验证样本结构有效性）")
    
    valid_samples = []
    for i, sample in enumerate(samples):
        atoms = sample['x1']['atoms']
        positions = sample['x1']['positions']
        bonds = sample['x1'].get('bonds', None)
        
        if isinstance(atoms, np.ndarray):
            atoms = atoms.flatten()
        
        print(f"\n  --- 样本 {i} ---")
        print(f"  原子数: {len(atoms)}, 坐标形状: {positions.shape}")
        
        try:
            conf_eval = ConfEval(atoms, positions, solvent='water', bonds=bonds, num_processes=1)
            print(f"  is_valid: {conf_eval.is_valid}")
            print(f"  is_valid_post_opt: {conf_eval.is_valid_post_opt}")
            print(f"  smiles: {conf_eval.smiles}")
            print(f"  strain_energy: {conf_eval.strain_energy}")
            
            if conf_eval.is_valid:
                valid_samples.append((i, sample))
                print(f"  ✅ ConfEval 通过")
            else:
                print(f"  ❌ ConfEval 未通过")
        except Exception as e:
            print(f"  ❌ ConfEval 异常: {e}")
            traceback.print_exc()
    
    if len(valid_samples) == 0:
        print("\n❌ 没有通过 ConfEval 的样本，无法测试 CondEval")
        return
    
    print(f"\n✅ {len(valid_samples)} 个样本通过 ConfEval")
    
    # ===== 步骤 4: 手动 ConditionalEval（逐步诊断） =====
    print_separator("步骤 4: 手动逐步 ConditionalEval")
    
    sample_idx, sample = valid_samples[0]
    print(f"  使用样本 {sample_idx}")
    
    # 4a: 创建 RDKit 分子
    print(f"\n  --- 4a: 创建 RDKit 分子 ---")
    try:
        rdkit_mol = create_rdkit_molecule(sample)
        if rdkit_mol is None:
            print(f"  ❌ create_rdkit_molecule 返回 None")
            return
        atoms = np.array([a.GetAtomicNum() for a in rdkit_mol.GetAtoms()])
        positions = rdkit_mol.GetConformer().GetPositions()
        print(f"  ✅ RDKit 分子创建成功: {len(atoms)} 原子")
    except Exception as e:
        print(f"  ❌ 创建 RDKit 分子失败: {e}")
        traceback.print_exc()
        return
    
    # 4b: 测试不同 solvent 参数下的 ConfEval
    print(f"\n  --- 4b: 对比不同 solvent 下的 ConfEval ---")
    for solvent in [None, 'water']:
        print(f"\n  solvent={solvent!r}:")
        try:
            ce = ConfEval(atoms, positions, solvent=solvent, num_processes=1)
            print(f"    is_valid: {ce.is_valid}")
            print(f"    is_valid_post_opt: {ce.is_valid_post_opt}")
            print(f"    smiles: {ce.smiles}")
            if ce.is_valid:
                print(f"    strain_energy: {ce.strain_energy}")
                print(f"    QED: {ce.QED}")
        except Exception as e:
            print(f"    ❌ 异常: {e}")
            traceback.print_exc()
    
    # 4c: 直接实例化 ConditionalEval
    print(f"\n  --- 4c: 直接实例化 ConditionalEval ---")
    for solvent in [None, 'water']:
        print(f"\n  solvent={solvent!r}:")
        try:
            cond_eval = ConditionalEval(
                ref_molec=ref_molec,
                atoms=atoms,
                positions=positions,
                condition='all',
                num_surf_points=200,
                pharm_multi_vector=False,
                solvent=solvent,
                num_processes=1
            )
            print(f"    is_valid: {cond_eval.is_valid}")
            print(f"    is_valid_post_opt: {cond_eval.is_valid_post_opt}")
            print(f"    smiles: {cond_eval.smiles}")
            print(f"    sim_surf_target: {cond_eval.sim_surf_target}")
            print(f"    sim_esp_target: {cond_eval.sim_esp_target}")
            print(f"    sim_pharm_target: {cond_eval.sim_pharm_target}")
            print(f"    sim_surf_target_relax: {cond_eval.sim_surf_target_relax}")
            print(f"    sim_esp_target_relax: {cond_eval.sim_esp_target_relax}")
            print(f"    sim_surf_target_relax_optimal: {cond_eval.sim_surf_target_relax_optimal}")
            print(f"    sim_esp_target_relax_optimal: {cond_eval.sim_esp_target_relax_optimal}")
            print(f"    sim_pharm_target_relax_optimal: {cond_eval.sim_pharm_target_relax_optimal}")
            
            if cond_eval.sim_surf_target is not None:
                print(f"    ✅ 相似度计算成功！")
            else:
                print(f"    ❌ 相似度为 None（is_valid={cond_eval.is_valid}）")
                
        except Exception as e:
            print(f"    ❌ 异常: {e}")
            traceback.print_exc()
    
    # ===== 步骤 5: ConditionalEvalPipeline（1 个样本测试） =====
    print_separator("步骤 5: ConditionalEvalPipeline（1 个样本）")
    
    generated_mols = [(atoms, positions)]
    
    for solvent in [None, 'water']:
        print(f"\n  --- solvent={solvent!r} ---")
        try:
            pipe = ConditionalEvalPipeline(
                ref_molec,
                generated_mols=generated_mols,
                condition='all',
                num_surf_points=200,
                pharm_multi_vector=False,
                solvent=solvent
            )
            pipe.evaluate(num_workers=1, num_processes=1, verbose=True)
            
            print(f"    num_valid: {pipe.num_valid}/{pipe.num_generated_mols}")
            print(f"    num_valid_post_opt: {pipe.num_valid_post_opt}/{pipe.num_generated_mols}")
            print(f"    sims_surf_target: {pipe.sims_surf_target}")
            print(f"    sims_esp_target: {pipe.sims_esp_target}")
            print(f"    sims_pharm_target: {pipe.sims_pharm_target}")
            print(f"    sims_surf_target_relax: {pipe.sims_surf_target_relax}")
            print(f"    sims_surf_target_relax_optimal: {pipe.sims_surf_target_relax_optimal}")
            print(f"    rmsds: {pipe.rmsds}")
            
            # 检查是否有非 NaN 值
            non_nan = sum(1 for v in pipe.sims_surf_target if not np.isnan(v))
            print(f"    sims_surf_target 非NaN数: {non_nan}/{len(pipe.sims_surf_target)}")
            
        except Exception as e:
            print(f"    ❌ Pipeline 异常: {e}")
            traceback.print_exc()
    
    print_separator("调试完成")
    print("  请将以上输出发给我进行分析。")


if __name__ == '__main__':
    main()
