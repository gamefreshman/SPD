"""
CondEval 调试脚本 v2：验证使用原始样本数据（含氢）传入 CondEval 的修复效果。

运行方式: python debug_cond_eval.py
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
from shepherd_score.evaluations.evaluate import ConfEval, ConditionalEvalPipeline
from shepherd_score.evaluations.evaluate.evals import ConditionalEval

# ========== 路径配置 ==========
MODEL_FILE = '/home1/zhh/workspace/SPD/evaluation/core_data/data/1/DIS/33/generated_samples_all_molecules_last_33epoch.ckpt.json'
REF_MOL_PKL = '/home1/zhh/workspace/SPD/data/conformers/np/molblock_charges_NPs.pkl'


def convert_sample_format(sample):
    modal_keys = ['x1', 'x2', 'x3', 'x4']
    for modal_key in modal_keys:
        if modal_key in sample and isinstance(sample[modal_key], dict):
            for data_key in sample[modal_key]:
                if isinstance(sample[modal_key][data_key], list):
                    sample[modal_key][data_key] = np.array(sample[modal_key][data_key])
    return sample


def load_first_n_samples(data, mol_idx=0, n=3):
    mol_key = f'molecule_{mol_idx}'
    if mol_key not in data:
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


def p(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    # ===== 步骤 1: 加载参考分子 =====
    p("步骤 1: 加载参考分子")
    with open(REF_MOL_PKL, 'rb') as f:
        molblocks_and_charges = pickle.load(f)
    mol = Chem.MolFromMolBlock(molblocks_and_charges[0][0], removeHs=False)
    ref_molec = Molecule(mol, num_surf_points=200, probe_radius=1.2, pharm_multi_vector=False)
    print(f"  ✅ 参考分子 0: {mol.GetNumAtoms()} 原子")
    print(f"     surf_pos: {ref_molec.surf_pos is not None}, surf_esp: {ref_molec.surf_esp is not None}")
    print(f"     pharm_ancs: {ref_molec.pharm_ancs is not None}, pharm_types: {ref_molec.pharm_types is not None}")

    # ===== 步骤 2: 加载样本 =====
    p("步骤 2: 加载用样本并 ConfEval")
    with open(MODEL_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = load_first_n_samples(data, mol_idx=0, n=5)
    print(f"  加载了 {len(samples)} 个样本")

    # 找一个通过 ConfEval 的
    valid_sample = None
    for i, sample in enumerate(samples):
        atoms = sample['x1']['atoms']
        positions = sample['x1']['positions']
        bonds = sample['x1'].get('bonds', None)
        if isinstance(atoms, np.ndarray):
            atoms = atoms.flatten()
        try:
            ce = ConfEval(atoms, positions, solvent='water', bonds=bonds, num_processes=1)
            print(f"  样本 {i}: {len(atoms)} 原子, is_valid={ce.is_valid}, smiles={ce.smiles}")
            if ce.is_valid and valid_sample is None:
                valid_sample = (i, sample)
        except Exception as e:
            print(f"  样本 {i}: ConfEval 异常: {e}")

    if valid_sample is None:
        print("\n❌ 没有通过 ConfEval 的样本")
        return

    idx, sample = valid_sample
    print(f"\n  ✅ 使用样本 {idx} 进行后续测试")

    # ===== 步骤 3: 对比两种输入方式 =====
    p("步骤 3: 对比 create_rdkit_molecule (去氢) vs 原始数据 (含氢)")

    # 方式 A: create_rdkit_molecule（去氢 — 旧方式，有 BUG）
    print("\n  --- 方式 A: create_rdkit_molecule (去氢) ---")
    rdkit_mol = create_rdkit_molecule(sample)
    if rdkit_mol is not None:
        atoms_a = np.array([a.GetAtomicNum() for a in rdkit_mol.GetAtoms()])
        positions_a = rdkit_mol.GetConformer().GetPositions()
        print(f"  原子数: {len(atoms_a)} (去氢)")
    else:
        atoms_a, positions_a = None, None
        print(f"  ❌ create_rdkit_molecule 返回 None")

    # 方式 B: 原始样本数据（含氢 — 修复后）
    print("\n  --- 方式 B: 原始样本数据 (含氢) ---")
    atoms_b = sample['x1']['atoms']
    positions_b = sample['x1']['positions']
    if isinstance(atoms_b, np.ndarray):
        atoms_b = atoms_b.flatten()
    if isinstance(positions_b, list):
        positions_b = np.array(positions_b)
    print(f"  原子数: {len(atoms_b)} (含氢)")

    # ===== 步骤 4: 对比 ConditionalEval =====
    p("步骤 4: 对比 ConditionalEval")

    for label, atoms, positions in [
        ("方式A (去氢)", atoms_a, positions_a),
        ("方式B (含氢)", atoms_b, positions_b),
    ]:
        print(f"\n  --- {label}: {len(atoms) if atoms is not None else 'N/A'} 原子 ---")
        if atoms is None:
            print(f"  ⚠️ 跳过（无数据）")
            continue
        try:
            cond = ConditionalEval(
                ref_molec=ref_molec,
                atoms=atoms,
                positions=positions,
                condition='all',
                num_surf_points=200,
                pharm_multi_vector=False,
                solvent='water',
                num_processes=1
            )
            print(f"  is_valid: {cond.is_valid}")
            print(f"  is_valid_post_opt: {cond.is_valid_post_opt}")
            print(f"  smiles: {cond.smiles}")
            print(f"  sim_surf_target: {cond.sim_surf_target}")
            print(f"  sim_esp_target: {cond.sim_esp_target}")
            print(f"  sim_pharm_target: {cond.sim_pharm_target}")
            print(f"  sim_surf_target_relax: {cond.sim_surf_target_relax}")
            print(f"  sim_surf_target_relax_optimal: {cond.sim_surf_target_relax_optimal}")
            print(f"  sim_esp_target_relax_optimal: {cond.sim_esp_target_relax_optimal}")

            if cond.sim_surf_target is not None:
                print(f"  ✅ 相似度计算成功！")
            else:
                print(f"  ❌ 相似度为 None")
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            traceback.print_exc()

    # ===== 步骤 5: Pipeline 对比 =====
    p("步骤 5: ConditionalEvalPipeline 对比")

    for label, atoms, positions in [
        ("方式A (去氢)", atoms_a, positions_a),
        ("方式B (含氢)", atoms_b, positions_b),
    ]:
        print(f"\n  --- {label} ---")
        if atoms is None:
            print(f"  ⚠️ 跳过")
            continue
        try:
            pipe = ConditionalEvalPipeline(
                ref_molec,
                generated_mols=[(atoms, positions)],
                condition='all',
                num_surf_points=200,
                pharm_multi_vector=False,
                solvent='water'
            )
            pipe.evaluate(num_workers=1, num_processes=1, verbose=True)
            print(f"  num_valid: {pipe.num_valid}/{pipe.num_generated_mols}")
            print(f"  sims_surf_target: {pipe.sims_surf_target}")
            print(f"  sims_esp_target: {pipe.sims_esp_target}")
            print(f"  sims_surf_target_relax_optimal: {pipe.sims_surf_target_relax_optimal}")
            non_nan = sum(1 for v in pipe.sims_surf_target if not np.isnan(v))
            print(f"  非NaN数: {non_nan}/{len(pipe.sims_surf_target)}")
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            traceback.print_exc()

    p("调试完成")
    print("  预期结果：方式A (去氢) 全部 NaN，方式B (含氢) 有实际值")
    print("  请将输出发给我确认修复效果。")


if __name__ == '__main__':
    main()
