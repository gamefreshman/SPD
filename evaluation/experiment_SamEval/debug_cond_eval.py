"""
CondEval 调试脚本 v3：验证 bonds 参数修复效果。

关键对比：无 bonds vs 有 bonds → ConditionalEval 结果差异

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

from shepherd_score.container import Molecule
from shepherd_score.evaluations.evaluate import ConfEval, ConditionalEvalPipeline
from shepherd_score.evaluations.evaluate.evals import ConditionalEval


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


def load_first_n_samples(data, mol_idx=0, n=5):
    mol_key = f'molecule_{mol_idx}'
    if mol_key not in data:
        return []
    samples = []
    for n_atoms_key in data[mol_key].keys():
        for sample in data[mol_key][n_atoms_key].get('samples', []):
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
    # ===== 加载参考分子 =====
    p("步骤 1: 加载参考分子")
    with open(REF_MOL_PKL, 'rb') as f:
        molblocks_and_charges = pickle.load(f)
    mol = Chem.MolFromMolBlock(molblocks_and_charges[0][0], removeHs=False)
    ref_molec = Molecule(mol, num_surf_points=200, probe_radius=1.2, pharm_multi_vector=False)
    print(f"  ✅ 参考分子: {mol.GetNumAtoms()} 原子")

    # ===== 加载样本 =====
    p("步骤 2: 找到一个 ConfEval 有效的样本")
    with open(MODEL_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = load_first_n_samples(data, mol_idx=0, n=10)

    valid_sample = None
    for i, sample in enumerate(samples):
        atoms = sample['x1']['atoms']
        positions = sample['x1']['positions']
        bonds = sample['x1'].get('bonds', None)
        if isinstance(atoms, np.ndarray):
            atoms = atoms.flatten()
        try:
            ce = ConfEval(atoms, positions, solvent='water', bonds=bonds, num_processes=1)
            has_bonds = bonds is not None
            print(f"  样本 {i}: {len(atoms)} 原子, bonds={'✅' if has_bonds else '❌'}, "
                  f"is_valid={ce.is_valid}")
            if ce.is_valid and valid_sample is None:
                valid_sample = (i, sample)
        except Exception as e:
            print(f"  样本 {i}: 异常: {e}")

    if valid_sample is None:
        print("\n❌ 没有有效样本")
        return

    idx, sample = valid_sample
    atoms = sample['x1']['atoms']
    positions = sample['x1']['positions']
    bonds = sample['x1'].get('bonds', None)
    if isinstance(atoms, np.ndarray):
        atoms = atoms.flatten()
    if isinstance(positions, list):
        positions = np.array(positions)
    if bonds is not None and isinstance(bonds, list):
        bonds = np.array(bonds)

    print(f"\n  ✅ 使用样本 {idx}: {len(atoms)} 原子, bonds={bonds is not None}")
    if bonds is not None:
        print(f"     bonds 形状: {bonds.shape}")

    # ===== 核心对比：无 bonds vs 有 bonds =====
    p("步骤 3: ConditionalEval 对比（无 bonds vs 有 bonds）")

    for label, use_bonds in [("无 bonds", None), ("有 bonds", bonds)]:
        print(f"\n  --- {label} ---")
        try:
            cond = ConditionalEval(
                ref_molec=ref_molec,
                atoms=atoms,
                positions=positions,
                condition='all',
                num_surf_points=200,
                pharm_multi_vector=False,
                solvent='water',
                num_processes=1,
                bonds=use_bonds
            )
            print(f"  is_valid: {cond.is_valid}")
            print(f"  is_valid_post_opt: {cond.is_valid_post_opt}")
            print(f"  smiles: {cond.smiles}")
            print(f"  sim_surf_target: {cond.sim_surf_target}")
            print(f"  sim_esp_target: {cond.sim_esp_target}")
            print(f"  sim_pharm_target: {cond.sim_pharm_target}")
            print(f"  sim_surf_target_relax_optimal: {cond.sim_surf_target_relax_optimal}")
            print(f"  sim_esp_target_relax_optimal: {cond.sim_esp_target_relax_optimal}")

            if cond.sim_surf_target is not None:
                print(f"  ✅ 相似度计算成功！")
            else:
                print(f"  ❌ 相似度为 None")
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            traceback.print_exc()

    # ===== Pipeline 对比 =====
    p("步骤 4: ConditionalEvalPipeline 对比")

    for label, gen_mol in [
        ("无 bonds (2元组)", (atoms, positions)),
        ("有 bonds (3元组)", (atoms, positions, bonds)),
    ]:
        if bonds is None and "有" in label:
            print(f"\n  --- {label}: ⚠️ 样本无 bonds，跳过 ---")
            continue
        print(f"\n  --- {label} ---")
        try:
            pipe = ConditionalEvalPipeline(
                ref_molec,
                generated_mols=[gen_mol],
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
            if non_nan > 0:
                print(f"  ✅ 成功获得相似度分数！")
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            traceback.print_exc()

    p("调试完成")
    print("  预期: 无 bonds → is_valid=False, 有 bonds → is_valid=True + 有效 sims")


if __name__ == '__main__':
    main()
