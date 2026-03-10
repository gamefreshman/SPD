#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在子进程中执行分子对接，仅将分数打印到 stdout。
用于隔离 meeko/vina 的 C++ 异常（如 internal_error），避免主进程被 Aborted 拖垮。
用法: python run_docking_subprocess.py <protein_pdbqt_path> <ref_smiles> <lig_smiles> [vina_dock_module_path] [score_out_file]
ref_smiles 可空，此时用 lig_smiles 作为参考。
若提供 score_out_file，将分数写入该文件（仅一行浮点数），避免 stdout 被库输出污染。
"""
from __future__ import annotations

import os
import sys


def main() -> None:
    if len(sys.argv) < 4:
        print("Usage: run_docking_subprocess.py <protein_pdbqt> <ref_smiles> <lig_smiles> [vina_dock_path] [score_out_file]", file=sys.stderr)
        sys.exit(1)
    protein_path = sys.argv[1]
    ref_smi = (sys.argv[2] or "").strip()
    lig_smi = (sys.argv[3] or "").strip()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    vina_module_path = sys.argv[4] if len(sys.argv) > 4 else os.path.join(project_root, "vina_dock.py")
    score_out_file = sys.argv[5] if len(sys.argv) > 5 else None

    if not os.path.isfile(protein_path):
        print("Protein file not found.", file=sys.stderr)
        sys.exit(1)
    if not lig_smi:
        print("Ligand SMILES is empty.", file=sys.stderr)
        sys.exit(1)

    import importlib.util
    from rdkit import Chem
    from rdkit.Chem import AllChem

    spec = importlib.util.spec_from_file_location("vina_dock", vina_module_path)
    if spec is None or spec.loader is None:
        print("Failed to load vina_dock module.", file=sys.stderr)
        sys.exit(1)
    vina_dock = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vina_dock)

    lig_mol = Chem.MolFromSmiles(lig_smi)
    if not lig_mol:
        print("Invalid ligand SMILES.", file=sys.stderr)
        sys.exit(1)
    lig_mol = Chem.AddHs(lig_mol)
    AllChem.EmbedMolecule(lig_mol, AllChem.ETKDG())

    ref_mol = None
    if ref_smi:
        ref_mol = Chem.MolFromSmiles(ref_smi)
        if ref_mol:
            ref_mol = Chem.AddHs(ref_mol)
            AllChem.EmbedMolecule(ref_mol, AllChem.ETKDG())
    if ref_mol is None:
        ref_mol = lig_mol

    score = vina_dock.vina_score(lig_mol, protein_path, ref_mol)
    score_float = float(score)
    if score_out_file:
        with open(score_out_file, "w") as f:
            f.write(str(score_float))
    else:
        print(score_float, flush=True)


if __name__ == "__main__":
    main()
