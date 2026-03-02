# -*- coding: utf-8 -*-
"""
Web 后端与数据工具：路径与设备、模型加载与推理、分子数据转换、检查点与实验列表、分子对接等。
"""

from __future__ import annotations

import io
import os
import sys
import glob
from typing import Any, Dict, List, Optional, Tuple

# 将项目根目录与 src 加入 Python 路径，以便导入 shepherd
_WEB_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_WEB_DIR)
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
_SCORE_DIR = os.path.join(_PROJECT_ROOT, "src", "score")
for _p in [_SCORE_DIR, _SRC_DIR, _PROJECT_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def get_project_root() -> str:
    """返回项目根目录的绝对路径（web 的上级目录）。"""
    return _PROJECT_ROOT


def get_device() -> str:
    """
    自动选择计算设备
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_gpu_memory_info() -> Optional[str]:
    """返回当前 GPU 显存简要信息"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"{allocated:.1f} / {total:.1f} GB"
    except Exception:
        return None


def get_gpu_memory_ratio() -> Optional[Tuple[float, float]]:
    """返回 (已用 GB, 总 GB)，用于状态栏进度条；无 GPU 返回 None。"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return (round(allocated, 2), round(total, 2))
    except Exception:
        return None


def get_sidebar_status_text(device: str, current_ckpt_path: Optional[str], n_mols: int) -> str:
    """组装侧边栏状态面板的简要文本（设备、显存、当前模型、会话分子数）。"""
    lines = [f"**运行设备**: {device.upper()}"]
    gpu_info = get_gpu_memory_info()
    if gpu_info:
        lines.append(f"**显存**: {gpu_info}")
    if current_ckpt_path:
        base = os.path.basename(current_ckpt_path)
        if len(base) > 24:
            base = base[:21] + "..."
        lines.append(f"**当前模型**: {base}")
    if n_mols > 0:
        lines.append(f"**会话分子数**: {n_mols}")
    return "  \n".join(lines)


def get_sidebar_status_html(device: str, current_ckpt_path: Optional[str], n_mols: int) -> str:
    """返回侧边栏状态面板 HTML：微型卡片、标签:数值左右对齐、显存进度条。"""
    rows: List[str] = []
    rows.append(f'<div class="status-row"><span class="label">运行设备</span><span>{device.upper()}</span></div>')
    gpu_ratio = get_gpu_memory_ratio()
    if gpu_ratio:
        used_gb, total_gb = gpu_ratio
        pct = min(100, int((used_gb / total_gb) * 100)) if total_gb > 0 else 0
        rows.append(f'<div class="status-row"><span class="label">显存</span><span>{used_gb:.1f} / {total_gb:.1f} GB</span></div>')
        rows.append(f'<div class="status-progress"><div class="status-progress-fill" style="width:{pct}%"></div></div>')
    if current_ckpt_path:
        base = os.path.basename(current_ckpt_path)
        if len(base) > 20:
            base = base[:17] + "..."
        rows.append(f'<div class="status-row"><span class="label">当前模型</span><span>{base}</span></div>')
    if n_mols > 0:
        rows.append(f'<div class="status-row"><span class="label">会话分子数</span><span>{n_mols}</span></div>')
    return f'<div class="sidebar-status-panel">{"".join(rows)}</div>'


def list_checkpoints() -> List[Tuple[str, str]]:
    """
    扫描项目内 checkpoint 目录，返回 (相对路径, 绝对路径) 列表。
    搜索目录：checkpoint/、data/shepherd_chkpts/、training/jobs/*/last.ckpt。
    Returns:
        List[Tuple[str, str]]: [(rel_path, abspath), ...]。
    """
    root = get_project_root()
    out: List[Tuple[str, str]] = []
    seen: set = set()

    ckpt_dir = os.path.join(root, "checkpoint")
    if os.path.isdir(ckpt_dir):
        for p in glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True):
            abspath = os.path.abspath(p)
            if abspath not in seen:
                seen.add(abspath)
                rel = os.path.relpath(p, root)
                out.append((rel, abspath))

    shepherd_ckpt = os.path.join(root, "data", "shepherd_chkpts")
    if os.path.isdir(shepherd_ckpt):
        for p in glob.glob(os.path.join(shepherd_ckpt, "**", "*.ckpt"), recursive=True):
            abspath = os.path.abspath(p)
            if abspath not in seen:
                seen.add(abspath)
                rel = os.path.relpath(p, root)
                out.append((rel, abspath))

    jobs_dir = os.path.join(root, "training", "jobs")
    if os.path.isdir(jobs_dir):
        for p in glob.glob(os.path.join(jobs_dir, "**", "last.ckpt"), recursive=True):
            abspath = os.path.abspath(p)
            if abspath not in seen:
                seen.add(abspath)
                rel = os.path.relpath(p, root)
                out.append((rel, abspath))

    return sorted(out, key=lambda x: x[0])


def checkpoint_to_readable_label(rel_path: str) -> str:
    """
    将检查点相对路径转为可读的模型名称。
    """
    if not rel_path:
        return "未命名模型"
    parts = rel_path.replace("\\", "/").split("/")
    name = parts[-2] if len(parts) >= 2 and parts[-1] == "last.ckpt" else parts[-1]
    name = name.replace(".ckpt", "").strip()
    if "dpo_finetune_nps" in name or "dpo_finetune_np" in name:
        return "DPO 微调 (天然产物)"
    if "dpo_fragment" in name:
        return "DPO 片段合并"
    if "dpo_finetune_pdb" in name:
        return "DPO 微调 (PDB)"
    if "diffusion" in name:
        return "扩散模型"
    return name or "模型"


def checkpoint_to_source_hint(rel_path: str) -> str:
    """
    从检查点相对路径提取简短来源提示，用于区分。
    """
    if not rel_path:
        return ""
    parts = rel_path.replace("\\", "/").strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "training" and parts[1] == "jobs":
        if len(parts) >= 4 and parts[-1] == "last.ckpt":
            return f"实验 {parts[2]} · {parts[3]}"
        if len(parts) >= 3:
            return f"实验 {parts[2]}"
        return "jobs"
    if len(parts) >= 1 and parts[0] == "data":
        if len(parts) >= 2 and "shepherd" in parts[1].lower():
            return f"预训练 · {parts[-1].replace('.ckpt', '')}" if len(parts) >= 3 else "预训练"
        return "data"
    if len(parts) >= 1 and parts[0] == "checkpoint":
        if len(parts) >= 2:
            return parts[1]
        return "checkpoint"
    return parts[0] if parts else ""


def build_unique_checkpoint_labels(checkpoints: List[Tuple[str, str]]) -> List[str]:
    from collections import Counter
    base_labels = [checkpoint_to_readable_label(rel) for rel, _ in checkpoints]
    hints = [checkpoint_to_source_hint(rel) for rel, _ in checkpoints]
    count = Counter(base_labels)
    out = []
    for i, base in enumerate(base_labels):
        if count[base] <= 1 or not hints[i]:
            out.append(base)
        else:
            out.append(f"{base} — {hints[i]}")
    out_count = Counter(out)
    seen: Dict[str, int] = {}
    for i in range(len(out)):
        if out_count[out[i]] > 1:
            seen[out[i]] = seen.get(out[i], 0) + 1
            out[i] = f"{out[i]} ({seen[out[i]]})"
    return out


def _cache_resource(f):
    """Streamlit 缓存装饰器：存在 st 时使用 st.cache_resource，否则不缓存。"""
    try:
        import streamlit as st
        return st.cache_resource(f)
    except Exception:
        return f


@_cache_resource
def load_model_pl(checkpoint_path: str, device: str) -> Any:
    """
    加载 Lightning 模型并移至指定设备（使用 Streamlit 缓存，避免重复加载）。
    """
    import torch
    from shepherd.lightning_module import LightningModule

    model_pl = LightningModule.load_from_checkpoint(checkpoint_path, map_location=device)
    model_pl.eval()
    model_pl.to(torch.device(device))
    model_pl.model.device = torch.device(device)
    return model_pl


def run_inference(
    model_pl: Any,
    batch_size: int,
    N_x1: int,
    N_x4: int,
    unconditional: bool,
    device: str,
    prior_noise_scale: float = 1.0,
    denoising_noise_scale: float = 1.0,
    harmonize: bool = False,
    harmonize_ts: Optional[List[int]] = None,
    harmonize_jumps: Optional[List[int]] = None,
    atom_marginals: Optional[Any] = None,
    bond_marginals: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    调用后端 inference_sample 进行采样；无条件生成时使用默认 inpainting 占位。
    """
    import numpy as np
    import torch
    from shepherd.inference import inference_sample

    params = model_pl.params
    x1 = params.get("dataset", {}).get("x1", {})
    num_atom_types = len(x1.get("atom_types", []))
    num_bond_types = len(x1.get("bond_types", []))
    if num_atom_types == 0:
        num_atom_types = 4
    if num_bond_types == 0:
        num_bond_types = 5
    if atom_marginals is None:
        atom_marginals = torch.ones(num_atom_types, dtype=torch.float32) / num_atom_types
    if bond_marginals is None:
        bond_marginals = torch.ones(num_bond_types, dtype=torch.float32) / num_bond_types

    n_pharm = N_x4
    pharm_types = np.zeros(n_pharm, dtype=int)
    pharm_pos = np.zeros((n_pharm, 3))
    pharm_direction = np.zeros((n_pharm, 3))

    samples = inference_sample(
        model_pl,
        batch_size=batch_size,
        N_x1=N_x1,
        N_x4=N_x4,
        unconditional=unconditional,
        prior_noise_scale=prior_noise_scale,
        denoising_noise_scale=denoising_noise_scale,
        inject_noise_at_ts=[],
        inject_noise_scales=[],
        harmonize=harmonize,
        harmonize_ts=harmonize_ts or [],
        harmonize_jumps=harmonize_jumps or [],
        inpaint_x2_pos=False,
        inpaint_x3_pos=False,
        inpaint_x3_x=False,
        inpaint_x4_pos=False,
        inpaint_x4_direction=False,
        inpaint_x4_type=False,
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
        surface=np.zeros((75, 3)),
        electrostatics=np.zeros(75),
        pharm_types=pharm_types,
        pharm_pos=pharm_pos,
        pharm_direction=pharm_direction,
        atom_marginals=atom_marginals,
        bond_marginals=bond_marginals,
    )
    return samples


def sample_to_rdkit_mol(sample: Dict[str, Any]) -> Optional[Any]:
    """
    将 inference 输出的单条 sample 转为 RDKit Mol。
    """
    try:
        from shepherd.extract_shepherd import create_rdkit_molecule
        return create_rdkit_molecule(sample)
    except Exception:
        return None


def mol_to_smiles(mol: Any) -> Optional[str]:
    """RDKit Mol 转 SMILES，失败返回 None。"""
    if mol is None:
        return None
    try:
        from rdkit import Chem
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def mol_to_xyz(mol: Any) -> Optional[str]:
    """RDKit Mol 转 XYZ 字符串（用于 3D 显示）。"""
    if mol is None:
        return None
    try:
        from rdkit import Chem
        conf = mol.GetConformer()
        n = mol.GetNumAtoms()
        lines = [str(n), ""]
        for i in range(n):
            a = mol.GetAtomWithIdx(i)
            p = conf.GetAtomPosition(i)
            lines.append(f"{a.GetSymbol()} {p.x:.4f} {p.y:.4f} {p.z:.4f}")
        return "\n".join(lines)
    except Exception:
        return None


def sample_to_xyz(sample: Dict[str, Any]) -> Optional[str]:
    """从 ShEPhERD sample 的 x1 生成 XYZ 字符串。"""
    if "x1" not in sample or "atoms" not in sample["x1"] or "positions" not in sample["x1"]:
        return None
    try:
        from rdkit import Chem
        atoms = sample["x1"]["atoms"]
        positions = sample["x1"]["positions"]
        lines = [str(len(atoms)), ""]
        for i in range(len(atoms)):
            sym = Chem.Atom(int(atoms[i])).GetSymbol()
            p = positions[i]
            lines.append(f"{sym} {float(p[0]):.4f} {float(p[1]):.4f} {float(p[2]):.4f}")
        return "\n".join(lines)
    except Exception:
        return None


def smiles_to_xyz(smiles: str) -> Optional[str]:
    """
    从 SMILES 解析分子、加氢、生成 3D 构象并转为 XYZ 字符串。
    成功返回 XYZ 字符串，失败返回 None。
    """
    if not (smiles and smiles.strip()):
        return None
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        return mol_to_xyz(mol)
    except Exception:
        return None


def run_docking(
    protein_bytes: bytes,
    ref_smi: str,
    lig_smi: str,
    vina_module_path: str,
) -> Tuple[bool, Any]:
    """
    执行分子对接：写入临时蛋白文件、加载 vina 模块、计算对接分数。
    Returns:
        (True, score_float) 成功；(False, error_message_str) 失败。
    """
    import tempfile
    import importlib.util
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        spec = importlib.util.spec_from_file_location("vina_dock", vina_module_path)
        if spec is None or spec.loader is None:
            return (False, "无法加载对接模块")
        vina_dock = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vina_dock)
        with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f:
            f.write(protein_bytes)
            protein_path = f.name
        try:
            ref_mol = None
            if ref_smi and ref_smi.strip():
                ref_mol = Chem.MolFromSmiles(ref_smi.strip())
                if ref_mol:
                    ref_mol = Chem.AddHs(ref_mol)
                    AllChem.EmbedMolecule(ref_mol, AllChem.ETKDG())
            lig_mol = Chem.MolFromSmiles(lig_smi)
            if not lig_mol:
                return (False, "配体 SMILES 无效。")
            lig_mol = Chem.AddHs(lig_mol)
            AllChem.EmbedMolecule(lig_mol, AllChem.ETKDG())
            if ref_mol is None:
                ref_mol = lig_mol
            score = vina_dock.vina_score(lig_mol, protein_path, ref_mol)
            return (True, float(score))
        finally:
            try:
                os.unlink(protein_path)
            except Exception:
                pass
    except Exception as e:
        return (False, str(e))


def mol_to_2d_image_bytes(sample_or_mol: Any, size: Tuple[int, int] = (200, 200)) -> Optional[bytes]:
    """
    将 sample 或 RDKit Mol 转为 2D 结构图 PNG 字节。
    用于 Gallery 卡片展示；失败返回 None。
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = None
        if hasattr(sample_or_mol, "GetNumAtoms"):
            mol = sample_or_mol
        elif isinstance(sample_or_mol, dict):
            mol = sample_to_rdkit_mol(sample_or_mol)
        if mol is None:
            return None
        img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def list_training_jobs() -> List[Tuple[str, str]]:
    """
    列出 training/jobs 下各任务目录及 last.ckpt 路径。
    Returns:
        [(job_name, last.ckpt_abspath), ...]，无 ckpt 的目录仅列目录名，路径为空字符串。
    """
    root = get_project_root()
    jobs_dir = os.path.join(root, "training", "jobs")
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(jobs_dir):
        return out
    for name in sorted(os.listdir(jobs_dir)):
        path = os.path.join(jobs_dir, name)
        if not os.path.isdir(path):
            continue
        ckpt = os.path.join(path, "last.ckpt")
        if os.path.isfile(ckpt):
            out.append((name, os.path.abspath(ckpt)))
        else:
            out.append((name, ""))
    return out


def list_eval_result_files() -> List[Tuple[str, str]]:
    """列出 evaluation/experiment_SamEval/eval_results 下的 JSON 结果文件。"""
    root = get_project_root()
    er_dir = os.path.join(root, "evaluation", "experiment_SamEval", "eval_results")
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(er_dir):
        return out
    for f in sorted(glob.glob(os.path.join(er_dir, "*.json"))):
        name = os.path.basename(f)
        out.append((name, os.path.abspath(f)))
    return out


def list_core_data_dirs() -> List[Tuple[str, str]]:
    """列出 evaluation/core_data/data 下实验目录（1/2/3 等）及其子类型。"""
    root = get_project_root()
    data_root = os.path.join(root, "evaluation", "core_data", "data")
    out: List[Tuple[str, str]] = []
    if not os.path.isdir(data_root):
        return out
    for exp in sorted(os.listdir(data_root)):
        exp_path = os.path.join(data_root, exp)
        if not os.path.isdir(exp_path):
            continue
        for sub in sorted(os.listdir(exp_path)):
            sub_path = os.path.join(exp_path, sub)
            if os.path.isdir(sub_path):
                rel = os.path.join("data", exp, sub)
                out.append((rel, os.path.abspath(sub_path)))
    return out
