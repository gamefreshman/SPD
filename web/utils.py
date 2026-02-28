# -*- coding: utf-8 -*-
"""
Web 前端逻辑工具库：模型加载、推理封装、数据格式转换、3D 可视化辅助等。
所有与后端 src/shepherd、shepherd_score 的交互均在本模块中封装，使用相对路径与缓存。
"""

from __future__ import annotations

import os
import sys
import glob
from typing import Any, Dict, List, Optional, Tuple

# 将项目根目录与 src 加入 Python 路径，以便导入 shepherd（不修改项目其他文件）
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
    自动选择计算设备：有 CUDA 则用 cuda，否则 cpu。
    Returns:
        str: 'cuda' 或 'cpu'。
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def list_checkpoints() -> List[Tuple[str, str]]:
    """
    扫描项目内 checkpoint 目录，返回 (显示名, 绝对路径) 列表。
    搜索目录：checkpoint/、data/shepherd_chkpts/、training/jobs/*/last.ckpt。
    Returns:
        List[Tuple[str, str]]: [(label, abspath), ...]。
    """
    root = get_project_root()
    out: List[Tuple[str, str]] = []
    seen: set = set()

    # 1) checkpoint/ 下的 .ckpt 文件
    ckpt_dir = os.path.join(root, "checkpoint")
    if os.path.isdir(ckpt_dir):
        for p in glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True):
            abspath = os.path.abspath(p)
            if abspath not in seen:
                seen.add(abspath)
                rel = os.path.relpath(p, root)
                out.append((rel, abspath))

    # 2) data/shepherd_chkpts/
    shepherd_ckpt = os.path.join(root, "data", "shepherd_chkpts")
    if os.path.isdir(shepherd_ckpt):
        for p in glob.glob(os.path.join(shepherd_ckpt, "**", "*.ckpt"), recursive=True):
            abspath = os.path.abspath(p)
            if abspath not in seen:
                seen.add(abspath)
                rel = os.path.relpath(p, root)
                out.append((rel, abspath))

    # 3) training/jobs/*/last.ckpt
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
    将检查点相对路径转为科研人员可读的模型名称（不展示路径）。
    Args:
        rel_path: 相对项目根的路径，如 training/jobs/33/x1x3x4_dpo_finetune_nps/last.ckpt
    Returns:
        可读名称，如 "DPO 微调 (NP)" 或最后目录名。
    """
    if not rel_path:
        return "未命名模型"
    # 取最后两级目录或文件名作为上下文
    parts = rel_path.replace("\\", "/").split("/")
    name = parts[-2] if len(parts) >= 2 and parts[-1] == "last.ckpt" else parts[-1]
    name = name.replace(".ckpt", "").strip()
    # 常见模式转可读名
    if "dpo_finetune_nps" in name or "dpo_finetune_np" in name:
        return "DPO 微调 (天然产物)"
    if "dpo_fragment" in name:
        return "DPO 片段合并"
    if "dpo_finetune_pdb" in name:
        return "DPO 微调 (PDB)"
    if "diffusion" in name:
        return "扩散模型"
    return name or "模型"


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
    Args:
        checkpoint_path: .ckpt 文件绝对路径。
        device: 'cuda' 或 'cpu'。
    Returns:
        LightningModule 实例（已 eval、已 to(device)）。
    Raises:
        Exception: 加载失败时抛出，由调用方 try/except 后通过 st.error 提示。
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
    Args:
        model_pl: LightningModule 实例。
        batch_size: 批次大小。
        N_x1: 原子数。
        N_x4: 药效团数。
        unconditional: 是否无条件生成。
        device: 设备字符串。
        prior_noise_scale: 先验噪声尺度。
        denoising_noise_scale: 去噪噪声尺度。
        harmonize: 是否协调化。
        harmonize_ts: 协调化时间步。
        harmonize_jumps: 协调化步长。
        atom_marginals: 原子边际（None 则后端使用默认）。
        bond_marginals: 键边际（None 则后端使用默认）。
    Returns:
        生成的样本列表，每项为 inference_sample 返回的字典结构（含 x1/x2/x3/x4）。
    Raises:
        Exception: 推理过程中任何异常会向上抛出，由 app 层捕获并 st.error。
    """
    import numpy as np
    import torch
    from shepherd.inference import inference_sample

    # 后端 inference 要求 atom_marginals / bond_marginals 非 None，否则 DiscreteFeatureDiffusion 会报错
    # 与 src/shepherd/discrete_diffuser.py 默认一致：4 类原子、5 类键，归一化后传入
    if atom_marginals is None:
        _atom = torch.tensor([0.7230, 0.1151, 0.1593, 0.0026], dtype=torch.float32)
        atom_marginals = _atom / _atom.sum()
    if bond_marginals is None:
        _bond = torch.tensor([0.7261, 0.2384, 0.0274, 0.0081, 0.0], dtype=torch.float32)
        bond_marginals = _bond / _bond.sum()

    # 无条件生成时 inpainting 目标使用零占位，与 basic_inference_test 一致
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
    Args:
        sample: 含 'x1']['atoms' 与 'x1']['positions' 的字典。
    Returns:
        RDKit Mol 或 None（转换失败时）。
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


def render_3d_mol_html(
    xyz_or_mol_block: str,
    style: str = "stick",
    width: int = 400,
    height: int = 400,
) -> str:
    """
    使用 py3Dmol 生成分子 3D 显示的 HTML 字符串，供 st.components.v1.html 使用。
    Args:
        xyz_or_mol_block: XYZ 格式字符串或 MOL block。
        style: 'stick' | 'sphere' | 'cartoon' 等。
        width: 视图宽度。
        height: 视图高度。
    Returns:
        完整 HTML 字符串。
    """
    try:
        import py3Dmol
    except ImportError:
        return "<p>需要安装 py3Dmol: pip install py3Dmol</p>"

    view = py3Dmol.view(width=width, height=height)
    # 根据前几行判断格式：首行为数字则视为 XYZ
    first_line = (xyz_or_mol_block.strip().split("\n")[0] or "").strip()
    if first_line.isdigit():
        view.addModel(xyz_or_mol_block, "xyz")
    else:
        view.addModel(xyz_or_mol_block, "sdf")
    view.setStyle({style: {}})
    view.zoomTo()
    return view.write_html()


def render_3d_two_mols_synced(
    xyz1: str,
    xyz2: str,
    style: str = "stick",
    width: int = 500,
    height: int = 450,
    offset_second: float = 12.0,
) -> str:
    """
    在同一 3D 视图中添加两个分子，实现同步旋转（旋转一侧，另一侧同步）。
    第二个分子沿 x 轴平移 offset_second，避免重叠。
    Args:
        xyz1: 分子 1 的 XYZ 字符串。
        xyz2: 分子 2 的 XYZ 字符串。
        style: 显示风格。
        width: 视图宽度。
        height: 视图高度。
        offset_second: 分子 2 的 x 轴偏移量（埃）。
    Returns:
        完整 HTML 字符串。
    """
    try:
        import py3Dmol
    except ImportError:
        return "<p>需要安装 py3Dmol</p>"

    def shift_xyz(xyz: str, dx: float) -> str:
        """将 XYZ 字符串中所有 x 坐标加上 dx。"""
        lines = xyz.strip().split("\n")
        if len(lines) < 3:
            return xyz
        try:
            n = int(lines[0])
        except ValueError:
            return xyz
        out = [lines[0], lines[1] if len(lines) > 1 else ""]
        for i in range(2, min(2 + n, len(lines))):
            parts = lines[i].split()
            if len(parts) >= 4:
                x = float(parts[1]) + dx
                out.append(f"{parts[0]} {x:.4f} {parts[2]} {parts[3]}")
            else:
                out.append(lines[i])
        return "\n".join(out)

    view = py3Dmol.view(width=width, height=height)
    first1 = (xyz1.strip().split("\n")[0] or "").strip()
    if first1.isdigit():
        view.addModel(xyz1, "xyz")
    else:
        view.addModel(xyz1, "sdf")
    xyz2_shifted = shift_xyz(xyz2, offset_second)
    first2 = (xyz2_shifted.strip().split("\n")[0] or "").strip()
    if first2.isdigit():
        view.addModel(xyz2_shifted, "xyz")
    else:
        view.addModel(xyz2_shifted, "sdf")
    view.setStyle({style: {}})
    view.zoomTo()
    return view.write_html()


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


