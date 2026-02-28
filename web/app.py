# -*- coding: utf-8 -*-
"""
Streamlit 主程序：科研人员交互平台。
核心功能：分子生成、训练与实验、评估与报告、3D 相似度、分子对接。
不展示代码/路径/目录；结果通过 st.session_state['generated_mols'] 在页面间互通。
"""

from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils import (
    checkpoint_to_readable_label,
    get_device,
    get_project_root,
    list_checkpoints,
    list_training_jobs,
    load_model_pl,
    mol_to_smiles,
    mol_to_xyz,
    render_3d_mol_html,
    render_3d_two_mols_synced,
    run_inference,
    sample_to_rdkit_mol,
    sample_to_xyz,
)

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="SPD 分子生成与评估",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Session State 初始化 ----------
if "device" not in st.session_state:
    st.session_state.device = get_device()
# 数据中转站：分子生成页的结果自动供评估与报告、3D 相似度使用
if "generated_mols" not in st.session_state:
    st.session_state.generated_mols = []  # List[Dict]: [{"sample": dict, "smiles": str|None}, ...]
if "generated_samples" not in st.session_state:
    st.session_state.generated_samples = []
if "generated_smiles" not in st.session_state:
    st.session_state.generated_smiles = []
if "selected_sample_idx" not in st.session_state:
    st.session_state.selected_sample_idx = 0
if "current_ckpt_path" not in st.session_state:
    st.session_state.current_ckpt_path = None
if "preselect_model_path" not in st.session_state:
    st.session_state.preselect_model_path = None  # 从「训练与实验」跳转时预选模型
if "goto_page" not in st.session_state:
    st.session_state.goto_page = None  # 跳转目标，在创建 radio 前写入 nav_radio 避免 API 报错


def sidebar_nav() -> str:
    """侧边栏导航：仅核心功能，不展示配置/数据目录。"""
    st.sidebar.title("SPD")
    st.sidebar.caption("分子生成与评估平台")
    st.sidebar.markdown("---")
    # 在创建 radio 之前处理跳转：Streamlit 不允许在 widget 创建后再改其 key 对应的 session_state
    if st.session_state.get("goto_page") is not None:
        st.session_state.nav_radio = st.session_state.goto_page
        st.session_state.goto_page = None
    page = st.sidebar.radio(
        "导航",
        [
            "分子生成",
            "训练与实验",
            "评估与报告",
            "3D 相似度",
            "分子对接",
        ],
        label_visibility="collapsed",
        key="nav_radio",
    )
    st.sidebar.markdown("---")
    n_mols = len(st.session_state.get("generated_mols", []))
    if n_mols > 0:
        st.sidebar.caption(f"当前会话: {n_mols} 个生成分子")
    st.sidebar.caption(f"设备: {st.session_state.device}")
    return page


def page_sampling() -> None:
    """分子生成：可读模型名、参数、生成（带进度条/动态提示）、结果列表与 3D。"""
    st.header("分子生成")
    st.caption("选择模型与参数，生成分子结构。生成过程可能较慢，请留意进度提示。")

    checkpoints = list_checkpoints()
    if not checkpoints:
        st.warning("未找到可用模型。请将检查点放在项目约定目录下。")
        return

    # 可读模型名，不展示路径
    labels = [checkpoint_to_readable_label(rel) for rel, _ in checkpoints]
    ckpt_paths = [p for _, p in checkpoints]
    # 若有从「训练与实验」来的预选，尝试匹配
    preselect_path = st.session_state.get("preselect_model_path")
    default_idx = 0
    if preselect_path and preselect_path in ckpt_paths:
        default_idx = ckpt_paths.index(preselect_path)
        st.session_state.preselect_model_path = None
    idx = st.selectbox("选择模型", range(len(labels)), format_func=lambda i: labels[i], index=default_idx, key="sb_ckpt")
    chosen_ckpt = ckpt_paths[idx]

    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.number_input("每批生成数量", min_value=1, max_value=32, value=4, key="ni_batch")
        N_x1 = st.number_input("原子数", min_value=1, max_value=80, value=12, key="ni_nx1", help="生成分子的原子数")
        N_x4 = st.number_input("药效团数", min_value=0, max_value=20, value=2, key="ni_nx4")
    with col2:
        unconditional = st.checkbox("无条件生成", value=True, key="cb_uncond")
        harmonize = st.checkbox("协调化 (harmonize)", value=False, key="cb_harmonize")
        prior_scale = st.number_input("先验噪声尺度", min_value=0.1, max_value=2.0, value=1.0, step=0.1, key="ni_prior")
        denoise_scale = st.number_input("去噪噪声尺度", min_value=0.1, max_value=2.0, value=1.0, step=0.1, key="ni_denoise")

    if st.button("生成分子", type="primary", key="btn_run_inference"):
        # 在主线程中执行推理，避免子线程缺少 ScriptRunContext 导致结果无法回传
        with st.spinner("生成中... 采样可能需 1–3 分钟，请勿关闭页面。"):
            try:
                model_pl = load_model_pl(chosen_ckpt, st.session_state.device)
                harmonize_ts = [80] if harmonize else []
                harmonize_jumps = [20] if harmonize else []
                samples = run_inference(
                    model_pl,
                    batch_size=batch_size,
                    N_x1=N_x1,
                    N_x4=N_x4,
                    unconditional=unconditional,
                    device=st.session_state.device,
                    prior_noise_scale=prior_scale,
                    denoising_noise_scale=denoise_scale,
                    harmonize=harmonize,
                    harmonize_ts=harmonize_ts,
                    harmonize_jumps=harmonize_jumps,
                )
                smiles_list = []
                for s in samples:
                    mol = sample_to_rdkit_mol(s)
                    smiles_list.append(mol_to_smiles(mol) if mol else None)
                generated_mols = [{"sample": s, "smiles": smi} for s, smi in zip(samples, smiles_list)]
                st.session_state.generated_samples = samples
                st.session_state.generated_smiles = smiles_list
                st.session_state.generated_mols = generated_mols
                st.session_state.current_ckpt_path = chosen_ckpt
                n_ok = sum(1 for smi in smiles_list if smi)
                st.success(f"已生成 {len(samples)} 个分子，其中 {n_ok} 个有效。结果已同步到「评估与报告」与「3D 相似度」。")
                # 仅成功时 rerun，以便下方「展示已生成结果」能渲染；失败时不 rerun，保留错误信息
                st.rerun()
            except Exception as e:
                st.error(f"推理失败: {e}")
                import traceback
                st.code(traceback.format_exc(), language="text")
                # 失败时不 rerun，让用户看到错误和堆栈，便于排查

    # 展示已生成结果
    if st.session_state.generated_mols:
        st.subheader("生成结果")
        mols = st.session_state.generated_mols
        idx_sel = st.selectbox(
            "选择分子",
            range(len(mols)),
            format_func=lambda i: f"#{i+1} {mols[i].get('smiles') or '(无效)'}",
            key="sel_mol_idx",
        )
        st.session_state.selected_sample_idx = idx_sel
        sel = mols[idx_sel]["sample"]
        smi = mols[idx_sel].get("smiles")
        if smi:
            st.text(f"SMILES: {smi}")
        xyz_str = sample_to_xyz(sel)
        if xyz_str:
            html = render_3d_mol_html(xyz_str, style="stick", width=500, height=450)
            components.html(html, height=470, scrolling=False)


def page_training() -> None:
    """训练与实验：实验卡片（可读名、状态）、用此模型生成（不展示路径）。"""
    st.header("训练与实验")
    st.caption("查看实验与对应模型，并可直接选用模型进行生成。")

    jobs = list_training_jobs()
    if not jobs:
        st.info("暂无实验记录。")
        return

    for name, ckpt_path in jobs:
        # 可读名称
        display_name = checkpoint_to_readable_label(name)
        with st.expander(f"实验: {display_name}"):
            if ckpt_path:
                st.caption("状态: 已完成（可选用此模型生成）")
                if st.button("用此模型生成", key=f"use_ckpt_{name.replace('/', '_')}"):
                    st.session_state.preselect_model_path = ckpt_path
                    st.session_state.goto_page = "分子生成"
                    st.rerun()
            else:
                st.caption("状态: 暂无检查点")


def page_evaluation() -> None:
    """评估与报告：优先使用会话中的 generated_mols，表格 + 导出 CSV/Excel。"""
    st.header("评估与报告")
    st.caption("对当前会话中的生成分子或上传结果进行统计，并导出报告。")

    # 优先使用 session_state['generated_mols'] 作为数据源
    mols = st.session_state.get("generated_mols", [])
    use_session = st.checkbox("使用当前会话中的生成结果", value=bool(mols), key="eval_use_session")

    if use_session and mols:
        st.write(f"共 **{len(mols)}** 条样本（来自「分子生成」页面）。")
        rows = []
        for i, m in enumerate(mols):
            s = m.get("sample", {})
            n_atoms = len(s.get("x1", {}).get("atoms", []))
            rows.append({"序号": i + 1, "SMILES": m.get("smiles") or "(无效)", "原子数": n_atoms})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        # 导出 CSV / Excel
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("导出 CSV", data=csv, file_name="generated_mols_report.csv", mime="text/csv", key="dl_csv")
        with col2:
            try:
                buf = io.BytesIO()
                df.to_excel(buf, index=False, engine="openpyxl")
                buf.seek(0)
                st.download_button("导出 Excel", data=buf, file_name="generated_mols_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx")
            except Exception:
                st.caption("导出 Excel 需要安装 openpyxl")
        return

    # 上传 JSON
    st.caption("或上传生成结果 JSON 文件进行统计。")
    uploaded = st.file_uploader("上传生成结果 JSON", type=["json"], key="eval_upload")
    if uploaded is not None:
        try:
            data = json.load(uploaded)
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict) and "samples" in data:
                items = data["samples"]
            else:
                items = [data]
            rows = []
            for i, it in enumerate(items):
                x1 = it.get("x1", {})
                n_atoms = len(x1.get("atoms", []))
                rows.append({"序号": i + 1, "原子数": n_atoms})
            df = pd.DataFrame(rows)
            st.write(f"解析到 **{len(items)}** 条记录。")
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("导出 CSV", data=csv, file_name="uploaded_report.csv", mime="text/csv", key="dl_csv_upload")
        except Exception as e:
            st.error(f"解析 JSON 失败: {e}")


def page_scoring_3d() -> None:
    """3D 相似度：双分子同视图展示（同步旋转），可从未生成分子中选择。"""
    st.header("3D 相似度")
    st.caption("并排对比两个分子的 3D 结构；旋转时两分子同步旋转。可从会话生成结果中选择或输入 SMILES。")

    mols = st.session_state.get("generated_mols", [])
    opts = ["手动输入 SMILES"] + [f"会话分子 #{i+1}: {m.get('smiles') or '(无效)'}" for i in range(len(mols))]

    mode = st.radio("分子来源", ["手动输入 SMILES", "从会话生成结果选择"], key="s3d_mode")
    smi1, smi2 = "CCO", "CC=O"
    xyz1, xyz2 = None, None

    if mode == "从会话生成结果选择" and mols:
        idx1 = st.selectbox("分子 A", range(len(mols)), format_func=lambda i: f"#{i+1} {mols[i].get('smiles') or '(无效)'}", key="s3d_a")
        idx2 = st.selectbox("分子 B", range(len(mols)), format_func=lambda i: f"#{i+1} {mols[i].get('smiles') or '(无效)'}", key="s3d_b")
        xyz1 = sample_to_xyz(mols[idx1]["sample"])
        xyz2 = sample_to_xyz(mols[idx2]["sample"])
        smi1 = mols[idx1].get("smiles") or ""
        smi2 = mols[idx2].get("smiles") or ""
    else:
        col1, col2 = st.columns(2)
        with col1:
            smi1 = st.text_input("分子 A SMILES", value="CCO", key="s3d_smi1")
        with col2:
            smi2 = st.text_input("分子 B SMILES", value="CC=O", key="s3d_smi2")

    if st.button("生成 3D 对比（同步旋转）", key="btn_s3d"):
        if xyz1 is None and smi1:
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem
                mol = Chem.MolFromSmiles(smi1)
                if mol:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                    xyz1 = mol_to_xyz(mol)
            except Exception:
                pass
        if xyz2 is None and smi2:
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem
                mol = Chem.MolFromSmiles(smi2)
                if mol:
                    mol = Chem.AddHs(mol)
                    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                    xyz2 = mol_to_xyz(mol)
            except Exception:
                pass
        if xyz1 and xyz2:
            html = render_3d_two_mols_synced(xyz1, xyz2, style="stick", width=600, height=500)
            st.caption("同一视图内两分子同步旋转。")
            components.html(html, height=520, scrolling=False)
        else:
            st.warning("请确保两个分子均可解析并生成 3D 构象。")


def page_docking() -> None:
    """分子对接：上传蛋白与配体，展示对接分数与简要解读。"""
    st.header("分子对接")
    st.caption("上传蛋白结构与配体，获取对接分数。参考分子用于确定结合腔中心。")

    root = get_project_root()
    vina_module_path = os.path.join(root, "vina_dock.py")
    if not os.path.isfile(vina_module_path):
        st.warning("当前环境未配置对接模块。")
        return

    protein_file = st.file_uploader("蛋白结构 (PDBQT)", type=["pdbqt"], key="dock_protein")
    ref_smi = st.text_input("参考分子 SMILES（用于确定盒子中心，可留空）", value="", key="dock_ref_smi")
    lig_smi = st.text_input("配体 SMILES", value="CCO", key="dock_lig_smi")

    if st.button("运行对接", key="btn_dock"):
        if not protein_file:
            st.error("请上传蛋白 PDBQT 文件。")
            return
        with st.spinner("对接中…"):
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem
                import tempfile
                import importlib.util
                spec = importlib.util.spec_from_file_location("vina_dock", vina_module_path)
                vina_dock = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(vina_dock)
                with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as f:
                    f.write(protein_file.getvalue())
                    protein_path = f.name
                try:
                    ref_mol = None
                    if ref_smi:
                        ref_mol = Chem.MolFromSmiles(ref_smi)
                        if ref_mol:
                            ref_mol = Chem.AddHs(ref_mol)
                            AllChem.EmbedMolecule(ref_mol, AllChem.ETKDG())
                    lig_mol = Chem.MolFromSmiles(lig_smi)
                    if not lig_mol:
                        st.error("配体 SMILES 无效。")
                        return
                    lig_mol = Chem.AddHs(lig_mol)
                    AllChem.EmbedMolecule(lig_mol, AllChem.ETKDG())
                    if ref_mol is None:
                        ref_mol = lig_mol
                    score = vina_dock.vina_score(lig_mol, protein_path, ref_mol)
                    st.success(f"**对接分数**: {score:.2f} kcal/mol")
                    if score < -7:
                        st.caption("解读: 分数较好，可能具有较好结合亲和力。")
                    elif score < -5:
                        st.caption("解读: 分数中等，可结合实验进一步验证。")
                    else:
                        st.caption("解读: 分数偏弱，可尝试其他配体或优化。")
                finally:
                    try:
                        os.unlink(protein_path)
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"对接失败: {e}")


def main() -> None:
    page = sidebar_nav()
    if page == "分子生成":
        page_sampling()
    elif page == "训练与实验":
        page_training()
    elif page == "评估与报告":
        page_evaluation()
    elif page == "3D 相似度":
        page_scoring_3d()
    else:
        page_docking()


if __name__ == "__main__":
    main()
