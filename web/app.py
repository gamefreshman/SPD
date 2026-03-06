# -*- coding: utf-8 -*-
"""
Streamlit 主程序：科研人员交互平台。
核心功能：分子生成、训练与实验、评估与报告、3D 相似度、分子对接。
不展示代码/路径/目录；结果通过 st.session_state['generated_mols'] 在页面间互通。
"""

from __future__ import annotations

import base64
import io
import json
import os
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ui_utils import (
    COLORS,
    LAYOUT,
    badge_color_for_value,
    badge_html,
    get_custom_css,
    parameter_card_marker_html,
    render_metric_cards_row,
    render_molecule_card,
    safe_render_3d_mol_html,
    safe_render_3d_two_mols_synced,
    smiles_display_html,
)
from backend_utils import (
    build_unique_checkpoint_labels,
    checkpoint_to_readable_label,
    get_device,
    get_project_root,
    get_sidebar_status_html,
    list_checkpoints,
    list_training_jobs,
    load_model_pl,
    mol_to_2d_image_bytes,
    mol_to_smiles,
    structure_to_pdbqt,
    run_docking,
    run_inference,
    sample_to_rdkit_mol,
    sample_to_xyz,
    smiles_to_xyz,
)

# ---------- 常量 ----------
PAGES = ("分子生成", "训练与实验", "评估与报告", "3D 相似度", "分子对接")
NAV_RADIO_KEY = "nav_radio"

# ---------- 页面配置 ----------
st.set_page_config(
    page_title="SPD 分子生成与评估",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)
# 注入设计系统 CSS（全局去噪、侧边栏、卡片、按钮）
st.markdown(get_custom_css(), unsafe_allow_html=True)

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
    """侧边栏：Logo 居中与标题精修、导航、状态仪表盘小组件。"""
    st.sidebar.markdown(
        '<div style="text-align:center; padding:0.5rem 0 1rem 0;">'
        f'<span style="font-size:{LAYOUT["sidebar_logo_emoji_size"]};">🧪</span><br/>'
        f'<strong style="font-size:{LAYOUT["sidebar_logo_title_size"]}; color:{COLORS["primary"]};">SPD</strong><br/>'
        f'<span style="font-size:{LAYOUT["sidebar_subtitle_size"]}; color:{COLORS["subtitle_gray"]};">分子生成与评估平台</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()
    if st.session_state.get("goto_page") is not None:
        st.session_state[NAV_RADIO_KEY] = st.session_state.goto_page
        st.session_state.goto_page = None
    page = st.sidebar.radio(
        "导航",
        list(PAGES),
        label_visibility="collapsed",
        key=NAV_RADIO_KEY,
    )
    st.sidebar.markdown("---")
    # 状态面板：紧凑仪表盘小组件（淡青背景由 CSS .sidebar-status-panel 提供）
    n_mols = len(st.session_state.get("generated_mols", []))
    status_html = get_sidebar_status_html(
        st.session_state.device,
        st.session_state.get("current_ckpt_path"),
        n_mols,
    )
    st.sidebar.markdown("**状态**")
    st.sidebar.markdown(status_html, unsafe_allow_html=True)
    return page


def page_sampling() -> None:
    """分子生成：参数卡片布局、生成、结果以 Gallery 网格展示（2D + 徽章 + 3D 详情）。"""
    st.header("分子生成")
    st.caption("选择模型与参数，生成分子结构。生成过程可能较慢，请留意进度提示。")

    checkpoints = list_checkpoints()
    if not checkpoints:
        st.warning("未找到可用模型。请将检查点放在项目约定目录下。")
        return

    labels = build_unique_checkpoint_labels(checkpoints)
    ckpt_paths = [p for _, p in checkpoints]
    preselect_path = st.session_state.get("preselect_model_path")
    default_idx = 0
    if preselect_path and preselect_path in ckpt_paths:
        default_idx = ckpt_paths.index(preselect_path)
        st.session_state.preselect_model_path = None
    idx = st.selectbox("选择模型", range(len(labels)), format_func=lambda i: labels[i], index=default_idx, key="sb_ckpt")
    chosen_ckpt = ckpt_paths[idx]

    st.markdown(parameter_card_marker_html(), unsafe_allow_html=True)
    _render_sampling_parameter_block(chosen_ckpt)
    if st.session_state.generated_mols:
        st.subheader("生成结果")
        _render_sampling_gallery(st.session_state.generated_mols)


def _render_sampling_parameter_block(chosen_ckpt: str) -> None:
    """分子生成页：参数区与生成按钮，触发后写 session_state 并 rerun。"""
    with st.container():
        col_batch, col_options, col_action = st.columns(3)
        with col_batch:
            batch_size = st.number_input("每批生成数量", min_value=1, max_value=32, value=4, key="ni_batch")
            N_x1 = st.number_input("原子数", min_value=1, max_value=80, value=12, key="ni_nx1", help="生成分子的原子数")
            N_x4 = st.number_input("药效团数", min_value=0, max_value=20, value=2, key="ni_nx4")
        with col_options:
            unconditional = st.checkbox("无条件生成", value=True, key="cb_uncond")
            harmonize = st.checkbox("协调化 ", value=False, key="cb_harmonize")
            if harmonize:
                st.caption("⚠️ 协调化在某些 checkpoint 下会导致图节点数与编码器期望不一致，若报错请取消勾选后重试。")
            prior_scale = st.number_input("先验噪声尺度", min_value=0.1, max_value=2.0, value=1.0, step=0.1, key="ni_prior")
            denoise_scale = st.number_input("去噪噪声尺度", min_value=0.1, max_value=2.0, value=1.0, step=0.1, key="ni_denoise")
        with col_action:
            st.markdown("&nbsp;")
            if st.button("生成分子", type="primary", key="btn_run_inference", use_container_width=True):
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
                        st.rerun()
                    except RuntimeError as e:
                        err_msg = str(e)
                        if "Expected size" in err_msg and ("batch2" in err_msg or "first two dimensions" in err_msg):
                            st.error("推理失败：当前模型在开启「协调化」时出现图维度不匹配。请取消勾选「协调化」后重试。")
                        else:
                            st.error(f"推理失败: {e}")
                        st.code(traceback.format_exc(), language="text")
                    except Exception as e:
                        st.error(f"推理失败: {e}")
                        st.code(traceback.format_exc(), language="text")


def _render_sampling_gallery(mols: List[Dict[str, Any]]) -> None:
    """分子生成页：Gallery 网格（2D 图 + 徽章 + SMILES + 3D expander）。"""
    n_cols = 4
    for start in range(0, len(mols), n_cols):
        cols = st.columns(n_cols)
        for k, col in enumerate(cols):
            i = start + k
            if i >= len(mols):
                break
            with col:
                m = mols[i]
                sample = m.get("sample", {})
                smi = m.get("smiles")
                n_atoms = len(sample.get("x1", {}).get("atoms", []))
                st.markdown('<div class="gallery-card-wrapper">', unsafe_allow_html=True)
                img_b64 = None
                try:
                    img_bytes = mol_to_2d_image_bytes(sample, size=(220, 220))
                    if img_bytes:
                        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                except Exception:
                    pass
                valid_color = COLORS["badge_high"] if smi else COLORS["badge_low"]
                atom_color = badge_color_for_value(min(n_atoms / 50.0, 1.0), low_is_bad=False)
                badges_inner = badge_html("有效" if smi else "无效", "是" if smi else "否", valid_color) + " " + badge_html("原子数", str(n_atoms), atom_color)
                card_html = render_molecule_card(img_b64, badges_inner, smi or "", card_label=f"#{i+1}")
                st.markdown(card_html, unsafe_allow_html=True)
                with st.expander(" 3D ", expanded=False):
                    xyz_str = sample_to_xyz(sample)
                    if xyz_str:
                        try:
                            html = safe_render_3d_mol_html(
                                xyz_str, style="stick", width=320, height=280,
                                background_color=COLORS["viewer_3d_bg"],
                            )
                            components.html(html, height=300, scrolling=False)
                        except Exception:
                            st.caption("3D 加载失败")
                    else:
                        st.caption("无 3D 数据")
                st.markdown("</div>", unsafe_allow_html=True)


def _render_eval_session_metrics(metric_htmls: List[str]) -> None:
    """评估页：顶部一排指标卡片。"""
    cols = st.columns(len(metric_htmls))
    for col, html in zip(cols, metric_htmls):
        with col:
            st.markdown(html, unsafe_allow_html=True)


def _render_eval_table_and_export(df: pd.DataFrame) -> None:
    """评估页：数据表 expander 与 CSV/Excel 导出。"""
    with st.expander("查看/导出数据表", expanded=False):
        styled = df.style.set_table_styles([
            {"selector": "th", "props": [("background-color", COLORS["primary"]), ("color", COLORS["download_btn_text"]), ("font-weight", "600")]},
            {"selector": "tr:nth-of-type(even)", "props": [("background-color", COLORS["table_even"])]},
            {"selector": "tr:hover", "props": [("background-color", COLORS["table_hover"])]},
        ])
        st.dataframe(styled, use_container_width=True)
        col_csv, col_xlsx = st.columns(2)
        with col_csv:
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("导出 CSV", data=csv, file_name="generated_mols_report.csv", mime="text/csv", key="dl_csv")
        with col_xlsx:
            try:
                buf = io.BytesIO()
                df.to_excel(buf, index=False, engine="openpyxl")
                buf.seek(0)
                st.download_button("导出 Excel", data=buf, file_name="generated_mols_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_xlsx")
            except Exception:
                st.caption("导出 Excel 需要安装 openpyxl")


def page_training() -> None:
    """训练与实验：实验卡片（可读名、状态）、用此模型生成。"""
    st.header("训练与实验")
    st.caption("查看实验与对应模型，并可直接选用模型进行生成。")

    jobs = list_training_jobs()
    if not jobs:
        st.info("暂无实验记录。")
        return

    for name, ckpt_path in jobs:
        display_name = checkpoint_to_readable_label(name)
        with st.expander(f"实验: {display_name}", expanded=False):
            if ckpt_path:
                st.caption("状态: 已完成（可选用此模型生成）")
                if st.button("用此模型生成", key=f"use_ckpt_{name.replace('/', '_')}"):
                    st.session_state.preselect_model_path = ckpt_path
                    st.session_state.goto_page = "分子生成"
                    st.rerun()
            else:
                st.caption("状态: 暂无检查点")


def page_evaluation() -> None:
    """评估与报告：顶部大指标、图表突出、表格收起，配色遵循设计规范。"""
    st.header("评估与报告")
    st.caption("对当前会话中的生成分子或上传结果进行统计，并导出报告。")

    mols = st.session_state.get("generated_mols", [])
    use_session = st.checkbox("使用当前会话中的生成结果", value=bool(mols), key="eval_use_session")

    if use_session and mols:
        n_total = len(mols)
        n_valid = sum(1 for m in mols if m.get("smiles"))
        rate = (n_valid / n_total * 100) if n_total else 0
        rows = []
        for i, m in enumerate(mols):
            s = m.get("sample", {})
            n_atoms = len(s.get("x1", {}).get("atoms", []))
            rows.append({"序号": i + 1, "SMILES": m.get("smiles") or "(无效)", "原子数": n_atoms})
        df = pd.DataFrame(rows)
        atoms_list = df["原子数"].tolist()
        avg_atoms = sum(atoms_list) / len(atoms_list) if atoms_list else 0

        metric_htmls = render_metric_cards_row([
            ("生成总数", str(n_total), "🧪"),
            ("有效分子数", str(n_valid), "✓"),
            ("有效率 (%)", f"{rate:.1f}", "📊"),
            ("平均原子数", f"{avg_atoms:.1f}", "⚛"),
        ])
        _render_eval_session_metrics(metric_htmls)
        _render_eval_table_and_export(df)
        return

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
            st.metric("解析记录数", len(items))
            with st.expander("查看数据表", expanded=False):
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("导出 CSV", data=csv, file_name="uploaded_report.csv", mime="text/csv", key="dl_csv_upload")
        except Exception as e:
            st.error(f"解析 JSON 失败: {e}")


def page_scoring_3d() -> None:
    """3D 相似度：参数胶囊、左右分屏、3D 浅灰背景；SMILES 等宽单行省略。"""
    st.header("3D 相似度")
    st.caption("并排对比两个分子的 3D 结构；旋转时两分子同步旋转。可从会话生成结果中选择或输入 SMILES。")

    st.markdown(parameter_card_marker_html(), unsafe_allow_html=True)
    with st.container():
        mols = st.session_state.get("generated_mols", [])
        mode = st.radio("分子来源", ["手动输入 SMILES", "从会话生成结果选择"], key="s3d_mode", horizontal=True)
        smi1, smi2 = "CCO", "CC=O"
        xyz1, xyz2 = None, None

        if mode == "从会话生成结果选择" and mols:
            idx1 = st.selectbox("分子 A（参考）", range(len(mols)), format_func=lambda i: f"#{i+1} {mols[i].get('smiles') or '(无效)'}", key="s3d_a")
            idx2 = st.selectbox("分子 B（生成）", range(len(mols)), format_func=lambda i: f"#{i+1} {mols[i].get('smiles') or '(无效)'}", key="s3d_b")
            xyz1 = sample_to_xyz(mols[idx1]["sample"]) if idx1 < len(mols) else None
            xyz2 = sample_to_xyz(mols[idx2]["sample"]) if idx2 < len(mols) else None
            smi1 = mols[idx1].get("smiles") or ""
            smi2 = mols[idx2].get("smiles") or ""
        else:
            left_in, right_in = st.columns(2)
            with left_in:
                smi1 = st.text_input("分子 A SMILES（参考）", value="CCO", key="s3d_smi1")
            with right_in:
                smi2 = st.text_input("分子 B SMILES（生成）", value="CC=O", key="s3d_smi2")

        if st.button("生成 3D 对比（同步旋转）", type="primary", key="btn_s3d", use_container_width=True):
            if xyz1 is None and smi1:
                xyz1 = smiles_to_xyz(smi1)
            if xyz2 is None and smi2:
                xyz2 = smiles_to_xyz(smi2)
            if xyz1 and xyz2:
                try:
                    html = safe_render_3d_two_mols_synced(
                        xyz1, xyz2, style="stick", width=500, height=450,
                        background_color=COLORS["viewer_3d_bg"],
                    )
                    st.caption("同一视图内两分子同步旋转。")
                    col_ref, col_gen = st.columns(2)
                    with col_ref:
                        st.markdown("**参考分子 A**")
                        st.markdown(smiles_display_html(smi1), unsafe_allow_html=True)
                    with col_gen:
                        st.markdown("**生成分子 B**")
                        st.markdown(smiles_display_html(smi2), unsafe_allow_html=True)
                    components.html(html, height=470, scrolling=False)
                except Exception:
                    st.warning("3D 渲染失败，请检查分子数据。")
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

    st.markdown(parameter_card_marker_html(), unsafe_allow_html=True)
    with st.container():
        protein_file = st.file_uploader("蛋白结构 (PDB / CIF / PDBQT / PKL)", type=["pdb", "cif", "pdbqt", "pkl"], key="dock_protein")
        ref_smi = st.text_input("参考分子 SMILES（用于确定盒子中心，可留空）", value="", key="dock_ref_smi")
        lig_smi = st.text_input("配体 SMILES", value="CCO", key="dock_lig_smi")
        if st.button("运行对接", type="primary", key="btn_dock", use_container_width=True):
            if not protein_file:
                st.error("请上传蛋白 PDB、CIF、PDBQT 或 PKL 文件。")
            else:
                raw = protein_file.getvalue()
                fname = (protein_file.name or "").lower()
                if fname.endswith(".pdb") or fname.endswith(".cif"):
                    fmt = "pdb" if fname.endswith(".pdb") else "cif"
                    with st.spinner(f"正在将 {fmt.upper()} 转为 PDBQT…"):
                        ok_conv, pdbqt_bytes, err_msg = structure_to_pdbqt(raw, fmt)
                    if not ok_conv:
                        st.error(f"{fmt.upper()} 转换失败: {err_msg}")
                        protein_bytes = None
                    else:
                        protein_bytes = pdbqt_bytes
                elif fname.endswith(".pkl"):
                    import pickle

                    def _looks_like_pdbqt(b: bytes) -> bool:
                        if len(b) < 80:
                            return False
                        text = b.decode("utf-8", errors="ignore")
                        return "ATOM" in text or "HETATM" in text

                    def _to_bytes(v) -> Optional[bytes]:
                        if v is None:
                            return None
                        if isinstance(v, bytes):
                            return v if _looks_like_pdbqt(v) else None
                        if isinstance(v, str):
                            b = v.encode("utf-8")
                            return b if _looks_like_pdbqt(b) else None
                        return None

                    def _extract_pdbqt_from_obj(obj) -> Optional[bytes]:
                        if isinstance(obj, bytes):
                            return obj if _looks_like_pdbqt(obj) else None
                        if isinstance(obj, str):
                            return _to_bytes(obj)
                        if isinstance(obj, dict):
                            for key in ("pdbqt", "pdbqt_string", "protein", "receptor", "protein_pdbqt", "structure"):
                                out = _to_bytes(obj.get(key))
                                if out is not None:
                                    return out
                            for k, v in obj.items():
                                if isinstance(k, str) and ("pdbqt" in k.lower() or "protein" in k.lower() or "receptor" in k.lower()):
                                    out = _to_bytes(v)
                                    if out is not None:
                                        return out
                            for v in obj.values():
                                out = _to_bytes(v)
                                if out is not None:
                                    return out
                            return None
                        if isinstance(obj, (list, tuple)):
                            for item in obj:
                                out = _to_bytes(item) if isinstance(item, (str, bytes)) else _extract_pdbqt_from_obj(item)
                                if out is not None:
                                    return out
                            return None
                        if hasattr(obj, "__dict__"):
                            return _extract_pdbqt_from_obj(obj.__dict__)
                        return None

                    try:
                        obj = pickle.loads(raw)
                        protein_bytes = _extract_pdbqt_from_obj(obj)
                        if protein_bytes is None:
                            st.error("PKL 中未找到可识别的 PDBQT 内容（需包含 ATOM/HETATM 行）。")
                    except Exception as e:
                        st.error(f"解析 PKL 失败: {e}")
                        protein_bytes = None
                else:
                    protein_bytes = raw
                if protein_bytes is not None:
                    with st.spinner("对接中…"):
                        ok, result = run_docking(
                            protein_bytes,
                            ref_smi or "",
                            lig_smi or "",
                            vina_module_path,
                        )
                    if ok:
                        score = result
                        st.success(f"**对接分数**: {score:.2f} kcal/mol")
                        if score < -7:
                            st.caption("解读: 分数较好，可能具有较好结合亲和力。")
                        elif score < -5:
                            st.caption("解读: 分数中等，可结合实验进一步验证。")
                        else:
                            st.caption("解读: 分数偏弱，可尝试其他配体或优化。")
                    else:
                        st.error(f"对接失败: {result}")


def main() -> None:
    page = sidebar_nav()
    if page == PAGES[0]:
        page_sampling()
    elif page == PAGES[1]:
        page_training()
    elif page == PAGES[2]:
        page_evaluation()
    elif page == PAGES[3]:
        page_scoring_3d()
    else:
        page_docking()


if __name__ == "__main__":
    main()
