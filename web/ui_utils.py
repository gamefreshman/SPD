# -*- coding: utf-8 -*-
"""
Web 界面与样式工具：设计系统（COLORS/LAYOUT）、自定义 CSS、HTML 组件、3D 可视化渲染。
"""

from __future__ import annotations
from typing import List, Optional, Tuple

# ---------- 设计规范 (Design System) ----------
COLORS = {
    "bg": "#F8F9FA",
    "primary": "#00838F",
    "accent": "#6A1B9A",
    "card_bg": "#FFFFFF",
    "text": "#2C3E50",
    "border": "#E0E0E0",
    "badge_high": "#2E7D32",
    "badge_mid": "#F9A825",
    "badge_low": "#C62828",
    "viewer_3d_bg": "#F0F2F5",
    "status_panel_bg": "#E0F7FA",
    "subtitle_gray": "#78909C",
    "input_bg": "#FFFFFF",
    "input_border": "#E0E0E0",
    "input_text": "#333333",
    "button_gradient_end": "#006064",
    "expander_bg": "#F5F5F5",
    "text_secondary": "#666666",
    "placeholder": "#9E9E9E",
    "gallery_img_bg": "#FAFAFA",
    "table_even": "#F5F5F5",
    "table_hover": "#E8F4F4",
    "number_btn_bg": "#E8E8E8",
    "number_btn_hover": "#E0E0E0",
    "nav_label": "#333333",
    "nav_hover_bg": "rgba(0,131,143,0.06)",
    "download_btn_text": "#FFFFFF",
    "chart_axis_label": "#333333",
    "chart_axis_title": "#333333",
    "chart_tick": "#666666",
    "viewer_loading_text": "#666666",
    "primary_alpha_2": "rgba(0,131,143,0.2)",
    "primary_shadow_3": "rgba(0,131,143,0.3)",
    "primary_shadow_4": "rgba(0,131,143,0.4)",
    "shadow_light": "rgba(0,0,0,0.06)",
    "shadow_medium": "rgba(0,0,0,0.1)",
    "shadow_header": "rgba(0,0,0,0.04)",
    "progress_track": "rgba(0,0,0,0.08)",
}

LAYOUT = {
    "sidebar_logo_emoji_size": "2.25rem",
    "sidebar_logo_title_size": "28px",
    "sidebar_subtitle_size": "0.95rem",
    "border_radius_std": 8,
    "border_radius_card": 12,
    "border_radius_lg": 16,
    "parameter_card_padding": 20,
    "chart_spacer_margin_top": "1.75rem",
    "gallery_img_height": 150,
    "metric_card_icon_size": 48,
    "metric_card_padding": 20,
    "nav_btn_min_height": 44,
    "primary_btn_min_height": 44,
    "nav_btn_padding": "10px 14px",
    "nav_btn_gap": 4,
}


def get_custom_css() -> str:
    """
    返回覆盖 Streamlit 原生样式的自定义 CSS，实现现代生物医药科研平台风格。
    包含：全局去噪、侧边栏与导航美化、卡片容器、主按钮样式。
    """
    return f"""
    <style>
    /* 全局背景 */
    .stApp {{ background-color: {COLORS['bg']}; }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    [data-testid="stSidebar"] > div:first-child {{
        background-color: {COLORS['card_bg']};
        box-shadow: 2px 0 8px {COLORS['shadow_light']};
        border-right: 1px solid {COLORS['border']};
    }}
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div[data-testid="stMarkdown"] {{ color: {COLORS['text']} !important; }}
    [data-testid="stSidebar"] .stCaption {{ color: {COLORS['text']} !important; }}

    [data-testid="stSidebar"] [role="radiogroup"] {{
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        display: flex !important;
        flex-direction: column !important;
        gap: {LAYOUT['nav_btn_gap']}px !important;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label {{
        background: transparent !important;
        color: {COLORS['nav_label']} !important;
        border-radius: {LAYOUT['border_radius_std']}px !important;
        padding: {LAYOUT['nav_btn_padding']} !important;
        margin: 0 !important;
        border-left: 4px solid transparent !important;
        width: 100% !important;
        min-height: {LAYOUT['nav_btn_min_height']}px !important;
        display: flex !important;
        align-items: center !important;
        box-sizing: border-box !important;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
        background: {COLORS['nav_hover_bg']} !important;
    }}
    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"] {{
        background: {COLORS['status_panel_bg']} !important;
        border-left-color: {COLORS['primary']} !important;
        color: {COLORS['primary']} !important;
        font-weight: 700 !important;
    }}

    .stCard, .card-container {{
        background: {COLORS['card_bg']};
        border-radius: {LAYOUT['border_radius_card']}px;
        padding: 1.25rem;
        box-shadow: 0 2px 12px {COLORS['shadow_light']};
        border: 1px solid {COLORS['border']};
        margin-bottom: 1rem;
    }}

    .stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['button_gradient_end']} 100%) !important;
        color: {COLORS['text']} !important;
        border: none !important;
        border-radius: {LAYOUT['border_radius_std']}px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 2px 6px {COLORS['primary_shadow_3']};
    }}
    .stButton > button[kind="primary"]:hover, .stButton > button[data-testid="baseButton-primary"]:hover {{
        box-shadow: 0 4px 12px {COLORS['primary_shadow_4']};
        filter: brightness(1.05);
    }}

    h1, h2, h3 {{ color: {COLORS['primary']} !important; margin-bottom: 0.5rem !important; }}
    h2 {{ margin-bottom: 0.75rem !important; }}
    h3 {{ margin-bottom: 0.5rem !important; }}

    [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input,
    [data-testid="stTextInput"] textarea,
    [data-testid="stSelectbox"] div[role="combobox"],
    [data-testid="stSelectbox"] input {{
        background-color: {COLORS['input_bg']} !important;
        border: 1px solid {COLORS['input_border']} !important;
        color: {COLORS['input_text']} !important;
        border-radius: {LAYOUT['border_radius_std']}px !important;
    }}
    [data-testid="stNumberInput"] button {{
        background: {COLORS['number_btn_bg']} !important;
        color: {COLORS['input_text']} !important;
        border: 1px solid {COLORS['input_border']} !important;
    }}
    [data-testid="stNumberInput"] button:hover {{
        background: {COLORS['number_btn_hover']} !important;
    }}
    [data-testid="stNumberInput"] button:first-of-type {{ border-radius: 0 !important; border-left: none !important; }}
    [data-testid="stNumberInput"] button:last-of-type {{
        border-radius: 0 {LAYOUT['border_radius_std']}px {LAYOUT['border_radius_std']}px 0 !important;
    }}
    [data-testid="stNumberInput"] input {{ border-radius: {LAYOUT['border_radius_std']}px 0 0 {LAYOUT['border_radius_std']}px !important; }}
    [data-testid="stNumberInput"] input:focus, [data-testid="stTextInput"] input:focus,
    [data-testid="stTextInput"] textarea:focus,
    [data-testid="stSelectbox"] div[role="combobox"]:focus-within,
    [data-testid="stSelectbox"] input:focus {{
        border-color: {COLORS['primary']} !important;
        box-shadow: 0 0 0 1px {COLORS['primary']} !important;
    }}
    [data-testid="stSelectbox"] [data-testid="stSelectboxContainer"],
    [data-testid="stSelectbox"] > div,
    [data-testid="stSelectbox"] div[style*="border"] {{
        background: {COLORS['input_bg']} !important;
        border: 1px solid {COLORS['input_border']} !important;
        color: {COLORS['input_text']} !important;
        border-radius: {LAYOUT['border_radius_std']}px !important;
    }}
    [data-testid="stSelectbox"] input {{ color: {COLORS['input_text']} !important; }}

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] > div,
    [data-testid="stFileUploader"] [data-testid] {{
        background: {COLORS['input_bg']} !important;
        border: 1px solid {COLORS['input_border']} !important;
        color: {COLORS['input_text']} !important;
        border-radius: {LAYOUT['border_radius_std']}px !important;
    }}
    [data-testid="stFileUploader"] section [data-testid],
    [data-testid="stFileUploader"] div[data-testid] {{
        background: {COLORS['input_bg']} !important;
        color: {COLORS['input_text']} !important;
    }}
    [data-testid="stFileUploader"]:focus-within section {{
        border-color: {COLORS['primary']} !important;
        box-shadow: 0 0 0 1px {COLORS['primary']} !important;
    }}
    [data-testid="stVerticalBlock"]:has([data-testid="stFileUploader"]) > div:first-child {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 0 0.5rem 0 !important;
    }}

    [data-testid="stCheckbox"] label {{ color: {COLORS['input_text']} !important; }}

    [data-testid="stDownloadButton"] button,
    .stDownloadButton button {{
        background: {COLORS['primary']} !important;
        color: {COLORS['download_btn_text']} !important;
        border: none !important;
        border-radius: {LAYOUT['border_radius_std']}px !important;
        padding: 0.5rem 1rem !important;
        letter-spacing: 0.02em !important;
    }}
    [data-testid="stDownloadButton"] button:hover,
    .stDownloadButton button:hover {{ filter: brightness(1.08); }}

    [data-testid="stExpander"]:not(.gallery-card-wrapper *) summary,
    [data-testid="stExpander"]:not(.gallery-card-wrapper *) > div:first-child {{
        background: {COLORS['expander_bg']} !important;
        border: 1px solid {COLORS['input_border']} !important;
        border-radius: {LAYOUT['border_radius_std']}px !important;
        color: {COLORS['input_text']} !important;
    }}
    [data-testid="stExpander"]:not(.gallery-card-wrapper *) > div[role="button"] {{
        background: {COLORS['expander_bg']} !important;
        color: {COLORS['input_text']} !important;
    }}

    [data-testid="stVerticalBlock"]:has(.parameter-card) {{
        background: {COLORS['card_bg']} !important;
        padding: {LAYOUT['parameter_card_padding']}px !important;
        border-radius: {LAYOUT['border_radius_lg']}px !important;
        box-shadow: 0 4px 20px {COLORS['shadow_light']} !important;
        border: 1px solid {COLORS['border']} !important;
        margin-bottom: 1rem !important;
    }}
    .parameter-card {{ display: block; height: 0; overflow: hidden; margin: 0; padding: 0; border: none !important; }}

    [data-testid="stVerticalBlock"]:has(.parameter-card) .stButton > button[kind="primary"] {{
        width: 100% !important;
        min-height: {LAYOUT['primary_btn_min_height']}px !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.25rem !important;
        letter-spacing: 0.03em !important;
    }}

    .gallery-card {{
        background: {COLORS['card_bg']};
        border-radius: {LAYOUT['border_radius_card']}px;
        overflow: hidden;
        box-shadow: 0 2px 12px {COLORS['shadow_light']};
        border: 1px solid {COLORS['border']};
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 0.5rem;
    }}
    .gallery-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px {COLORS['shadow_medium']};
    }}
    .gallery-card-img-wrap {{
        height: {LAYOUT['gallery_img_height']}px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: {COLORS['gallery_img_bg']};
    }}
    .gallery-card-img {{
        object-fit: contain;
        max-height: {LAYOUT['gallery_img_height']}px;
        width: 100%;
    }}
    .card-smiles {{
        font-family: monospace;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-size: 0.75rem;
        color: {COLORS['text']};
        padding: 4px 8px;
    }}
    .gallery-card-wrapper [data-testid="stExpander"] {{
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }}
    .gallery-card-wrapper [data-testid="stExpander"] summary {{
        border-radius: 6px;
        background: {COLORS['bg']};
    }}

    .sidebar-status-panel {{
        background: {COLORS['status_panel_bg']};
        border-radius: 10px;
        padding: 0.75rem;
        margin-top: 0.5rem;
        border: 1px solid {COLORS['primary_alpha_2']};
        box-shadow: 0 2px 8px {COLORS['shadow_header']};
    }}
    .sidebar-status-panel .status-row {{ display: flex; justify-content: space-between; align-items: center; margin: 6px 0; font-size: 0.8rem; color: {COLORS['input_text']}; }}
    .sidebar-status-panel .status-row .label {{ color: {COLORS['text_secondary']}; }}
    .sidebar-status-panel .status-progress {{ height: 6px; background: {COLORS['progress_track']}; border-radius: 3px; overflow: hidden; margin-top: 4px; }}
    .sidebar-status-panel .status-progress-fill {{ height: 100%; background: {COLORS['primary']}; border-radius: 3px; transition: width 0.3s ease; }}

    .gallery-card-wrapper [data-testid="stExpander"],
    .gallery-card-wrapper .streamlit-expander {{
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
    }}
    .gallery-card-wrapper [data-testid="stExpander"] > div {{
        border: none !important;
        background: transparent !important;
    }}
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stCaption,
    [data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] span,
    [data-testid="stAppViewContainer"] label {{ color: {COLORS['text']} !important; }}
    [data-testid="stAppViewContainer"] .stNumberInput label, [data-testid="stAppViewContainer"] .stTextInput label,
    [data-testid="stAppViewContainer"] .stSelectbox label, [data-testid="stAppViewContainer"] .stCheckbox label {{ color: {COLORS['text']} !important; }}

    .metric-big {{ font-size: 1.75rem; font-weight: 700; color: {COLORS['primary']}; }}

    .chart-spacer {{ margin-top: {LAYOUT['chart_spacer_margin_top']} !important; display: block !important; }}
    </style>
    """


def parameter_card_marker_html() -> str:
    """返回用于触发参数区白卡样式的占位 div（class=parameter-card）。"""
    return '<div class="parameter-card"></div>'
def chart_spacer_html() -> str:
    """返回评估页指标与图表之间的留白 div。"""
    return '<div class="chart-spacer"></div>'


def smiles_display_html(smiles: str, max_len: int = 60, extra_style: str = "margin-top:2px;") -> str:
    """
    生成 SMILES 单行省略展示的 HTML（card-smiles 样式，带 title 完整值）。
    """
    s = (smiles or "").strip()
    title = s.replace('"', "&quot;")
    display = (s[:max_len] + ("..." if len(s) > max_len else "")) if s else "—"
    style_attr = f' style="{extra_style}"' if extra_style else ""
    return f'<div class="card-smiles"{style_attr} title="{title}">{display}</div>'


def badge_color_for_value(value: float, low_is_bad: bool = True, thresholds: Optional[Tuple[float, float]] = None) -> str:
    """
    根据数值返回徽章颜色。
    """
    if thresholds is not None:
        low_ok, high_ok = thresholds
        if value < low_ok:
            return COLORS["badge_low"] if low_is_bad else COLORS["badge_high"]
        if value > high_ok:
            return COLORS["badge_high"] if low_is_bad else COLORS["badge_low"]
        return COLORS["badge_mid"]
    if value >= 0.5:
        return COLORS["badge_high"] if low_is_bad else COLORS["badge_low"]
    return COLORS["badge_low"] if low_is_bad else COLORS["badge_high"]


def badge_html(label: str, value: str, color_hex: str) -> str:
    """生成胶囊型徽章 HTML（Pill-shaped, 10px 加粗）。"""
    return f'<span style="display:inline-block;background:{color_hex};color:{COLORS["text"]};padding:2px 10px;border-radius:999px;font-size:10px;font-weight:700;margin:2px;">{label}: {value}</span>'


def render_metric_card(label: str, value: str, icon: str = "📊") -> str:
    """
    评估页大屏指标卡片：白底、左侧圆形图标、右侧数值。
    返回 HTML，用于 st.markdown(..., unsafe_allow_html=True)。
    """
    icon_size = LAYOUT["metric_card_icon_size"]
    padding = LAYOUT["metric_card_padding"]
    return f'''
    <div class="metric-card" style="background:{COLORS['card_bg']};border-radius:{LAYOUT['border_radius_card']}px;padding:{padding}px;box-shadow:0 2px 12px {COLORS['shadow_light']};border:1px solid {COLORS['border']};display:flex;align-items:center;gap:16px;">
        <div style="width:{icon_size}px;height:{icon_size}px;border-radius:50%;background:{COLORS['primary']}15;display:flex;align-items:center;justify-content:center;font-size:1.5rem;flex-shrink:0;">{icon}</div>
        <div style="flex:1;min-width:0;">
            <div style="font-size:0.8rem;color:{COLORS['text_secondary']};margin-bottom:4px;">{label}</div>
            <div style="font-size:1.5rem;font-weight:700;color:{COLORS['primary']};">{value}</div>
        </div>
    </div>
    '''


def render_metric_cards_row(items: List[Tuple[str, str, str]]) -> List[str]:
    """
    为评估页生成一排指标卡片的 HTML 列表（每项 label, value, icon）。
    """
    return [render_metric_card(label, value, icon) for label, value, icon in items]


def render_molecule_card(
    img_base64: Optional[str],
    badges_inner_html: str,
    smiles_text: str,
    card_label: str = "",
) -> str:
    """
    生成单张分子卡片的完整 HTML（2D 图固定高度、徽章、SMILES 单行省略）。
    用于 Gallery 网格；3D 部分由调用方用 st.expander 接在下方。
    """
    smiles_esc = (smiles_text or "").replace("<", "&lt;").replace(">", "&gt;")
    if img_base64:
        body = f'<div class="gallery-card-img-wrap"><img src="data:image/png;base64,{img_base64}" class="gallery-card-img" alt="{card_label}" /></div>'
    else:
        body = f'<div class="gallery-card-img-wrap" style="align-items:center;justify-content:center;color:{COLORS["placeholder"]};font-size:0.85rem;">{card_label or "无 2D 图"}</div>'
    return f'<div class="gallery-card">{body}<div style="padding:6px 8px;">{badges_inner_html}</div><div class="card-smiles" title="{smiles_esc}">{smiles_esc or "—"}</div></div>'


def _wrap_3d_with_loading(html: str, width: int, height: int) -> str:
    """在 3D 视图 HTML 外包裹加载占位（Loading 文案 + 1.5s 后隐藏）。"""
    loading = (
        '<div class="viewer-3d-loading" style="'
        f"position:absolute;top:0;left:0;right:0;bottom:0;background:{COLORS['viewer_3d_bg']};"
        f"display:flex;align-items:center;justify-content:center;font-size:14px;color:{COLORS['viewer_loading_text']};z-index:2;"
        '">Loading...</div>'
    )
    script = (
        "<script>setTimeout(function(){"
        'var e=document.querySelector(".viewer-3d-loading");if(e)e.style.display="none";'
        "},1500);</script>"
    )
    return (
        f'<div class="viewer-3d-wrapper" style="position:relative;min-width:{width}px;min-height:{height}px;">'
        + loading
        + html
        + script
        + "</div>"
    )


def render_3d_mol_html(
    xyz_or_mol_block: str,
    style: str = "stick",
    width: int = 400,
    height: int = 400,
    background_color: Optional[str] = None,
) -> str:
    """
    使用 py3Dmol 生成分子 3D 显示的 HTML 字符串，供 st.components.v1.html 使用。
    """
    try:
        import py3Dmol
    except ImportError:
        return "<p>需要安装 py3Dmol: pip install py3Dmol</p>"

    view = py3Dmol.view(width=width, height=height)
    first_line = (xyz_or_mol_block.strip().split("\n")[0] or "").strip()
    if first_line.isdigit():
        view.addModel(xyz_or_mol_block, "xyz")
    else:
        view.addModel(xyz_or_mol_block, "sdf")
    if style == "stick":
        view.setStyle({"stick": {"radius": 0.2}, "sphere": {"scale": 0.25, "opacity": 0.7}})
    else:
        view.setStyle({style: {}})
    if background_color:
        hex_val = background_color.lstrip("#")
        if len(hex_val) == 6:
            view.setBackgroundColor(f"0x{hex_val}")
    view.zoomTo()
    return view.write_html()


def render_3d_two_mols_synced(
    xyz1: str,
    xyz2: str,
    style: str = "stick",
    width: int = 500,
    height: int = 450,
    offset_second: float = 12.0,
    background_color: Optional[str] = None,
) -> str:
    """
    在同一 3D 视图中添加两个分子，实现同步旋转。
    第二个分子沿 x 轴平移 offset_second，避免重叠。
    """
    try:
        import py3Dmol
    except ImportError:
        return "<p>需要安装 py3Dmol</p>"

    def shift_xyz(xyz: str, dx: float) -> str:
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
    if style == "stick":
        view.setStyle({"stick": {"radius": 0.2}, "sphere": {"scale": 0.25, "opacity": 0.7}})
    else:
        view.setStyle({style: {}})
    if background_color:
        hex_val = background_color.lstrip("#")
        if len(hex_val) == 6:
            view.setBackgroundColor(f"0x{hex_val}")
    view.zoomTo()
    return view.write_html()


def safe_render_3d_mol_html(
    xyz_or_mol_block: str,
    style: str = "stick",
    width: int = 400,
    height: int = 400,
    background_color: Optional[str] = None,
) -> str:
    """带异常捕获的 3D 分子 HTML 渲染；外裹 Loading 占位；失败时返回简短错误提示 HTML。"""
    try:
        html = render_3d_mol_html(xyz_or_mol_block, style=style, width=width, height=height, background_color=background_color)
        if not html or "需要安装" in html or "失败" in html or "异常" in html:
            return html or "<p>3D 渲染失败</p>"
        return _wrap_3d_with_loading(html, width, height)
    except Exception:
        return "<p>3D 渲染异常，请检查分子数据。</p>"


def safe_render_3d_two_mols_synced(
    xyz1: str,
    xyz2: str,
    style: str = "stick",
    width: int = 500,
    height: int = 450,
    offset_second: float = 12.0,
    background_color: Optional[str] = None,
) -> str:
    """带异常捕获的双分子 3D 同步视图；失败时返回错误提示 HTML。"""
    try:
        return render_3d_two_mols_synced(
            xyz1, xyz2, style=style, width=width, height=height,
            offset_second=offset_second, background_color=background_color,
        )
    except Exception:
        return "<p>3D 双分子渲染异常。</p>"
