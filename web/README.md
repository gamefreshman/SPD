# web/

本文件夹为 **SPD 分子生成与评估** 的 Web 前端，基于 Streamlit 实现科研人员交互界面，提供分子生成、训练与实验查看、评估与报告、3D 相似度对比、分子对接等功能。界面风格统一为现代生物医药科研平台样式，结果通过 `st.session_state` 在页面间共享。

---

## 文件说明

| 文件 | 说明 |
|------|------|
| **app.py** | Streamlit 主入口。负责页面配置、侧边栏导航、Session State 初始化，以及各功能页的编排与调用（分子生成、训练与实验、评估与报告、3D 相似度、分子对接）。 |
| **ui_utils.py** | 界面与样式工具。包含设计系统（`COLORS`、`LAYOUT`）、全局自定义 CSS（`get_custom_css`）、参数区/图表留白/SMILES 展示等 HTML 组件、指标卡片与分子卡片渲染、3D 分子视图（py3Dmol）的封装与安全渲染。 |
| **backend_utils.py** | 后端与数据工具。包含项目路径与设备选择、侧边栏状态 HTML、检查点扫描与可读标签、模型加载与推理（shepherd）、分子数据转换（RDKit/sample↔SMILES/XYZ）、2D 图生成、分子对接（vina_dock）、训练/评估结果目录列表等。 |
| **requirements_web.txt** | Web 前端额外依赖（Streamlit、py3Dmol、pandas、openpyxl 等）。主项目已有依赖（如 torch、rdkit）需另行满足。 |
| **README.md** | 本说明文件。 |
| **contest/** | 竞赛提交材料：作品报告、作品信息概要表、配图说明及生图脚本（见 `contest/README.md`）。 |

---

## 运行方式

在项目根目录下执行：

```bash
streamlit run web/app.py
```

运行前请确保已安装 `requirements_web.txt` 中的依赖及项目主环境（如 `src`、torch、rdkit 等）。
