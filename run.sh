#!/bin/bash
source .venv/bin/activate
cd training

export CUDA_VISIBLE_DEVICES="1,2"

# ============================================================
# SPD 基模型重训（从头训练，使用 v1.9 修复后的推理代码）
# ============================================================
# nohup python new_train.py params_x1x3x4_diffusion_mosesaq_retrain > logs/retrain.log 2>&1 &

# ============================================================
# DPO v2.0 — Partial Denoising DPO（从 GT 加噪 t=0.5 出发）
# ============================================================
# nohup python DPO1_0_triSim.py params_x1x3x4_dpo_partial_denoise_nps 0 > logs/dpo_partial_denoise.log 2>&1 &

# ============================================================
# DPO 标准训练（三指标：Surf + ESP + Pharm）
# ============================================================
# nohup python DPO1_0_triSim.py params_x1x3x4_dpo_finetune_nps 0 > logs/dpo_triSim.log 2>&1 &

# ============================================================
# 可视化 DPO 训练指标
# ============================================================
# python visualize_dpo_metrics.py <json_path> --output dpo_metrics.png

# ============================================================
# 评估采样
# ============================================================
# cd /home1/zhh/workspace/SPD/evaluation/experiment_SamEval
# nohup python sample_NP.py > sample_NP.log 2>&1 &
