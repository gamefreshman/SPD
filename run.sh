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
# DPO v2.1 — 修复超参数重启训练
# dpo_max_weight=0.05, ramp_up=30, real_data_ratio=0.85
# num_samples=32, score_gap=0.10, 保护机制已验证可用
# ============================================================
nohup python DPO1_0_triSim.py params_x1x3x4_dpo_finetune_nps 0 > logs/dpo_v2.1_triSim.log 2>&1 &

# ============================================================
# 可视化 DPO 训练指标
# ============================================================
# python visualize_dpo_metrics.py /home1/zhh/workspace/SPD/training/jobs/33/x1x3x4_dpo_finetune_nps/dpo_round_metrics.json --output dpo_metrics.png

# ============================================================
# 评估采样（当前正在运行 epoch-009 大规模采样）
# ============================================================
# cd /home1/zhh/workspace/SPD/evaluation/experiment_SamEval
# nohup python sample_NP.py > sample_NP.log 2>&1 &
# nohup python sample_NP.py > sample.log 2>&1 &