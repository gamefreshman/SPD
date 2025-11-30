#!/bin/bashs
source .venv/bin/activate

# 切换到 training 目录
cd training

# 运行 Python 训练脚本
export CUDA_VISIBLE_DEVICES="1,2"

nohup python dpo_trainer.py params_x1x3x4_dpo_finetune_nps 0 &
