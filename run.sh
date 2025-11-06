#!/bin/bashs
source .venv/bin/activate

# 切换到 training 目录
cd training

# 运行 Python 训练脚本
CUDA_VISIBLE_DEVICES="0,1"

python new_train.py params_x1x3x4_dpo_finetune_nps 0
