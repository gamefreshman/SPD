#!/bin/bash

# 激活虚拟环境
source .venv/bin/activate

# 切换到 training 目录
cd training

# 运行 Python 训练脚本
python new_train.py params_x1x3x4_diffusion_mosesaq_20240824 0