#!/bin/bashs
source .venv/bin/activate

# 切换到 training 目录
cd training

# 运行 Python 训练脚本
export CUDA_VISIBLE_DEVICES="1,2"

nohup python DPO1_0_partlyFrozen.py params_x1x3x4_dpo_finetune_nps 0 > logs/dpo_nps_0_partlyFrozen.log 2>&1 &

nohup python DPO1_0.py params_x1x3x4_dpo_finetune_nps 0 > logs/dpo_nps_0.log 2>&1 &

nohup python DPO1.py params_x1x3x4_dpo_finetune_nps 0 > logs/dpo_nps.log 2>&1 &

nohup python DPO2.py params_x1x3x4_dpo_finetune_pdb 0 > logs/dpo_pdb.log 2>&1 &

nohup python DPO3.py params_x1x3x4_dpo_fragment_merging 0 > logs/dpo_fragment_merging.log 2>&1 & 

---

cd /home1/zhh/workspace/SPD/evaluation/experiment

nohup python sample200.py > sample200.log 2>&1 &

---

# 杀掉所有你的 DPO 相关进程
pkill -9 -u zhh -f "DPO1_0_partlyFrozen.py"
# 检查是否清理干净
nvidia-smi

nohup python sample_fragment.py > fragDPO.log 2>&1 &

git fetch origin new-clean-branch

python visualize_dpo_metrics.py /home1/zhh/workspace/SPD/training/jobs/33/x1x3x4_dpo_finetune_nps/dpo_round_metrics.json --output dpo_metrics.png
