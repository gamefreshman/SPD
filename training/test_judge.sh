#!/bin/bash
# DPO Judge 测试脚本 - 使用UV虚拟环境

echo "=================================================="
echo "DPO分子评估测试 (UV环境)"
echo "=================================================="

# # 评估output_all_mols0.json中的分子
# python dpo_judge.py /home1/zhh/workspace/SPD/training/jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251207_210610.json \
#     --output output1.json \
#     --verbose \
#     --top-k 20

# echo ""
# echo "✅ 测试完成"
# echo "查看结果文件: output1.json"

# # 评估output_all_mols0.json中的分子
# python dpo_judge.py /home1/zhh/workspace/SPD/training/jobs/x1x3x4_dpo_finetune_nps/generated_mols_20251207_212615.json \
#     --output output2.json \
#     --verbose \
#     --top-k 20

# echo ""
# echo "✅ 测试完成"
# echo "查看结果文件: output2.json"

python dpo_judge.py /home1/zhh/workspace/SPD/training/output_all_mols.json \
    --output output0.json \
    --verbose \
    --top-k 20

echo ""
echo "✅ 测试完成"
echo "查看结果文件: output0.json"