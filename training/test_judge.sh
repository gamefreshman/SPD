#!/bin/bash
# DPO Judge 测试脚本 - 使用UV虚拟环境

echo "=================================================="
echo "DPO分子评估测试 (UV环境)"
echo "=================================================="

# 评估output_all_mols0.json中的分子
python dpo_judge.py output_all_mols0.json \
    --output output_all_mols0_evaluated.json \
    --verbose \
    --top-k 20

echo ""
echo "✅ 测试完成"
echo "查看结果文件: output_all_mols0_evaluated.json"
