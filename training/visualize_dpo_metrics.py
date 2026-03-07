#!/usr/bin/env python3
"""
DPO 训练指标可视化脚本

读取 dpo_round_metrics.json，绘制每轮训练的 winner/loser 指标对比图、
total score 变化、score gap 趋势和训练损失曲线。

使用方式:
    python visualize_dpo_metrics.py <json_path> [--output <output_png>]

示例:
    python visualize_dpo_metrics.py jobs/33/dpo_output/dpo_round_metrics.json
    python visualize_dpo_metrics.py dpo_round_metrics.json --output my_plot.png
"""

import json
import argparse
import sys

import matplotlib
matplotlib.use('Agg')  # 无需 GUI 后端
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_metrics(json_path: str) -> list:
    """加载 JSON 指标文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        print("❌ JSON 文件为空或格式不正确")
        sys.exit(1)
    return data


def plot_metrics(metrics: list, output_path: str):
    """绘制所有指标图表"""
    rounds = [m['round'] for m in metrics]
    epochs = [m['epoch'] for m in metrics]
    
    # 使用 round 作为 x 轴标签
    x = np.arange(len(rounds))
    x_labels = [f"R{r}\n(E{e})" for r, e in zip(rounds, epochs)]
    
    # ==================== 图表配置 ====================
    # 6 个子图：5 个指标 + 1 个 total_score + score_gap + train_loss = 8 个
    # 布局：4 行 2 列
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle('DPO Training Metrics per Round', fontsize=18, fontweight='bold', y=0.98)
    
    # 颜色方案
    winner_color = '#2196F3'   # 蓝色 - Winner
    loser_color = '#F44336'    # 红色 - Loser
    gap_color = '#4CAF50'      # 绿色 - Gap
    loss_color = '#FF9800'     # 橙色 - Loss
    
    # ==================== 1. Surface Similarity ====================
    ax = axes[0, 0]
    w_vals = [m['winner'].get('sims_surf_target', 0) for m in metrics]
    l_vals = [m['loser'].get('sims_surf_target', 0) for m in metrics]
    ax.plot(x, w_vals, 'o-', color=winner_color, label='Winner', linewidth=2, markersize=6)
    ax.plot(x, l_vals, 's--', color=loser_color, label='Loser', linewidth=2, markersize=6)
    ax.set_title('Surface Similarity (sims_surf_target)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    
    # ==================== 2. ESP Similarity ====================
    ax = axes[0, 1]
    w_vals = [m['winner'].get('sims_esp_target', 0) for m in metrics]
    l_vals = [m['loser'].get('sims_esp_target', 0) for m in metrics]
    ax.plot(x, w_vals, 'o-', color=winner_color, label='Winner', linewidth=2, markersize=6)
    ax.plot(x, l_vals, 's--', color=loser_color, label='Loser', linewidth=2, markersize=6)
    ax.set_title('ESP Similarity (sims_esp_target)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    
    # ==================== 3. Pharmacophore Similarity ====================
    ax = axes[1, 0]
    w_vals = [m['winner'].get('sims_pharm_target', 0) for m in metrics]
    l_vals = [m['loser'].get('sims_pharm_target', 0) for m in metrics]
    ax.plot(x, w_vals, 'o-', color=winner_color, label='Winner', linewidth=2, markersize=6)
    ax.plot(x, l_vals, 's--', color=loser_color, label='Loser', linewidth=2, markersize=6)
    ax.set_title('Pharmacophore Similarity (sims_pharm_target)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    
    # ==================== 4. SA Score ====================
    ax = axes[1, 1]
    w_vals = [m['winner'].get('sa_score', 0) for m in metrics]
    l_vals = [m['loser'].get('sa_score', 0) for m in metrics]
    ax.plot(x, w_vals, 'o-', color=winner_color, label='Winner', linewidth=2, markersize=6)
    ax.plot(x, l_vals, 's--', color=loser_color, label='Loser', linewidth=2, markersize=6)
    ax.set_title('SA Score (lower is better)', fontsize=12, fontweight='bold')
    ax.set_ylabel('SA Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    
    # ==================== 5. LogP ====================
    ax = axes[2, 0]
    w_vals = [m['winner'].get('logp', 0) for m in metrics]
    l_vals = [m['loser'].get('logp', 0) for m in metrics]
    ax.plot(x, w_vals, 'o-', color=winner_color, label='Winner', linewidth=2, markersize=6)
    ax.plot(x, l_vals, 's--', color=loser_color, label='Loser', linewidth=2, markersize=6)
    # 目标范围标注
    ax.axhspan(0.0, 6.0, alpha=0.08, color='green', label='Target range (0~6)')
    ax.set_title('LogP', fontsize=12, fontweight='bold')
    ax.set_ylabel('LogP')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    
    # ==================== 6. Total Score ====================
    ax = axes[2, 1]
    w_vals = [m['winner'].get('total_score', 0) for m in metrics]
    l_vals = [m['loser'].get('total_score', 0) for m in metrics]
    ax.plot(x, w_vals, 'o-', color=winner_color, label='Winner', linewidth=2, markersize=6)
    ax.plot(x, l_vals, 's--', color=loser_color, label='Loser', linewidth=2, markersize=6)
    ax.fill_between(x, w_vals, l_vals, alpha=0.15, color=gap_color)
    ax.set_title('Total Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    
    # ==================== 7. Score Gap ====================
    ax = axes[3, 0]
    gaps = [m.get('score_gap', 0) for m in metrics]
    ax.bar(x, gaps, color=gap_color, alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.plot(x, gaps, 'o-', color=gap_color, linewidth=2, markersize=6)
    # 在柱上标注数值
    for i, g in enumerate(gaps):
        ax.annotate(f'{g:.2f}', (x[i], g), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=8, fontweight='bold')
    ax.set_title('Score Gap (Winner - Loser)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gap')
    ax.set_xlabel('Round (Epoch)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    
    # ==================== 8. Train Loss ====================
    ax = axes[3, 1]
    losses = [m.get('train_loss', None) for m in metrics]
    valid_idx = [i for i, l in enumerate(losses) if l is not None]
    valid_losses = [losses[i] for i in valid_idx]
    
    if len(valid_losses) > 0:
        ax.plot([x[i] for i in valid_idx], valid_losses, 'o-', 
                color=loss_color, linewidth=2, markersize=6)
        ax.fill_between([x[i] for i in valid_idx], valid_losses, 
                        alpha=0.15, color=loss_color)
        ax.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Round (Epoch)')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No loss data available\n(Round 0 has no loss)', 
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='gray')
        ax.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图表已保存到: {output_path}")
    
    # 同时打印数据摘要表格
    print("\n" + "=" * 90)
    print(f"{'Round':>6} {'Epoch':>6} {'Pairs':>6} {'W_Surf':>8} {'L_Surf':>8} "
          f"{'W_Total':>8} {'L_Total':>8} {'Gap':>8} {'Loss':>10}")
    print("-" * 90)
    for m in metrics:
        loss_str = f"{m['train_loss']:.4f}" if m.get('train_loss') is not None else "N/A"
        print(f"{m['round']:>6} {m['epoch']:>6} {m['num_pairs']:>6} "
              f"{m['winner'].get('sims_surf_target', 0):>8.4f} "
              f"{m['loser'].get('sims_surf_target', 0):>8.4f} "
              f"{m['winner'].get('total_score', 0):>8.3f} "
              f"{m['loser'].get('total_score', 0):>8.3f} "
              f"{m.get('score_gap', 0):>8.3f} "
              f"{loss_str:>10}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="可视化 DPO 训练每轮的 winner/loser 指标"
    )
    parser.add_argument('json_path', type=str,
                        help='dpo_round_metrics.json 文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出图片路径（默认与 JSON 同目录，后缀改为 .png）')
    args = parser.parse_args()
    
    # 确定输出路径
    if args.output is None:
        import os
        base = os.path.splitext(args.json_path)[0]
        output_path = base + '.png'
    else:
        output_path = args.output
    
    # 加载数据
    metrics = load_metrics(args.json_path)
    print(f"📊 加载了 {len(metrics)} 轮的指标数据")
    
    # 绘制图表
    plot_metrics(metrics, output_path)


if __name__ == '__main__':
    main()
