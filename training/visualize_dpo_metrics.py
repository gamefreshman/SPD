#!/usr/bin/env python3
"""
DPO 训练指标可视化脚本 (v2)

优化改进:
  1. EMA 平滑曲线 —— 为折线图叠加指数移动平均虚线
  2. 平坦/缺失指标自动检测 —— 方差接近 0 的子图标注警告
  3. Model vs Ref Loss Diff 使用 Symlog 坐标
  4. 评分曲线背景叠加 Pair 计数灰色柱形 —— 标示置信度

使用方式:
    python visualize_dpo_metrics.py <json_path> [--output <output_png>]
"""

import json
import argparse
import sys
import os

import matplotlib
matplotlib.use('Agg')
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


def ema(values, alpha=0.3):
    """指数移动平均"""
    result = []
    s = values[0] if len(values) > 0 else 0
    for v in values:
        s = alpha * v + (1 - alpha) * s
        result.append(s)
    return result


def is_flat(values, tol=1e-4):
    """检测数据是否平坦/全为 0（缺失数据）"""
    if len(values) == 0:
        return True
    arr = np.array(values, dtype=float)
    return np.std(arr) < tol


def add_pair_count_bars(ax, x, pair_counts, max_y=None):
    """在子图背景添加淡灰色 Pair 计数柱状图"""
    if max_y is None:
        max_y = ax.get_ylim()[1]
    # 将柱状高度归一化到 y 轴的约 30%
    max_pairs = max(pair_counts) if max(pair_counts) > 0 else 1
    bar_heights = [p / max_pairs * 0.3 * max_y for p in pair_counts]
    ax.bar(x, bar_heights, color='#BDBDBD', alpha=0.25, width=0.8,
           bottom=ax.get_ylim()[0], zorder=0)


def mark_flat(ax, label="(Data Missing / Flat)"):
    """在子图上叠加半透明蒙版和警告文字"""
    ax.patch.set_facecolor('#FFEBEE')
    ax.patch.set_alpha(0.5)
    ax.text(0.5, 0.5, label, transform=ax.transAxes,
            ha='center', va='center', fontsize=16, color='#B71C1C',
            fontweight='bold', alpha=0.6, zorder=10)


def plot_winner_loser(ax, x, x_labels, w_vals, l_vals, title, ylabel,
                      winner_color, loser_color, pair_counts=None,
                      target_band=None, ema_alpha=0.3):
    """通用的 Winner/Loser 折线图绘制（含 EMA 和平坦检测）"""
    ax.plot(x, w_vals, 'o-', color=winner_color, label='Winner', linewidth=2, markersize=5)
    ax.plot(x, l_vals, 's--', color=loser_color, label='Loser', linewidth=2, markersize=5)

    # EMA 平滑线
    if len(w_vals) >= 3:
        ax.plot(x, ema(w_vals, ema_alpha), '-', color=winner_color,
                alpha=0.35, linewidth=3, label='Winner EMA')
        ax.plot(x, ema(l_vals, ema_alpha), '-', color=loser_color,
                alpha=0.35, linewidth=3, label='Loser EMA')

    # 目标范围
    if target_band:
        ax.axhspan(*target_band, alpha=0.08, color='green', label=f'Target range {target_band}')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7)

    # 平坦检测
    if is_flat(w_vals) and is_flat(l_vals):
        mark_flat(ax)

    # 背景 Pair 计数
    if pair_counts is not None:
        add_pair_count_bars(ax, x, pair_counts)


def plot_metrics(metrics: list, output_path: str):
    """绘制所有指标图表"""
    rounds = [m['round'] for m in metrics]
    epochs = [m['epoch'] for m in metrics]
    pair_counts = [m.get('num_pairs', 1) for m in metrics]

    x = np.arange(len(rounds))
    x_labels = [f"R{r}\n(E{e})" for r, e in zip(rounds, epochs)]

    # 颜色方案
    W = '#2196F3'   # 蓝色 - Winner
    L = '#F44336'   # 红色 - Loser
    gap_color = '#4CAF50'
    loss_color = '#FF9800'
    dpo_color = '#9C27B0'
    acc_color = '#00BCD4'

    has_training_metrics = any(m.get('training_metrics', {}) for m in metrics)

    rows = 5 if has_training_metrics else 4
    fig, axes = plt.subplots(rows, 2, figsize=(17, 5.5 * rows))
    fig.suptitle('DPO Training Metrics per Round', fontsize=18, fontweight='bold', y=0.98)

    # ==================== 1. Surface Similarity ====================
    plot_winner_loser(
        axes[0, 0], x, x_labels,
        [m['winner'].get('sims_surf_target', 0) for m in metrics],
        [m['loser'].get('sims_surf_target', 0) for m in metrics],
        'Surface Similarity (sims_surf_target)', 'Score', W, L,
        pair_counts=pair_counts,
    )

    # ==================== 2. ESP Similarity ====================
    plot_winner_loser(
        axes[0, 1], x, x_labels,
        [m['winner'].get('sims_esp_target', 0) for m in metrics],
        [m['loser'].get('sims_esp_target', 0) for m in metrics],
        'ESP Similarity (sims_esp_target)', 'Score', W, L,
        pair_counts=pair_counts,
    )

    # ==================== 3. Pharmacophore Similarity ====================
    plot_winner_loser(
        axes[1, 0], x, x_labels,
        [m['winner'].get('sims_pharm_target', 0) for m in metrics],
        [m['loser'].get('sims_pharm_target', 0) for m in metrics],
        'Pharmacophore Similarity (sims_pharm_target)', 'Score', W, L,
        pair_counts=pair_counts,
    )

    # ==================== 4. SA Score ====================
    plot_winner_loser(
        axes[1, 1], x, x_labels,
        [m['winner'].get('sa_score', 0) for m in metrics],
        [m['loser'].get('sa_score', 0) for m in metrics],
        'SA Score (lower is better)', 'SA Score', W, L,
        pair_counts=pair_counts,
    )

    # ==================== 5. LogP ====================
    plot_winner_loser(
        axes[2, 0], x, x_labels,
        [m['winner'].get('logp', 0) for m in metrics],
        [m['loser'].get('logp', 0) for m in metrics],
        'LogP', 'LogP', W, L,
        pair_counts=pair_counts,
        target_band=(0.0, 6.0),
    )

    # ==================== 6. Total Score ====================
    ax = axes[2, 1]
    w_vals = [m['winner'].get('total_score', 0) for m in metrics]
    l_vals = [m['loser'].get('total_score', 0) for m in metrics]
    ax.plot(x, w_vals, 'o-', color=W, label='Winner', linewidth=2, markersize=5)
    ax.plot(x, l_vals, 's--', color=L, label='Loser', linewidth=2, markersize=5)
    if len(w_vals) >= 3:
        ax.plot(x, ema(w_vals), '-', color=W, alpha=0.35, linewidth=3, label='Winner EMA')
        ax.plot(x, ema(l_vals), '-', color=L, alpha=0.35, linewidth=3, label='Loser EMA')
    ax.fill_between(x, w_vals, l_vals, alpha=0.12, color=gap_color)
    ax.set_title('Total Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7)
    add_pair_count_bars(ax, x, pair_counts)

    # ==================== 7. Score Gap + Pair Count ====================
    ax = axes[3, 0]
    gaps = [m.get('score_gap', 0) for m in metrics]
    # 主柱：Gap（绿色）
    ax.bar(x, gaps, color=gap_color, alpha=0.7, edgecolor='white', linewidth=0.5, label='Score Gap')
    ax.plot(x, gaps, 'o-', color=gap_color, linewidth=2, markersize=5)
    for i, g in enumerate(gaps):
        ax.annotate(f'{g:.2f}', (x[i], g), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=7, fontweight='bold')
    # 双轴：Pair 计数
    ax2 = ax.twinx()
    ax2.bar(x + 0.3, pair_counts, width=0.25, color='#90A4AE', alpha=0.5, label='Num Pairs')
    ax2.set_ylabel('Num Pairs', color='#607D8B', fontsize=9)
    ax2.tick_params(axis='y', labelcolor='#607D8B')
    ax.set_title('Score Gap & Pair Count', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gap')
    ax.set_xlabel('Round (Epoch)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7)
    # 合并图例
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper right')

    # ==================== 8. Training Losses ====================
    ax = axes[3, 1]
    losses = [m.get('train_loss', None) for m in metrics]
    dpo_losses = [m.get('training_metrics', {}).get('loss_dpo', None) for m in metrics]
    std_losses = [m.get('training_metrics', {}).get('loss_std_on_winner', None) for m in metrics]

    def plot_series(ax, vals, color, label, marker='o-'):
        idx = [i for i, v in enumerate(vals) if v is not None]
        if len(idx) > 0:
            ys = [vals[i] for i in idx]
            ax.plot([x[i] for i in idx], ys, marker, color=color, linewidth=2, markersize=5, label=label)
            if len(ys) >= 3:
                ax.plot([x[i] for i in idx], ema(ys), '-', color=color, alpha=0.3, linewidth=3)

    plot_series(ax, losses, loss_color, 'Total Loss')
    plot_series(ax, dpo_losses, dpo_color, 'DPO Loss', 's--')
    plot_series(ax, std_losses, '#607D8B', 'Std Loss', '^:')

    has_any_loss = any(v is not None for v in losses + dpo_losses)
    if has_any_loss:
        ax.set_title('Training Losses', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Round (Epoch)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No loss data available\n(Round 0 has no loss)',
                transform=ax.transAxes, ha='center', va='center', fontsize=14, color='gray')
        ax.set_title('Training Losses', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7)

    # ==================== 9-10. DPO 训练指标 ====================
    if has_training_metrics:
        # 9. Implicit Accuracy + DPO Weight
        ax = axes[4, 0]
        accs = [m.get('training_metrics', {}).get('implicit_acc', None) for m in metrics]
        weights = [m.get('training_metrics', {}).get('dpo_weight', None) for m in metrics]

        plot_series(ax, accs, acc_color, 'Implicit Accuracy')
        plot_series(ax, weights, '#E91E63', 'DPO Weight', 's--')
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='50% baseline')
        ax.set_ylim(-0.05, 1.05)
        ax.set_title('Implicit Accuracy & DPO Weight', fontsize=12, fontweight='bold')
        ax.set_ylabel('Value')
        ax.set_xlabel('Round (Epoch)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7)

        # 10. Model vs Ref Loss Diff — 使用 Symlog
        ax = axes[4, 1]
        model_diffs = [m.get('training_metrics', {}).get('model_loss_diff', None) for m in metrics]
        ref_diffs = [m.get('training_metrics', {}).get('ref_loss_diff', None) for m in metrics]

        plot_series(ax, model_diffs, '#FF5722', 'Model Diff (w-l)')
        plot_series(ax, ref_diffs, '#795548', 'Ref Diff (w-l)', 's--')
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

        # Symlog 坐标：压缩异常尖峰，保留 0 附近细节
        ax.set_yscale('symlog', linthresh=1.0)
        ax.set_title('Model vs Ref Loss Diff (symlog)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss Diff')
        ax.set_xlabel('Round (Epoch)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图表已保存到: {output_path}")

    # ==================== 数据摘要表格 ====================
    print("\n" + "=" * 150)
    print(f"{'Round':>6} {'Epoch':>6} {'Status':>7} {'Pairs':>6} {'W_Surf':>8} {'L_Surf':>8} "
          f"{'W_Total':>8} {'L_Total':>8} {'Gap':>8} {'Loss':>10} {'DPO_Loss':>10} {'Acc':>6} "
          f"{'AvgScore':>9} {'BestScore':>10} {'RefUpd':>7}")
    print("-" * 150)
    for m in metrics:
        loss_str = f"{m['train_loss']:.4f}" if m.get('train_loss') is not None else "N/A"
        tm = m.get('training_metrics', {})
        dpo_str = f"{tm['loss_dpo']:.4f}" if tm.get('loss_dpo') is not None else "N/A"
        acc_str = f"{tm['implicit_acc']:.3f}" if tm.get('implicit_acc') is not None else "N/A"
        
        # Iterative DPO 字段
        status = m.get('status', 'ok')
        avg_score = m.get('current_avg_score')
        avg_str = f"{avg_score:.4f}" if avg_score is not None else "N/A"
        best_score = m.get('best_score')
        best_str = f"{best_score:.4f}" if best_score is not None else "N/A"
        ref_upd = m.get('ref_model_updated')
        ref_str = "✓" if ref_upd else ("✗" if ref_upd is not None else "N/A")
        
        # 状态标记
        status_str = status.upper() if status != 'ok' else 'OK'
        error_info = m.get('sampling_error', '')
        
        print(f"{m['round']:>6} {m['epoch']:>6} {status_str:>7} {m['num_pairs']:>6} "
              f"{m['winner'].get('sims_surf_target', 0):>8.4f} "
              f"{m['loser'].get('sims_surf_target', 0):>8.4f} "
              f"{m['winner'].get('total_score', 0):>8.3f} "
              f"{m['loser'].get('total_score', 0):>8.3f} "
              f"{m.get('score_gap', 0):>8.3f} "
              f"{loss_str:>10} "
              f"{dpo_str:>10} "
              f"{acc_str:>6} "
              f"{avg_str:>9} "
              f"{best_str:>10} "
              f"{ref_str:>7}")
        if error_info:
            print(f"       ⚠️  ERROR: {error_info}")
    print("=" * 150)
    
    # Iterative DPO 汇总
    ref_updates = [m for m in metrics if m.get('ref_model_updated') == True]
    errors = [m for m in metrics if m.get('status') == 'error']
    if ref_updates or errors:
        print(f"\n📊 Iterative DPO 汇总:")
        print(f"   参考模型更新次数: {len(ref_updates)}")
        if ref_updates:
            print(f"   更新 epochs: {[m['epoch'] for m in ref_updates]}")
        if errors:
            print(f"   ⚠️  采样失败次数: {len(errors)} (epochs: {[m['epoch'] for m in errors]})")


def main():
    parser = argparse.ArgumentParser(
        description="可视化 DPO 训练每轮的 winner/loser 指标"
    )
    parser.add_argument('json_path', type=str,
                        help='dpo_round_metrics.json 文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出图片路径（默认与 JSON 同目录，后缀改为 .png）')
    args = parser.parse_args()

    if args.output is None:
        base = os.path.splitext(args.json_path)[0]
        output_path = base + '.png'
    else:
        output_path = args.output

    metrics = load_metrics(args.json_path)
    print(f"📊 加载了 {len(metrics)} 轮的指标数据")

    plot_metrics(metrics, output_path)


if __name__ == '__main__':
    main()
