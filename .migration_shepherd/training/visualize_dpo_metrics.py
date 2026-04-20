#!/usr/bin/env python3
"""
DPO training metrics visualization for Shepherd.

Usage:
    python visualize_dpo_metrics.py <json_path> [--output <output_png>]
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(json_path):
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not data:
        print("JSON file is empty or malformed.", file=sys.stderr)
        sys.exit(1)
    return data


def ema(values, alpha=0.3):
    if not values:
        return []
    result = []
    state = values[0]
    for value in values:
        state = alpha * value + (1.0 - alpha) * state
        result.append(state)
    return result


def is_flat(values, tol=1e-4):
    valid = [value for value in values if value is not None]
    if not valid:
        return True
    return float(np.std(np.asarray(valid, dtype=float))) < tol


def add_pair_count_bars(ax, x, pair_counts, max_y=None):
    if max_y is None:
        max_y = ax.get_ylim()[1]
    max_pairs = max(pair_counts) if pair_counts and max(pair_counts) > 0 else 1
    bar_heights = [(count / max_pairs) * 0.3 * max_y for count in pair_counts]
    ax.bar(
        x,
        bar_heights,
        color="#BDBDBD",
        alpha=0.25,
        width=0.8,
        bottom=ax.get_ylim()[0],
        zorder=0,
    )


def mark_flat(ax, label="(Data Missing / Flat)"):
    ax.patch.set_facecolor("#FFEBEE")
    ax.patch.set_alpha(0.5)
    ax.text(
        0.5,
        0.5,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        color="#B71C1C",
        fontweight="bold",
        alpha=0.6,
        zorder=10,
    )


def plot_series(ax, x, values, color, label, marker="o-", ema_alpha=0.3):
    indices = [idx for idx, value in enumerate(values) if value is not None]
    if not indices:
        return False
    xs = [x[idx] for idx in indices]
    ys = [values[idx] for idx in indices]
    ax.plot(xs, ys, marker, color=color, linewidth=2, markersize=5, label=label)
    if len(ys) >= 3:
        ax.plot(xs, ema(ys, ema_alpha), "-", color=color, alpha=0.3, linewidth=3)
    return True


def fill_between_valid(ax, x, first, second, **kwargs):
    indices = [
        idx
        for idx, (first_value, second_value) in enumerate(zip(first, second))
        if first_value is not None and second_value is not None
    ]
    if not indices:
        return
    xs = [x[idx] for idx in indices]
    ys_first = [first[idx] for idx in indices]
    ys_second = [second[idx] for idx in indices]
    ax.fill_between(xs, ys_first, ys_second, **kwargs)


def plot_winner_loser(
    ax,
    x,
    x_labels,
    winner_values,
    loser_values,
    title,
    ylabel,
    winner_color,
    loser_color,
    pair_counts=None,
    target_band=None,
    ema_alpha=0.3,
):
    has_winner = plot_series(ax, x, winner_values, winner_color, "Winner", ema_alpha=ema_alpha)
    has_loser = plot_series(
        ax,
        x,
        loser_values,
        loser_color,
        "Loser",
        marker="s--",
        ema_alpha=ema_alpha,
    )

    if target_band is not None:
        ax.axhspan(*target_band, alpha=0.08, color="green", label=f"Target range {target_band}")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7)

    if pair_counts is not None:
        add_pair_count_bars(ax, x, pair_counts)

    if not has_winner and not has_loser:
        mark_flat(ax)
    elif is_flat(winner_values) and is_flat(loser_values):
        mark_flat(ax)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7, loc="best")


def get_training_metric(metric, key):
    metrics = metric.get("training_metrics") or {}
    return metrics.get(key)


def fmt(value, precision):
    if value is None:
        return "N/A"
    return format(value, precision)


def fmt_percent(value):
    if value is None:
        return "N/A"
    return f"{value:.0f}%"


def plot_metrics(metrics, output_path):
    rounds = [metric["round"] for metric in metrics]
    epochs = [metric["epoch"] for metric in metrics]
    pair_counts = [metric.get("num_pairs", 0) for metric in metrics]
    x = np.arange(len(rounds))
    x_labels = [f"R{round_idx}\n(E{epoch})" for round_idx, epoch in zip(rounds, epochs)]

    winner_color = "#2196F3"
    loser_color = "#F44336"
    gap_color = "#4CAF50"
    loss_color = "#FF9800"
    dpo_color = "#9C27B0"
    acc_color = "#00BCD4"

    has_training_metrics = any(metric.get("training_metrics") for metric in metrics)
    rows = 6 if has_training_metrics else 5
    fig, axes = plt.subplots(rows, 2, figsize=(17, 5.5 * rows))
    fig.suptitle("Shepherd DPO Training Metrics", fontsize=18, fontweight="bold", y=0.98)

    plot_winner_loser(
        axes[0, 0],
        x,
        x_labels,
        [metric["winner"].get("sims_surf_target") for metric in metrics],
        [metric["loser"].get("sims_surf_target") for metric in metrics],
        "Surface Similarity",
        "Score",
        winner_color,
        loser_color,
        pair_counts=pair_counts,
    )
    plot_winner_loser(
        axes[0, 1],
        x,
        x_labels,
        [metric["winner"].get("sims_esp_target") for metric in metrics],
        [metric["loser"].get("sims_esp_target") for metric in metrics],
        "ESP Similarity",
        "Score",
        winner_color,
        loser_color,
        pair_counts=pair_counts,
    )
    plot_winner_loser(
        axes[1, 0],
        x,
        x_labels,
        [metric["winner"].get("sims_pharm_target") for metric in metrics],
        [metric["loser"].get("sims_pharm_target") for metric in metrics],
        "Pharmacophore Similarity",
        "Score",
        winner_color,
        loser_color,
        pair_counts=pair_counts,
    )
    plot_winner_loser(
        axes[1, 1],
        x,
        x_labels,
        [metric["winner"].get("sa_score") for metric in metrics],
        [metric["loser"].get("sa_score") for metric in metrics],
        "SA Score (lower is better)",
        "SA Score",
        winner_color,
        loser_color,
        pair_counts=pair_counts,
    )
    plot_winner_loser(
        axes[2, 0],
        x,
        x_labels,
        [metric["winner"].get("logp") for metric in metrics],
        [metric["loser"].get("logp") for metric in metrics],
        "LogP",
        "LogP",
        winner_color,
        loser_color,
        pair_counts=pair_counts,
        target_band=(0.0, 6.0),
    )

    ax = axes[2, 1]
    winner_total = [metric["winner"].get("total_score") for metric in metrics]
    loser_total = [metric["loser"].get("total_score") for metric in metrics]
    plot_series(ax, x, winner_total, winner_color, "Winner")
    plot_series(ax, x, loser_total, loser_color, "Loser", marker="s--")
    fill_between_valid(ax, x, winner_total, loser_total, alpha=0.12, color=gap_color)
    ax.set_title("Total Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7)
    add_pair_count_bars(ax, x, pair_counts)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7, loc="best")
    if is_flat(winner_total) and is_flat(loser_total):
        mark_flat(ax)

    ax = axes[3, 0]
    gaps = [metric.get("score_gap") for metric in metrics]
    gap_indices = [idx for idx, value in enumerate(gaps) if value is not None]
    if gap_indices:
        gap_x = [x[idx] for idx in gap_indices]
        gap_y = [gaps[idx] for idx in gap_indices]
        ax.bar(
            gap_x,
            gap_y,
            color=gap_color,
            alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
            label="Score Gap",
        )
        ax.plot(gap_x, gap_y, "o-", color=gap_color, linewidth=2, markersize=5)
        for idx, value in zip(gap_x, gap_y):
            ax.annotate(
                f"{value:.2f}",
                (idx, value),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
                fontweight="bold",
            )
    else:
        mark_flat(ax)
    ax2 = ax.twinx()
    ax2.bar(x + 0.3, pair_counts, width=0.25, color="#90A4AE", alpha=0.5, label="Num Pairs")
    ax2.set_ylabel("Num Pairs", color="#607D8B", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#607D8B")
    ax.set_title("Score Gap & Pair Count", fontsize=12, fontweight="bold")
    ax.set_ylabel("Gap")
    ax.set_xlabel("Round (Epoch)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7)
    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    if handles_1 or handles_2:
        ax.legend(handles_1 + handles_2, labels_1 + labels_2, fontsize=7, loc="upper right")

    ax = axes[3, 1]
    total_losses = [get_training_metric(metric, "train_loss") for metric in metrics]
    dpo_losses = [get_training_metric(metric, "loss_dpo") for metric in metrics]
    std_losses = [get_training_metric(metric, "loss_std_on_winner") for metric in metrics]
    plotted = False
    plotted |= plot_series(ax, x, total_losses, loss_color, "Total Loss")
    plotted |= plot_series(ax, x, dpo_losses, dpo_color, "DPO Loss", marker="s--")
    plotted |= plot_series(ax, x, std_losses, "#607D8B", "Std Loss", marker="^:")
    ax.set_title("Training Losses", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Round (Epoch)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7)
    if plotted:
        ax.legend(fontsize=7)
    else:
        mark_flat(ax)

    if has_training_metrics:
        ax = axes[4, 0]
        implicit_acc = [get_training_metric(metric, "implicit_acc") for metric in metrics]
        dpo_weights = [get_training_metric(metric, "dpo_weight") for metric in metrics]
        plotted = False
        plotted |= plot_series(ax, x, implicit_acc, acc_color, "Implicit Accuracy")
        plotted |= plot_series(ax, x, dpo_weights, "#E91E63", "DPO Weight", marker="s--")
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="50% baseline")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Implicit Accuracy & DPO Weight", fontsize=12, fontweight="bold")
        ax.set_ylabel("Value")
        ax.set_xlabel("Round (Epoch)")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7)
        if plotted:
            ax.legend(fontsize=7)
        else:
            mark_flat(ax)

        ax = axes[4, 1]
        model_diff = [get_training_metric(metric, "model_loss_diff") for metric in metrics]
        ref_diff = [get_training_metric(metric, "ref_loss_diff") for metric in metrics]
        plotted = False
        plotted |= plot_series(ax, x, model_diff, "#FF5722", "Model Diff (w-l)")
        plotted |= plot_series(ax, x, ref_diff, "#795548", "Ref Diff (w-l)", marker="s--")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title("Model vs Ref Loss Diff (symlog)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Loss Diff")
        ax.set_xlabel("Round (Epoch)")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7)
        if plotted:
            ax.legend(fontsize=7)
        else:
            mark_flat(ax)

    validity_row = 5 if has_training_metrics else 4
    ax = axes[validity_row, 0]
    validity_rates = []
    for metric in metrics:
        validity = metric.get("validity_stats") or {}
        value = validity.get("validity_rate")
        validity_rates.append(value * 100.0 if value is not None else None)

    valid_indices = [idx for idx, value in enumerate(validity_rates) if value is not None]
    if valid_indices:
        xs = [x[idx] for idx in valid_indices]
        ys = [validity_rates[idx] for idx in valid_indices]
        ax.bar(xs, ys, color="#26A69A", alpha=0.6, edgecolor="white", linewidth=0.5)
        ax.plot(xs, ys, "o-", color="#00796B", linewidth=2, markersize=6, label="Validity Rate")
        if len(ys) >= 3:
            ax.plot(xs, ema(ys), "-", color="#00796B", alpha=0.35, linewidth=3, label="EMA")
        for idx, value in zip(valid_indices, ys):
            validity = metrics[idx].get("validity_stats") or {}
            ax.annotate(
                f"{value:.0f}%\n({validity.get('num_valid', '?')}/{validity.get('num_total', '?')})",
                (x[idx], value),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=7,
                fontweight="bold",
            )
        ax.legend(fontsize=7)
    else:
        mark_flat(ax)
    ax.set_title("Molecule Validity Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validity (%)")
    ax.set_xlabel("Round (Epoch)")
    ax.set_ylim(0, 105)
    ax.axhline(y=50, color="gray", linestyle=":", alpha=0.4)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7)

    ax = axes[validity_row, 1]
    winner_sa = [metric["winner"].get("sa_score") for metric in metrics]
    loser_sa = [metric["loser"].get("sa_score") for metric in metrics]
    plotted = False
    plotted |= plot_series(ax, x, winner_sa, winner_color, "Winner SA")
    plotted |= plot_series(ax, x, loser_sa, loser_color, "Loser SA", marker="s--")
    fill_between_valid(ax, x, winner_sa, loser_sa, alpha=0.1, color="#4CAF50")
    ax.axhspan(1, 4, alpha=0.06, color="green", label="Easy to synthesize (1-4)")
    ax.set_title("SA Score (lower = easier to synthesize)", fontsize=12, fontweight="bold")
    ax.set_ylabel("SA Score")
    ax.set_xlabel("Round (Epoch)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7)
    if plotted:
        ax.legend(fontsize=7, loc="best")
    else:
        mark_flat(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")

    print("\n" + "=" * 148)
    print(
        f"{'Round':>6} {'Epoch':>6} {'Status':>7} {'Pairs':>6} {'Valid%':>7} "
        f"{'W_Surf':>8} {'L_Surf':>8} {'W_SA':>7} {'L_SA':>7} "
        f"{'W_Total':>8} {'L_Total':>8} {'Gap':>8} {'Loss':>10} "
        f"{'DPO_Loss':>10} {'Acc':>6} {'AvgScore':>9} {'RefUpd':>7}"
    )
    print("-" * 148)
    for metric in metrics:
        validity = metric.get("validity_stats") or {}
        print(
            f"{metric['round']:>6} "
            f"{metric['epoch']:>6} "
            f"{metric.get('status', 'ok').upper():>7} "
            f"{metric.get('num_pairs', 0):>6} "
            f"{fmt_percent(validity.get('validity_rate') * 100.0 if validity.get('validity_rate') is not None else None):>7} "
            f"{fmt(metric['winner'].get('sims_surf_target'), '.4f'):>8} "
            f"{fmt(metric['loser'].get('sims_surf_target'), '.4f'):>8} "
            f"{fmt(metric['winner'].get('sa_score'), '.2f'):>7} "
            f"{fmt(metric['loser'].get('sa_score'), '.2f'):>7} "
            f"{fmt(metric['winner'].get('total_score'), '.3f'):>8} "
            f"{fmt(metric['loser'].get('total_score'), '.3f'):>8} "
            f"{fmt(metric.get('score_gap'), '.3f'):>8} "
            f"{fmt(get_training_metric(metric, 'train_loss'), '.4f'):>10} "
            f"{fmt(get_training_metric(metric, 'loss_dpo'), '.4f'):>10} "
            f"{fmt(get_training_metric(metric, 'implicit_acc'), '.3f'):>6} "
            f"{fmt(metric.get('avg_score'), '.4f'):>9} "
            f"{('✓' if metric.get('ref_model_updated') else '✗'):>7}"
        )
        if metric.get("sampling_error"):
            print(f"       ERROR: {metric['sampling_error']}")
    print("=" * 148)

    ref_updates = [metric for metric in metrics if metric.get("ref_model_updated")]
    errors = [metric for metric in metrics if metric.get("status") == "error"]
    empties = [metric for metric in metrics if metric.get("status") == "empty"]
    if ref_updates or errors or empties:
        print("\nSummary:")
        print(f"  ref model updates: {len(ref_updates)}")
        print(f"  error rounds: {len(errors)}")
        print(f"  empty rounds: {len(empties)}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Shepherd DPO metrics")
    parser.add_argument("json_path", type=str, help="Path to dpo_round_metrics.json")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output image path (defaults to <json>.png)",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = os.path.splitext(args.json_path)[0] + ".png"

    metrics = load_metrics(args.json_path)
    print(f"Loaded {len(metrics)} rounds from {args.json_path}")
    plot_metrics(metrics, output_path)


if __name__ == "__main__":
    main()
