import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

ODE_COLOR = "#CC6666"
SDE_COLOR = "#8B0000"
FM_WEAK_COLOR = "#FF0000"
FM_MEDIUM_COLOR = "#D6604D"
FM_STRONG_COLOR = "#7F0000"

SELECTIONS = ["weak", "medium", "strong"]
SELECTION_TITLE = {"weak": "Weak", "medium": "Medium", "strong": "Strong"}
FM_COLORS = {"weak": FM_WEAK_COLOR, "medium": FM_MEDIUM_COLOR, "strong": FM_STRONG_COLOR}

AXIS_LABEL_FONTSIZE = 28
TICK_LABEL_FONTSIZE = 22
LEGEND_FONTSIZE = 24


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stack_metric(histories: List[List[dict]], key: str) -> np.ndarray:
    rows = [[float(row[key]) for row in hist] for hist in histories]
    min_len = min(len(r) for r in rows)
    return np.array([r[:min_len] for r in rows], dtype=np.float64)


def plot_with_band(
    ax: plt.Axes,
    updates: np.ndarray,
    values: np.ndarray,
    label: str,
    color: str,
) -> None:
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    ax.plot(updates, mean, label=label, color=color, linewidth=2.6)
    ax.fill_between(updates, mean - std, mean + std, color=color, alpha=0.20, linewidth=0)


def plot_reward_panel(selection: str, histories: List[List[dict]], output_path: Path) -> None:
    ode = stack_metric(histories, "monitor_reward_ode")
    sde = stack_metric(histories, "monitor_reward_sde")
    n_updates = ode.shape[1]
    updates = np.arange(1, n_updates + 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    plot_with_band(ax, updates, ode, label="ODE", color=ODE_COLOR)
    plot_with_band(ax, updates, sde, label="SDE", color=SDE_COLOR)
    ax.set_xlabel("Fine-tune Update", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Evaluation Reward", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_fm_panel(per_selection_histories: Dict[str, List[List[dict]]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    n_seeds = None
    for selection in SELECTIONS:
        histories = per_selection_histories[selection]
        if not all("monitor_val_fm_loss" in row for hist in histories for row in hist):
            print(f"  warning: {selection} missing monitor_val_fm_loss; skipping")
            continue
        fm = stack_metric(histories, "monitor_val_fm_loss")
        if n_seeds is None:
            n_seeds = fm.shape[0]
        updates = np.arange(1, fm.shape[1] + 1)
        plot_with_band(ax, updates, fm, label=SELECTION_TITLE[selection], color=FM_COLORS[selection])
    ax.set_xlabel("Fine-tune Update", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Validation FM Loss", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(alpha=0.3)
    _ = n_seeds
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot multi-seed mean ± std trajectories.")
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("results/CIFAR_Exp_seed42"),
            Path("results/CIFAR_Exp_seed43"),
            Path("results/CIFAR_Exp_seed44"),
        ],
        help="One directory per seed; each must contain weak/medium/strong subdirs with history.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/CIFAR_Exp_plots"),
        help="Directory to write the four output figures into.",
    )
    args = parser.parse_args()

    per_selection: Dict[str, List[List[dict]]] = {sel: [] for sel in SELECTIONS}
    for run_dir in args.run_dirs:
        for selection in SELECTIONS:
            history_path = run_dir / selection / "history.jsonl"
            if not history_path.exists():
                raise FileNotFoundError(f"Missing {history_path}")
            per_selection[selection].append(load_jsonl(history_path))

    for selection in SELECTIONS:
        plot_reward_panel(
            selection=selection,
            histories=per_selection[selection],
            output_path=args.output_dir / f"reward_{selection}.pdf",
        )

    plot_fm_panel(per_selection, args.output_dir / "fm_loss.pdf")


if __name__ == "__main__":
    main()
