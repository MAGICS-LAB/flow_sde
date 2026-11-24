import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

WEAK_COLOR = "#C97C5D"
MEDIUM_COLOR = "#7A9E9F"
STRONG_COLOR = "#2F5D62"
OVERALL_COLOR = "#1F4E79"

SELECTIONS = ["weak", "medium", "strong"]
SELECTION_TITLE = {"weak": "Weak", "medium": "Medium", "strong": "Strong"}
SELECTION_COLOR = {"weak": WEAK_COLOR, "medium": MEDIUM_COLOR, "strong": STRONG_COLOR}


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stack_metric(histories: List[List[dict]], key: str) -> np.ndarray:
    rows = [[float(row[key]) for row in hist] for hist in histories]
    min_len = min(len(r) for r in rows)
    return np.array([r[:min_len] for r in rows], dtype=np.float64)


def seed_averaged_deltas(run_dirs: List[Path]) -> Dict[str, Dict[str, np.ndarray]]:
    """Returns per-selection seed-averaged trajectories and their deltas:
        {sel: {"sde": Δ̄, "ode": Δ̄, "sde_mean": x̄, "ode_mean": x̄}}
    where x̄ has shape (n_updates,) and Δ̄ has shape (n_updates - 1,)."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for selection in SELECTIONS:
        histories = [load_jsonl(rd / selection / "history.jsonl") for rd in run_dirs]
        sde = stack_metric(histories, "monitor_reward_sde").mean(axis=0)
        ode = stack_metric(histories, "monitor_reward_ode").mean(axis=0)
        out[selection] = {
            "sde_mean": sde,
            "ode_mean": ode,
            "sde": np.diff(sde),
            "ode": np.diff(ode),
        }
    return out


def conditional_summary(sde_delta: np.ndarray, ode_delta: np.ndarray) -> Dict[str, float]:
    sde_up = sde_delta > 0
    ode_up = ode_delta > 0
    n_events = int(sde_delta.size)
    n_sde = int(sde_up.sum())
    n_both = int((sde_up & ode_up).sum())
    rate = n_both / n_sde if n_sde > 0 else float("nan")
    return {
        "n_events": n_events,
        "n_sde_improve": n_sde,
        "n_both_improve": n_both,
        "p_ode_given_sde": rate,
        "p_sde_improve": n_sde / n_events if n_events else float("nan"),
    }


def plot_delta_signs(
    averaged: Dict[str, Dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(SELECTIONS), 1, figsize=(9.0, 6.0), sharex=True)
    for ax, sel in zip(axes, SELECTIONS):
        sde = averaged[sel]["sde"]
        ode = averaged[sel]["ode"]
        x = np.arange(2, 2 + sde.size)
        sde_up = sde > 0
        ode_up = ode > 0

        ax.fill_between(x, 0, 1, where=sde_up & ode_up, step="mid",
                        color=SELECTION_COLOR[sel], alpha=0.35,
                        label="SDE↑ & ODE↑")
        ax.fill_between(x, 0, 1, where=sde_up & ~ode_up, step="mid",
                        color=SELECTION_COLOR[sel], alpha=0.10,
                        label="SDE↑ & ODE↓")
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        s = conditional_summary(sde, ode)
        ax.set_title(f"{SELECTION_TITLE[sel]}: P(ODE↑ | SDE↑) = "
                     f"{s['n_both_improve']}/{s['n_sde_improve']} = {s['p_ode_given_sde']:.3f}")
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Fine-tune Update t")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P(ODE improves | SDE improves) per-update across seeds.",
    )
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("results/CIFAR_Exp_seed42"),
            Path("results/CIFAR_Exp_seed43"),
            Path("results/CIFAR_Exp_seed44"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/CIFAR_Exp_plots"),
    )
    args = parser.parse_args()

    averaged = seed_averaged_deltas(args.run_dirs)

    summary = {sel: conditional_summary(averaged[sel]["sde"], averaged[sel]["ode"]) for sel in SELECTIONS}
    sde_all = np.concatenate([averaged[s]["sde"] for s in SELECTIONS])
    ode_all = np.concatenate([averaged[s]["ode"] for s in SELECTIONS])
    summary["pooled"] = conditional_summary(sde_all, ode_all)

    print("\nP(ODE improves | SDE improves) on seed-averaged trajectories (Δ vs. previous update):")
    print(f"  {'group':<10} {'n_events':>10} {'n_sde+':>10} {'n_both+':>10} {'P(ODE+|SDE+)':>14} {'P(SDE+)':>10}")
    for key in [*SELECTIONS, "pooled"]:
        s = summary[key]
        print(f"  {key:<10} {s['n_events']:>10} {s['n_sde_improve']:>10} {s['n_both_improve']:>10} "
              f"{s['p_ode_given_sde']:>14.3f} {s['p_sde_improve']:>10.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "ode_given_sde_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary to {summary_path}")

    plot_delta_signs(averaged, args.output_dir / "p_ode_given_sde.pdf")


if __name__ == "__main__":
    main()
