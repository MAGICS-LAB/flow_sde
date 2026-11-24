"""Recreate Figures 3 and 4 from the distributed logs without training."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args):
    script = ROOT / script
    subprocess.run([sys.executable, str(script), *map(str, args)], cwd=script.parent, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["all", "mujoco", "cifar"], default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--ablation", action="store_true", help="Also recreate Figure 5; requires additional NoAux logs.")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.suite in ("all", "mujoco"):
        run("FMPO_Evaluation/plot_gym_tasks.py", "--output-dir", out / "mujoco")
        if args.ablation:
            run("FMPO_Evaluation/plot_gym_tasks.py", "--output-dir", out / "noaux", "--ablation")
    if args.suite in ("all", "cifar"):
        run("CIFAR32Exp/plot_multiseed_trajectories.py", "--output-dir", out / "cifar")
        run("CIFAR32Exp/analyze_ode_sde_conditional.py", "--output-dir", out / "cifar")
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
