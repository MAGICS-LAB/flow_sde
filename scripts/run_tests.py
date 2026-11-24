"""Run independent experiment test suites in separate Python processes."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = {"mujoco": ROOT / "FMPO_Evaluation/FMPO-CPPO", "cifar": ROOT / "CIFAR32Exp",
          "numerical": ROOT / "NumericalExp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["all", *SUITES], default="all")
    args, pytest_args = parser.parse_known_args()
    for name, directory in SUITES.items():
        if args.suite not in ("all", name):
            continue
        print(f"Running {name} tests", flush=True)
        subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", *pytest_args], cwd=directory, check=True)


if __name__ == "__main__":
    main()
