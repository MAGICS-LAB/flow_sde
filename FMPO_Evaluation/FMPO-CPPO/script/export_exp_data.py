"""
Export gym fine-tuning runs into the exp_data/ layout used by plot_gym_tasks.py,
together with the provenance of every curve.

For each run directory (a Hydra logdir containing result.pkl and .hydra/config.yaml):
  exp_data/<task>_<method>/seed_<seed>_progress.csv     eval curve (type,itr,reward,std,total_step_M)
  exp_data/<task>_<method>/seed_<seed>_config.yaml      resolved Hydra config of the run
  exp_data/<task>_<method>/seed_<seed>_provenance.json  overrides, git commit, sha256 of result.pkl,
                                                        base checkpoint sha256, latency summary
and a summary of the training-time latency per method and task in exp_data/latency.csv.

Usage (from the FMPO-CPPO directory):
  python script/export_exp_data.py --runs "$REINFLOW_LOG_DIR/gym/finetune/*/*" --out ../exp_data
  python script/export_exp_data.py --split-noise-level --runs "<search run dirs>" --out <dir>   # noise-level search
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import pickle
import re
import subprocess
from collections import defaultdict

import numpy as np
import yaml

TASK_DIRS = {
    "hopper-medium-v2": "hopperv2",
    "walker2d-medium-v2": "walker2d",
    "ant-medium-expert-v2": "antv2",
    "Humanoid-medium-v3": "humanoidv3",
}


def sha256(path):
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand_env(path):
    """Resolve ${oc.env:VAR} interpolations left in the saved Hydra config."""
    if not isinstance(path, str):
        return path
    return re.sub(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.environ.get(m.group(1), m.group(0)), path)


def git_commit(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def method_name(model_cfg):
    target = model_cfg.get("_target_", "")
    if target.endswith("ppoflow.PPOFlow"):
        return "reinflow"
    if not model_cfg.get("use_chain_ppo_loss", False):
        return "FlowSDE"
    if float(model_cfg.get("chain_ppo_aux_coef", 0.25)) == 0.0:
        return "chainppo_noaux"
    return "chainppo"


def export_run(run_dir, out_root, repo_dir, split_noise_level=False):
    result_path = os.path.join(run_dir, "result.pkl")
    config_path = os.path.join(run_dir, "resolved_config.yaml")
    if not os.path.isfile(config_path):
        config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not (os.path.isfile(result_path) and os.path.isfile(config_path)):
        return None
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    with open(result_path, "rb") as handle:
        run_results = pickle.load(handle)

    task = TASK_DIRS.get(cfg["env_name"], cfg["env_name"].replace("-", ""))
    method = method_name(cfg["model"])
    seed = int(cfg["seed"])
    folder = f"{task}_{method}"
    if split_noise_level and method != "reinflow":
        folder += f"_a{cfg['model'].get('noise_level_a')}"
        method = folder.split("_", 1)[1]
    out_dir = os.path.join(out_root, folder)
    os.makedirs(out_dir, exist_ok=True)

    rows, train_times = [], []
    for record in run_results:
        if "eval/avg episode reward" in record:
            rows.append(
                {
                    "type": "eval",
                    "itr": record["itr"],
                    "reward": round(float(record["eval/avg episode reward"]), 2),
                    "std": round(float(record["eval/std episode reward"]), 2),
                    "total_step_M": round(record["step"] / 1e6, 6),
                }
            )
        elif "time" in record and record["itr"] > 0:
            train_times.append(float(record["time"]))

    with open(os.path.join(out_dir, f"seed_{seed}_progress.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["type", "itr", "reward", "std", "total_step_M"])
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, f"seed_{seed}_config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    overrides_path = os.path.join(run_dir, ".hydra", "overrides.yaml")
    overrides = None
    if os.path.isfile(overrides_path):
        with open(overrides_path, "r", encoding="utf-8") as handle:
            overrides = yaml.safe_load(handle)
    latency = {
        "train_iterations_timed": len(train_times),
        "mean_seconds_per_train_iteration": float(np.mean(train_times)) if train_times else None,
        "std_seconds_per_train_iteration": float(np.std(train_times)) if train_times else None,
    }
    metadata_path = os.path.join(run_dir, "run_metadata.json")
    training_metadata = None
    if os.path.isfile(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            training_metadata = json.load(handle)
    provenance = {
        "run_dir": os.path.abspath(run_dir),
        "task": task,
        "method": method,
        "seed": seed,
        "overrides": overrides,
        "training_metadata": training_metadata,
        "git_commit": training_metadata.get("git_commit") if training_metadata else None,
        "export_git_commit": git_commit(repo_dir),
        "result_pkl_sha256": sha256(result_path),
        "base_policy_path": expand_env(cfg.get("base_policy_path")),
        "base_policy_sha256": (training_metadata.get("base_policy_sha256") if training_metadata
                               else sha256(expand_env(cfg.get("base_policy_path")))),
        "num_eval_points": len(rows),
        "latency": latency,
    }
    with open(os.path.join(out_dir, f"seed_{seed}_provenance.json"), "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
    return task, method, seed, len(rows), latency


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories or glob patterns.")
    parser.add_argument("--out", default="../exp_data", help="Output exp_data directory.")
    parser.add_argument("--split-noise-level", action="store_true",
                        help="Write one folder per noise level (<task>_<method>_a<value>), e.g. for search_noise_level.sh runs.")
    args = parser.parse_args()

    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_dirs = sorted({d for pattern in args.runs for d in glob.glob(pattern) if os.path.isdir(d)})
    latency = defaultdict(list)
    exported = 0
    for run_dir in run_dirs:
        result = export_run(run_dir, args.out, repo_dir, split_noise_level=args.split_noise_level)
        if result is None:
            continue
        task, method, seed, n_points, run_latency = result
        exported += 1
        print(f"{task:>11s} {method:>15s} seed {seed}: {n_points} eval points  <- {run_dir}")
        if run_latency["mean_seconds_per_train_iteration"] is not None:
            latency[(method, task)].append(run_latency["mean_seconds_per_train_iteration"])

    if not exported:
        parser.error("No exportable runs found. Check --runs and the presence of result.pkl and config files.")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "latency.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "task", "num_runs", "mean_seconds_per_train_iteration", "std_across_runs"])
        for (method, task), values in sorted(latency.items()):
            writer.writerow([method, task, len(values), f"{np.mean(values):.3f}", f"{np.std(values):.3f}"])
    print(f"Exported {exported} runs to {args.out}")


if __name__ == "__main__":
    main()
