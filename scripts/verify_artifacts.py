"""Validate the distributed records against their inventory and paper protocol."""
import csv
import json
import math
from pathlib import Path

from make_manifest import ROOT, build_manifest


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    expected = json.loads((ROOT / "docs/artifacts.json").read_text(encoding="utf-8"))
    actual = build_manifest()
    for key in ("paper", "results", "missing_artifacts"):
        require(actual[key] == expected[key], f"Artifact inventory differs: {key}")
    for entry in actual["results"]["Figure 3"]:
        with (ROOT / entry["path"]).open(encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["type"] == "eval"]
        require([int(row["itr"]) for row in rows] == list(range(0, 601, 10)), f"Incomplete evaluation grid: {entry['path']}")
        require(all(math.isfinite(float(row["reward"])) for row in rows), f"Nonfinite reward: {entry['path']}")
    histories = {}
    for entry in actual["results"]["Figure 4"]:
        path = ROOT / entry["path"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        require(len(rows) == 80, f"Expected 80 updates: {path}")
        for row in rows:
            require(all(math.isfinite(float(row[key])) for key in ("monitor_reward_ode", "monitor_reward_sde", "monitor_val_fm_loss")),
                    f"Nonfinite CIFAR metric: {path}")
        histories.setdefault(path.parent.name, []).append(rows)
    for selection, counts in {"weak": (21, 45), "medium": (18, 43), "strong": (19, 46)}.items():
        runs = histories[selection]
        mean = {key: [sum(run[i][key] for run in runs) / len(runs) for i in range(80)]
                for key in ("monitor_reward_sde", "monitor_reward_ode")}
        deltas = {key: [values[i+1] - values[i] for i in range(79)] for key, values in mean.items()}
        pairs = list(zip(deltas["monitor_reward_sde"], deltas["monitor_reward_ode"]))
        observed = (sum(sde > 0 and ode < 0 for sde, ode in pairs), sum(sde > 0 for sde, ode in pairs))
        require(observed == counts, f"Violation statistic differs for {selection}: {observed}")
        print(f"{selection}: {observed[0]}/{observed[1]} = {observed[0]/observed[1]:.6f}")
    print("Verified paper hash, 60 MuJoCo CSVs, nine CIFAR histories, and reported violation counts.")


if __name__ == "__main__":
    main()
