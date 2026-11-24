"""Inventory paper artifacts and optional checkpoints with SHA-256 hashes."""
import argparse
from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATTERNS = {
    "Figure 2(a)": ["NumericalExp/theorem4_1_results.json"],
    "Figure 2(b)": ["NumericalExp/violation_rate_results.json"],
    "Figure 2(c)": ["NumericalExp/trajectory_results.json"],
    "Figure 3": [f"FMPO_Evaluation/exp_data/{task}_{method}/seed_*_progress.csv"
                 for task in ("antv2", "hopperv2", "humanoidv3", "walker2d")
                 for method in ("FlowSDE", "reinflow", "chainppo")],
    "Figure 4": ["CIFAR32Exp/results/CIFAR_Exp_seed*/**/history.jsonl"],
    "Figure 5 (NoAux inputs)": ["FMPO_Evaluation/exp_data/*_chainppo_noaux/seed_*_progress.csv"],
    "Table 2": ["FMPO_Evaluation/exp_data/latency.csv"],
}
EXPECTED = {"Figure 2(a)": 1, "Figure 2(b)": 1, "Figure 2(c)": 1, "Figure 3": 60,
            "Figure 4": 9, "Figure 5 (NoAux inputs)": 20, "Table 2": 1}


def file_entry(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    try:
        label = path.relative_to(ROOT).as_posix()
    except ValueError:
        label = str(path)
    return {"path": label, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def expand(patterns):
    paths = {Path(p).resolve() for pattern in patterns
             for p in glob.glob(str(ROOT / pattern), recursive=True) if Path(p).is_file()}
    return [file_entry(path) for path in sorted(paths)]


def build_manifest(checkpoints=()):
    results = {figure: expand(patterns) for figure, patterns in RESULT_PATTERNS.items()}
    missing = {figure: {"expected_files": EXPECTED[figure], "found_files": len(entries)}
               for figure, entries in results.items() if len(entries) != EXPECTED[figure]}
    return {"paper": file_entry(ROOT / "docs/paper.pdf"), "results": results,
            "missing_artifacts": missing, "checkpoints": expand(checkpoints)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="*", default=[], help="Checkpoint paths or glob patterns.")
    parser.add_argument("--out", type=Path, default=ROOT / "MANIFEST.json")
    args = parser.parse_args()
    manifest = build_manifest(args.checkpoints)
    manifest["generated_utc"] = datetime.now(timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for figure, entries in manifest["results"].items():
        print(f"{figure}: {len(entries)}/{EXPECTED[figure]} files")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
