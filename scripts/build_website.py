"""Build the static GitHub Pages artifact from website/ and canonical paper assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from statistics import fmean, pstdev

ROOT = Path(__file__).resolve().parents[1]


def build_trajectory_data():
    """Aggregate the distributed CIFAR histories for the browser viewer."""
    seeds = [42, 43, 44]
    result = {"seeds": seeds, "aggregation": "Mean and population standard deviation across seeds before taking reward differences.", "models": {}}
    keys = {"sde": "monitor_reward_sde", "ode": "monitor_reward_ode", "fm_loss": "monitor_val_fm_loss"}
    for selection in ("weak", "medium", "strong"):
        paths = [ROOT / f"CIFAR32Exp/results/CIFAR_Exp_seed{seed}/{selection}/history.jsonl" for seed in seeds]
        runs = [[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] for path in paths]
        updates = [row["update"] for row in runs[0]]
        if any([row["update"] for row in run] != updates for run in runs) or updates != list(range(1, 81)):
            raise ValueError(f"Incomplete or mismatched recorded update grid: {selection}")
        model = {"updates": updates, "source_paths": [path.relative_to(ROOT).as_posix() for path in paths]}
        for label, key in keys.items():
            values = [[float(run[i][key]) for run in runs] for i in range(len(updates))]
            model[label] = {"mean": [fmean(row) for row in values], "std": [pstdev(row) for row in values]}
        deltas = [(model["sde"]["mean"][i+1] - model["sde"]["mean"][i],
                   model["ode"]["mean"][i+1] - model["ode"]["mean"][i]) for i in range(len(updates)-1)]
        model["sde_improvements"] = sum(sde > 0 for sde, ode in deltas)
        model["violations"] = sum(sde > 0 and ode < 0 for sde, ode in deltas)
        model["violation_rate"] = model["violations"] / model["sde_improvements"]
        result["models"][selection] = model
    return result


def build(out):
    source = ROOT / "website"
    config = json.loads((source / "site.json").read_text(encoding="utf-8"))
    provenance = json.loads((source / "assets/provenance.json").read_text(encoding="utf-8"))
    paper = ROOT / "docs/paper.pdf"
    if hashlib.sha256(paper.read_bytes()).hexdigest() != provenance["paper_sha256"]:
        raise ValueError("The paper changed. Refresh the website figures and their provenance before publishing.")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "style.css", "app.js", "citation.bib"):
        text = (source / name).read_text(encoding="utf-8")
        text = text.replace("{{SITE_URL}}", config["site_url"]).replace("{{REPOSITORY_URL}}", config["repository_url"])
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    shutil.copytree(source / "assets", out / "assets", dirs_exist_ok=True)
    (out / "assets/cifar-trajectories.json").write_text(json.dumps(build_trajectory_data(), separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    # Fingerprint the CSS and script URLs so repeat visitors receive a restyle immediately.
    html_path = out / "index.html"
    html = html_path.read_text(encoding="utf-8")
    for name, attr in (("style.css", "href"), ("app.js", "src")):
        version = hashlib.sha256((out / name).read_bytes()).hexdigest()[:12]
        html = html.replace(f'{attr}="{name}"', f'{attr}="{name}?v={version}"')
    html_path.write_text(html, encoding="utf-8", newline="\n")
    shutil.copy2(paper, out / "paper.pdf")
    shutil.copy2(ROOT / "docs/assets/framework.png", out / "assets/framework.png")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Built {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / ".site")
    args = parser.parse_args()
    build(args.out.resolve())


if __name__ == "__main__":
    main()
