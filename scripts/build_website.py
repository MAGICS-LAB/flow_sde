"""Build the static GitHub Pages artifact from website/ and canonical paper assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


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
