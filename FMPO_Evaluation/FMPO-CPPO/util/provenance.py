"""Capture the code and inputs at launch time for newly trained Gym runs."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone

from omegaconf import OmegaConf


def file_sha256(path):
    path = Path(path) if path else None
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(root, *args):
    try:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def save_run_metadata(cfg):
    root = Path(__file__).resolve().parents[1]
    source = hashlib.sha256()
    sources = [p for folder in ("agent", "env", "model", "util", "script", "cfg")
               for p in (root / folder).rglob("*")
               if p.is_file() and p.suffix in {".py", ".yaml", ".sh"}]
    sources.append(root / "pyproject.toml")
    for path in sorted(sources):
        source.update(path.relative_to(root).as_posix().encode())
        source.update(bytes.fromhex(file_sha256(path)))
    versions = {}
    for package in ("torch", "numpy", "gym", "hydra-core"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    status = git_output(root, "status", "--porcelain", "--untracked-files=normal")
    metadata = {
        "captured_at_launch_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output(root, "rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "source_sha256": source.hexdigest(),
        "source_file_count": len(sources),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "device": str(cfg.device),
        "base_policy_path": cfg.get("base_policy_path"),
        "base_policy_sha256": file_sha256(cfg.get("base_policy_path")),
    }
    out = Path(cfg.logdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (out / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
