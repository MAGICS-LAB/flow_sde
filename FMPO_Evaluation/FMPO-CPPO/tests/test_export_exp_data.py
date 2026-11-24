"""The exporter rebuilds exp_data CSVs (plot_gym_tasks.py format) with provenance from Hydra run dirs."""
import csv
import importlib.util
import json
import os
import pickle
from pathlib import Path

import yaml

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "script", "export_exp_data.py")
spec = importlib.util.spec_from_file_location("export_exp_data", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def make_run(root, name, model, seed, ckpt):
    run = root / name
    (run / ".hydra").mkdir(parents=True)
    cfg = {"env_name": "hopper-medium-v2", "seed": seed, "base_policy_path": "${oc.env:FAKE_LOG_DIR}/base.pt", "model": model}
    (run / ".hydra" / "config.yaml").write_text(yaml.safe_dump(cfg))
    (run / ".hydra" / "overrides.yaml").write_text(yaml.safe_dump([f"seed={seed}"]))
    results = []
    for itr in range(0, 21):
        rec = {"itr": itr, "step": itr * 80_000, "time": 2.0 + 0.1 * (itr % 2)}
        if itr % 10 == 0:
            rec.update({"eval/avg episode reward": 1000.0 + itr, "eval/std episode reward": 50.0})
        results.append(rec)
    with open(run / "result.pkl", "wb") as handle:
        pickle.dump(results, handle)
    return run


def test_export(tmp_path, monkeypatch):
    (tmp_path / "base.pt").write_bytes(b"checkpoint")
    monkeypatch.setenv("FAKE_LOG_DIR", str(tmp_path))
    flowsde = {"_target_": "model.flow.ft_ppo.ppoflow_fmppo.FlowSDE", "use_chain_ppo_loss": False}
    chain = {"_target_": "model.flow.ft_ppo.ppoflow_fmppo.FlowSDE", "use_chain_ppo_loss": True, "chain_ppo_aux_coef": 0.25}
    noaux = dict(chain, chain_ppo_aux_coef=0.0)
    reinflow = {"_target_": "model.flow.ft_ppo.ppoflow.PPOFlow"}
    runs = tmp_path / "runs"
    for i, model in enumerate([flowsde, chain, noaux, reinflow]):
        make_run(runs, f"run{i}", model, 42, tmp_path / "base.pt")
    out = tmp_path / "exp_data"
    for run in sorted(runs.iterdir()):
        export.export_run(str(run), str(out), str(tmp_path))

    assert sorted(os.listdir(out)) == ["hopperv2_FlowSDE", "hopperv2_chainppo", "hopperv2_chainppo_noaux", "hopperv2_reinflow"]
    rows = list(csv.DictReader(open(out / "hopperv2_chainppo" / "seed_42_progress.csv")))
    assert [r["itr"] for r in rows] == ["0", "10", "20"] and rows[1]["reward"] == "1010.0"
    assert list(rows[0]) == ["type", "itr", "reward", "std", "total_step_M"] and rows[0]["type"] == "eval"
    prov = json.load(open(out / "hopperv2_chainppo" / "seed_42_provenance.json"))
    assert prov["method"] == "chainppo" and Path(prov["base_policy_path"]) == tmp_path / "base.pt"
    assert prov["base_policy_sha256"] is not None and prov["result_pkl_sha256"] is not None
    assert prov["latency"]["train_iterations_timed"] == 18  # 20 iterations minus the eval ones (itr 10, 20)
    assert (out / "hopperv2_chainppo" / "seed_42_config.yaml").exists()


def test_export_split_noise_level(tmp_path):
    runs = tmp_path / "runs"
    for i, a in enumerate([0.1, 0.2]):
        model = {"_target_": "model.flow.ft_ppo.ppoflow_fmppo.FlowSDE", "use_chain_ppo_loss": False, "noise_level_a": a}
        make_run(runs, f"run{i}", model, 42, tmp_path / "base.pt")
    out = tmp_path / "search"
    for run in sorted(runs.iterdir()):
        export.export_run(str(run), str(out), str(tmp_path), split_noise_level=True)
    assert sorted(os.listdir(out)) == ["hopperv2_FlowSDE_a0.1", "hopperv2_FlowSDE_a0.2"]


def test_export_preserves_training_revision(tmp_path, monkeypatch):
    model = {"_target_": "model.flow.ft_ppo.ppoflow_fmppo.FlowSDE"}
    run = make_run(tmp_path, "run", model, 42, None)
    metadata = {"git_commit": "training-revision", "git_dirty": True,
                "source_sha256": "source-digest", "base_policy_sha256": "loaded-checkpoint-digest"}
    (run / "run_metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(export, "git_commit", lambda root: "export-revision")
    export.export_run(str(run), str(tmp_path / "out"), str(tmp_path))
    provenance = json.loads((tmp_path / "out/hopperv2_FlowSDE/seed_42_provenance.json").read_text())
    assert provenance["git_commit"] == "training-revision"
    assert provenance["export_git_commit"] == "export-revision"
    assert provenance["base_policy_sha256"] == "loaded-checkpoint-digest"
    assert provenance["training_metadata"] == metadata
