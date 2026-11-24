"""Seeding and raw-result saving of the numerical experiments."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tv = load("TV_eps_scaling")


def mini_sweep(pre_loop_seed):
    """Same structure as validate_theorem_4_1: build net, train (reseeds), then measure."""
    torch.manual_seed(pre_loop_seed)
    ustar = tv.TrueVelocityGaussianBridge(mu=np.array([2.0, -1.0], dtype=np.float32),
                                          Sigma=np.array([[0.7, 0.25], [0.25, 0.5]], dtype=np.float32))
    hashes = []
    for sd in range(3):
        net = tv.VelocityNetwork(dim=2, hidden_dim=96, n_layers=3).to(tv.device)
        hashes.append(hashlib.md5(b"".join(p.detach().cpu().numpy().tobytes() for p in net.parameters())).hexdigest())
        tv.train_to_true_velocity(net, ustar, n_iters=10, batch_size=64, T=0.99, seed=10_000 * sd)
        tv.estimate_epsilon(net, ustar, n_samples=200, n_time_steps=3, T=0.99, seed=20_000 * sd)
        tv.sde_rollout(net, torch.randn(100, 2), n_steps=10, T=0.99, seed=30_000 * sd)
    return hashes


def test_only_first_network_depends_on_pre_loop_seed():
    a, b = mini_sweep(0), mini_sweep(1)
    assert a[0] != b[0] and a[1:] == b[1:]
    assert mini_sweep(0) == a
    source = (ROOT / "TV_eps_scaling.py").read_text()
    loop = source.index("for n_iter in training_iters:")
    assert "torch.manual_seed(0)" in source[source.rindex("results = []", 0, loop):loop]


@pytest.mark.parametrize("name", ["TV_eps_scaling", "Trajectory", "Violation_rate"])
def test_save_records_json(tmp_path, name):
    module = tv if name == "TV_eps_scaling" else load(name)
    path = tmp_path / "records.json"
    module.save_records_json(str(path), {"a": np.float32(1.5), "b": np.arange(3), "c": {2, 1}})
    assert json.loads(path.read_text()) == {"a": 1.5, "b": [0, 1, 2], "c": [1, 2]}
