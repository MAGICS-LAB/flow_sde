"""Validation-split, device, and reward-model guards (no CIFAR download or pretrained weights needed)."""
import json
import os
import sys
import types
from pathlib import Path

import pytest
import torch

sys.modules.setdefault("timm", types.ModuleType("timm"))  # only needed for timm-hosted classifiers
from torchvision import datasets  # noqa: E402

import models.classifier_utils as C  # noqa: E402
import models.utils as U  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class FakeCIFAR(torch.utils.data.Dataset):
    def __init__(self, root, train, transform=None, download=False):
        g = torch.Generator().manual_seed(0)
        self.targets = torch.randint(0, 10, (50000 if train else 10000,), generator=g).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return torch.zeros(3, 32, 32), self.targets[i]


@pytest.fixture(autouse=True)
def fake_cifar(monkeypatch):
    monkeypatch.setattr(datasets, "CIFAR10", FakeCIFAR)


def finetune_subset():
    pre = json.load(open(ROOT / "configs" / "fm_cifar.json"))
    ft = json.load(open(ROOT / "configs" / "grpo_finetune.json"))
    train, val, _ = U.create_cifar_splits("data", pre["data"]["val_fraction"], pre["seed"])
    sub = U.create_balanced_class_subset(val, ft["data"]["fm_loss_samples_per_class"], 10, ft["data"]["fm_loss_seed"])
    return set(train.indices), sub


def test_fm_loss_subset_is_held_out_and_fixed():
    pre_train, sub = finetune_subset()
    assert not set(sub.indices) & pre_train
    assert sorted(sub.indices) == sorted(finetune_subset()[1].indices)
    targets = FakeCIFAR(None, True).targets
    assert torch.bincount(torch.tensor([targets[i] for i in sub.indices]), minlength=10).tolist() == [50] * 10


def test_finetune_config_points_to_pretraining_split():
    ft = json.load(open(ROOT / "configs" / "grpo_finetune.json"))
    assert ft["pretrain_config"] == "configs/fm_cifar.json" and "fm_loss_seed" in ft["data"]


def test_device_selection():
    assert U.get_device("cpu").type == "cpu"
    with pytest.raises(ValueError):
        U.get_device("tpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            U.get_device("cuda")
    else:
        assert U.get_device(None).type == "cuda"


def test_reward_model_guards(tmp_path):
    cfg = dict(json.load(open(ROOT / "configs" / "classifier.json")), output_dir=str(tmp_path), pretrained_candidates=[])
    with pytest.raises(RuntimeError, match="Refusing to train a fallback"):
        C.ensure_reward_classifier(cfg, torch.device("cpu"))
    fallback = C.ClassifierWrapper(C.build_fallback_classifier(cfg), "identity")
    C.save_classifier_wrapper(tmp_path / "classifier.pt", fallback, cfg, extra={})
    with pytest.raises(RuntimeError, match="fallback classifier"):
        C.ensure_reward_classifier(cfg, torch.device("cpu"))
    resnet = C.ClassifierWrapper(C.build_torchvision_classifier("torchvision_resnet18", 10), "unit_interval")
    C.save_classifier_wrapper(tmp_path / "classifier.pt", resnet, cfg,
                              extra={"model_type": "torchvision_resnet18", "source": {"type": "hf_state_dict"},
                                     "num_classes": 10, "preprocess_mode": "unit_interval"})
    _, report = C.ensure_reward_classifier(cfg, torch.device("cpu"))
    assert report["model_type"] == "torchvision_resnet18" and len(report["checkpoint_sha256"]) == 64
    with pytest.raises(RuntimeError, match="sha256"):
        C.ensure_reward_classifier(dict(cfg, expected_checkpoint_sha256="0" * 64), torch.device("cpu"))


def test_monitor_split_uses_checkpoint_config(monkeypatch):
    import finetune_sde_grpo as ft
    config = {"selections": ["weak", "strong"], "data": {"val_fraction": 0.1}}
    manifest = {name: {"selected_checkpoint_path": name} for name in config["selections"]}
    # Static JSON uses seed 123; actual pretraining used --seed 999.
    payloads = {name: {"config": {"seed": 999, "data": {"val_fraction": 0.1}}} for name in manifest}
    monkeypatch.setattr(ft, "load_checkpoint", lambda path, **kwargs: payloads[path])
    assert ft.checkpoint_split_protocol(config, manifest) == (999, 0.1)
    train, val, _ = U.create_cifar_splits("data", 0.1, 999)
    sub = U.create_balanced_class_subset(val, 50, 10, 123)
    assert not set(sub.indices) & set(train.indices)
    payloads["strong"]["config"]["seed"] = 123
    with pytest.raises(ValueError, match="same pretraining split"):
        ft.checkpoint_split_protocol(config, manifest)
