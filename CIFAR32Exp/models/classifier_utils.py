import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import timm
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torch.nn import functional as F
from torchvision import models as tv_models
from tqdm import tqdm

from models.utils import (
    clamp_to_image_range,
    compute_accuracy,
    create_cifar_splits,
    create_loader,
    denorm_to_unit_interval,
    ensure_dir,
    load_checkpoint,
    logsumexp_entropy,
    per_class_accuracy,
    save_checkpoint,
    save_json,
)


class SmallCIFARClassifier(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x.flatten(1))


class ClassifierWrapper(nn.Module):
    def __init__(self, model: nn.Module, preprocess_mode: str = "identity"):
        super().__init__()
        self.model = model
        self.preprocess_mode = preprocess_mode

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = clamp_to_image_range(x)
        if self.preprocess_mode == "identity":
            return x
        if self.preprocess_mode == "unit_interval":
            return denorm_to_unit_interval(x)
        raise ValueError(f"Unsupported preprocess mode: {self.preprocess_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.preprocess(x))


def target_reward_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reward_type: str = "log_prob",
) -> torch.Tensor:
    idx = torch.arange(labels.shape[0], device=logits.device)
    if reward_type == "log_prob":
        return torch.log_softmax(logits, dim=-1)[idx, labels]
    if reward_type == "prob":
        return torch.softmax(logits, dim=-1)[idx, labels]
    if reward_type == "margin":
        target = logits[idx, labels]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[idx, labels] = False
        competitor = logits.masked_fill(~mask, float("-inf")).max(dim=-1).values
        return target - competitor
    raise ValueError(f"Unsupported reward_type '{reward_type}'.")


def build_fallback_classifier(config: Dict[str, Any]) -> nn.Module:
    return SmallCIFARClassifier(
        in_channels=config.get("in_channels", 3),
        base_channels=config["model"].get("base_channels", 64),
        num_classes=config["model"].get("num_classes", 10),
    )


def build_torchvision_classifier(arch: str, num_classes: int) -> nn.Module:
    if arch == "torchvision_resnet18":
        return tv_models.resnet18(weights=None, num_classes=num_classes)
    if arch == "torchvision_resnet34":
        return tv_models.resnet34(weights=None, num_classes=num_classes)
    if arch == "torchvision_resnet50":
        return tv_models.resnet50(weights=None, num_classes=num_classes)
    raise ValueError(f"Unsupported torchvision architecture '{arch}'.")


def _load_timm_candidate(candidate: Dict[str, Any], device: torch.device) -> Optional[ClassifierWrapper]:
    model_name = candidate["model_name"]
    pretrained = bool(candidate.get("pretrained", True))
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=candidate.get("num_classes", 10))
    model.to(device)
    model.eval()
    return ClassifierWrapper(model=model, preprocess_mode=candidate.get("preprocess_mode", "identity"))


def _load_hf_state_dict_candidate(candidate: Dict[str, Any], device: torch.device) -> Optional[ClassifierWrapper]:
    arch = candidate.get("arch", "small_cifar")
    if arch == "small_cifar":
        model = SmallCIFARClassifier(
            in_channels=candidate.get("in_channels", 3),
            base_channels=candidate.get("base_channels", 64),
            num_classes=candidate.get("num_classes", 10),
        )
    elif arch.startswith("torchvision_"):
        model = build_torchvision_classifier(arch, num_classes=candidate.get("num_classes", 10))
    else:
        raise ValueError(f"Unsupported HF architecture '{arch}' for classifier loading.")
    file_path = hf_hub_download(repo_id=candidate["repo_id"], filename=candidate["filename"])
    state = torch.load(file_path, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return ClassifierWrapper(model=model, preprocess_mode=candidate.get("preprocess_mode", "identity"))


def try_load_pretrained_classifier(config: Dict[str, Any], device: torch.device) -> Tuple[Optional[ClassifierWrapper], Dict[str, Any]]:
    report = {"attempted": [], "loaded": False}
    candidates: List[Dict[str, Any]] = config.get("pretrained_candidates", [])
    for candidate in candidates:
        entry = {"candidate": candidate, "status": "failed", "error": None}
        try:
            if candidate["type"] == "timm":
                wrapper = _load_timm_candidate(candidate, device=device)
            elif candidate["type"] == "hf_state_dict":
                wrapper = _load_hf_state_dict_candidate(candidate, device=device)
            else:
                raise ValueError(f"Unsupported candidate type: {candidate['type']}")
            entry["status"] = "loaded"
            report["attempted"].append(entry)
            report["loaded"] = True
            report["source"] = candidate
            return wrapper, report
        except Exception as exc:  # pragma: no cover - intended graceful fallback.
            entry["error"] = str(exc)
            report["attempted"].append(entry)
    return None, report


def train_fallback_classifier(config: Dict[str, Any], device: torch.device, output_dir: Path) -> Tuple[ClassifierWrapper, Dict[str, Any]]:
    data_cfg = config["data"]
    train_set, val_set, _ = create_cifar_splits(
        data_dir=data_cfg["data_dir"],
        val_fraction=data_cfg.get("val_fraction", 0.1),
        seed=config["seed"],
    )
    train_loader = create_loader(
        train_set,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
    )
    val_loader = create_loader(
        val_set,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
    )

    model = build_fallback_classifier(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"].get("weight_decay", 1e-4),
    )
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_acc = -1.0
    best_path = output_dir / "classifier.pt"
    start = time.time()

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"classifier epoch {epoch}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.shape[0]

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate_classifier(model, val_loader, device=device, num_classes=config["model"].get("num_classes", 10))
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["acc"]),
        }
        history.append(epoch_record)
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = float(val_metrics["acc"])
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                config=config,
                extra={"history": history, "best_val_acc": best_val_acc},
            )

    best_payload = load_checkpoint(str(best_path), map_location="cpu")
    model.load_state_dict(best_payload["model_state"])
    model.to(device)
    model.eval()
    elapsed = time.time() - start

    report = {
        "mode": "trained_fallback",
        "checkpoint_path": str(best_path),
        "history": history,
        "best_val_acc": best_val_acc,
        "train_time_sec": elapsed,
    }
    save_json(output_dir / "classifier_train_history.json", report)
    return ClassifierWrapper(model=model, preprocess_mode="identity"), report


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    logits_all = []
    labels_all = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_loss += float(criterion(logits, labels).item())
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
    logits_cat = torch.cat(logits_all, dim=0)
    labels_cat = torch.cat(labels_all, dim=0)
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": compute_accuracy(logits_cat, labels_cat),
        "per_class_acc": per_class_accuracy(logits_cat, labels_cat, num_classes=num_classes),
        "entropy": float(logsumexp_entropy(logits_cat).mean().item()),
    }


def load_classifier_checkpoint(path: str, device: torch.device) -> ClassifierWrapper:
    payload = load_checkpoint(path, map_location="cpu")
    extra = payload.get("extra", {})
    model_type = extra.get("model_type", "small_cifar")
    if model_type == "timm":
        source = extra["source"]
        model = timm.create_model(
            source["model_name"],
            pretrained=False,
            num_classes=source.get("num_classes", 10),
        )
        preprocess_mode = source.get("preprocess_mode", "identity")
    elif model_type.startswith("torchvision_"):
        model = build_torchvision_classifier(model_type, num_classes=extra.get("num_classes", 10))
        preprocess_mode = extra.get("preprocess_mode", "identity")
    else:
        config = payload.get("config", {})
        model_cfg = config.get("model", {})
        if not model_cfg:
            model_cfg = {
                "in_channels": extra.get("in_channels", 3),
                "base_channels": extra.get("base_channels", 64),
                "num_classes": extra.get("num_classes", 10),
            }
        model = SmallCIFARClassifier(
            in_channels=model_cfg.get("in_channels", 3),
            base_channels=model_cfg.get("base_channels", 64),
            num_classes=model_cfg.get("num_classes", 10),
        )
        preprocess_mode = extra.get("preprocess_mode", "identity")
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return ClassifierWrapper(model=model, preprocess_mode=preprocess_mode)


def save_classifier_wrapper(path: Path, wrapper: ClassifierWrapper, config: Dict[str, Any], extra: Dict[str, Any]) -> None:
    payload = {
        "model_state": wrapper.model.state_dict(),
        "config": config,
        "extra": extra,
    }
    torch.save(payload, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_reward_checkpoint(config: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """Refuse a cached reward checkpoint that is not the configured pretrained classifier."""
    payload = load_checkpoint(str(path), map_location="cpu")
    extra = payload.get("extra", {})
    model_type = extra.get("model_type", "small_cifar")
    if model_type == "small_cifar" and not config.get("allow_fallback_classifier", False):
        raise RuntimeError(
            f"{path} holds a locally trained fallback classifier, not a pretrained ResNet-18. "
            "Delete it to reload a pretrained candidate, or set allow_fallback_classifier=true "
            "to use it on purpose (this changes the reward model)."
        )
    sha256 = _file_sha256(path)
    expected = config.get("expected_checkpoint_sha256")
    if expected and sha256 != expected:
        raise RuntimeError(f"{path} has sha256 {sha256}, expected {expected}.")
    return {"model_type": model_type, "source": extra.get("source"), "checkpoint_sha256": sha256}


def ensure_reward_classifier(config: Dict[str, Any], device: torch.device) -> Tuple[ClassifierWrapper, Dict[str, Any]]:
    output_dir = ensure_dir(config["output_dir"])
    local_ckpt = output_dir / "classifier.pt"
    report_path = output_dir / "classifier_report.json"

    if local_ckpt.exists():
        checkpoint_info = _check_reward_checkpoint(config, local_ckpt)
        wrapper = load_classifier_checkpoint(str(local_ckpt), device=device)
        report = {
            "mode": "existing_checkpoint",
            "checkpoint_path": str(local_ckpt),
            **checkpoint_info,
        }
        save_json(report_path, report)
        return wrapper, report

    wrapper, load_report = try_load_pretrained_classifier(config, device=device)
    if wrapper is not None:
        source = load_report["source"]
        model_type = "timm" if source["type"] == "timm" else source.get("arch", "small_cifar")
        save_classifier_wrapper(
            path=local_ckpt,
            wrapper=wrapper,
            config=config,
            extra={
                "model_type": model_type,
                "source": source,
                "preprocess_mode": source.get("preprocess_mode", "identity"),
                "in_channels": config["model"].get("in_channels", 3),
                "base_channels": config["model"].get("base_channels", 64),
                "num_classes": config["model"].get("num_classes", 10),
            },
        )
        report = {
            "mode": "pretrained_loaded",
            "checkpoint_path": str(local_ckpt),
            **load_report,
            **_check_reward_checkpoint(config, local_ckpt),
        }
        save_json(report_path, report)
        return wrapper, report

    if not config.get("allow_fallback_classifier", False):
        raise RuntimeError(
            "No pretrained reward classifier could be loaded (attempted: "
            f"{load_report['attempted']}). Refusing to train a fallback classifier because it would "
            "change the reward model; set allow_fallback_classifier=true to allow it."
        )
    print("Training a fallback reward classifier; the reward differs from the pretrained ResNet-18.")
    wrapper, train_report = train_fallback_classifier(config, device=device, output_dir=output_dir)
    combined = {
        "mode": "trained_fallback",
        "pretrained_attempts": load_report["attempted"],
        **train_report,
    }
    save_json(report_path, combined)
    return wrapper, combined


def freeze_classifier(model: nn.Module) -> nn.Module:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model
