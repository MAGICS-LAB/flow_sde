import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms


def parse_common_args(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Path to a JSON config file.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override: cuda, cuda:N, mps, or cpu.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional output directory override.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    return parser


def load_json_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def ensure_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Return the requested device, or the best available one (cuda > mps > cpu).

    A requested device that is not available raises instead of silently falling back to CPU.
    """
    if preferred is not None:
        try:
            device = torch.device(preferred)
        except RuntimeError as exc:
            raise ValueError(f"Unsupported device '{preferred}'. Use cuda, cuda:N, mps, or cpu.") from exc
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"Requested device '{preferred}' but CUDA is not available.")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Requested device '{preferred}' but only {torch.cuda.device_count()} CUDA device(s) are visible."
                )
        elif device.type == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(f"Requested device '{preferred}' but MPS is not available.")
        elif device.type != "cpu":
            raise ValueError(f"Unsupported device '{preferred}'. Use cuda, cuda:N, mps, or cpu.")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info(device: torch.device) -> Dict[str, Any]:
    info = {
        "device": str(device),
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "float32_matmul_precision": None,
    }
    if hasattr(torch, "get_float32_matmul_precision"):
        info["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    return info


def save_run_metadata(output_dir: Path, config: Dict[str, Any], device: torch.device, seed: int) -> None:
    payload = {
        "config": config,
        "device_info": get_device_info(device),
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_json(output_dir / "run_metadata.json", payload)


def get_autocast_context(device: torch.device, enabled: bool):
    del device
    del enabled
    return torch.autocast(device_type="cpu", enabled=False)


def clamp_to_image_range(images: torch.Tensor) -> torch.Tensor:
    return images.clamp(-1.0, 1.0)


def denorm_to_unit_interval(images: torch.Tensor) -> torch.Tensor:
    return ((images + 1.0) * 0.5).clamp(0.0, 1.0)


def build_cifar_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def create_cifar_splits(
    data_dir: str,
    val_fraction: float,
    seed: int,
) -> Tuple[Dataset, Dataset, Dataset]:
    transform = build_cifar_transform()
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)

    val_size = int(len(train_dataset) * val_fraction)
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset, test_dataset


def create_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )


def create_balanced_class_subset(
    dataset: Dataset,
    samples_per_class: int,
    num_classes: int,
    seed: int,
) -> Subset:
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base = dataset.dataset
        candidate_indices = list(dataset.indices)
    else:
        base = dataset
        candidate_indices = list(range(len(dataset)))

    if hasattr(base, "targets"):
        targets = [int(t) for t in base.targets]
        labels_for_candidates = [targets[i] for i in candidate_indices]
    else:
        labels_for_candidates = [int(base[i][1]) for i in candidate_indices]

    by_class: Dict[int, list] = {c: [] for c in range(num_classes)}
    for idx, label in zip(candidate_indices, labels_for_candidates):
        if 0 <= label < num_classes:
            by_class[label].append(idx)

    rng = random.Random(seed)
    chosen: list = []
    for class_idx in range(num_classes):
        bucket = by_class[class_idx]
        if len(bucket) < samples_per_class:
            raise ValueError(
                f"Class {class_idx} has only {len(bucket)} samples in the val split, need {samples_per_class}"
            )
        rng.shuffle(bucket)
        chosen.extend(bucket[:samples_per_class])

    return Subset(base, chosen)


def create_fixed_generation_inputs(
    num_samples: int,
    image_shape: Tuple[int, int, int],
    num_classes: int,
    seed: int,
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x0 = torch.randn((num_samples,) + image_shape, generator=generator, dtype=torch.float32)
    labels = torch.randint(0, num_classes, (num_samples,), generator=generator, dtype=torch.long)
    return {
        "x0": x0,
        "labels": labels,
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    config: Dict[str, Any],
    extra: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": config,
        "extra": extra,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def model_parameter_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def fm_time_grid(num_steps: int, device: torch.device) -> torch.Tensor:
    dt = 1.0 / num_steps
    return torch.arange(0, num_steps, device=device, dtype=torch.float32) * dt


def cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return float((preds == labels).float().mean().item())


def per_class_accuracy(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> Dict[str, float]:
    preds = logits.argmax(dim=-1)
    result: Dict[str, float] = {}
    for idx in range(num_classes):
        mask = labels == idx
        if mask.any():
            acc = (preds[mask] == labels[mask]).float().mean().item()
        else:
            acc = float("nan")
        result[str(idx)] = float(acc)
    return result


def logsumexp_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def flatten_for_log_prob(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], -1)


def gaussian_log_prob(x: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma = sigma.clamp_min(1e-4)
    sigma_flat = flatten_for_log_prob(sigma)
    x_flat = flatten_for_log_prob(x)
    mean_flat = flatten_for_log_prob(mean)
    var = sigma_flat.pow(2)
    quadratic = ((x_flat - mean_flat).pow(2) / var).sum(dim=-1)
    log_det = torch.log(2.0 * math.pi * var).sum(dim=-1)
    return -0.5 * (quadratic + log_det)
