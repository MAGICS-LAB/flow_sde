import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.nn import functional as F
from tqdm import tqdm

from models.flow_model import build_flow_model
from models.utils import (
    create_cifar_splits,
    create_loader,
    ensure_dir,
    get_device,
    load_json_config,
    model_parameter_count,
    parse_common_args,
    save_checkpoint,
    save_json,
    save_jsonl,
    save_run_metadata,
    seed_everything,
)


def evaluate_fm_loss(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 0,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            x0 = torch.randn_like(images)
            t = torch.rand(images.shape[0], device=device, dtype=torch.float32)
            xt = (1.0 - t[:, None, None, None]) * x0 + t[:, None, None, None] * images
            target = images - x0
            pred = model(xt, t, labels)
            batch_loss = F.mse_loss(pred, target, reduction="sum")
            total_loss += float(batch_loss.item())
            total_count += target.numel()
            if max_batches and (batch_idx + 1) >= max_batches:
                break
    return total_loss / max(total_count, 1)


def select_checkpoints(
    records: List[Dict[str, Any]],
    output_dir: Path,
    selection_epochs: Dict[str, int] | None = None,
) -> Dict[str, Dict[str, Any]]:
    if len(records) < 3:
        raise RuntimeError("Need at least 3 saved checkpoints to select weak/medium/strong.")

    if selection_epochs:
        by_epoch = {r["epoch"]: r for r in records}
        missing = [(name, ep) for name, ep in selection_epochs.items() if ep not in by_epoch]
        if missing:
            raise RuntimeError(
                f"selection_epochs references epochs not saved: {missing}. "
                f"Saved epochs: {sorted(by_epoch.keys())}."
            )
        selected = {name: by_epoch[ep] for name, ep in selection_epochs.items()}
    else:
        selected = {
            "weak": records[0],
            "medium": records[len(records) // 2],
            "strong": min(records, key=lambda item: item["val_loss"]),
        }

    selected_dir = output_dir / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Dict[str, Any]] = {}
    for name, record in selected.items():
        src = Path(record["checkpoint_path"])
        dst = selected_dir / f"{name}.pt"
        shutil.copy2(src, dst)
        manifest[name] = {
            **record,
            "selection_name": name,
            "selected_checkpoint_path": str(dst),
        }
    save_json(output_dir / "checkpoint_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = parse_common_args("Train a conditional CIFAR-10 flow-matching model.")
    args = parser.parse_args()

    config = load_json_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.seed is not None:
        config["seed"] = args.seed

    output_dir = ensure_dir(config["output_dir"])
    ckpt_dir = ensure_dir(str(output_dir / "checkpoints"))
    device = get_device(args.device or config.get("device"))
    seed_everything(config["seed"])
    save_run_metadata(Path(output_dir), config=config, device=device, seed=config["seed"])

    train_set, val_set, _ = create_cifar_splits(
        data_dir=config["data"]["data_dir"],
        val_fraction=config["data"].get("val_fraction", 0.1),
        seed=config["seed"],
    )
    train_loader = create_loader(
        train_set,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["data"].get("num_workers", 2),
    )
    val_loader = create_loader(
        val_set,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        num_workers=config["data"].get("num_workers", 2),
    )

    model = build_flow_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["train"]["epochs"])

    history = []
    checkpoint_records = []
    best_val = float("inf")
    start_time = time.time()

    print(f"Training on {device} with {model_parameter_count(model):,} parameters")

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        progress = tqdm(train_loader, desc=f"fm epoch {epoch}", leave=False)
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            x0 = torch.randn_like(images)
            t = torch.rand(images.shape[0], device=device, dtype=torch.float32)
            xt = (1.0 - t[:, None, None, None]) * x0 + t[:, None, None, None] * images
            target = images - x0

            optimizer.zero_grad(set_to_none=True)
            pred = model(xt, t, labels)
            loss = F.mse_loss(pred, target)
            loss.backward()
            if config["train"].get("grad_clip", 0.0) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
            optimizer.step()

            running_loss += float(loss.item()) * images.shape[0]
            sample_count += images.shape[0]
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / max(sample_count, 1)
        val_loss = evaluate_fm_loss(
            model,
            val_loader,
            device=device,
            max_batches=config["eval"].get("max_batches", 0),
        )
        wall_time = time.time() - start_time

        record: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "wall_time_sec": wall_time,
        }

        if (epoch % config["train"]["save_every"] == 0) or (epoch == config["train"]["epochs"]):
            checkpoint_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            record["checkpoint_path"] = str(checkpoint_path)
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                config=config,
                extra=record,
            )
            checkpoint_records.append(record.copy())
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    path=output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    extra=record,
                )

        history.append(record)
        save_jsonl(output_dir / "train_history.jsonl", history)
        save_json(output_dir / "train_history.json", {"history": history, "num_parameters": model_parameter_count(model)})
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.8f} wall_time_sec={wall_time:.1f}"
        )
    manifest = select_checkpoints(
        checkpoint_records,
        output_dir=Path(output_dir),
        selection_epochs=config.get("selection_epochs"),
    )

    summary = {
        "output_dir": str(output_dir),
        "best_val_loss": best_val,
        "num_saved_checkpoints": len(checkpoint_records),
        "selected": manifest,
        "total_train_time_sec": time.time() - start_time,
    }
    save_json(output_dir / "summary.json", summary)
    print(f"Saved FM checkpoints and selection manifest to {output_dir}")


if __name__ == "__main__":
    main()
