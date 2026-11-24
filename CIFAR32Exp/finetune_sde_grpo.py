import copy
import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.nn import functional as F
from tqdm import tqdm

from models.classifier_utils import ensure_reward_classifier, freeze_classifier, target_reward_from_logits
from models.flow_model import build_flow_model
from models.samplers import flow_grpo_velocity_kl, ode_rollout, sde_rollout, sde_step_statistics
from models.utils import (
    create_balanced_class_subset,
    create_cifar_splits,
    create_fixed_generation_inputs,
    create_loader,
    ensure_dir,
    get_device,
    load_checkpoint,
    load_json_config,
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
    was_training = model.training
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
    if was_training:
        model.train()
    return total_loss / max(total_count, 1)


def group_normalized_advantages(rewards: torch.Tensor, group_size: int, clip_value: float) -> torch.Tensor:
    rewards = rewards.view(-1, group_size)
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    scaled = centered / (rewards.std(dim=1, keepdim=True, unbiased=False) + 1e-4)
    return scaled.clamp(-clip_value, clip_value).reshape(-1)


def policy_terms_from_trajectory(
    model: torch.nn.Module,
    reference_model: torch.nn.Module,
    trajectory: torch.Tensor,
    old_step_log_probs: torch.Tensor,
    times: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int,
    schedule_type: str,
    lambda_scale: float,
    path_orientation: str,
) -> Dict[str, torch.Tensor]:
    dt = 1.0 / num_steps
    current_step_log_probs = []
    kl_terms = []
    for step_idx in range(num_steps):
        x_t = trajectory[:, step_idx]
        x_next = trajectory[:, step_idx + 1]
        t_value = torch.full(
            (trajectory.shape[0],),
            float(times[step_idx].item()),
            device=trajectory.device,
            dtype=torch.float32,
        )
        velocity = model(x_t, t_value, labels)
        step_stats = sde_step_statistics(
            x=x_t,
            velocity=velocity,
            t_batch=t_value,
            dt=dt,
            schedule_type=schedule_type,
            lambda_scale=lambda_scale,
            path_orientation=path_orientation,
        )
        step_log_prob = -0.5 * (
            ((x_next - step_stats["mean"]) / step_stats["sigma"].clamp_min(1e-4)).pow(2).flatten(1).sum(dim=-1)
            + torch.log(2.0 * torch.pi * step_stats["sigma"].clamp_min(1e-4).pow(2)).flatten(1).sum(dim=-1)
        )
        current_step_log_probs.append(step_log_prob)

        with torch.no_grad():
            ref_velocity = reference_model(x_t, t_value, labels)
        kl_terms.append(
            flow_grpo_velocity_kl(
                velocity=velocity,
                ref_velocity=ref_velocity,
                t_batch=t_value,
                sigma_t=step_stats["sigma_t"],
                dt=dt,
                path_orientation=path_orientation,
            )
        )

    step_log_probs = torch.stack(current_step_log_probs, dim=1)
    ratios = torch.exp(step_log_probs - old_step_log_probs)
    return {
        "ratios": ratios,
        "current_step_log_probs": step_log_probs,
        "kl_terms": torch.stack(kl_terms, dim=1),
    }


@torch.no_grad()
def monitor_sampler_rewards(
    model: torch.nn.Module,
    classifier: torch.nn.Module,
    device: torch.device,
    sampler_cfg: Dict[str, Any],
    image_shape: tuple[int, int, int],
    num_classes: int,
    num_samples: int,
    batch_size: int,
    reward_type: str,
    fixed_inputs: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    classifier.eval()

    sde_rewards = []
    ode_rewards = []
    steps = (num_samples + batch_size - 1) // batch_size
    monitor_steps = sampler_cfg.get("monitor_num_steps", sampler_cfg["num_steps"])
    path_orientation = sampler_cfg.get("path_orientation", "noise_to_data")

    for step_idx in range(steps):
        current_batch = min(batch_size, num_samples - step_idx * batch_size)
        if fixed_inputs is None:
            labels = torch.randint(0, num_classes, (current_batch,), device=device)
            x0 = torch.randn((current_batch,) + image_shape, device=device, dtype=torch.float32)
        else:
            start = step_idx * batch_size
            end = start + current_batch
            labels = fixed_inputs["labels"][start:end].to(device)
            x0 = fixed_inputs["x0"][start:end].to(device)

        ode_samples = ode_rollout(
            model=model,
            x0=x0,
            labels=labels,
            num_steps=monitor_steps,
            clamp=True,
        )["samples"]
        sde_samples = sde_rollout(
            model=model,
            x0=x0,
            labels=labels,
            num_steps=monitor_steps,
            schedule_type=sampler_cfg["lambda_type"],
            lambda_scale=sampler_cfg["lambda_scale"],
            clamp=True,
            return_log_probs=False,
            path_orientation=path_orientation,
        )["samples"]

        ode_logits = classifier(ode_samples)
        sde_logits = classifier(sde_samples)
        ode_rewards.append(target_reward_from_logits(ode_logits, labels, reward_type=reward_type).cpu())
        sde_rewards.append(target_reward_from_logits(sde_logits, labels, reward_type=reward_type).cpu())

    if was_training:
        model.train()
    return {
        "monitor_reward_ode": float(torch.cat(ode_rewards, dim=0).mean().item()),
        "monitor_reward_sde": float(torch.cat(sde_rewards, dim=0).mean().item()),
    }


def initialize_model_for_selection(
    selection_name: str,
    checkpoint_path: Optional[str],
    config: Dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    if selection_name == "random_init":
        reference_path = config["random_init"]["reference_checkpoint"]
        reference_payload = load_checkpoint(reference_path, map_location="cpu")
        base_config = reference_payload["config"]
        model = build_flow_model(base_config).to(device)
        metadata = {
            "source_checkpoint_path": None,
            "reference_checkpoint_path": reference_path,
            "source_val_loss": None,
            "source_type": "random_init",
        }
        return model, base_config, metadata

    if checkpoint_path is None:
        raise ValueError(f"Missing checkpoint path for selection {selection_name}")

    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    base_config = payload["config"]
    model = build_flow_model(base_config).to(device)
    model.load_state_dict(payload["model_state"])
    metadata = {
        "source_checkpoint_path": checkpoint_path,
        "reference_checkpoint_path": checkpoint_path,
        "source_val_loss": None,
        "source_type": "checkpoint",
    }
    return model, base_config, metadata


def finetune_checkpoint(
    selection_name: str,
    checkpoint_path: Optional[str],
    checkpoint_record: Dict[str, Any],
    config: Dict[str, Any],
    classifier: torch.nn.Module,
    device: torch.device,
    output_root: Path,
    val_loader: torch.utils.data.DataLoader,
    run_seed: int,
) -> Dict[str, Any]:
    seed_everything(run_seed)
    model, base_config, model_metadata = initialize_model_for_selection(
        selection_name=selection_name,
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
    )
    model.train()
    reference_model = copy.deepcopy(model).eval()
    for param in reference_model.parameters():
        param.requires_grad_(False)

    ft_dir = output_root / selection_name
    ft_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"].get("weight_decay", 0.0),
    )

    history: List[Dict[str, Any]] = []
    start_time = time.time()
    num_classes = base_config["model"].get("num_classes", 10)
    image_shape = (
        config["data"].get("channels", 3),
        config["data"].get("image_size", 32),
        config["data"].get("image_size", 32),
    )
    train_num_steps = config["sampler"].get("train_num_steps", config["sampler"]["num_steps"])
    policy_epochs = config["train"].get("policy_epochs", 1)
    reward_type = config.get("reward", {}).get("type", "log_prob")
    monitor_fixed_inputs = None
    if config["train"].get("fixed_monitor_inputs", True):
        monitor_fixed_inputs = create_fixed_generation_inputs(
            num_samples=config["train"].get("monitor_num_samples", 50),
            image_shape=image_shape,
            num_classes=num_classes,
            seed=config["train"].get("monitor_seed", run_seed + 10_000),
        )
    pre_val_fm_loss = evaluate_fm_loss(
        model=model,
        loader=val_loader,
        device=device,
        max_batches=config["train"].get("max_val_batches", 0),
    )
    monitor_eval_fm_loss = bool(config["train"].get("monitor_eval_fm_loss", False))
    monitor_fm_max_batches = int(config["train"].get("monitor_fm_max_batches", config["train"].get("max_val_batches", 0)))

    group_size = int(config["train"]["group_size"])
    adv_clip = float(config["train"].get("adv_clip", 3.0))
    clip_ratio = float(config["train"]["clip_ratio"])
    beta_kl = float(config["train"]["beta_kl"])
    grad_clip = float(config["train"].get("grad_clip", 0.0))
    schedule_type = config["sampler"]["lambda_type"]
    lambda_scale = config["sampler"]["lambda_scale"]
    path_orientation = config["sampler"].get("path_orientation", "noise_to_data")

    progress = tqdm(range(1, config["train"]["updates"] + 1), desc=f"grpo {selection_name}", leave=False)
    for update in progress:
        class_order = torch.randperm(num_classes, device=device)

        step_losses: List[float] = []
        step_reward_means: List[float] = []
        step_reward_stds: List[float] = []
        step_accs: List[float] = []
        step_ratios: List[float] = []
        step_kls: List[float] = []

        for class_idx in class_order.tolist():
            old_model = copy.deepcopy(model).eval()
            for param in old_model.parameters():
                param.requires_grad_(False)

            labels = torch.full((group_size,), class_idx, dtype=torch.long, device=device)
            x0 = torch.randn((group_size,) + image_shape, device=device, dtype=torch.float32)

            rollout = sde_rollout(
                model=old_model,
                x0=x0,
                labels=labels,
                num_steps=train_num_steps,
                schedule_type=schedule_type,
                lambda_scale=lambda_scale,
                clamp=False,
                return_log_probs=True,
                path_orientation=path_orientation,
            )
            samples = rollout["samples"].clamp(-1.0, 1.0)
            with torch.no_grad():
                logits = classifier(samples)
                rewards = target_reward_from_logits(logits, labels, reward_type=reward_type)
            advantages = group_normalized_advantages(
                rewards=rewards,
                group_size=group_size,
                clip_value=adv_clip,
            ).detach()

            step_loss_val = None
            step_ratio_means: List[float] = []
            step_kl_means: List[float] = []
            for _ in range(policy_epochs):
                policy_terms = policy_terms_from_trajectory(
                    model=model,
                    reference_model=reference_model,
                    trajectory=rollout["trajectory"].detach(),
                    old_step_log_probs=rollout["step_log_probs"].detach(),
                    times=rollout["times"].detach(),
                    labels=labels,
                    num_steps=train_num_steps,
                    schedule_type=schedule_type,
                    lambda_scale=lambda_scale,
                    path_orientation=path_orientation,
                )
                ratio = policy_terms["ratios"]
                clipped_ratio_tensor = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
                advantages_expanded = advantages[:, None]
                clipped_objective = torch.min(ratio * advantages_expanded, clipped_ratio_tensor * advantages_expanded)
                objective = clipped_objective - beta_kl * policy_terms["kl_terms"]
                loss = -objective.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                step_loss_val = float(loss.item())
                step_ratio_means.append(float(ratio.mean().item()))
                step_kl_means.append(float(policy_terms["kl_terms"].mean().item()))

            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                step_losses.append(step_loss_val if step_loss_val is not None else 0.0)
                step_reward_means.append(float(rewards.mean().item()))
                step_reward_stds.append(float(rewards.std(unbiased=False).item()))
                step_accs.append(float((preds == labels).float().mean().item()))
                step_ratios.append(float(sum(step_ratio_means) / max(len(step_ratio_means), 1)))
                step_kls.append(float(sum(step_kl_means) / max(len(step_kl_means), 1)))

            del old_model
            gc.collect()
            if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

        loss_mean = float(sum(step_losses) / max(len(step_losses), 1))
        reward_mean = float(sum(step_reward_means) / max(len(step_reward_means), 1))
        reward_std = float(sum(step_reward_stds) / max(len(step_reward_stds), 1))
        acc = float(sum(step_accs) / max(len(step_accs), 1))
        ratio_mean = float(sum(step_ratios) / max(len(step_ratios), 1))
        kl_mean = float(sum(step_kls) / max(len(step_kls), 1))

        monitor_metrics = monitor_sampler_rewards(
            model=model,
            classifier=classifier,
            device=device,
            sampler_cfg=config["sampler"],
            image_shape=image_shape,
            num_classes=num_classes,
            num_samples=config["train"].get("monitor_num_samples", 50),
            batch_size=config["train"].get("monitor_batch_size", 50),
            reward_type=reward_type,
            fixed_inputs=monitor_fixed_inputs,
        )
        if monitor_eval_fm_loss:
            monitor_metrics["monitor_val_fm_loss"] = float(
                evaluate_fm_loss(
                    model=model,
                    loader=val_loader,
                    device=device,
                    max_batches=monitor_fm_max_batches,
                )
            )
        record = {
            "update": update,
            "loss": loss_mean,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "acc": acc,
            "ratio_mean": ratio_mean,
            "kl_mean": kl_mean,
            "policy_epochs": policy_epochs,
            "class_order": class_order.tolist(),
            "pre_val_fm_loss": float(pre_val_fm_loss),
            "wall_time_sec": time.time() - start_time,
        }
        record.update(monitor_metrics)
        history.append(record)
        save_jsonl(ft_dir / "history.jsonl", history)
        save_json(ft_dir / "history.json", {"history": history})
        postfix = {
            "loss": f"{record['loss']:.4f}",
            "reward": f"{reward_mean:.3f}",
            "acc": f"{acc:.3f}",
            "kl": f"{kl_mean:.4f}",
            "ode": f"{record['monitor_reward_ode']:.3f}",
            "sde": f"{record['monitor_reward_sde']:.3f}",
        }
        if "monitor_val_fm_loss" in record:
            postfix["fm"] = f"{record['monitor_val_fm_loss']:.4f}"
        progress.set_postfix(postfix)

    final_path = ft_dir / "final.pt"
    save_checkpoint(
        path=final_path,
        model=model,
        optimizer=optimizer,
        config=base_config,
        extra={
            "selection_name": selection_name,
            "source_checkpoint_path": model_metadata["source_checkpoint_path"],
            "reference_checkpoint_path": model_metadata["reference_checkpoint_path"],
            "source_val_loss": checkpoint_record.get("val_loss"),
            "pre_val_fm_loss": float(pre_val_fm_loss),
            "finetune_history": history,
            "finetune_time_sec": time.time() - start_time,
        },
    )

    return {
        "selection_name": selection_name,
        "source_checkpoint_path": model_metadata["source_checkpoint_path"],
        "reference_checkpoint_path": model_metadata["reference_checkpoint_path"],
        "source_val_loss": checkpoint_record.get("val_loss"),
        "pre_val_fm_loss": float(pre_val_fm_loss),
        "checkpoint_path": str(final_path),
        "finetune_time_sec": time.time() - start_time,
        "last_reward_mean": history[-1]["reward_mean"],
        "last_acc": history[-1]["acc"],
        "last_monitor_reward_sde": history[-1]["monitor_reward_sde"],
        "last_monitor_reward_ode": history[-1]["monitor_reward_ode"],
        "last_monitor_val_fm_loss": history[-1].get("monitor_val_fm_loss"),
    }


def checkpoint_split_protocol(config: Dict[str, Any], manifest: Dict[str, Any]) -> tuple[int, float]:
    """Bind the held-out split to the configs stored in all selected checkpoints."""
    protocols = set()
    for selection in config.get("selections", ["weak", "medium", "strong"]):
        if selection == "random_init":
            checkpoint_path = config["random_init"]["reference_checkpoint"]
        else:
            checkpoint_path = manifest.get(selection, {}).get("selected_checkpoint_path")
        if not checkpoint_path:
            raise ValueError(f"Missing checkpoint path for selection {selection}")
        payload = load_checkpoint(checkpoint_path, map_location="cpu")
        base_config = payload["config"]
        protocols.add((int(base_config["seed"]), float(base_config["data"].get("val_fraction", 0.1))))
        del payload
    if len(protocols) != 1:
        raise ValueError("Selected checkpoints must share the same pretraining split seed and validation fraction.")
    split_seed, val_fraction = protocols.pop()
    if float(config["data"].get("val_fraction", val_fraction)) != val_fraction:
        raise ValueError("Fine-tuning validation fraction differs from the selected checkpoint split.")
    return split_seed, val_fraction


def main() -> None:
    parser = parse_common_args("Fine-tune flow-matching checkpoints with SDE GRPO.")
    args = parser.parse_args()

    config = load_json_config(args.config)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.seed is not None:
        config["seed"] = args.seed

    output_dir = ensure_dir(config["output_dir"])
    device = get_device(args.device or config.get("device"))
    seed_everything(config["seed"])
    save_run_metadata(Path(output_dir), config=config, device=device, seed=config["seed"])

    classifier_config = load_json_config(config["classifier_config"])
    classifier, classifier_report = ensure_reward_classifier(classifier_config, device=device)
    classifier = freeze_classifier(classifier)

    # The validation slice must be the one held out during flow-matching pre-training, so the
    # split uses the pre-training split seed and the monitoring subset uses a fixed seed; neither
    # depends on the RL seed. This keeps the FM-loss images disjoint from pre-training data and
    # identical across RL seeds.
    manifest = load_json_config(config["checkpoint_manifest"])
    split_seed, val_fraction = checkpoint_split_protocol(config, manifest)
    train_set, val_set, _ = create_cifar_splits(
        data_dir=config["data"]["data_dir"],
        val_fraction=val_fraction,
        seed=split_seed,
    )
    fm_loss_subset = create_balanced_class_subset(
        dataset=val_set,
        samples_per_class=config["data"].get("fm_loss_samples_per_class", 50),
        num_classes=config["data"].get("num_classes", 10),
        seed=config["data"]["fm_loss_seed"],
    )
    overlap = set(fm_loss_subset.indices) & set(train_set.indices)
    if overlap:
        raise RuntimeError(f"{len(overlap)} FM-loss monitoring images overlap the pre-training set.")
    save_json(
        Path(output_dir) / "fm_loss_subset.json",
        {
            "split_seed": split_seed,
            "val_fraction": val_fraction,
            "fm_loss_seed": config["data"]["fm_loss_seed"],
            "indices": sorted(int(i) for i in fm_loss_subset.indices),
        },
    )
    val_loader = create_loader(
        fm_loss_subset,
        batch_size=config["data"].get("val_batch_size", 128),
        shuffle=False,
        num_workers=config["data"].get("num_workers", 0),
    )

    results = []
    for selection_idx, selection_name in enumerate(config.get("selections", ["weak", "medium", "strong"])):
        record = manifest.get(selection_name, {"selection_name": selection_name})
        checkpoint_path = record.get("selected_checkpoint_path")
        result = finetune_checkpoint(
            selection_name=selection_name,
            checkpoint_path=checkpoint_path,
            checkpoint_record=record,
            config=config,
            classifier=classifier,
            device=device,
            output_root=Path(output_dir),
            val_loader=val_loader,
            run_seed=config["seed"] + selection_idx,
        )
        results.append(result)

    summary = {
        "classifier_report": classifier_report,
        "results": results,
    }
    save_json(Path(output_dir) / "finetune_manifest.json", summary)
    print(f"Saved fine-tuned checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
