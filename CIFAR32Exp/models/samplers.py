from typing import Dict, Optional

import torch

from models.utils import clamp_to_image_range, gaussian_log_prob


def lambda_schedule(
    t: torch.Tensor,
    schedule_type: str,
    scale: float,
    path_orientation: str = "noise_to_data",
) -> torch.Tensor:
    if schedule_type == "constant":
        return torch.full_like(t, fill_value=scale)
    if schedule_type == "linear_decay":
        return scale * (1.0 - t)
    if schedule_type == "flow_grpo":
        if path_orientation == "noise_to_data":
            return scale * torch.sqrt((1.0 - t).clamp_min(0.0) / t)
        if path_orientation == "data_to_noise":
            return scale * torch.sqrt(t.clamp_min(0.0) / (1.0 - t))
        raise ValueError(f"Unsupported path orientation: {path_orientation}")
    raise ValueError(f"Unsupported lambda schedule: {schedule_type}")


def rollout_time_grid(num_steps: int, device: torch.device) -> torch.Tensor:
    return torch.arange(num_steps, device=device, dtype=torch.float32) / num_steps


def sde_step_statistics(
    x: torch.Tensor,
    velocity: torch.Tensor,
    t_batch: torch.Tensor,
    dt: float,
    schedule_type: str,
    lambda_scale: float,
    path_orientation: str = "noise_to_data",
) -> Dict[str, torch.Tensor]:
    # Noise level and score scaling for the step [t, t + dt] are evaluated at the midpoint,
    # which keeps the flow_grpo terms finite at both endpoints without special-casing them.
    t_mid = t_batch + 0.5 * dt
    sigma_t = lambda_schedule(
        t_mid,
        schedule_type=schedule_type,
        scale=lambda_scale,
        path_orientation=path_orientation,
    )

    if schedule_type == "flow_grpo":
        if path_orientation == "noise_to_data":
            drift = velocity - (sigma_t.pow(2) / (2.0 * (1.0 - t_mid)))[:, None, None, None] * (
                x - t_batch[:, None, None, None] * velocity
            )
        elif path_orientation == "data_to_noise":
            drift = velocity + (sigma_t.pow(2) / (2.0 * t_mid))[:, None, None, None] * (
                x + (1.0 - t_batch)[:, None, None, None] * velocity
            )
        else:
            raise ValueError(f"Unsupported path orientation: {path_orientation}")
    else:
        drift = velocity

    mean = x + dt * drift
    sigma_step = sigma_t * (dt ** 0.5)
    sigma = sigma_step[:, None, None, None].expand_as(mean)
    return {
        "mean": mean,
        "sigma_t": sigma_t,
        "sigma_step": sigma_step,
        "sigma": sigma,
        "drift": drift,
    }


def flow_grpo_velocity_kl(
    velocity: torch.Tensor,
    ref_velocity: torch.Tensor,
    t_batch: torch.Tensor,
    sigma_t: torch.Tensor,
    dt: float,
    path_orientation: str = "noise_to_data",
) -> torch.Tensor:
    sigma_t = sigma_t.clamp_min(1e-4)
    t_mid = t_batch + 0.5 * dt
    if path_orientation == "noise_to_data":
        coeff = (sigma_t * t_batch / (2.0 * (1.0 - t_mid))) + (1.0 / sigma_t)
    elif path_orientation == "data_to_noise":
        coeff = (sigma_t * (1.0 - t_batch) / (2.0 * t_mid)) + (1.0 / sigma_t)
    else:
        raise ValueError(f"Unsupported path orientation: {path_orientation}")
    diff = (velocity - ref_velocity).reshape(velocity.shape[0], -1)
    return 0.5 * dt * coeff.pow(2) * diff.pow(2).sum(dim=-1)


@torch.no_grad()
def ode_rollout(
    model: torch.nn.Module,
    x0: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int,
    clamp: bool = True,
) -> Dict[str, torch.Tensor]:
    model.eval()
    x = x0
    dt = 1.0 / num_steps
    time_grid = rollout_time_grid(num_steps, x.device)
    for t in time_grid:
        t_batch = torch.full((x.shape[0],), float(t.item()), device=x.device, dtype=torch.float32)
        velocity = model(x, t_batch, labels)
        x = x + dt * velocity
    if clamp:
        x = clamp_to_image_range(x)
    return {"samples": x}


def sde_rollout(
    model: torch.nn.Module,
    x0: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int,
    schedule_type: str,
    lambda_scale: float,
    clamp: bool = True,
    return_log_probs: bool = False,
    noise: Optional[torch.Tensor] = None,
    path_orientation: str = "noise_to_data",
) -> Dict[str, torch.Tensor]:
    x = x0
    dt = 1.0 / num_steps
    time_grid = rollout_time_grid(num_steps, x.device)

    total_log_prob = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
    step_log_probs = []
    all_means = []
    all_sigmas = []
    all_sigma_steps = []
    all_samples = [x]

    if noise is None:
        noise = torch.randn((num_steps,) + tuple(x.shape), device=x.device, dtype=x.dtype)

    for step_idx, t in enumerate(time_grid):
        t_batch = torch.full((x.shape[0],), float(t.item()), device=x.device, dtype=torch.float32)
        velocity = model(x, t_batch, labels)
        stats = sde_step_statistics(
            x=x,
            velocity=velocity,
            t_batch=t_batch,
            dt=dt,
            schedule_type=schedule_type,
            lambda_scale=lambda_scale,
            path_orientation=path_orientation,
        )
        x = stats["mean"] + stats["sigma"] * noise[step_idx]
        if return_log_probs:
            step_lp = gaussian_log_prob(x, stats["mean"], stats["sigma"])
            total_log_prob = total_log_prob + step_lp
            step_log_probs.append(step_lp)
            all_means.append(stats["mean"])
            all_sigmas.append(stats["sigma_t"])
            all_sigma_steps.append(stats["sigma_step"])
        all_samples.append(x)

    outputs: Dict[str, torch.Tensor] = {"samples": x}
    if clamp:
        outputs["samples"] = clamp_to_image_range(outputs["samples"])
    if return_log_probs:
        outputs["total_log_prob"] = total_log_prob
        outputs["step_log_probs"] = torch.stack(step_log_probs, dim=1)
        outputs["means"] = torch.stack(all_means, dim=1)
        outputs["sigmas"] = torch.stack(all_sigmas, dim=1)
        outputs["step_sigmas"] = torch.stack(all_sigma_steps, dim=1)
        outputs["noise"] = noise
        outputs["times"] = time_grid
    outputs["trajectory"] = torch.stack(all_samples, dim=1)
    return outputs
