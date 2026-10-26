"""Direct-target FLEX ablation.

This variant keeps the backbone and all conditioning identical to FLEX, but
diffuses the target snapshot itself instead of a refinement around the linear
interpolation of the two endpoint snapshots.
"""

from typing import Optional, Tuple

import torch

from src.diffusion_model import DiffusionModel as RefinementDiffusionModel
from src.diffusion_model import get_logsnr_alpha_sigma


class DiffusionModel(RefinementDiffusionModel):
    """Generate ``u_tau`` directly while conditioning on both endpoints."""

    def forward(
            self,
            target_snapshot: torch.Tensor,
            cond_snapshot_start: torch.Tensor,
            cond_snapshot_end: torch.Tensor,
            fluid_condition: torch.Tensor,
            target_interp_step=torch.Tensor,
            total_interp_steps=torch.Tensor,
        ) -> torch.Tensor:
        clean_target = target_snapshot
        t = torch.rand(clean_target.shape[0], device=clean_target.device)
        _, alpha, sigma = get_logsnr_alpha_sigma(t, shift=self.logsnr_shift)
        eps = torch.randn_like(clean_target, device=clean_target.device)
        target_t = alpha * clean_target + sigma * eps

        scalar_cond = self._build_scalar_cond(
            fluid_condition=fluid_condition,
            target_interp_step=target_interp_step,
            total_interp_steps=total_interp_steps,
        )
        effective_fluid_condition = self._effective_fluid_condition(fluid_condition)
        effective_total_interp_steps = self._effective_total_interp_steps(total_interp_steps)
        spatial_cond_maps = self._run_task_encoder(
            cond_snapshot_start=cond_snapshot_start,
            cond_snapshot_end=cond_snapshot_end,
            fluid_condition=effective_fluid_condition,
            target_interp_step=target_interp_step,
            total_interp_steps=effective_total_interp_steps,
        )
        h, skips = self._run_encoder(
            residual_t=target_t,
            t=t,
            fluid_condition=effective_fluid_condition,
            target_interp_step=target_interp_step,
            total_interp_steps=effective_total_interp_steps,
            scalar_cond=scalar_cond,
            spatial_cond_maps=spatial_cond_maps,
        )
        pred, aux_losses = self._run_decoder(
            h=h,
            skips=skips,
            t=t,
            fluid_condition=effective_fluid_condition,
            target_interp_step=target_interp_step,
            total_interp_steps=effective_total_interp_steps,
            scalar_cond=scalar_cond,
            spatial_cond_maps=spatial_cond_maps,
        )

        if self.prediction_type == "x":
            target = clean_target
        elif self.prediction_type == "eps":
            pred = alpha * pred + sigma * target_t
            target = eps
        else:
            target = alpha * eps - sigma * clean_target

        loss = self.criterion(pred, target)
        if aux_losses:
            loss = loss + sum(aux_losses.values())
        return loss

    @torch.no_grad()
    def sample(
            self,
            n_sample: int,
            size: Tuple[int, int, int],
            cond_snapshot_start: torch.Tensor,
            cond_snapshot_end: torch.Tensor,
            fluid_condition: torch.Tensor,
            target_interp_step: torch.Tensor,
            total_interp_steps: torch.Tensor,
            device: str = "cuda",
            snapshots_i: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
        if snapshots_i is None:
            snapshots_i = torch.randn(n_sample, *size, device=device)

        cond_snapshot_start = cond_snapshot_start.to(device)
        cond_snapshot_end = cond_snapshot_end.to(device)
        fluid_condition = fluid_condition.to(device)
        target_interp_step = target_interp_step.to(device)
        total_interp_steps = total_interp_steps.to(device)

        scalar_cond = self._build_scalar_cond(
            fluid_condition=fluid_condition,
            target_interp_step=target_interp_step,
            total_interp_steps=total_interp_steps,
        )
        effective_fluid_condition = self._effective_fluid_condition(fluid_condition)
        effective_total_interp_steps = self._effective_total_interp_steps(total_interp_steps)
        spatial_cond_maps = self._run_task_encoder(
            cond_snapshot_start=cond_snapshot_start,
            cond_snapshot_end=cond_snapshot_end,
            fluid_condition=effective_fluid_condition,
            target_interp_step=target_interp_step,
            total_interp_steps=effective_total_interp_steps,
        )

        mean = snapshots_i
        for time_step in range(self.diff_steps, 0, -1):
            t = torch.full((n_sample,), time_step / self.diff_steps, device=device)
            t_prev = torch.full((n_sample,), (time_step - 1) / self.diff_steps, device=device)
            _, alpha, sigma = get_logsnr_alpha_sigma(t, shift=self.logsnr_shift)
            _, alpha_prev, sigma_prev = get_logsnr_alpha_sigma(
                t_prev, shift=self.logsnr_shift
            )
            h, skips = self._run_encoder(
                residual_t=snapshots_i,
                t=t,
                fluid_condition=effective_fluid_condition,
                target_interp_step=target_interp_step,
                total_interp_steps=effective_total_interp_steps,
                scalar_cond=scalar_cond,
                spatial_cond_maps=spatial_cond_maps,
            )
            pred, _ = self._run_decoder(
                h=h,
                skips=skips,
                t=t,
                fluid_condition=effective_fluid_condition,
                target_interp_step=target_interp_step,
                total_interp_steps=effective_total_interp_steps,
                scalar_cond=scalar_cond,
                spatial_cond_maps=spatial_cond_maps,
            )
            if self.prediction_type == "v":
                mean = alpha * snapshots_i - sigma * pred
                eps = alpha * pred + sigma * snapshots_i
            elif self.prediction_type == "x":
                mean = pred
                eps = (alpha * pred - snapshots_i) / sigma
            else:
                mean = alpha * snapshots_i - sigma * pred
                eps = alpha * pred + sigma * snapshots_i

            snapshots_i = alpha_prev * mean + sigma_prev * eps

        return mean
