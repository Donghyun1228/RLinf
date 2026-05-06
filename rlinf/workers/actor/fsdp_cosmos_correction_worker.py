# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""TD3+BC FSDP worker for the cosmos+correction policy.

Subclasses ``EmbodiedSACFSDPPolicy`` and reuses its replay buffer,
target-network management, FSDP wrapping, optimizer plumbing, and
checkpoint flow. Only the algorithmic differences are overridden:

  * ``setup_model_and_optimizer`` -- skip entropy-temperature init.
  * ``build_lr_schedulers`` -- no alpha scheduler.
  * ``forward_critic`` -- TD3 target with clipped policy noise on the
    target actor, no entropy backup.
  * ``forward_actor`` -- compute ``Q(s, mu(s))`` from the actor's mean
    and dispatch to the registered ``correction_td3_bc`` loss for the
    final ``-min(Q1, Q2) + bc_coef * ||mu||^2`` value.
  * ``update_one_epoch`` -- same critic/actor/target cadence as SAC,
    minus the alpha update.

The actor's ``log_std`` is frozen on the wrapper side (see Phase 4),
so target-action noise comes entirely from the worker-controlled
``target_policy_noise`` / ``target_noise_clip`` config keys, matching
the standard TD3 recipe.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from rlinf.algorithms.registry import get_policy_loss
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.utils import drq
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class EmbodiedCosmosCorrectionFSDPPolicy(EmbodiedSACFSDPPolicy):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # SAC parent already sets alpha_optimizer / entropy_temp = None;
        # we just keep them None permanently and rely on inherited save/load
        # gates ("if self.alpha_optimizer is not None:") to skip alpha state.

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        """Same as SAC's setup but without entropy-temperature plumbing.

        We still build a critic optimizer group (filter ``q_head`` /
        encoder names) and the main actor optimizer; entropy alpha is
        not learned for TD3+BC.
        """
        module = self.model_provider_func()
        if initialize_target:
            target_module = self.model_provider_func()

        if self.cfg.actor.model.get("gradient_checkpointing", False):
            self.logger.info("[FSDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
            if initialize_target:
                target_module.gradient_checkpointing_enable()

        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        if self.torch_dtype is None:
            self.torch_dtype = next(self.model.parameters()).dtype
        if initialize_target:
            self.target_model = self._strategy.wrap_model(
                model=target_module, device_mesh=self._device_mesh
            )
            self.target_model.requires_grad_(False)
            self.target_model_initialized = True

        # CosmosCorrectionPolicy puts the twin Q under ``policy.critic``
        # as ``qs.<i>.<j>``; matching by ``critic`` / ``qs`` separates
        # the critic params from the actor params at optimizer time.
        param_filters = {"critic": ["critic", "qs"]}
        filtered_optim_config = {"critic": self.cfg.actor.critic_optim}
        optimizers = self.build_optimizers(
            model=self.model,
            main_optim_config=self.cfg.actor.optim,
            param_filters=param_filters,
            filtered_optim_config=filtered_optim_config,
        )
        self.optimizer = optimizers[0]
        self.qf_optimizer = optimizers[1]

        # Explicitly mark "no entropy alpha" so inherited save/load skip it.
        self.entropy_temp = None
        self.alpha_optimizer = None

        # SAC parent's init_worker checks ``self.use_dsrl`` after setup;
        # cosmos+correction never uses the openpi DSRL path.
        self.use_dsrl = False

        self.build_lr_schedulers()
        self.grad_scaler = self.build_grad_scaler(
            self.cfg.actor.fsdp_config.grad_scaler
        )

    def build_lr_schedulers(self):
        self.lr_scheduler = self.build_lr_scheduler(
            self.optimizer, self.cfg.actor.optim
        )
        self.qf_lr_scheduler = self.build_lr_scheduler(
            self.qf_optimizer, self.cfg.actor.critic_optim
        )

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        """TD3 critic loss with target policy smoothing.

        Target = ``r + gamma * (1 - done) * min(Q1', Q2')(s', mu'(s') + clip(N(0, sigma_pi), -c, c))``.
        No entropy term. ``sigma_pi`` (``target_policy_noise``) and ``c``
        (``target_noise_clip``) come from the algorithm config.
        """
        gamma = self.cfg.algorithm.gamma
        policy_noise = self.cfg.algorithm.get("target_policy_noise", 0.2)
        noise_clip = self.cfg.algorithm.get("target_noise_clip", 0.5)
        action_clip = self.cfg.algorithm.get("target_action_clip", 1.0)

        rewards_for_bootstrap = (
            batch["rewards"].sum(dim=-1, keepdim=True).to(self.torch_dtype)
        )
        terminations = batch["terminations"].to(self.torch_dtype)

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]

        with torch.no_grad():
            # Target actor: take the mean (deterministic) and add clipped noise.
            _, _, target_info = self.target_model(
                forward_type=ForwardType.SAC, obs=next_obs
            )
            next_mean = target_info["mean"]
            noise = torch.randn_like(next_mean) * policy_noise
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_mean + noise).clamp(-action_clip, action_clip)

            # Twin Q on target: take the min for the TD target.
            all_qf_next_target = self.target_model(
                forward_type=ForwardType.SAC_Q,
                obs=next_obs,
                actions=next_action,
            )
            qf_next_target = all_qf_next_target.min(dim=-1, keepdim=True).values

            # done mask: any termination across the chunk
            not_done = ~terminations.any(dim=-1, keepdim=True)
            target_q_values = rewards_for_bootstrap + not_done * gamma * qf_next_target

        all_data_q_values = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=actions,
        )

        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )
        metrics = {
            "q_data": all_data_q_values.mean().item(),
            "q_target": target_q_values.mean().item(),
        }
        return critic_loss, metrics

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        """TD3+BC actor loss via the registered ``correction_td3_bc``.

        Uses the actor's mean (not a sample) for ``Q(s, mu(s))`` and
        the ``||mu||^2`` BC anchor.
        """
        curr_obs = batch["curr_obs"]
        _, _, info = self.model(forward_type=ForwardType.SAC, obs=curr_obs)
        mean = info["mean"]

        all_qf_pi = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=mean,
        )

        bc_coef = self.cfg.algorithm.get("bc_coef", 1.0)
        loss_fn = get_policy_loss("correction_td3_bc")
        actor_loss, loss_metrics = loss_fn(
            mean_action=mean, q_values=all_qf_pi, bc_coef=bc_coef
        )

        metrics = {
            f"q_value_{i}": all_qf_pi[..., i].mean().item()
            for i in range(all_qf_pi.shape[-1])
        }
        # Pull a few scalars out of the registered loss's metrics.
        for key in ("actor/q_term", "actor/bc_term", "actor/mean_action_abs"):
            if key in loss_metrics:
                metrics[key.split("/", 1)[1]] = loss_metrics[key].item()
        # Entropy is meaningless without a learned dist; report 0 to keep
        # the inherited update_one_epoch metric shape stable.
        entropy = torch.tensor(0.0, device=actor_loss.device)
        return actor_loss, entropy, metrics

    def forward_alpha(self, batch):
        raise NotImplementedError(
            "TD3+BC has no entropy temperature; this should never be called."
        )

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self, train_actor: bool = True):
        """Critic step every iteration; actor + target soft-update on
        the configured cadence. No alpha update."""
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )

        with self.worker_timer("sample"):
            global_batch = next(self.buffer_dataloader_iter)

        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )
        for i, batch in enumerate(train_micro_batch_list):
            batch = put_tensor_device(batch, device=self.device)
            if self.enable_drq:
                drq.apply_drq(batch["curr_obs"], pad=4)
                drq.apply_drq(batch["next_obs"], pad=4)
            train_micro_batch_list[i] = batch

        # Critic update
        self.qf_optimizer.zero_grad()
        gbs_critic_loss = []
        all_critic_metrics = {}
        for batch in train_micro_batch_list:
            critic_loss, critic_metrics = self.forward_critic(batch)
            critic_loss = critic_loss / self.gradient_accumulation
            critic_loss.backward()
            gbs_critic_loss.append(critic_loss.item() * self.gradient_accumulation)
            append_to_dict(all_critic_metrics, critic_metrics)
        all_critic_metrics = {
            f"critic/{k}": np.mean(v) for k, v in all_critic_metrics.items()
        }
        qf_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.critic_optim.clip_grad
        )
        self.qf_optimizer.step()
        self.qf_lr_scheduler.step()

        metrics_data = {
            "td3bc/critic_loss": np.mean(gbs_critic_loss),
            "critic/lr": self.qf_optimizer.param_groups[0]["lr"],
            "critic/grad_norm": qf_grad_norm,
            **all_critic_metrics,
        }

        # Actor update on critic_actor_ratio cadence (TD3 policy delay)
        if self.update_step % self.critic_actor_ratio == 0 and train_actor:
            self.optimizer.zero_grad()
            gbs_actor_loss = []
            all_actor_metrics = {}
            for batch in train_micro_batch_list:
                actor_loss, _entropy, q_metrics = self.forward_actor(batch)
                actor_loss = actor_loss / self.gradient_accumulation
                actor_loss.backward()
                gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)
                append_to_dict(all_actor_metrics, q_metrics)
            all_actor_metrics = {
                f"actor/{k}": np.mean(v) for k, v in all_actor_metrics.items()
            }
            actor_grad_norm = self.model.clip_grad_norm_(
                max_norm=self.cfg.actor.optim.clip_grad
            )
            self.optimizer.step()
            self.lr_scheduler.step()

            metrics_data.update(
                {
                    "td3bc/actor_loss": np.mean(gbs_actor_loss),
                    "actor/lr": self.optimizer.param_groups[0]["lr"],
                    "actor/grad_norm": actor_grad_norm,
                    **all_actor_metrics,
                }
            )

        # Target network soft update on the configured cadence.
        if (
            self.target_model_initialized
            and self.update_step
            % self.cfg.algorithm.get("target_update_freq", 1)
            == 0
        ):
            self.soft_update_target_model()

        return metrics_data
