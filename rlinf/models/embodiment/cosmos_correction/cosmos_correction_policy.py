# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""RLinf wrapper around the cosmos-policy ``CorrectionPolicy``.

The underlying ``CorrectionPolicy`` (in the cosmos-policy repo) bundles
a 6-DOF Gaussian actor and a twin-Q critic over RL-token observations
(``z_obs``, ``z_goal``). This wrapper adapts that module to RLinf's
``BasePolicy`` interface so it can be driven by an off-policy actor-critic
worker.

Loss form is TD3+BC (anchor toward zero = cosmos baseline). The actor
returns its mean alongside the sampled action so the loss can use
``Q(s, mu(s))`` and ``||mu(s)||^2`` without a second forward pass.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


class CosmosCorrectionPolicy(nn.Module, BasePolicy):
    def __init__(
        self,
        token_dim: int = 768,
        action_dim: int = 6,
        chunk_size: int = 1,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        num_critics: int = 2,
        log_std_init: float = -1.0,
        learnable_log_std: bool = False,
    ):
        super().__init__()
        from cosmos_policy._src.predict2.correction.correction_policy import (
            CorrectionPolicy,
        )

        self.token_dim = token_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_critics = num_critics

        self.policy = CorrectionPolicy(
            token_dim=token_dim,
            chunk_size=chunk_size,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_critics=num_critics,
            log_std_init=log_std_init,
            learnable_log_std=learnable_log_std,
        )

        self.cuda_graph_manager = None
        self.torch_compile_enabled = False

    @property
    def actor(self) -> nn.Module:
        return self.policy.actor

    @property
    def critic(self) -> nn.Module:
        return self.policy.critic

    def preprocess_env_obs(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Move ``{z_obs, z_goal}`` to the model device as float32 tensors."""
        device = next(self.parameters()).device
        out = {}
        for key in ("z_obs", "z_goal"):
            v = env_obs[key]
            if not torch.is_tensor(v):
                v = torch.as_tensor(v)
            out[key] = v.to(device=device, dtype=torch.float32)
        return out

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        obs = kwargs.get("obs")
        if obs is not None:
            kwargs["obs"] = self.preprocess_env_obs(obs)
        next_obs = kwargs.get("next_obs")
        if next_obs is not None:
            kwargs["next_obs"] = self.preprocess_env_obs(next_obs)

        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"forward_type {forward_type} not supported")

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "CosmosCorrectionPolicy is off-policy; use sac_forward / sac_q_forward."
        )

    def sac_forward(self, obs: dict[str, torch.Tensor], **kwargs):
        """Actor pass.

        Returns ``(action_sampled, log_prob, info)``:
          - ``action_sampled``: (B, chunk_size, action_dim), reparam sample
          - ``log_prob``: (B,) summed over chunk and action dims
          - ``info``: ``{"mean", "log_std"}`` so a TD3+BC loss can compute
            ``Q(s, mu(s))`` and ``||mu||^2`` without a second pass.
        """
        mean, log_std = self.policy.actor(obs["z_obs"], obs["z_goal"])
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=(-2, -1))
        return action, log_prob, {"mean": mean, "log_std": log_std}

    def sac_q_forward(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Twin Q over (z_obs, z_goal, action). Returns ``(B, num_critics)``.

        Phase-5 loss takes ``min`` along the last dim for the TD target
        and the actor objective.
        """
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)  # (B, action_dim) -> (B, 1, action_dim)
        return self.policy.critic(obs["z_obs"], obs["z_goal"], actions)

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "train",
        return_obs: bool = True,
        **kwargs,
    ):
        """Sample an action chunk for env rollout.

        ``mode='train'`` samples ``mu + sigma * eps`` (rollout exploration).
        ``mode='eval'`` returns the deterministic mean.
        """
        obs = self.preprocess_env_obs(env_obs)
        mean, log_std = self.policy.actor(obs["z_obs"], obs["z_goal"])

        if mode == "train":
            std = log_std.exp()
            action = mean + torch.randn_like(mean) * std
        elif mode == "eval":
            action = mean
        else:
            raise ValueError(f"unknown mode {mode!r}")

        forward_inputs = {"action": action, "model_action": action}
        if return_obs:
            forward_inputs["obs"] = obs

        result = {
            "prev_logprobs": torch.zeros(
                action.shape[0], device=action.device, dtype=action.dtype
            ),
            "prev_values": torch.zeros(
                action.shape[0], device=action.device, dtype=action.dtype
            ),
            "forward_inputs": forward_inputs,
        }
        return action, result
