# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Ray-actor wrapper for the cosmos VLA + RL-token AE.

Lets multiple env worker processes share a single cosmos load. Each
env worker drives its own LIBERO sim locally and dispatches cosmos
predictions / VAE encodes to this actor over RPC. The actor is single
threaded by Ray default, so cosmos forwards serialize on the GPU --
which they would anyway, since the GPU runs one cosmos call at a
time. The win comes from env workers running their LIBERO physics on
CPU (and their own EGL contexts) in parallel with the actor's cosmos
work, so the GPU stays busier and overall wall time drops.

The actor is created lazily (named, ``get_if_exists=True``) by the
first env worker that asks for it; subsequent workers reuse the same
handle. ``lifetime="detached"`` keeps it alive across runner
restarts within the same Ray cluster.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import ray
import torch


@ray.remote(num_gpus=0)
class CosmosActor:
    """Owns cosmos VLA + AE on the GPU; exposes RPC methods."""

    def __init__(
        self,
        cosmos_ckpt_repo: str,
        cosmos_chunk_size: int,
        cosmos_num_open_loop_steps: int,
        cosmos_num_denoising_steps_action: int,
        rl_token_ae_ckpt_path: str,
    ):
        # Lazy imports keep the actor's serialized closure small.
        from cosmos_policy._src.predict2.correction.rl_token_autoencoder import (
            RLTokenAutoencoder,
        )

        from rlinf.envs.libero_cosmos_correction.libero_cosmos_correction_env import (
            _CosmosCfg,
            _load_shared_cosmos,
        )

        self._device = torch.device("cuda")

        cosmos_cfg = _CosmosCfg(
            chunk_size=cosmos_chunk_size,
            num_open_loop_steps=cosmos_num_open_loop_steps,
            num_denoising_steps_action=cosmos_num_denoising_steps_action,
        )
        self.cosmos_model, self.dataset_stats, self.cosmos_cfg = _load_shared_cosmos(
            cosmos_ckpt_repo, cosmos_cfg
        )

        self.rl_token_ae = RLTokenAutoencoder()
        if rl_token_ae_ckpt_path:
            sd = torch.load(rl_token_ae_ckpt_path, map_location="cpu")
            self.rl_token_ae.load_state_dict(sd)
        self.rl_token_ae = self.rl_token_ae.eval().to(self._device)
        for p in self.rl_token_ae.parameters():
            p.requires_grad_(False)

        # Cache the action-scale vector (env workers need it but it's a
        # function of dataset_stats only, computed once here).
        a_min = np.asarray(self.dataset_stats["actions_min"], dtype=np.float32)
        a_max = np.asarray(self.dataset_stats["actions_max"], dtype=np.float32)
        self._action_scale = (0.5 * (a_max - a_min))[:6]  # ACTION_DIM

    def get_action_scale(self) -> np.ndarray:
        """One-shot RPC at env construction so workers cache locally."""
        return self._action_scale.copy()

    def predict_chunk_and_goal(
        self,
        observation_np: dict[str, Any],
        task_description: str,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cosmos predict + AE encode of the goal latent.

        ``observation_np`` is the cosmos-style obs dict produced by
        ``libero_utils.prepare_observation`` (image arrays, proprio,
        etc.) with values as numpy arrays. We move them onto the
        actor's GPU device, run the existing ``get_action_with_goal_state``,
        and return numpy so the calling env worker doesn't need to
        share a CUDA context with us.

        Returns:
            ``(action_chunk_np, goal_z_rl_np)``:
              * action_chunk_np: ``(1, T, 7)`` cosmos action chunk
                (already unnormalized)
              * goal_z_rl_np:    ``(1, 768)`` AE-encoded goal token
        """
        from cosmos_policy.experiments.robot.correction_utils import (
            get_action_with_goal_state,
        )

        # Pass numpy through as-is: cosmos's ``get_action`` does its own
        # tensorization internally and emits a list of numpy actions
        # which ``_stack_action_chunk`` expects. Forcing the obs to GPU
        # here makes cosmos emit CUDA tensors instead, which then
        # crashes ``np.stack`` inside the upstream helper.
        action_chunk, goal = get_action_with_goal_state(
            cfg=self.cosmos_cfg,
            cosmos_model=self.cosmos_model,
            rl_token_ae=self.rl_token_ae,
            dataset_stats=self.dataset_stats,
            observation=observation_np,
            task_description=task_description,
            seed=seed,
            randomize_seed=False,
            num_denoising_steps_action=self.cosmos_cfg.num_denoising_steps_action,
        )
        return (
            action_chunk.detach().cpu().numpy(),
            goal.goal_z_rl.detach().cpu().numpy(),
        )

    def encode_image_to_rl_token(self, image_np: np.ndarray) -> np.ndarray:
        """Cosmos VAE + AE encode of a preprocessed image.

        ``image_np`` is the already-jpeg-roundtripped, resized, center-
        cropped uint8 image of shape ``(B, 3, H, W)`` produced by
        ``_preprocess_image_for_correction`` -- the same preprocessing
        cosmos saw at training time.
        """
        from cosmos_policy.experiments.robot.correction_utils import (
            encode_image_to_rl_token,
        )

        image = torch.from_numpy(image_np).to(self._device)
        z_obs = encode_image_to_rl_token(self.cosmos_model, self.rl_token_ae, image)
        return z_obs.detach().cpu().numpy()


_COSMOS_ACTOR_NAME = "rlinf_cosmos_correction_cosmos_actor"


def get_or_create_cosmos_actor(cfg) -> "ray.actor.ActorHandle":
    """Return the named cosmos actor, creating it on the first call.

    Uses RLinf's cluster namespace so the named actor is reachable
    from every worker in the same Ray cluster.
    """
    from rlinf.scheduler.cluster.cluster import Cluster

    namespace = Cluster.NAMESPACE
    try:
        return ray.get_actor(_COSMOS_ACTOR_NAME, namespace=namespace)
    except ValueError:
        pass
    return CosmosActor.options(
        name=_COSMOS_ACTOR_NAME,
        namespace=namespace,
        lifetime="detached",
        get_if_exists=True,
    ).remote(
        cosmos_ckpt_repo=cfg.cosmos_ckpt_repo,
        cosmos_chunk_size=cfg.cosmos_chunk_size,
        cosmos_num_open_loop_steps=cfg.cosmos_num_open_loop_steps,
        cosmos_num_denoising_steps_action=cfg.cosmos_num_denoising_steps_action,
        rl_token_ae_ckpt_path=cfg.rl_token_ae_ckpt_path,
    )
