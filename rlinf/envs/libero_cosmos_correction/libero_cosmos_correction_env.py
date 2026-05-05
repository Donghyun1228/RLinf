# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LIBERO env that bundles each cosmos cycle into one RL step.

From the agent's view, a single ``step(correction_action_6d)`` is:

  1. Compose the correction with cosmos's last action's gripper bit
     (the EE part is sum, gripper passes through) -> 7-DOF env action.
  2. Run that action for one underlying LIBERO step.
  3. Compute reward
        r = dense_coef * cos_sim(z_obs_after, z_goal)
            + success_bonus * (1 if env terminated successfully else 0)
     where ``z_goal`` is the cosmos-predicted future RL token from
     the current cycle (cached on the env).
  4. If not done, roll a new cosmos cycle: cosmos prediction +
     execute its 16-step chunk open-loop in the env. Cache the new
     ``z_goal`` and ``cosmos_action_chunk``.
  5. Return ``(state=(z_obs, z_goal), reward, done, info)``. When
     ``done`` is True the next state's ``z_obs`` / ``z_goal`` are
     set from the post-correction obs and the **previous** goal --
     they are unused by Q-learning since the next-Q is masked by
     ``(1 - done)``.

Cosmos and the WanVAE never run inside the trainer; only this env
holds them. Replay-buffer transitions only carry the small
(``z_obs``, ``z_goal``, action, reward, done) tensors.

Single-env for the MVP. Multi-env can come later via a vectorized
wrapper or by sharing a batched cosmos call across N envs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import gym
import numpy as np
import torch
import torchvision.transforms.functional as TF
from libero.libero import benchmark

from cosmos_policy.datasets.dataset_utils import apply_jpeg_compression_np
from cosmos_policy.experiments.robot.correction_utils import (
    GoalState,
    encode_image_to_rl_token,
    get_action_with_goal_state,
)
from cosmos_policy.experiments.robot.cosmos_utils import (
    DEVICE,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
from cosmos_policy.experiments.robot.libero.run_libero_eval import (
    prepare_observation,
)
from cosmos_policy._src.predict2.correction.rl_token_autoencoder import (
    RLTokenAutoencoder,
)
from rlinf.data.cosmos_correction import ACTION_DIM, TOKEN_DIM


def _find_hf_snapshot(repo_id: str) -> Path:
    cache = Path.home() / ".cache/huggingface/hub"
    repo_dir = cache / f"models--{repo_id.replace('/', '--')}"
    snaps = list((repo_dir / "snapshots").glob("*"))
    if not snaps:
        raise FileNotFoundError(
            f"No HF snapshot for {repo_id}; "
            f"run `huggingface-cli download {repo_id}` first."
        )
    return snaps[0]


@dataclass
class _CosmosCfg:
    """Flat cfg matching the fields cosmos_utils.{get_action, get_model} read.
    Mirrors the structure used in ``run_libero_correction_eval``."""

    config: str = "cosmos_predict2_2b_480p_libero__inference_only"
    ckpt_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
    config_file: str = "cosmos_policy/config/config.py"
    suite: str = "libero"
    chunk_size: int = 16
    num_open_loop_steps: int = 16
    num_denoising_steps_action: int = 5
    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True
    flip_images: bool = True
    use_jpeg_compression: bool = True
    use_variance_scale: bool = False
    trained_with_image_aug: bool = True
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    model_family: str = "cosmos"
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""


# Module-level singletons so a single process running many envs sharing the
# same cosmos checkpoint doesn't pay the 2B-param load cost N times.
_SHARED_COSMOS: dict[str, Any] = {}


def _load_shared_cosmos(cosmos_ckpt_repo: str, cosmos_cfg: _CosmosCfg):
    """Load cosmos + dataset_stats once per process, keyed on the HF repo id.

    Subsequent calls in the same process hit the cache. Saves roughly 30s
    and ~8GB of RAM per additional env.
    """
    key = cosmos_ckpt_repo
    if key in _SHARED_COSMOS:
        return _SHARED_COSMOS[key]

    snap = _find_hf_snapshot(cosmos_ckpt_repo)
    cosmos_cfg.dataset_stats_path = str(snap / "libero_dataset_statistics.json")
    cosmos_cfg.t5_text_embeddings_path = str(snap / "libero_t5_embeddings.pkl")

    init_t5_text_embeddings_cache(cosmos_cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cosmos_cfg.dataset_stats_path)
    cosmos_model, _ = get_model(cosmos_cfg)
    _SHARED_COSMOS[key] = (cosmos_model, dataset_stats, cosmos_cfg)
    return _SHARED_COSMOS[key]


def _preprocess_image_for_correction(
    image_np: np.ndarray, image_size: int = 224
) -> torch.Tensor:
    """``(H, W, 3)`` uint8 -> ``(1, 3, H', W')`` uint8 on DEVICE.

    JPEG q=95 round-trip + resize 224 + 90% center crop + resize 224.
    Matches the build_libero_vae_cache pipeline so the AE sees the
    same input distribution at runtime as it did during training.
    """
    image_np = apply_jpeg_compression_np(image_np, quality=95)
    img = torch.from_numpy(np.ascontiguousarray(image_np))
    img = img.permute(2, 0, 1).unsqueeze(0).float()
    img = TF.resize(img, [image_size, image_size], antialias=True)
    crop = int(image_size * (0.9**0.5))
    top = (image_size - crop) // 2
    img = img[:, :, top : top + crop, top : top + crop]
    img = TF.resize(img, [image_size, image_size], antialias=True)
    return img.clamp(0, 255).to(torch.uint8).to(DEVICE)


@dataclass
class LiberoCosmosCorrectionEnvCfg:
    """Single-env config. Multi-env adds a thin vec wrapper over this."""

    # Task
    task_suite_name: str = "libero_10"
    task_id: int = 0
    max_episode_steps: int = 520

    # Cosmos checkpoint (HF repo id; assets resolved from local HF cache)
    cosmos_ckpt_repo: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
    cosmos_chunk_size: int = 16
    cosmos_num_open_loop_steps: int = 16
    cosmos_num_denoising_steps_action: int = 5
    cosmos_seed: int = 0

    # AE
    rl_token_ae_ckpt_path: str = ""

    # Reward
    dense_coef: float = 1.0
    success_bonus: float = 100.0

    # Stabilization
    num_steps_wait_after_reset: int = 10

    # Render
    env_resolution: int = 256


class LiberoCosmosCorrectionEnv(gym.Env):
    """Single LIBERO env wrapped with the cosmos+correction loop."""

    metadata = {"render.modes": []}

    def __init__(self, cfg: LiberoCosmosCorrectionEnvCfg):
        super().__init__()
        self.cfg = cfg

        # gym spaces -- the agent sees pre-encoded RL tokens, not raw obs.
        self.observation_space = gym.spaces.Dict(
            {
                "z_obs": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(TOKEN_DIM,), dtype=np.float32
                ),
                "z_goal": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(TOKEN_DIM,), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(ACTION_DIM,), dtype=np.float32
        )

        # Cosmos + dataset_stats (shared across envs in this process)
        cosmos_cfg = _CosmosCfg(
            chunk_size=cfg.cosmos_chunk_size,
            num_open_loop_steps=cfg.cosmos_num_open_loop_steps,
            num_denoising_steps_action=cfg.cosmos_num_denoising_steps_action,
        )
        self.cosmos_model, self.dataset_stats, self.cosmos_cfg = _load_shared_cosmos(
            cfg.cosmos_ckpt_repo, cosmos_cfg
        )

        # AE: per-env instance is cheap (~290MB), so don't bother sharing.
        self.rl_token_ae = RLTokenAutoencoder()
        if cfg.rl_token_ae_ckpt_path:
            sd = torch.load(cfg.rl_token_ae_ckpt_path, map_location="cpu")
            self.rl_token_ae.load_state_dict(sd)
        self.rl_token_ae = self.rl_token_ae.eval().to(DEVICE)
        for p in self.rl_token_ae.parameters():
            p.requires_grad_(False)

        # LIBERO env
        os.environ.setdefault("MUJOCO_GL", "egl")
        bench = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
        if not (0 <= cfg.task_id < bench.n_tasks):
            raise ValueError(
                f"task_id {cfg.task_id} out of range [0, {bench.n_tasks}) "
                f"for suite {cfg.task_suite_name}"
            )
        task = bench.get_task(cfg.task_id)
        self._libero_env, self._task_description = get_libero_env(
            task, "cosmos", resolution=cfg.env_resolution
        )

        # Per-cycle state (filled in reset / step)
        self._goal: Optional[GoalState] = None
        self._cosmos_action_chunk: Optional[torch.Tensor] = None  # (1, T, 7)
        self._step_count: int = 0
        self._last_obs: Any = None  # raw libero obs, kept for prepare_observation

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------
    def reset(self) -> dict[str, np.ndarray]:
        """Reset libero env + run the first cosmos cycle.

        Returns the initial state ``{"z_obs", "z_goal"}`` as numpy
        float32 arrays, matching the configured observation space.
        """
        obs = self._libero_env.reset()
        for _ in range(self.cfg.num_steps_wait_after_reset):
            obs, _, _, _ = self._libero_env.step(
                get_libero_dummy_action(self.cosmos_cfg.model_family)
            )
        self._last_obs = obs
        self._step_count = 0

        # First cosmos cycle: predict + execute chunk -> post-chunk obs.
        self._roll_cosmos_cycle()
        return self._build_state()

    def step(
        self, action: np.ndarray | torch.Tensor
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, Any]]:
        """One RL step = correction + reward + next cosmos cycle.

        Returns ``(state, reward, done, info)``. ``info`` exposes the
        reward components so the runner can log them separately.
        """
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        if action.shape != (ACTION_DIM,):
            raise ValueError(
                f"action shape must be ({ACTION_DIM},); got {tuple(action.shape)}"
            )
        if self._goal is None or self._cosmos_action_chunk is None:
            raise RuntimeError("Call reset() before step().")

        # 1. Compose correction with cosmos last action's gripper.
        last_cosmos = self._cosmos_action_chunk[0, -1].cpu().numpy()  # (7,)
        composed = np.concatenate(
            [last_cosmos[:6] + action.astype(last_cosmos.dtype), last_cosmos[6:7]],
            axis=-1,
        )

        # 2. Execute correction step.
        obs, _reward, done, _info = self._libero_env.step(composed.tolist())
        self._last_obs = obs
        self._step_count += 1
        success = bool(done)
        truncated = (not success) and (self._step_count >= self.cfg.max_episode_steps)
        terminal = success or truncated

        # 3. Dense reward = cos_sim(z_obs_after, z_goal) on the
        #    *current* cycle's goal (cached at the start of this step).
        z_obs_after = self._encode_obs(obs)
        z_goal = self._goal.goal_z_rl
        dense = float(
            torch.nn.functional.cosine_similarity(
                z_obs_after, z_goal, dim=-1
            ).item()
        )
        sparse = self.cfg.success_bonus if success else 0.0
        reward = self.cfg.dense_coef * dense + sparse

        info = {
            "dense_reward": dense,
            "sparse_reward": sparse,
            "success": success,
            "truncated": truncated,
            "step": self._step_count,
        }

        # 4. If terminal, the next state is unused (Q masks via 1-done);
        #    fill in placeholders to keep shape contracts intact.
        if terminal:
            next_state = {
                "z_obs": z_obs_after.squeeze(0).cpu().numpy(),
                "z_goal": z_goal.squeeze(0).cpu().numpy(),
            }
            return next_state, reward, True, info

        # 5. Otherwise: roll the next cosmos cycle.
        self._roll_cosmos_cycle()
        return self._build_state(), reward, False, info

    def close(self) -> None:
        try:
            self._libero_env.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _roll_cosmos_cycle(self) -> None:
        """From ``self._last_obs``: cosmos predict -> execute its chunk in
        env -> cache new goal + chunk + post-chunk obs."""
        observation = prepare_observation(
            self._last_obs,
            resize_size=self.cfg.env_resolution,
            flip_images=self.cosmos_cfg.flip_images,
        )

        action_chunk, goal = get_action_with_goal_state(
            cfg=self.cosmos_cfg,
            cosmos_model=self.cosmos_model,
            rl_token_ae=self.rl_token_ae,
            dataset_stats=self.dataset_stats,
            observation=observation,
            task_description=self._task_description,
            seed=self.cfg.cosmos_seed,
            randomize_seed=False,
            num_denoising_steps_action=self.cosmos_cfg.num_denoising_steps_action,
        )
        self._cosmos_action_chunk = action_chunk  # (1, T, 7), DEVICE
        self._goal = goal

        # Execute the cosmos chunk open-loop in env.
        chunk_np = action_chunk[0].cpu().numpy()  # (T, 7)
        n_steps = min(self.cfg.cosmos_num_open_loop_steps, chunk_np.shape[0])
        for t in range(n_steps):
            obs, _, done, _ = self._libero_env.step(chunk_np[t].tolist())
            self._step_count += 1
            self._last_obs = obs
            if done:
                # Chunk solved the task before we could even fire a
                # correction. We let the *next* step() see done=True via
                # the success / truncation logic; here we just stop the
                # chunk early. The cached goal/chunk are still valid for
                # the upcoming correction step.
                break
            if self._step_count >= self.cfg.max_episode_steps:
                break

    def _encode_obs(self, raw_obs: Any) -> torch.Tensor:
        """raw libero obs -> ``(1, 768)`` z_obs on DEVICE."""
        observation = prepare_observation(
            raw_obs,
            resize_size=self.cfg.env_resolution,
            flip_images=self.cosmos_cfg.flip_images,
        )
        img_t = _preprocess_image_for_correction(observation["primary_image"])
        return encode_image_to_rl_token(self.cosmos_model, self.rl_token_ae, img_t)

    def _build_state(self) -> dict[str, np.ndarray]:
        """Encode current obs into the agent-facing state dict."""
        z_obs = self._encode_obs(self._last_obs).squeeze(0).cpu().numpy()
        z_goal = self._goal.goal_z_rl.squeeze(0).cpu().numpy()
        return {"z_obs": z_obs, "z_goal": z_goal}
