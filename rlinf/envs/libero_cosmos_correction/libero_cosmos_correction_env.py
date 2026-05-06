# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LIBERO env that bundles each cosmos cycle into one RL step.

From the agent's view, a single ``step(correction_action_6d)`` is:

  1. Compose a 7-DOF env action: the 6-DOF EE part is the correction
     itself (post-chunk standalone delta-EE, expressed in cosmos's
     normalized action space and unnormalized via the dataset stats);
     the gripper bit passes through from cosmos's last action.
     ``correction = 0`` therefore means "no extra movement, hold the
     last gripper command" -- the cosmos chunk is trusted as-is.
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
from dataclasses import dataclass, field, fields
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
    # If non-empty, ``reset()`` samples a task id from this list each
    # episode, rebuilding the underlying LIBERO sim. Lets one env worker
    # cycle through the full suite without a vec wrapper.
    task_ids: list[int] = field(default_factory=list)
    # LIBERO stores ~50 init states per task; ``reset()`` randomly picks
    # one each episode unless ``trial_id >= 0`` is set (single-trial,
    # used by smoke tests).
    trial_id: int = -1
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


def _coerce_env_cfg(cfg: Any) -> LiberoCosmosCorrectionEnvCfg:
    """Accept either the dataclass directly (smoke tests) or a DictConfig
    coming from the RLinf YAML pipeline (EnvWorker)."""
    if isinstance(cfg, LiberoCosmosCorrectionEnvCfg):
        return cfg
    from omegaconf import OmegaConf

    if hasattr(cfg, "_content") or hasattr(cfg, "to_container"):
        as_dict = OmegaConf.to_container(cfg, resolve=True)
    elif isinstance(cfg, dict):
        as_dict = dict(cfg)
    else:
        raise TypeError(f"unsupported env cfg type: {type(cfg).__name__}")
    valid_fields = {f.name for f in fields(LiberoCosmosCorrectionEnvCfg)}
    filtered = {k: v for k, v in as_dict.items() if k in valid_fields}
    return LiberoCosmosCorrectionEnvCfg(**filtered)


class LiberoCosmosCorrectionEnv(gym.Env):
    """Single LIBERO env wrapped with the cosmos+correction loop."""

    metadata = {"render.modes": []}

    def __init__(
        self,
        cfg: LiberoCosmosCorrectionEnvCfg | Any,
        num_envs: int = 1,
        seed_offset: int = 0,
        total_num_processes: int = 1,
        worker_info: Any = None,
    ):
        super().__init__()
        if num_envs != 1:
            raise NotImplementedError(
                f"LiberoCosmosCorrectionEnv only supports num_envs=1; got {num_envs}. "
                "Multi-env support requires sharing the cosmos model across envs "
                "and is deferred to a follow-up."
            )
        self.cfg = _coerce_env_cfg(cfg)
        self._seed_offset = seed_offset
        self._total_num_processes = total_num_processes
        self._worker_info = worker_info
        cfg = self.cfg  # rebind so the rest of __init__ sees the dataclass

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

        # Cosmos VLA + AE live in a single shared Ray actor so multiple
        # env workers can hit the same GPU-resident model without each
        # paying for its own 8GB cosmos load. Keep a local
        # ``_CosmosCfg`` for the flags we still need on the env side
        # (``flip_images``, ``model_family``, etc.); the actor builds
        # its own copy.
        self.cosmos_cfg = _CosmosCfg(
            chunk_size=cfg.cosmos_chunk_size,
            num_open_loop_steps=cfg.cosmos_num_open_loop_steps,
            num_denoising_steps_action=cfg.cosmos_num_denoising_steps_action,
        )
        from rlinf.envs.libero_cosmos_correction.cosmos_actor import (
            get_or_create_cosmos_actor,
        )

        self.cosmos_actor = get_or_create_cosmos_actor(cfg)
        # One-shot RPC at construction so steady-state step() doesn't
        # round-trip just to read the action scale.
        import ray as _ray

        self._action_scale = np.asarray(
            _ray.get(self.cosmos_actor.get_action_scale.remote()),
            dtype=np.float32,
        )

        # LIBERO benchmark: keep the suite handle so ``reset()`` can
        # rebuild the sim against a different task on demand.
        os.environ.setdefault("MUJOCO_GL", "egl")
        self._bench = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
        n_tasks = self._bench.n_tasks
        if cfg.task_ids:
            for tid in cfg.task_ids:
                if not (0 <= tid < n_tasks):
                    raise ValueError(
                        f"task_id {tid} out of range [0, {n_tasks}) "
                        f"for suite {cfg.task_suite_name}"
                    )
            self._task_pool = list(cfg.task_ids)
        else:
            if not (0 <= cfg.task_id < n_tasks):
                raise ValueError(
                    f"task_id {cfg.task_id} out of range [0, {n_tasks}) "
                    f"for suite {cfg.task_suite_name}"
                )
            self._task_pool = [cfg.task_id]
        self._task_rng = np.random.default_rng(seed_offset)
        self._current_task_id: Optional[int] = None
        self._current_trial_id: Optional[int] = None
        self._init_states = []
        self._libero_env = None
        self._task_description = ""
        self._build_libero_for_task(self._task_pool[0])

        # Per-cycle state (filled in reset / step)
        self._goal: Optional[GoalState] = None
        self._cosmos_action_chunk: Optional[np.ndarray] = None  # (1, T, 7) numpy
        self._step_count: int = 0
        self._last_obs: Any = None  # raw libero obs, kept for prepare_observation
        # Episode-level accumulators surfaced via final_info["episode"] so
        # env_worker can log env/success / env/return / env/episode_len.
        # Dense and sparse are split out so we can tell whether the
        # policy is gaming cos-sim hovering vs. actually hitting success.
        self._episode_return: float = 0.0
        self._episode_dense_return: float = 0.0
        self._episode_sparse_return: float = 0.0
        self._episode_success: bool = False
        # Set by env_worker after the RecordVideo wrapper is constructed.
        # When present, we push a frame per LIBERO sim step (16 cosmos +
        # 1 correction, ~17 frames per chunk_step) directly into the
        # wrapper instead of the wrapper's default 1-frame-per-chunk_step
        # capture. flush_video() is called per episode so each mp4 is
        # one episode rather than one outer epoch.
        self._video_wrapper: Optional[Any] = None

    def _build_libero_for_task(self, task_id: int) -> None:
        """(Re)build the underlying LIBERO sim for ``task_id``.

        We close the previous sim if any so MuJoCo doesn't leak GL
        contexts when cycling tasks across episodes. The list of init
        states for the new task is cached for ``reset()`` to sample
        from.
        """
        if self._libero_env is not None:
            try:
                self._libero_env.close()
            except Exception:
                pass
        task = self._bench.get_task(task_id)
        self._libero_env, self._task_description = get_libero_env(
            task, "cosmos", resolution=self.cfg.env_resolution
        )
        self._current_task_id = task_id
        self._init_states = self._bench.get_task_init_states(task_id)

    # ------------------------------------------------------------------
    # RLinf vectorized env API (single-env: leading num_envs dim is 1)
    # ------------------------------------------------------------------
    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Reset libero + run the first cosmos cycle.

        Returns ``(obs, info)`` matching the gymnasium-style 2-tuple.
        ``obs`` keys (``z_obs`` / ``z_goal``) carry a leading num_envs=1
        dim so the EnvWorker / replay-buffer pipeline sees the same
        layout as standard vec envs.
        """
        if len(self._task_pool) > 1:
            new_task_id = int(self._task_rng.choice(self._task_pool))
            if new_task_id != self._current_task_id:
                self._build_libero_for_task(new_task_id)

        # LIBERO ships ~50 init states per task; randomize unless cfg
        # pins a specific trial id (smoke tests).
        if self.cfg.trial_id >= 0:
            trial_id = self.cfg.trial_id
        else:
            trial_id = int(self._task_rng.integers(0, len(self._init_states)))
        self._current_trial_id = trial_id

        self._libero_env.reset()
        obs = self._libero_env.set_init_state(self._init_states[trial_id])
        for _ in range(self.cfg.num_steps_wait_after_reset):
            obs, _, _, _ = self._libero_env.step(
                get_libero_dummy_action(self.cosmos_cfg.model_family)
            )
        self._last_obs = obs
        self._step_count = 0
        self._episode_return = 0.0
        self._episode_dense_return = 0.0
        self._episode_sparse_return = 0.0
        self._episode_success = False

        # First frame of the new episode so the per-episode mp4 starts
        # at the post-stabilization initial state, not at the cosmos
        # rollout's first chunk step.
        self._push_video_frame(
            info_overlay={
                "step": 0,
                "phase": "reset",
                "task_id": self._current_task_id,
                "trial_id": self._current_trial_id,
            },
            reward=None,
            termination=False,
        )

        self._roll_cosmos_cycle()
        info = {
            "task_id": self._current_task_id,
            "trial_id": self._current_trial_id,
            "task_description": self._task_description,
        }
        return self._batched_state(), info

    def step(
        self,
        actions: np.ndarray | torch.Tensor,
        auto_reset: bool = True,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """One RL step = correction + reward + next cosmos cycle.

        ``actions`` is a ``(1, ACTION_DIM)`` tensor / array (num_envs=1).
        Returns ``(obs, reward, terminations, truncations, info)`` with
        all tensor fields shaped ``(1,)`` (or ``(1, ...)`` for obs values)
        to match the RLinf vec env contract that ``EnvWorker.chunk_step``
        consumes.
        """
        action = self._unbatch_action(actions)
        if self._goal is None or self._cosmos_action_chunk is None:
            raise RuntimeError("Call reset() before step().")

        # 1. Compose the 7-DOF env action: correction in normalized
        #    space, unnormalized via dataset stats; cosmos gripper
        #    passes through.
        last_cosmos = self._cosmos_action_chunk[0, -1]  # numpy (7,)
        correction_raw = action.astype(np.float32) * self._action_scale  # (6,)
        composed = np.concatenate(
            [correction_raw.astype(last_cosmos.dtype), last_cosmos[6:7]],
            axis=-1,
        )

        # 2. Execute one libero step.
        obs, _reward, done, _info = self._libero_env.step(composed.tolist())
        self._last_obs = obs
        self._step_count += 1
        success = bool(done)
        truncated = (not success) and (self._step_count >= self.cfg.max_episode_steps)
        terminal = success or truncated

        # 3. Reward: dense cos-sim on post-correction obs vs cached goal,
        #    plus sparse success bonus.
        z_obs_after = self._encode_obs(obs)
        z_goal = self._goal.goal_z_rl
        dense = float(
            torch.nn.functional.cosine_similarity(
                z_obs_after, z_goal, dim=-1
            ).item()
        )
        sparse = self.cfg.success_bonus if success else 0.0
        dense_term = self.cfg.dense_coef * dense
        reward = dense_term + sparse
        self._episode_return += float(reward)
        self._episode_dense_return += float(dense_term)
        self._episode_sparse_return += float(sparse)
        if success:
            self._episode_success = True

        info: dict[str, Any] = {
            "dense_reward": dense,
            "sparse_reward": sparse,
            "success": success,
            "truncated": truncated,
            "step": self._step_count,
            "task_id": self._current_task_id,
            "trial_id": self._current_trial_id,
            "task_description": self._task_description,
        }

        # Capture the post-correction frame with full overlay info.
        self._push_video_frame(
            info_overlay=info, reward=reward, termination=success
        )

        if terminal:
            # Q is masked by ``(1 - done)``, so the next-state contents
            # are inert. Use the post-correction encoding + cached goal
            # so the dict shape stays consistent.
            terminal_obs_np = {
                "z_obs": z_obs_after.squeeze(0).cpu().numpy(),
                "z_goal": z_goal.squeeze(0).cpu().numpy(),
            }
            info["final_observation"] = self._batch_obs_dict(terminal_obs_np)

            if auto_reset:
                next_obs, _ = self.reset()
            else:
                next_obs = info["final_observation"]
        else:
            self._roll_cosmos_cycle()
            next_obs = self._batched_state()

        rewards = torch.tensor([reward], dtype=torch.float32)
        terminations = torch.tensor([success], dtype=torch.bool)
        truncations = torch.tensor([truncated], dtype=torch.bool)
        return next_obs, rewards, terminations, truncations, info

    def chunk_step(
        self, chunk_actions: torch.Tensor
    ) -> tuple[list[dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        """Roll a chunk of actions through ``step``.

        ``chunk_actions`` has shape ``(num_envs=1, chunk_size, ACTION_DIM)``.
        Inside the chunk we step with ``auto_reset=False`` so terminations
        don't reset mid-chunk; once the chunk is done, if anything
        terminated and ``cfg.auto_reset`` is set we tack on a reset and
        rewrite the last entry to expose ``final_observation`` while
        carrying the post-reset obs forward (mirrors LiberoEnv).
        """
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions_np = chunk_actions.detach().cpu().numpy()
        else:
            chunk_actions_np = np.asarray(chunk_actions)
        if chunk_actions_np.ndim != 3 or chunk_actions_np.shape[0] != 1:
            raise ValueError(
                f"chunk_actions must be (1, T, {ACTION_DIM}); got {chunk_actions_np.shape}"
            )

        chunk_size = chunk_actions_np.shape[1]
        obs_list: list[dict[str, torch.Tensor]] = []
        infos_list: list[dict[str, Any]] = []
        rewards_list: list[torch.Tensor] = []
        terminations_list: list[torch.Tensor] = []
        truncations_list: list[torch.Tensor] = []
        for t in range(chunk_size):
            o, r, term, trunc, info = self.step(chunk_actions_np[:, t], auto_reset=False)
            obs_list.append(o)
            rewards_list.append(r)
            terminations_list.append(term)
            truncations_list.append(trunc)
            infos_list.append(info)
            # Stop rolling further actions through a sim that's already
            # terminated -- libero/robosuite raises on step-after-done.
            if bool(term.item()) or bool(trunc.item()):
                break

        # Pad the chunk back out to ``chunk_size`` if we broke early so
        # downstream tensor shapes stay consistent.
        while len(rewards_list) < chunk_size:
            obs_list.append(obs_list[-1])
            infos_list.append(infos_list[-1])
            rewards_list.append(torch.zeros_like(rewards_list[-1]))
            terminations_list.append(torch.zeros_like(terminations_list[-1]))
            truncations_list.append(torch.zeros_like(truncations_list[-1]))

        chunk_rewards = torch.stack(rewards_list, dim=1)  # (1, T)
        chunk_terminations = torch.stack(terminations_list, dim=1)
        chunk_truncations = torch.stack(truncations_list, dim=1)

        # Post-chunk auto-reset: if anything terminated, reset and stash
        # the terminal obs into ``final_observation`` so the worker can
        # bootstrap correctly; the last ``obs_list`` entry becomes the
        # post-reset state so the next chunk starts fresh.
        any_done = bool(chunk_terminations.any().item() or chunk_truncations.any().item())
        if any_done and getattr(self.cfg, "auto_reset", True):
            terminal_obs = obs_list[-1]
            # Snapshot episode-level metrics BEFORE reset() zeros the
            # accumulators. env_worker indexes these with
            # ``chunk_dones[:, -1]`` so values must be num_envs-shaped (=1).
            episode_metrics = {
                "success": torch.tensor([self._episode_success], dtype=torch.bool),
                "return": torch.tensor([self._episode_return], dtype=torch.float32),
                "dense_return": torch.tensor(
                    [self._episode_dense_return], dtype=torch.float32
                ),
                "sparse_return": torch.tensor(
                    [self._episode_sparse_return], dtype=torch.float32
                ),
                "episode_len": torch.tensor([self._step_count], dtype=torch.int32),
            }
            # Per-episode mp4: flush the wrapper's frame buffer BEFORE
            # resetting so this video covers exactly the just-finished
            # episode. Subsequent reset frames go into the next mp4.
            if self._video_wrapper is not None:
                sub_dir = f"task_{self._current_task_id}"
                try:
                    self._video_wrapper.flush_video(video_sub_dir=sub_dir)
                except Exception as exc:  # noqa: BLE001
                    # Don't crash the env on a flush failure -- training
                    # is the priority; the wrapper's internal warnings
                    # already log the cause.
                    print(f"[libero_cosmos_correction] flush_video failed: {exc}")
            reset_obs, _ = self.reset()
            infos_list[-1] = {
                **infos_list[-1],
                "final_observation": terminal_obs,
                "final_info": {"episode": episode_metrics},
            }
            obs_list[-1] = reset_obs

        return obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list

    def close(self) -> None:
        try:
            self._libero_env.close()
        except Exception:
            pass

    @property
    def seed(self) -> int:
        """Used by the RecordVideo wrapper to namespace output files."""
        return int(self._seed_offset)

    @property
    def num_envs(self) -> int:
        return 1

    def capture_image(self) -> Optional[np.ndarray]:
        """No-op for the RecordVideo wrapper's automatic chunk_step capture.

        We push frames per LIBERO sim step ourselves via ``_push_video_frame``
        (called inside ``step`` and ``_roll_cosmos_cycle``), so returning
        None here disables the wrapper's default 1-frame-per-chunk_step
        capture and avoids a duplicate at the end of each chunk.
        """
        return None

    def register_video_wrapper(self, wrapper: Any) -> None:
        """Receive the RecordVideo wrapper handle from env_worker.

        Called once after construction so subsequent sim-step frames can
        be pushed straight into ``wrapper.render_images``. With no
        wrapper registered (e.g., ``save_video: False``), all the video
        machinery in this file becomes a no-op.
        """
        self._video_wrapper = wrapper

    def _render_libero_frame(self) -> Optional[np.ndarray]:
        """Render the current libero scene as ``(H, W, 3)`` uint8 RGB.

        Returns None if no obs is cached. libero stores images flipped
        vertically; flipud + ascontiguous to match playback expectations.
        """
        if self._last_obs is None or "agentview_image" not in self._last_obs:
            return None
        img = self._last_obs["agentview_image"]
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        img = np.flipud(img)
        return np.ascontiguousarray(img.astype(np.uint8))

    def _push_video_frame(
        self,
        info_overlay: Optional[dict] = None,
        reward: Optional[float] = None,
        termination: Optional[bool] = None,
    ) -> None:
        """Push one frame to the wrapper. Skips if no wrapper registered."""
        if self._video_wrapper is None:
            return
        frame = self._render_libero_frame()
        if frame is None:
            return
        # Reuse the wrapper's overlay/tile path so the result lines up
        # with how it would have rendered through capture_image normally.
        try:
            self._video_wrapper._append_frame(
                images=[frame],
                infos=info_overlay,
                rewards=reward,
                terminations=termination,
            )
        except Exception as exc:  # noqa: BLE001
            # Skip a bad frame rather than killing the rollout.
            print(f"[libero_cosmos_correction] _append_frame failed: {exc}")

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _roll_cosmos_cycle(self) -> None:
        """From ``self._last_obs``: cosmos predict (RPC) -> execute its
        chunk in env -> cache new goal + chunk + post-chunk obs."""
        import ray as _ray

        observation = prepare_observation(
            self._last_obs,
            resize_size=self.cfg.env_resolution,
            flip_images=self.cosmos_cfg.flip_images,
        )
        # The cosmos actor wants numpy on the wire; tensors get pickled
        # and bounced through CPU anyway, so do it explicitly here.
        observation_np = {
            k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v)
            for k, v in observation.items()
        }

        action_chunk_np, goal_z_rl_np = _ray.get(
            self.cosmos_actor.predict_chunk_and_goal.remote(
                observation_np,
                self._task_description,
                int(self.cfg.cosmos_seed),
            )
        )
        # Keep the chunk as numpy (we feed it to the LIBERO sim as a
        # python list anyway) and only lift the goal token onto the
        # GPU when reward time comes.
        self._cosmos_action_chunk = action_chunk_np  # (1, T, 7) numpy
        self._goal = GoalState(
            goal_vae_latent=None,
            goal_z_rl=torch.from_numpy(goal_z_rl_np).to(DEVICE),
        )

        chunk_np = action_chunk_np[0]  # (T, 7) numpy
        n_steps = min(self.cfg.cosmos_num_open_loop_steps, chunk_np.shape[0])
        for t in range(n_steps):
            obs, _, done, _ = self._libero_env.step(chunk_np[t].tolist())
            self._step_count += 1
            self._last_obs = obs
            self._push_video_frame(
                info_overlay={
                    "step": self._step_count,
                    "phase": "cosmos",
                    "cosmos_chunk_step": t,
                    "task_id": self._current_task_id,
                    "trial_id": self._current_trial_id,
                },
                reward=None,
                termination=bool(done),
            )
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
        """raw libero obs -> ``(1, 768)`` z_obs on DEVICE.

        Image preprocessing (jpeg roundtrip + resize + center crop) runs
        locally on the env worker so we ship a small uint8 (1, 3, 224,
        224) tensor over RPC instead of the full obs dict.
        """
        import ray as _ray

        observation = prepare_observation(
            raw_obs,
            resize_size=self.cfg.env_resolution,
            flip_images=self.cosmos_cfg.flip_images,
        )
        img_t = _preprocess_image_for_correction(observation["primary_image"])
        z_obs_np = _ray.get(
            self.cosmos_actor.encode_image_to_rl_token.remote(
                img_t.detach().cpu().numpy()
            )
        )
        return torch.from_numpy(z_obs_np).to(DEVICE)

    def _batched_state(self) -> dict[str, torch.Tensor]:
        """Encode current obs into the agent-facing state dict with a
        leading num_envs=1 dim. Tensors live on CPU; the worker pipeline
        moves them to device as needed."""
        z_obs = self._encode_obs(self._last_obs).detach().cpu()  # (1, 768)
        z_goal = self._goal.goal_z_rl.detach().cpu()  # (1, 768)
        return {"z_obs": z_obs.contiguous(), "z_goal": z_goal.contiguous()}

    def _batch_obs_dict(self, obs_np: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        """Wrap a raw {z_obs, z_goal} numpy dict (no batch dim) into
        the (1, ...) torch dict the worker pipeline expects."""
        return {
            k: torch.as_tensor(v, dtype=torch.float32).unsqueeze(0).contiguous()
            for k, v in obs_np.items()
        }

    def _unbatch_action(self, actions: np.ndarray | torch.Tensor) -> np.ndarray:
        """Validate ``(1, ACTION_DIM)`` shape and pull out the single env's
        action as a numpy array."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.shape == (ACTION_DIM,):
            return actions  # tolerate already-flat input from chunk_step
        if actions.shape != (1, ACTION_DIM):
            raise ValueError(
                f"action shape must be (1, {ACTION_DIM}); got {actions.shape}"
            )
        return actions[0]
