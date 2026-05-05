# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replay-buffer schema for the cosmos + correction RL pipeline.

States are pre-encoded -- a 768-dim ``z_obs`` (RL token of the
post-cosmos-chunk observation) and a 768-dim ``z_goal`` (RL token of
cosmos's predicted future image). Storing tokens, not raw images,
keeps cosmos and the WanVAE out of the training-time hot path: the
replay buffer and training worker only ever see small float tensors.

Schema (slotted into ``rlinf.data.embodied_io_struct.Trajectory``)::

    actions:    (T, B, 6)    float -- 6-DOF delta-EE correction
    rewards:    (T, B, 1)    float -- dense (cos-sim to goal) + sparse
    dones:      (T, B, 1)    bool
    curr_obs:   {"z_obs": (T, B, 768), "z_goal": (T, B, 768)}  float
    next_obs:   {"z_obs": (T, B, 768), "z_goal": (T, B, 768)}  float

T is the number of correction transitions in the rollout window
(i.e. the number of cosmos cycles, NOT the number of env steps), and
B is the parallel-env batch size.
"""

from __future__ import annotations

from typing import Optional

import torch

from rlinf.data.embodied_io_struct import Trajectory

TOKEN_DIM = 768
ACTION_DIM = 6
Z_OBS_KEY = "z_obs"
Z_GOAL_KEY = "z_goal"


def _ensure_3d(t: torch.Tensor) -> torch.Tensor:
    """``(T, B)`` → ``(T, B, 1)``; pass-through for already-3D."""
    return t.unsqueeze(-1) if t.dim() == 2 else t


def build_correction_trajectory(
    *,
    z_obs: torch.Tensor,
    z_goal: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    next_z_obs: torch.Tensor,
    next_z_goal: torch.Tensor,
    terminations: Optional[torch.Tensor] = None,
    truncations: Optional[torch.Tensor] = None,
    max_episode_length: int = 0,
    model_weights_id: str = "",
) -> Trajectory:
    """Bundle correction-step transitions into the rlinf ``Trajectory`` format.

    All tensor inputs are expected with leading axes ``(T, B, ...)``.
    ``rewards`` / ``dones`` may be ``(T, B)`` or ``(T, B, 1)``; they are
    normalized to ``(T, B, 1)`` here. ``dones`` is cast to ``bool``.

    The returned ``Trajectory`` plugs directly into
    ``TrajectoryReplayBuffer.add_trajectories``.
    """
    T, B = actions.shape[:2]

    if actions.shape != (T, B, ACTION_DIM):
        raise ValueError(
            f"actions: expected (T, B, {ACTION_DIM}), got {tuple(actions.shape)}"
        )
    for name, t in (
        ("z_obs", z_obs),
        ("z_goal", z_goal),
        ("next_z_obs", next_z_obs),
        ("next_z_goal", next_z_goal),
    ):
        if t.shape != (T, B, TOKEN_DIM):
            raise ValueError(
                f"{name}: expected (T, B, {TOKEN_DIM}), got {tuple(t.shape)}"
            )

    rewards = _ensure_3d(rewards).contiguous()
    dones = _ensure_3d(dones).to(torch.bool).contiguous()

    return Trajectory(
        max_episode_length=max_episode_length,
        model_weights_id=model_weights_id,
        actions=actions.contiguous(),
        rewards=rewards,
        dones=dones,
        terminations=terminations,
        truncations=truncations,
        curr_obs={
            Z_OBS_KEY: z_obs.contiguous(),
            Z_GOAL_KEY: z_goal.contiguous(),
        },
        next_obs={
            Z_OBS_KEY: next_z_obs.contiguous(),
            Z_GOAL_KEY: next_z_goal.contiguous(),
        },
    )


def validate_correction_trajectory(traj: Trajectory) -> None:
    """Raise ``ValueError`` if ``traj`` does not match the cosmos-correction
    schema; cheap runtime sanity check before pushing to the buffer."""
    if traj.actions is None:
        raise ValueError("trajectory missing `actions`")
    if traj.actions.dim() < 2:
        raise ValueError(f"actions: expected >=2D, got {tuple(traj.actions.shape)}")

    T, B = traj.actions.shape[:2]
    if traj.actions.shape != (T, B, ACTION_DIM):
        raise ValueError(
            f"actions: expected (T, B, {ACTION_DIM}), got {tuple(traj.actions.shape)}"
        )

    for slot_name, slot in (("curr_obs", traj.curr_obs), ("next_obs", traj.next_obs)):
        for key in (Z_OBS_KEY, Z_GOAL_KEY):
            if key not in slot:
                raise ValueError(f"{slot_name} missing key {key!r}")
            t = slot[key]
            if t.shape != (T, B, TOKEN_DIM):
                raise ValueError(
                    f"{slot_name}[{key!r}]: expected (T, B, {TOKEN_DIM}), "
                    f"got {tuple(t.shape)}"
                )

    for name, t in (("rewards", traj.rewards), ("dones", traj.dones)):
        if t is None:
            raise ValueError(f"trajectory missing `{name}`")
        if t.shape[:2] != (T, B):
            raise ValueError(
                f"{name}: expected leading shape (T, B), got {tuple(t.shape)}"
            )
