# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Round-trip smoke tests for the cosmos+correction Trajectory schema.

Confirms ``build_correction_trajectory`` produces a ``Trajectory`` that
``TrajectoryReplayBuffer.add_trajectories`` accepts, that
``sample_chunks`` returns the same shapes / keys we put in, and that
``validate_correction_trajectory`` catches the obvious schema errors.
"""

from __future__ import annotations

import pytest
import torch

from rlinf.data.cosmos_correction import (
    ACTION_DIM,
    TOKEN_DIM,
    Z_GOAL_KEY,
    Z_OBS_KEY,
    build_correction_trajectory,
    validate_correction_trajectory,
)
from rlinf.data.replay_buffer import TrajectoryReplayBuffer


def _make_traj(T: int = 4, B: int = 2):
    return build_correction_trajectory(
        z_obs=torch.randn(T, B, TOKEN_DIM),
        z_goal=torch.randn(T, B, TOKEN_DIM),
        actions=torch.randn(T, B, ACTION_DIM),
        rewards=torch.randn(T, B),  # exercises (T, B) -> (T, B, 1) expansion
        dones=torch.zeros(T, B, dtype=torch.float32),  # exercises bool cast
        next_z_obs=torch.randn(T, B, TOKEN_DIM),
        next_z_goal=torch.randn(T, B, TOKEN_DIM),
    )


def test_build_normalizes_shapes_and_dtypes():
    traj = _make_traj(T=5, B=3)

    assert traj.actions.shape == (5, 3, ACTION_DIM)
    assert traj.rewards.shape == (5, 3, 1)
    assert traj.dones.shape == (5, 3, 1)
    assert traj.dones.dtype == torch.bool
    assert traj.curr_obs[Z_OBS_KEY].shape == (5, 3, TOKEN_DIM)
    assert traj.curr_obs[Z_GOAL_KEY].shape == (5, 3, TOKEN_DIM)
    assert traj.next_obs[Z_OBS_KEY].shape == (5, 3, TOKEN_DIM)
    assert traj.next_obs[Z_GOAL_KEY].shape == (5, 3, TOKEN_DIM)


def test_validate_passes_for_well_formed_trajectory():
    validate_correction_trajectory(_make_traj())


def test_validate_rejects_wrong_action_dim():
    traj = _make_traj()
    traj.actions = torch.randn(traj.actions.shape[0], traj.actions.shape[1], 7)
    with pytest.raises(ValueError, match="actions"):
        validate_correction_trajectory(traj)


def test_validate_rejects_missing_obs_key():
    traj = _make_traj()
    del traj.curr_obs[Z_GOAL_KEY]
    with pytest.raises(ValueError, match="z_goal"):
        validate_correction_trajectory(traj)


def test_replay_buffer_round_trip():
    """Push one trajectory, sample N transitions, verify schema preserved."""
    T, B = 8, 2
    traj = _make_traj(T=T, B=B)

    buffer = TrajectoryReplayBuffer(
        seed=0,
        enable_cache=True,
        cache_size=2,
        sample_window_size=2,
    )
    buffer.add_trajectories([traj])

    # __len__ counts trajectories, not transitions; one push => one trajectory.
    assert len(buffer) == 1
    assert buffer.is_ready(min_size=1)

    n = 6
    batch = buffer.sample(num_chunks=n)

    # Action / reward / done shape preserved (B-major after sampling)
    assert batch["actions"].shape[-1] == ACTION_DIM
    assert batch["actions"].shape[0] >= n

    # Nested obs dicts came through with our keys + correct trailing dim
    for slot in ("curr_obs", "next_obs"):
        assert Z_OBS_KEY in batch[slot]
        assert Z_GOAL_KEY in batch[slot]
        assert batch[slot][Z_OBS_KEY].shape[-1] == TOKEN_DIM
        assert batch[slot][Z_GOAL_KEY].shape[-1] == TOKEN_DIM
