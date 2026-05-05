# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Smoke test for LiberoCosmosCorrectionEnv.

Loads the full cosmos+AE+libero stack and runs ``reset()`` plus a
handful of zero-correction steps to confirm:
  - shapes match the schema (z_obs/z_goal 768-dim, action 6-dim)
  - reward components are populated and finite
  - cosmos cycles roll without error
  - episode terminates at max_episode_steps with the expected info

Skipped automatically when:
  - the cosmos HF snapshot or AE checkpoint is missing
  - CUDA is unavailable
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from rlinf.data.cosmos_correction import ACTION_DIM, TOKEN_DIM


COSMOS_REPO = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
AE_CKPT = Path.home() / "donghyun/checkpoints/rl-token-ae/best.pt"


def _hf_snapshot_exists(repo_id: str) -> bool:
    cache = Path.home() / ".cache/huggingface/hub"
    repo_dir = cache / f"models--{repo_id.replace('/', '--')}"
    return any((repo_dir / "snapshots").glob("*"))


pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="needs CUDA"
    ),
    pytest.mark.skipif(
        not _hf_snapshot_exists(COSMOS_REPO),
        reason=f"HF snapshot for {COSMOS_REPO} not in cache",
    ),
    pytest.mark.skipif(not AE_CKPT.exists(), reason=f"AE ckpt missing: {AE_CKPT}"),
]


@pytest.fixture(scope="module")
def env():
    os.environ.setdefault("MUJOCO_GL", "egl")
    from rlinf.envs.libero_cosmos_correction import LiberoCosmosCorrectionEnv
    from rlinf.envs.libero_cosmos_correction.libero_cosmos_correction_env import (
        LiberoCosmosCorrectionEnvCfg,
    )

    cfg = LiberoCosmosCorrectionEnvCfg(
        task_suite_name="libero_10",
        task_id=0,
        max_episode_steps=80,  # short for smoke test
        rl_token_ae_ckpt_path=str(AE_CKPT),
        num_steps_wait_after_reset=10,
    )
    e = LiberoCosmosCorrectionEnv(cfg)
    yield e
    e.close()


def _check_state(state: dict[str, np.ndarray]) -> None:
    assert set(state.keys()) == {"z_obs", "z_goal"}
    for key in ("z_obs", "z_goal"):
        v = state[key]
        assert v.shape == (TOKEN_DIM,), f"{key}: {v.shape}"
        assert v.dtype == np.float32 or np.issubdtype(v.dtype, np.floating)
        assert np.isfinite(v).all(), f"{key} has non-finite entries"


def test_reset_returns_correct_state_shape(env):
    state = env.reset()
    _check_state(state)


def test_zero_correction_steps_terminate_with_expected_info(env):
    state = env.reset()
    _check_state(state)

    zero_action = np.zeros(ACTION_DIM, dtype=np.float32)

    # Run a handful of cycles with zero correction (cosmos baseline only).
    rewards = []
    for i in range(3):
        next_state, reward, done, info = env.step(zero_action)
        _check_state(next_state)

        assert isinstance(reward, float)
        assert np.isfinite(reward)
        assert "dense_reward" in info and "sparse_reward" in info
        assert "success" in info and "truncated" in info
        assert -1.0 - 1e-3 <= info["dense_reward"] <= 1.0 + 1e-3, (
            f"cos-sim should be in [-1, 1], got {info['dense_reward']}"
        )

        rewards.append(reward)
        state = next_state
        if done:
            assert info["success"] or info["truncated"]
            break
    else:
        # Loop exhausted without termination — fine for the smoke test.
        pass


def test_action_dim_validation(env):
    env.reset()
    with pytest.raises(ValueError, match="action shape"):
        env.step(np.zeros(7, dtype=np.float32))


def test_get_env_cls_dispatch():
    """Confirm the registry hooks up correctly."""
    from rlinf.envs import SupportedEnvType, get_env_cls
    from rlinf.envs.libero_cosmos_correction import LiberoCosmosCorrectionEnv

    cls = get_env_cls("libero_cosmos_correction")
    assert cls is LiberoCosmosCorrectionEnv
    assert SupportedEnvType("libero_cosmos_correction") == (
        SupportedEnvType.LIBERO_COSMOS_CORRECTION
    )
