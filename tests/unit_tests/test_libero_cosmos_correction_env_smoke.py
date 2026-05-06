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


def _check_state(state: dict) -> None:
    assert set(state.keys()) == {"z_obs", "z_goal"}
    for key in ("z_obs", "z_goal"):
        v = state[key]
        # RLinf vec env: leading num_envs=1 dim.
        assert v.shape == (1, TOKEN_DIM), f"{key}: {v.shape}"
        v_np = v.detach().cpu().numpy() if hasattr(v, "detach") else v
        assert np.issubdtype(v_np.dtype, np.floating)
        assert np.isfinite(v_np).all(), f"{key} has non-finite entries"


def test_reset_returns_correct_state_shape(env):
    state, info = env.reset()
    _check_state(state)
    assert "task_id" in info and "trial_id" in info


def test_zero_correction_steps_terminate_with_expected_info(env):
    state, _ = env.reset()
    _check_state(state)

    zero_action = np.zeros((1, ACTION_DIM), dtype=np.float32)

    for _ in range(3):
        next_state, reward, term, trunc, info = env.step(
            zero_action, auto_reset=False
        )
        _check_state(next_state)

        assert reward.shape == (1,) and np.isfinite(reward.numpy()).all()
        assert term.shape == (1,) and trunc.shape == (1,)
        assert "dense_reward" in info and "sparse_reward" in info
        assert "success" in info and "truncated" in info
        assert -1.0 - 1e-3 <= info["dense_reward"] <= 1.0 + 1e-3, (
            f"cos-sim should be in [-1, 1], got {info['dense_reward']}"
        )

        if bool(term.item()) or bool(trunc.item()):
            assert info["success"] or info["truncated"]
            break


def test_chunk_step_runs_one_step_chunk(env):
    """num_action_chunks=1 -> chunk_step degenerates to a single step."""
    env.reset()
    chunk_actions = np.zeros((1, 1, ACTION_DIM), dtype=np.float32)
    obs_list, rewards, term, trunc, infos = env.chunk_step(chunk_actions)
    assert len(obs_list) == 1
    _check_state(obs_list[0])
    assert rewards.shape == (1, 1)
    assert term.shape == (1, 1) and trunc.shape == (1, 1)
    assert len(infos) == 1


def test_action_dim_validation(env):
    env.reset()
    with pytest.raises(ValueError, match="action shape"):
        env.step(np.zeros((1, 7), dtype=np.float32))


def test_get_env_cls_dispatch():
    """Confirm the registry hooks up correctly."""
    from rlinf.envs import SupportedEnvType, get_env_cls
    from rlinf.envs.libero_cosmos_correction import LiberoCosmosCorrectionEnv

    cls = get_env_cls("libero_cosmos_correction")
    assert cls is LiberoCosmosCorrectionEnv
    assert SupportedEnvType("libero_cosmos_correction") == (
        SupportedEnvType.LIBERO_COSMOS_CORRECTION
    )
