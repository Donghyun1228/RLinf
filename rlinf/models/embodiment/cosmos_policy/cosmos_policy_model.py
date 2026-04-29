# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.utils.logging import get_logger


class CosmosPolicyForRLActionPrediction(nn.Module, BasePolicy):
    """Thin wrapper around NVIDIA Cosmos Policy for RLinf integration.

    The Cosmos base policy itself is treated as a frozen prior; RL training is
    expected to happen via a separate residual module attached on top. This
    wrapper therefore only loads the cosmos checkpoint and exposes
    ``predict_action_batch`` so the rollout side can call cosmos inference.
    ``default_forward`` is intentionally left unimplemented — subclass or
    attach a residual module to provide the training-time forward.
    """

    def __init__(
        self,
        cosmos_model: nn.Module,
        cosmos_config: Any,
        dataset_stats: Optional[dict] = None,
        t5_text_embeddings_cache: Optional[dict] = None,
        runtime_cfg: Optional[DictConfig] = None,
    ):
        nn.Module.__init__(self)
        self.cosmos_model = cosmos_model
        self.cosmos_config = cosmos_config
        self.dataset_stats = dataset_stats
        self.t5_text_embeddings_cache = t5_text_embeddings_cache
        self.runtime_cfg = runtime_cfg
        self.logger = get_logger()
        self.global_step = 0

    @classmethod
    def from_config(
        cls,
        cfg: DictConfig,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "CosmosPolicyForRLActionPrediction":
        """Load a cosmos-policy checkpoint and wrap it.

        Expected ``cfg`` keys (under ``cfg``):
            - ``model_path``: HuggingFace repo id or local checkpoint dir.
            - ``cosmos`` (optional): nested DictConfig forwarded to cosmos as
              the ``cfg`` argument expected by ``cosmos_utils.get_model``.
            - ``dataset_stats_path`` (optional): path to per-dataset
              normalization stats (json/npz/pkl per cosmos convention).
            - ``t5_text_embeddings_path`` (optional): path to precomputed T5
              embeddings cache pickle.
        """
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_model as cosmos_get_model,
            init_t5_text_embeddings_cache,
            load_dataset_stats,
        )

        cosmos_cfg = cls._build_cosmos_cfg(cfg)
        cosmos_model, cosmos_config = cosmos_get_model(cosmos_cfg)

        dataset_stats = None
        dataset_stats_path = cfg.get("dataset_stats_path", None)
        if dataset_stats_path:
            dataset_stats = load_dataset_stats(dataset_stats_path)

        t5_cache = None
        t5_path = cfg.get("t5_text_embeddings_path", None)
        if t5_path:
            t5_cache = init_t5_text_embeddings_cache(t5_path)

        if torch_dtype is not None:
            cosmos_model = cosmos_model.to(dtype=torch_dtype)

        return cls(
            cosmos_model=cosmos_model,
            cosmos_config=cosmos_config,
            dataset_stats=dataset_stats,
            t5_text_embeddings_cache=t5_cache,
            runtime_cfg=cfg,
        )

    @staticmethod
    def _build_cosmos_cfg(cfg: DictConfig) -> Any:
        """Translate RLinf-style cfg to the shape cosmos ``get_model`` expects.

        RLinf YAML keeps the checkpoint path at the top level under
        ``model_path`` and groups cosmos-specific options under a ``cosmos:``
        block. Cosmos itself expects a flat cfg with these keys:

        - ``ckpt_path``      : HF repo id or local checkpoint dir
        - ``config``         : experiment name registered in cosmos ConfigStore
        - ``config_file``    : path to ``cosmos_policy/config/config.py``

        Any extra keys under the user's ``cosmos:`` block are forwarded as-is
        so cosmos can read them (e.g. ``planning_model_*``, sampling knobs).
        """
        cosmos_block = cfg.get("cosmos", None)
        if cosmos_block is None:
            extras: dict[str, Any] = {}
        else:
            extras = OmegaConf.to_container(cosmos_block, resolve=True) or {}

        # Cosmos's config_helper.get_config_module converts ``config_file``
        # to an importable dotted module name via str.replace("/", "."), so
        # the value must be a *relative* path that maps to an importable
        # module (e.g. "cosmos_policy/config/config.py" -> cosmos_policy.config.config).
        default_config_file = "cosmos_policy/config/config.py"

        cosmos_cfg = {
            "ckpt_path": extras.pop("ckpt_path", cfg.model_path),
            "config": extras.pop("config", extras.pop("experiment_name", "")),
            "config_file": extras.pop("config_file", default_config_file),
        }
        cosmos_cfg.update(extras)
        return OmegaConf.create(cosmos_cfg)

    def set_global_step(self, global_step: int) -> None:
        self.global_step = global_step

    def freeze_base(self) -> None:
        """Freeze the cosmos base policy; intended when only a residual module
        on top is being trained."""
        for param in self.cosmos_model.parameters():
            param.requires_grad = False
        self.cosmos_model.eval()

    def default_forward(self, **kwargs):
        """Training-time forward.

        Not implemented at the base level — RL training is expected to happen
        through a residual module attached on top. Override in a subclass or
        compose with the residual module's own forward.
        """
        raise NotImplementedError(
            "CosmosPolicyForRLActionPrediction.default_forward is not implemented. "
            "Attach a residual module and override default_forward, or call the "
            "residual module's forward directly from the actor worker."
        )

    @torch.no_grad()
    def predict_action_batch(
        self,
        observation: dict,
        task_description: Any,
        seed: int = 1,
        randomize_seed: bool = False,
        num_denoising_steps_action: int = 5,
        generate_future_state_and_value_in_parallel: bool = True,
        worker_id: int = 0,
        batch_size: int = 1,
        **kwargs,
    ) -> dict:
        """Run cosmos inference on a batch and return action / value /
        future-image predictions.

        Args:
            observation: Multi-modal observation dict expected by cosmos
                (images, proprio, etc.).
            task_description: Either a language string or a precomputed T5
                embedding (numpy array).
        """
        from cosmos_policy.experiments.robot.cosmos_utils import (
            get_action as cosmos_get_action,
        )

        if self.dataset_stats is None:
            raise RuntimeError(
                "dataset_stats must be loaded before calling predict_action_batch. "
                "Set cfg.dataset_stats_path or assign to model.dataset_stats."
            )

        outputs = cosmos_get_action(
            self.cosmos_config,
            self.cosmos_model,
            self.dataset_stats,
            observation,
            task_description,
            seed=seed,
            randomize_seed=randomize_seed,
            num_denoising_steps_action=num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=generate_future_state_and_value_in_parallel,
            worker_id=worker_id,
            batch_size=batch_size,
        )
        return outputs
