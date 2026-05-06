# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=torch.float32):
    from rlinf.models.embodiment.cosmos_correction.cosmos_correction_policy import (
        CosmosCorrectionPolicy,
    )

    return CosmosCorrectionPolicy(
        token_dim=cfg.get("token_dim", 768),
        action_dim=cfg.get("action_dim", 6),
        chunk_size=cfg.get("num_action_chunks", 1),
        hidden_dim=cfg.get("hidden_dim", 256),
        num_hidden_layers=cfg.get("num_hidden_layers", 2),
        num_critics=cfg.get("num_critics", 2),
        log_std_init=cfg.get("log_std_init", -1.0),
        learnable_log_std=cfg.get("learnable_log_std", False),
    )
