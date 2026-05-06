# `libero_cosmos_correction` — environment setup

LIBERO + Cosmos Policy + correction MLP. This env wrapper drives a frozen
Cosmos Policy through LIBERO and exposes RL-token observations
(`z_obs`, `z_goal`) so an external correction policy can train on top.

## venv install (one shot)

```bash
bash requirements/install.sh embodied --model cosmos-policy --env maniskill_libero --install-rlinf
```

This is the **only** supported install path for this env. It:

- creates `.venv/` (uv, Python 3.10.18)
- pulls torch 2.7.0+cu128 / flash-attn / TE / natten via the cosmos `[cu128]` extra
- clones `Donghyun1228/cosmos-policy-RL.git` → `third_party/cosmos-policy/` and pip-installs editable
- clones `RLinf/LIBERO.git` → `.venv/libero/` and pip-installs editable (PYTHONPATH wired in `activate`)
- installs `rlinf` editable (because of `--install-rlinf`)

After it finishes, `source .venv/bin/activate` and you're done.

## Manual prep (install.sh does not handle these)

| What | Where it must end up | How |
|---|---|---|
| Cosmos HF snapshot | `~/.cache/huggingface/hub/models--nvidia--Cosmos-Policy-LIBERO-Predict2-2B/` | `huggingface-cli download nvidia/Cosmos-Policy-LIBERO-Predict2-2B` |
| RL-token AE checkpoint | `~/donghyun/checkpoints/rl-token-ae/best.pt` (default in `LiberoCosmosCorrectionEnvCfg`) | Train via `cosmos-policy` repo: `build_libero_vae_cache.py` → `train_rl_token_ae.py` |

The smoke test [tests/unit_tests/test_libero_cosmos_correction_env_smoke.py](../../../tests/unit_tests/test_libero_cosmos_correction_env_smoke.py) auto-skips when either is missing, so you'll see a skip — not a failure — until both are in place.

## Runtime env vars

`MUJOCO_GL=egl` is required for headless rendering; the env fixture sets it
automatically, but set it yourself when running outside pytest.
