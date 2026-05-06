#!/bin/bash
#
# Launch RL training for the cosmos+correction TD3+BC policy on
# LIBERO-10. Spawns a Ray cluster locally and runs
# train_embodied_agent.py with cosmos_correction_libero_10.yaml.
#
# Usage:
#   bash examples/embodiment/train_libero_cosmos_correction.sh [CONFIG_NAME]
#
# Logs are tee'd into logs/<timestamp>-<config>/train.log. Wandb
# requires `wandb login` (or WANDB_API_KEY) once -- the YAML enables
# wandb in addition to local tensorboard.

set -euo pipefail

export EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

# LIBERO + MuJoCo on a headless box: EGL for GL, no Omni weirdness.
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export PYTHONPATH=${REPO_PATH}:${PYTHONPATH:-}

# Cosmos+correction is LIBERO-only; the action shape constants in
# cosmos-policy depend on this.
export ROBOT_PLATFORM=${ROBOT_PLATFORM:-LIBERO}

CONFIG_NAME=${1:-cosmos_correction_libero_10}

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
LOG_FILE="${LOG_DIR}/train.log"
mkdir -p "${LOG_DIR}"

CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"

echo "============================================================" | tee "${LOG_FILE}"
echo "Cosmos+correction TD3+BC training (LIBERO-10)" | tee -a "${LOG_FILE}"
echo "  config:   ${CONFIG_NAME}" | tee -a "${LOG_FILE}"
echo "  log dir:  ${LOG_DIR}" | tee -a "${LOG_FILE}"
echo "  python:   $(which python)" | tee -a "${LOG_FILE}"
echo "  command:" | tee -a "${LOG_FILE}"
echo "    ${CMD}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

exec ${CMD} 2>&1 | tee -a "${LOG_FILE}"
