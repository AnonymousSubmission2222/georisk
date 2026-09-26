#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASSETS_ROOT="${GEORISK_ASSETS_ROOT:-${ROOT_DIR}/..}"
EVAL_ENTRY="${ROOT_DIR}/src/vlnce_src/eval.py"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/src:${PYTHONPATH:-}"
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PYTHON_EXE="${PYTHON_EXE:-python3}"
GPU_ID="${GPU_ID:-1}"
SIM_GPU_IDS="${SIM_GPU_IDS:-1,1}"
SIMULATOR_TOOL_PORT="${SIMULATOR_TOOL_PORT:-25000}"
DDP_MASTER_PORT="${DDP_MASTER_PORT:-20001}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_WAYPOINTS="${MAX_WAYPOINTS:-200}"
VIEW_MODE="${VIEW_MODE:-dual}"
ALWAYS_HELP="${ALWAYS_HELP:-True}"
USE_GT="${USE_GT:-True}"

STOP_PROB_THRESH="${STOP_PROB_THRESH:-0.85}"            
STOP_PROB_CONSECUTIVE="${STOP_PROB_CONSECUTIVE:-1}"    

DATASET_PATH="${DATASET_PATH:-${ASSETS_ROOT}/TravelUAV_decompressed}"
EVAL_JSON_PATH="${EVAL_JSON_PATH:-${ASSETS_ROOT}/TravelUAV_data_json/data/uav_dataset/seen_valset.json}"
EVAL_SAVE_PATH="${EVAL_SAVE_PATH:-${ROOT_DIR}/eval_test_qwen3vl_dual_2epoch}"

MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/work_dirs/qwen3vl-uav-2b-lora/checkpoint-7350}"
MODEL_BASE="${MODEL_BASE:-${ASSETS_ROOT}/Qwen3-VL-2B-Instruct}"

MAP_SPAWN_AREA_JSON_PATH="${MAP_SPAWN_AREA_JSON_PATH:-${ASSETS_ROOT}/TravelUAV_data_json/data/meta/map_spawnarea_info.json}"
OBJECT_NAME_JSON_PATH="${OBJECT_NAME_JSON_PATH:-${ASSETS_ROOT}/TravelUAV_data_json/data/meta/object_description.json}"
GROUNDINGDINO_CONFIG="${GROUNDINGDINO_CONFIG:-${ROOT_DIR}/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
GROUNDINGDINO_MODEL_PATH="${GROUNDINGDINO_MODEL_PATH:-${ROOT_DIR}/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth}"

if ! command -v "${PYTHON_EXE}" >/dev/null 2>&1; then
  echo "ERROR: python executable not found: ${PYTHON_EXE}" >&2
  exit 1
fi
if [[ ! -f "${EVAL_ENTRY}" ]]; then
  echo "ERROR: eval entry not found: ${EVAL_ENTRY}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model checkpoint dir not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_BASE}/config.json" ]]; then
  echo "ERROR: MODEL_BASE is not a valid HuggingFace model directory: ${MODEL_BASE}" >&2
  echo "Hint: set MODEL_BASE to Qwen3-VL-2B-Instruct path (contains config.json)." >&2
  exit 1
fi

mkdir -p "${EVAL_SAVE_PATH}"

export GEORISK_STOP_PROB_THRESH="${STOP_PROB_THRESH}"
export GEORISK_STOP_PROB_CONSECUTIVE="${STOP_PROB_CONSECUTIVE}"

"${PYTHON_EXE}" "${EVAL_ENTRY}" \
  --run_type eval \
  --name GeoRisk \
  --gpu_id "${GPU_ID}" \
  --sim_gpu_ids "${SIM_GPU_IDS}" \
  --simulator_tool_port "${SIMULATOR_TOOL_PORT}" \
  --DDP_MASTER_PORT "${DDP_MASTER_PORT}" \
  --batchSize "${BATCH_SIZE}" \
  --always_help "${ALWAYS_HELP}" \
  --use_gt "${USE_GT}" \
  --maxWaypoints "${MAX_WAYPOINTS}" \
  --view_mode "${VIEW_MODE}" \
  --dataset_path "${DATASET_PATH}" \
  --eval_save_path "${EVAL_SAVE_PATH}" \
  --model_path "${MODEL_PATH}" \
  --model_base "${MODEL_BASE}" \
  --eval_json_path "${EVAL_JSON_PATH}" \
  --map_spawn_area_json_path "${MAP_SPAWN_AREA_JSON_PATH}" \
  --object_name_json_path "${OBJECT_NAME_JSON_PATH}" \
  --groundingdino_config "${GROUNDINGDINO_CONFIG}" \
  --groundingdino_model_path "${GROUNDINGDINO_MODEL_PATH}"
