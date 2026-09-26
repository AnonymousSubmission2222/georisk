#!/usr/bin/env bash


set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ASSETS_ROOT="${GEORISK_ASSETS_ROOT:-${ROOT_DIR}/..}"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export WANDB_DISABLED=true
export USE_LIBUV=0


NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29201}"
PYTHON_EXE="${PYTHON_EXE:-}"


MODEL_PATH="${MODEL_PATH:-${ASSETS_ROOT}/Qwen3-VL-2B-Instruct}"
DATA_JSON="${DATA_JSON:-${ASSETS_ROOT}/TravelUAV_data_json/data/uav_dataset/trainset.json}"
DATASET_PATH="${DATASET_PATH:-${ASSETS_ROOT}/TravelUAV_webdataset}"
DENSE_DATASET_PATH="${DENSE_DATASET_PATH:-${ASSETS_ROOT}/TravelUAV_original_decompressed_merged_all}"
DEPTH_DATASET_PATH="${DEPTH_DATASET_PATH:-${ASSETS_ROOT}/TravelUAV_depth_trainset}"
DA3_JOINT_TEACHER_PATH="${DA3_JOINT_TEACHER_PATH:-${ASSETS_ROOT}/TravelUAV_da3_large_joint_teacher}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/work_dirs/qwen3vl-uav-2b-lora}"


DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-16}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-True}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-1}"
DEPTH_CACHE_SIZE="${GEORISK_DEPTH_CACHE_SIZE:-${DEPTH_CACHE_SIZE:-16}}"
DA3_JOINT_CACHE_SIZE="${GEORISK_DA3_JOINT_CACHE_SIZE:-${DA3_JOINT_CACHE_SIZE:-64}}"
WDS_CACHE_SIZE="${GEORISK_WDS_CACHE_SIZE:-${WDS_CACHE_SIZE:-16}}"
RESAMPLE_TRAJ_CACHE_SIZE="${GEORISK_RESAMPLE_TRAJ_CACHE_SIZE:-${RESAMPLE_TRAJ_CACHE_SIZE:-64}}"
ENABLE_TF32="${ENABLE_TF32:-True}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-True}"
ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-False}"
TORCH_COMPILE_BACKEND="${TORCH_COMPILE_BACKEND:-inductor}"
TORCH_COMPILE_MODE="${TORCH_COMPILE_MODE:-default}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"


NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-8192}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-128}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
NON_LORA_TRAINABLE_LEARNING_RATE="${NON_LORA_TRAINABLE_LEARNING_RATE:-5e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_STEPS="${SAVE_STEPS:-7349}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
HEAD_WARMUP_ENABLE="${HEAD_WARMUP_ENABLE:-True}"
HEAD_WARMUP_STEPS="${HEAD_WARMUP_STEPS:-5}"
HEAD_WARMUP_LEARNING_RATE="${HEAD_WARMUP_LEARNING_RATE:-1e-4}"


LORA_R="${LORA_R:-64}"
LORA_ALPHA=$((LORA_R * 2))
MERGER_LORA_R="${MERGER_LORA_R:-64}"
MERGER_LORA_ALPHA=$((MERGER_LORA_R * 2))
ENABLE_LORA="${ENABLE_LORA:-True}"
USE_QLORA="${USE_QLORA:-False}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-False}"
BNB_4BIT_QUANT_TYPE="${BNB_4BIT_QUANT_TYPE:-nf4}"     
BNB_4BIT_COMPUTE_DTYPE="${BNB_4BIT_COMPUTE_DTYPE:-bfloat16}"
BNB_4BIT_USE_DOUBLE_QUANT="${BNB_4BIT_USE_DOUBLE_QUANT:-True}"


TUNE_MM_LLM="${TUNE_MM_LLM:-False}"
TUNE_MM_VISION="${TUNE_MM_VISION:-True}"
TUNE_MM_MLP="${TUNE_MM_MLP:-True}"
TUNE_MM_MLP_WITH_LORA="${TUNE_MM_MLP_WITH_LORA:-False}"
TUNE_TRAJECTORY_HEAD="${TUNE_TRAJECTORY_HEAD:-True}"
VISUAL_LORA_BLOCK_INDICES="${VISUAL_LORA_BLOCK_INDICES:-20,21,22,23}"
TRAJ_HORIZON="${TRAJ_HORIZON:-10}"
TRAJ_EXECUTE_POINTS="${TRAJ_EXECUTE_POINTS:-5}"
USE_DA3_JOINT_SUPERVISION="${USE_DA3_JOINT_SUPERVISION:-True}"
USE_CLEARANCE_SUPERVISION="${USE_CLEARANCE_SUPERVISION:-True}"
CLEARANCE_BAD_DEPTH_MAPS="${CLEARANCE_BAD_DEPTH_MAPS:-BrushifyCountryRoads,NordicHarbour}"
CLEARANCE_LOSS_WEIGHT="${CLEARANCE_LOSS_WEIGHT:-10.0}"
CLEARANCE_SAFE_MARGIN_M="${CLEARANCE_SAFE_MARGIN_M:-2.0}"
CLEARANCE_DEPTH_MAX_M="${CLEARANCE_DEPTH_MAX_M:-100.0}"
CLEARANCE_VOXEL_SIZE_M="${CLEARANCE_VOXEL_SIZE_M:-0.39215686274509803}"
CLEARANCE_LOCAL_RANGE_M="${CLEARANCE_LOCAL_RANGE_M:-30.0}"
CLEARANCE_DEPTH_POOL_SIZE="${CLEARANCE_DEPTH_POOL_SIZE:-8}"
CLEARANCE_MAX_VOXELS="${CLEARANCE_MAX_VOXELS:-2048}"
CLEARANCE_PATH_SAMPLES_PER_SEGMENT="${CLEARANCE_PATH_SAMPLES_PER_SEGMENT:-4}"
CLEARANCE_TEMPERATURE="${CLEARANCE_TEMPERATURE:-0.25}"
DA3_JOINT_WEIGHT="${DA3_JOINT_WEIGHT:-0.5}"
DA3_JOINT_TEACHER_DIM="${DA3_JOINT_TEACHER_DIM:-1024}"
STRICT_DA3_JOINT_CACHE="${STRICT_DA3_JOINT_CACHE:-True}"
STOP_LOSS_WEIGHT="${STOP_LOSS_WEIGHT:-0.1}"
STOP_RANK_LOSS_WEIGHT="${STOP_RANK_LOSS_WEIGHT:-0.0}"
STOP_RANK_MARGIN="${STOP_RANK_MARGIN:-0.1}"
STOP_RANK_MIN_GAP="${STOP_RANK_MIN_GAP:-2.0}"
STOP_SOFT_R="${STOP_SOFT_R:-20.0}"
STOP_SOFT_TAU="${STOP_SOFT_TAU:-5.0}"
STOP_LABEL_CLIP_EPS="${STOP_LABEL_CLIP_EPS:-1e-4}"
STOP_PHASE2_DISABLE_DIST="${STOP_PHASE2_DISABLE_DIST:-${STOP_SOFT_R}}"
STOP_NEAR_RESAMPLE_RULES="${STOP_NEAR_RESAMPLE_RULES:-0,5,2;5,10,2;10,20,2}"
STOP_TERMINAL_REPEAT="${STOP_TERMINAL_REPEAT:-1}"


VIEW_MODE="${VIEW_MODE:-dual}"


LITE_CAMERA_INDICES="${LITE_CAMERA_INDICES:-0,4}"

DRY_RUN="${DRY_RUN:-False}"

if [[ -z "${PYTHON_EXE}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_EXE="${ROOT_DIR}/.venv/bin/python"
  else
    PYTHON_EXE="python"
  fi
fi

view_mode_lc="$(echo "${VIEW_MODE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${view_mode_lc}" != "dual" ]]; then
  echo "ERROR: GeoRisk requires VIEW_MODE=dual (got '${VIEW_MODE}')." >&2
  exit 1
fi
VIEW_MODE="${view_mode_lc}"

if [[ "${DATALOADER_NUM_WORKERS}" -le 0 ]]; then
  DATALOADER_PERSISTENT_WORKERS="False"
fi

clamp_cache_size() {
  local value="${1:-0}"
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    value=0
  fi
  if (( value > 1024 )); then
    value=1024
  fi
  echo "${value}"
}


export GEORISK_LITE_CAMERA_INDICES="${GEORISK_LITE_CAMERA_INDICES:-${LITE_CAMERA_INDICES}}"
export GEORISK_DEPTH_CACHE_SIZE="$(clamp_cache_size "${DEPTH_CACHE_SIZE}")"
export GEORISK_DA3_JOINT_CACHE_SIZE="$(clamp_cache_size "${DA3_JOINT_CACHE_SIZE}")"
export GEORISK_DA3_JOINT_TEACHER_DIM="${DA3_JOINT_TEACHER_DIM}"
export GEORISK_WDS_CACHE_SIZE="$(clamp_cache_size "${WDS_CACHE_SIZE}")"
export GEORISK_RESAMPLE_TRAJ_CACHE_SIZE="$(clamp_cache_size "${RESAMPLE_TRAJ_CACHE_SIZE}")"

mkdir -p "${OUTPUT_DIR}"
TRAIN_LOG="${OUTPUT_DIR}/train.log"
exec > >(tee -a "${TRAIN_LOG}") 2>&1
echo "Train log   : ${TRAIN_LOG}"

echo "Python      : ${PYTHON_EXE}"
echo "RootDir     : ${ROOT_DIR}"
echo "MODEL_PATH  : ${MODEL_PATH}"
echo "DATA_JSON   : ${DATA_JSON}"
echo "DATASET_PATH: ${DATASET_PATH}"
echo "DENSE_DATA  : ${DENSE_DATASET_PATH}"
echo "DEPTH_DATA  : ${DEPTH_DATASET_PATH}"
echo "DA3_JOINT   : ${DA3_JOINT_TEACHER_PATH}"
echo "OUTPUT_DIR  : ${OUTPUT_DIR}"
echo "ViewMode    : ${VIEW_MODE}"
echo "TF32        : ${ENABLE_TF32}"
echo "GradCkpt    : ${ENABLE_GRADIENT_CHECKPOINTING}"
echo "Compile     : ${ENABLE_TORCH_COMPILE}"
echo "CompileBackend: ${TORCH_COMPILE_BACKEND}"
echo "CompileMode : ${TORCH_COMPILE_MODE}"
echo "MaxLen      : ${MODEL_MAX_LENGTH}"
echo "AttnImpl    : ${ATTN_IMPLEMENTATION}"
echo "DL workers  : ${DATALOADER_NUM_WORKERS}"
echo "DL persist  : ${DATALOADER_PERSISTENT_WORKERS}"
echo "DL prefetch : ${DATALOADER_PREFETCH_FACTOR}"
echo "DL pin mem  : True"
echo "DepthCache  : ${GEORISK_DEPTH_CACHE_SIZE}"
echo "DA3JCache   : ${GEORISK_DA3_JOINT_CACHE_SIZE}"
echo "WDSCache    : ${GEORISK_WDS_CACHE_SIZE}"
echo "ResampCache : ${GEORISK_RESAMPLE_TRAJ_CACHE_SIZE}"
echo "LogSteps    : ${LOGGING_STEPS}"
echo "4bit        : ${LOAD_IN_4BIT}"
echo "QuantType   : ${BNB_4BIT_QUANT_TYPE}"
echo "QuantDtype  : ${BNB_4BIT_COMPUTE_DTYPE}"
echo "DoubleQuant : ${BNB_4BIT_USE_DOUBLE_QUANT}"
echo "StopLossW   : ${STOP_LOSS_WEIGHT}"
echo "StopRankW   : ${STOP_RANK_LOSS_WEIGHT}"
echo "StopRankM   : ${STOP_RANK_MARGIN}"
echo "StopRankGap : ${STOP_RANK_MIN_GAP}"
echo "StopSoftR   : ${STOP_SOFT_R}"
echo "StopSoftTau : ${STOP_SOFT_TAU}"
echo "StopClipEps : ${STOP_LABEL_CLIP_EPS}"
echo "StopPhase2D : ${STOP_PHASE2_DISABLE_DIST}"
echo "StopResamp  : ${STOP_NEAR_RESAMPLE_RULES}"
echo "StopRepeat  : ${STOP_TERMINAL_REPEAT}"
echo "TrajHorizon : ${TRAJ_HORIZON}"
echo "TrajExecPts : ${TRAJ_EXECUTE_POINTS}"
echo "Clearance   : ${USE_CLEARANCE_SUPERVISION} safe_margin=${CLEARANCE_SAFE_MARGIN_M} weight=${CLEARANCE_LOSS_WEIGHT}"
echo "ClearPool   : local_min_argmin_v1 size=${CLEARANCE_DEPTH_POOL_SIZE}"
echo "DA3 enabled : ${USE_DA3_JOINT_SUPERVISION}"
echo "DA3 weight  : ${DA3_JOINT_WEIGHT}"
echo "DA3 dim     : ${DA3_JOINT_TEACHER_DIM}"
echo "DA3 strict  : ${STRICT_DA3_JOINT_CACHE}"
echo "Head LR     : ${NON_LORA_TRAINABLE_LEARNING_RATE}"
echo "HeadWarmup  : ${HEAD_WARMUP_ENABLE}"
echo "WarmupSteps : ${HEAD_WARMUP_STEPS}"
echo "WarmupLR    : ${HEAD_WARMUP_LEARNING_RATE}"
echo "LiteCamIdx  : ${GEORISK_LITE_CAMERA_INDICES}"

train_args=(
  -m georisk.llm.train_uav_qwen
  --model_name_or_path "${MODEL_PATH}"
  --data_path "${DATA_JSON}"
  --dataset_path "${DATASET_PATH}"
  --dense_dataset_path "${DENSE_DATASET_PATH}"
  --depth_dataset_path "${DEPTH_DATASET_PATH}"
  --da3_joint_teacher_path "${DA3_JOINT_TEACHER_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --bf16 True
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --save_strategy steps
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --learning_rate "${LEARNING_RATE}"
  --non_lora_trainable_learning_rate "${NON_LORA_TRAINABLE_LEARNING_RATE}"
  --optim adamw_torch_fused
  --max_grad_norm 1.0
  --weight_decay 0.0
  --warmup_ratio "${WARMUP_RATIO}"
  --lr_scheduler_type cosine
  --logging_steps "${LOGGING_STEPS}"
  --head_warmup_enable "${HEAD_WARMUP_ENABLE}"
  --head_warmup_steps "${HEAD_WARMUP_STEPS}"
  --head_warmup_learning_rate "${HEAD_WARMUP_LEARNING_RATE}"
  --model_max_length "${MODEL_MAX_LENGTH}"
  --gradient_checkpointing "${ENABLE_GRADIENT_CHECKPOINTING}"
  --tf32 "${ENABLE_TF32}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --dataloader_persistent_workers "${DATALOADER_PERSISTENT_WORKERS}"
  --dataloader_pin_memory True
  --remove_unused_columns False
  --report_to none
  --lora_enable "${ENABLE_LORA}"
  --lora_r "${LORA_R}"
  --lora_alpha "${LORA_ALPHA}"
  --merger_lora_r "${MERGER_LORA_R}"
  --merger_lora_alpha "${MERGER_LORA_ALPHA}"
  --use_qlora "${USE_QLORA}"
  --load_in_4bit "${LOAD_IN_4BIT}"
  --bnb_4bit_quant_type "${BNB_4BIT_QUANT_TYPE}"
  --bnb_4bit_compute_dtype "${BNB_4BIT_COMPUTE_DTYPE}"
  --bnb_4bit_use_double_quant "${BNB_4BIT_USE_DOUBLE_QUANT}"
  --tune_mm_llm "${TUNE_MM_LLM}"
  --tune_mm_vision "${TUNE_MM_VISION}"
  --tune_mm_mlp "${TUNE_MM_MLP}"
  --tune_mm_mlp_with_lora "${TUNE_MM_MLP_WITH_LORA}"
  --tune_trajectory_head "${TUNE_TRAJECTORY_HEAD}"
  --visual_lora_block_indices "${VISUAL_LORA_BLOCK_INDICES}"
  --traj_horizon "${TRAJ_HORIZON}"
  --traj_execute_points "${TRAJ_EXECUTE_POINTS}"
  --use_da3_joint_supervision "${USE_DA3_JOINT_SUPERVISION}"
  --use_clearance_supervision "${USE_CLEARANCE_SUPERVISION}"
  --clearance_bad_depth_maps "${CLEARANCE_BAD_DEPTH_MAPS}"
  --clearance_loss_weight "${CLEARANCE_LOSS_WEIGHT}"
  --clearance_safe_margin_m "${CLEARANCE_SAFE_MARGIN_M}"
  --clearance_depth_max_m "${CLEARANCE_DEPTH_MAX_M}"
  --clearance_voxel_size_m "${CLEARANCE_VOXEL_SIZE_M}"
  --clearance_local_range_m "${CLEARANCE_LOCAL_RANGE_M}"
  --clearance_depth_pool_size "${CLEARANCE_DEPTH_POOL_SIZE}"
  --clearance_max_voxels "${CLEARANCE_MAX_VOXELS}"
  --clearance_path_samples_per_segment "${CLEARANCE_PATH_SAMPLES_PER_SEGMENT}"
  --clearance_temperature "${CLEARANCE_TEMPERATURE}"
  --da3_joint_weight "${DA3_JOINT_WEIGHT}"
  --da3_joint_teacher_dim "${DA3_JOINT_TEACHER_DIM}"
  --strict_da3_joint_cache "${STRICT_DA3_JOINT_CACHE}"
  --stop_loss_weight "${STOP_LOSS_WEIGHT}"
  --stop_rank_loss_weight "${STOP_RANK_LOSS_WEIGHT}"
  --stop_rank_margin "${STOP_RANK_MARGIN}"
  --stop_rank_min_gap "${STOP_RANK_MIN_GAP}"
  --stop_terminal_repeat "${STOP_TERMINAL_REPEAT}"
  --stop_soft_r "${STOP_SOFT_R}"
  --stop_soft_tau "${STOP_SOFT_TAU}"
  --stop_label_clip_eps "${STOP_LABEL_CLIP_EPS}"
  --stop_phase2_disable_dist "${STOP_PHASE2_DISABLE_DIST}"
  --stop_near_resample_rules "${STOP_NEAR_RESAMPLE_RULES}"
  --attn_implementation "${ATTN_IMPLEMENTATION}"
  --view_mode "${VIEW_MODE}"
)


if [[ "${DATALOADER_NUM_WORKERS}" -gt 0 ]]; then
  train_args+=(--dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}")
fi

enable_compile_lc="$(echo "${ENABLE_TORCH_COMPILE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${enable_compile_lc}" == "true" || "${enable_compile_lc}" == "1" || "${enable_compile_lc}" == "yes" ]]; then
  train_args+=(
    --torch_compile True
    --torch_compile_backend "${TORCH_COMPILE_BACKEND}"
    --torch_compile_mode "${TORCH_COMPILE_MODE}"
  )
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  cmd=(
    "${PYTHON_EXE}"
    -m torch.distributed.run
    --nproc_per_node="${NPROC_PER_NODE}"
    --master_port="${MASTER_PORT}"
    "${train_args[@]}"
  )
  echo "Launch mode : torch.distributed.run (nproc=${NPROC_PER_NODE})"
else
  cmd=(
    "${PYTHON_EXE}"
    "${train_args[@]}"
  )
  echo "Launch mode : single process (no torchrun)"
fi

dry_run_lc="$(echo "${DRY_RUN}" | tr '[:upper:]' '[:lower:]')"
if [[ "${dry_run_lc}" == "true" || "${dry_run_lc}" == "1" || "${dry_run_lc}" == "yes" ]]; then
  echo "DryRun enabled. Command prepared but not executed."
  exit 0
fi

"${cmd[@]}"
