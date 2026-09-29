# GeoRisk-IL

Geometry and collision-risk supervision for RGB-only UAV vision-language
navigation on OpenUAV/TravelUAV, built on Qwen3-VL-2B.

## Method

- **Input:** current front/down RGB images, instruction, stage, previous
  displacement and current position.
- **Output:** a stop probability from `<stop>` and ten future 3D trajectory
  points from `<traj>`. The simulator executes the first five points before
  the next observation. There is no separate trajectory-completion model.
- **Geometry alignment:** an alignment head projects final-layer image-token
  states to frozen DA3 joint-view features. 
- **Collision-risk penalty:** current and historical front/down depth and UAV
  poses construct local obstacle geometry. The predicted execution path is
  penalized for insufficient clearance. 
- **Inference:** only the RGB navigation model and trajectory/stop heads run.
  The DA3 teacher, alignment head and geometry losses are not executed.

The default objective is
`L_traj + 0.1 L_stop + 10 L_risk + 0.5 L_geo`, with a 2 m clearance margin.

## Layout

```text
GeoRisk/
  src/georisk/llm/          # model, dataset, losses, prompts and training
  src/georisk/model_wrapper/ # two-phase navigation inference
  src/vlnce_src/           # closed-loop evaluation and collision events
  airsim_plugin/          # simulator server and client
  scripts/                # training and main evaluation launchers
  tools/                  # DA3 cache generation and validation
  utils/metric.py          # separate AirSim/depth collision metrics
  tests/                  # model, checkpoint and geometry regression tests
```

## Environment And Assets

Training uses Python 3.12 and CUDA PyTorch (reference: 2.8.0+cu128).
Install the CUDA build of PyTorch for your machine first, then:

```bash
pip install -r requirements.txt
```

Closed-loop evaluation also requires `requirements_eval.txt`, the AirSim/UE
environments and the OpenUAV metadata. FlashAttention is optional; set
`ATTN_IMPLEMENTATION=sdpa` for training or
`GEORISK_ATTN_IMPLEMENTATION=sdpa` for evaluation when it is unavailable.

Place the following shared training assets beside `GeoRisk/`, or set
`GEORISK_ASSETS_ROOT` to their parent directory. Individual paths can also
be overridden in the launch scripts:

```text
Qwen3-VL-2B-Instruct/
TravelUAV_data_json/data/uav_dataset/trainset.json
TravelUAV_webdataset/
TravelUAV_original_decompressed_merged_all/
TravelUAV_depth_trainset/
TravelUAV_da3_large_joint_teacher/
```

Depth sidecars contain `map/uuid/depth_imgs_uint8.npz`. DA3 caches contain
`map/uuid/frames/<raw_frame_id>/da3_sliced.npy` and `meta.json`.
To build the teacher cache, also provide `Depth-Anything-3-main/` and
`Depth-Anything-3-Checkpoints/DA3-LARGE-1.1/`.

## Training

```bash
cd GeoRisk
# Only needed when a teacher cache has not been generated:
ASSETS_ROOT="${GEORISK_ASSETS_ROOT:-..}"
python tools/precompute_da3_teacher_features.py \
  --input_root "$ASSETS_ROOT/TravelUAV_webdataset" \
  --output_root "$ASSETS_ROOT/TravelUAV_da3_large_joint_teacher" \
  --data_json "$ASSETS_ROOT/TravelUAV_data_json/data/uav_dataset/trainset.json" \
  --da3_repo "$ASSETS_ROOT/Depth-Anything-3-main" \
  --da3_checkpoint "$ASSETS_ROOT/Depth-Anything-3-Checkpoints/DA3-LARGE-1.1" \
  --expected_feat_dim 1024 --cameras frontcamera,downcamera \
  --process_res 252 --output_grid_hw 8 --gpus 0,1,2,3 \
  --workers_per_gpu 1 --batch_size 32 --strict
python tools/validate_da3_teacher_cache.py \
  --cache_root "$ASSETS_ROOT/TravelUAV_da3_large_joint_teacher" \
  --data_json "$ASSETS_ROOT/TravelUAV_data_json/data/uav_dataset/trainset.json" \
  --dense_dataset_root "$ASSETS_ROOT/TravelUAV_original_decompressed_merged_all" \
  --expected_views 2 --expected_feat_dim 1024 --expected_tokens_per_view 64 \
  --report_json "$ASSETS_ROOT/TravelUAV_da3_large_joint_teacher/validation_report.json"
```

```bash
cd GeoRisk
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 MASTER_PORT=29213 \
  PER_DEVICE_TRAIN_BATCH_SIZE=32 GRADIENT_ACCUMULATION_STEPS=1 \
  bash scripts/train_llm_qwen3vl.sh
```

The training command uses four GPUs, two epochs, batch 32/GPU, accumulation 1, learning rate
`5e-5`, LoRA rank 64/alpha 128, five head-warmup steps, 16 workers, and
WDS/depth/trajectory/DA3 cache sizes `16/16/64/64`.
Outputs and `train.log` go to `work_dirs/qwen3vl-uav-2b-lora/`.
Valid checkpoints in that output directory resume automatically, including
the trained non-LoRA parameters and special-token embeddings.

## Closed-Loop Evaluation

Evaluation reads assets from `GEORISK_ASSETS_ROOT` (the project's parent
directory by default) and writes results inside `GeoRisk/`. Start the
simulator server and evaluation in separate terminals:

Evaluation also needs `TravelUAV_decompressed/`, `TravelUAV_envs/` and
`hf_cache/bert-base-uncased/` under the assets root. Set
`GEORISK_BERT_MODEL_PATH` to override the detector text model location and
`GROUNDINGDINO_MODEL_PATH` to locate its weights.

```bash
cd GeoRisk
python airsim_plugin/AirVLNSimulatorServerTool.py --gpus 0,1 --port 25000 \
  --root_path "${GEORISK_ASSETS_ROOT:-..}/TravelUAV_envs"
```

```bash
cd GeoRisk
CUDA_VISIBLE_DEVICES=0 SIM_GPU_IDS=0,1 SIMULATOR_TOOL_PORT=25000 \
  bash scripts/eval_qwen3vl.sh
```

`MODEL_PATH`, `MODEL_BASE`, `DATASET_PATH`, `EVAL_JSON_PATH` and `EVAL_SAVE_PATH`
are overridable. The default split is `seen_valset.json`; the output is
`eval_test_qwen3vl_dual_2epoch`. The execution helper defaults are
`USE_GT=True` and `ALWAYS_HELP=True`, matching the reference evaluation.
Set `EVAL_JSON_PATH` to the desired unseen/full/half JSON explicitly.

