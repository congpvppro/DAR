#!/usr/bin/env bash
# Fresh full SFT per arm: frozen vision, train LLM + aligner, like DAR baseline.
set -euo pipefail
: "${BASE_MODEL:?Set BASE_MODEL to the local Qwen2.5-VL-3B-Instruct directory}"
: "${DATA_DIR:?Set DATA_DIR to prepare.py build output}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a fresh output directory}"
ARM="${ARM:-stsg}"
case "$ARM" in baseline|caption|stsg|stsg_no_links) ;; *) exit 2 ;; esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Use the installed, version-pinned runtime; do not shadow it with the repo's
# independent 4.0 development checkout when using the Kaggle 3.12.5 wheelhouse.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export FPS_MIN_FRAMES=16 FPS_MAX_FRAMES=16 VIDEO_MIN_PIXELS=100352 VIDEO_MAX_PIXELS=100352
export VIDEO_MIN_TOKEN_NUM=128 VIDEO_MAX_TOKEN_NUM=128
export TOKENIZERS_PARALLELISM=false
SEED="${SEED:-1234}"
DEST="$OUTPUT_ROOT/$ARM-seed$SEED"
if [[ -e "$DEST" ]]; then echo "Output already exists: $DEST" >&2; exit 2; fi
python -m swift.cli.sft \
  --model "$BASE_MODEL" --model_type qwen2_5_vl --template qwen2_5_vl \
  --dataset "$DATA_DIR/$ARM.jsonl" --split_dataset_ratio 0 \
  --train_type full --freeze_vit true --freeze_llm false --freeze_aligner false \
  --torch_dtype bfloat16 --fp16 false --bf16 true --attn_impl sdpa \
  --max_length 8192 --truncation_strategy delete --strict true \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-0.5}" --learning_rate "${LEARNING_RATE:-1e-5}" \
  --optim adamw_torch --weight_decay 0 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-32}" \
  --gradient_checkpointing true --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --seed "$SEED" --data_seed "$SEED" --dataloader_num_workers 0 \
  --save_strategy epoch --save_total_limit 1 --save_only_model true \
  --logging_steps 1 --report_to none --output_dir "$DEST"
