#!/usr/bin/env bash
set -Eeuo pipefail

# Single Search-R1 entry point. Override paths and budgets with environment variables.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_TOOL_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PROJECT_DIR="$(cd -- "$VERL_TOOL_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="${MODEL_PATH:-/ssd2/llm_models/Qwen3-1.7B}"
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_DIR/data/search_r1/training_data/train.parquet}"
VAL_DATA="${VAL_DATA:-$PROJECT_DIR/data/search_r1/test/nq.parquet,$PROJECT_DIR/data/search_r1/test/hotpotqa.parquet}"
RETRIEVER_PATH="${RETRIEVER_PATH:-}"
N_GPUS="${N_GPUS:-8}"
ROLLOUT_N="${ROLLOUT_N:-5}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
PROJECT_NAME="${PROJECT_NAME:-Search-OPD}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-search-r1-qwen3-1.7b}"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/train-$(date +%Y%m%d-%H%M%S).log}"

[[ -d "$MODEL_PATH" ]] || { echo "Model path not found: $MODEL_PATH" >&2; exit 1; }
[[ -f "$TRAIN_DATA" ]] || { echo "Train data not found: $TRAIN_DATA" >&2; exit 1; }
[[ -n "$RETRIEVER_PATH" ]] && export RETRIEVER_PATH

exec > >(tee -a "$LOG_FILE") 2>&1
echo "[$(date -Is)] Search-R1 start"
echo "model=$MODEL_PATH train=$TRAIN_DATA val=$VAL_DATA gpus=$N_GPUS"
echo "log=$LOG_FILE"

export PYTHONPATH="$VERL_TOOL_DIR/verl:$VERL_TOOL_DIR:$PROJECT_DIR/code:${PYTHONPATH:-}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-$PROJECT_NAME}"

EXTRA_ARGS=()
if (( $# )); then
  EXTRA_ARGS=("$@")
fi

python -m verl_tool.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-128}" \
  data.val_batch_size="${VAL_BATCH_SIZE:-256}" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=true \
  data.truncation=left \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr="${LR:-1e-6}" \
  actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY:-0.1}" \
  actor_rollout_ref.actor.optim.optimizer=SearchOPDMuon \
  actor_rollout_ref.actor.optim.optimizer_impl=muon \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE:-1}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.7}" \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.logger="['console','swanlab']" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.save_freq="${SAVE_FREQ:-100}" \
  trainer.test_freq="${TEST_FREQ:-100}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  "${EXTRA_ARGS[@]}"

echo "[$(date -Is)] Search-R1 finished"
