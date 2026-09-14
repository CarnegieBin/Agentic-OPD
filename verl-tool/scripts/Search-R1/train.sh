#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_TOOL_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PROJECT_DIR="$(cd -- "$VERL_TOOL_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="${MODEL_PATH:-/ssd2/llm_models/Qwen2.5-3B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_DIR/data/search_r1/training_data/train.parquet}"
VAL_DATA="${VAL_DATA:-$PROJECT_DIR/data/search_r1/test/nq.parquet}"
TEST_DATA_HOTPOTQA="${TEST_DATA_HOTPOTQA:-$PROJECT_DIR/data/search_r1/test/hotpotqa.parquet}"

RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8181/retrieve}"
TOOL_CONFIG="${TOOL_CONFIG:-$SCRIPT_DIR/search_tool_config.yaml}"
AGENT_LOOP_CONFIG="${AGENT_LOOP_CONFIG:-$SCRIPT_DIR/agent_loop_config.yaml}"
PROJECT_NAME="${PROJECT_NAME:-Search-OPD}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Qwen2.5-3B-Instruct}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/ssd1/tcbian/$PROJECT_NAME/${EXPERIMENT_NAME}}"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/${PROJECT_NAME}_${EXPERIMENT_NAME}.log}"
RESUME_FROM_STEP="${RESUME_FROM_STEP:-0}"
RESUME_MODEL_PATH="${RESUME_MODEL_PATH:-}"
RESUME_DATA_STEPS="${RESUME_DATA_STEPS:-}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-}"

if [[ ! "$RESUME_FROM_STEP" =~ ^[0-9]+$ ]]; then
  echo "RESUME_FROM_STEP must be a non-negative integer, got: $RESUME_FROM_STEP" >&2
  exit 1
fi

if (( RESUME_FROM_STEP > 0 )); then
  if [[ -z "$RESUME_MODEL_PATH" ]]; then
    RESUME_MODEL_PATH="$CHECKPOINT_DIR/global_step_$RESUME_FROM_STEP"
  fi
  MODEL_PATH="$RESUME_MODEL_PATH"
  if [[ -z "$RESUME_DATA_STEPS" ]]; then
    RESUME_DATA_STEPS="$RESUME_FROM_STEP"
  fi
  if [[ -z "$VAL_BEFORE_TRAIN" ]]; then
    VAL_BEFORE_TRAIN=false
  fi
else
  if [[ -z "$RESUME_DATA_STEPS" ]]; then
    RESUME_DATA_STEPS=0
  fi
  if [[ -z "$VAL_BEFORE_TRAIN" ]]; then
    VAL_BEFORE_TRAIN=true
  fi
fi

if [[ ! "$RESUME_DATA_STEPS" =~ ^[0-9]+$ ]]; then
  echo "RESUME_DATA_STEPS must be a non-negative integer, got: $RESUME_DATA_STEPS" >&2
  exit 1
fi

N_GPUS="${N_GPUS:-8}"
ROLLOUT_N="${ROLLOUT_N:-5}"
LR="${LR:-1e-6}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.285}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-500}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-4}"
MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-500}"
MAX_GENERATED_RESPONSE_LENGTH="${MAX_GENERATED_RESPONSE_LENGTH:-$((MAX_RESPONSE_LENGTH * MAX_ASSISTANT_TURNS))}"
MAX_TRAJECTORY_LENGTH="${MAX_TRAJECTORY_LENGTH:-4096}"
SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:-5}"

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

export RETRIEVER_URL
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-$PROJECT_NAME}"
export SEARCH_R1_PRINT_TRAJECTORIES="${SEARCH_R1_PRINT_TRAJECTORIES:-1}"
export SEARCH_R1_PRINT_STEP_TRAJECTORY="${SEARCH_R1_PRINT_STEP_TRAJECTORY:-1}"
export RAY_DEDUP_LOGS=0
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"
export PYTHONPATH="$VERL_TOOL_DIR/verl:$VERL_TOOL_DIR:$PROJECT_DIR/code:${PYTHONPATH:-}"

if [[ "${SEARCH_R1_SKIP_RETRIEVER_PREFLIGHT:-0}" != "1" ]]; then
  python - "$RETRIEVER_URL" <<'PY'
import sys

import requests

url = sys.argv[1]
response = requests.post(
    url,
    json={"queries": ["who won the first Nobel Prize in Physics"], "topk": 3, "return_scores": True},
    timeout=60,
)
response.raise_for_status()
payload = response.json()
results = payload.get("result")
if not isinstance(results, list) or not results or not results[0]:
    raise RuntimeError(f"Retriever preflight returned no passages: {payload}")
print(f"Retriever preflight passed: status={response.status_code}, passages={len(results[0])}")
PY
fi

echo "[$(date -Is)] Search-R1 start"
echo "model=$MODEL_PATH train=$TRAIN_DATA validation=[$VAL_DATA,$TEST_DATA_HOTPOTQA] seed=${DATA_SEED:-1} rollout_n=$ROLLOUT_N"
echo "lr=$LR lr_warmup_steps_ratio=$LR_WARMUP_STEPS_RATIO total_training_steps=$TOTAL_TRAINING_STEPS"
echo "prompt_length=$MAX_PROMPT_LENGTH response_length_per_turn=$MAX_RESPONSE_LENGTH generated_response_length=$MAX_GENERATED_RESPONSE_LENGTH observation_length=$MAX_OBS_LENGTH trajectory_length=$MAX_TRAJECTORY_LENGTH retriever=$RETRIEVER_URL"
echo "checkpoint_dir=$CHECKPOINT_DIR log=$LOG_FILE save_freq=$SAVE_FREQ test_freq=$TEST_FREQ"
echo "resume_from_step=$RESUME_FROM_STEP resume_model_path=$MODEL_PATH initial_data_steps=$RESUME_DATA_STEPS val_before_train=$VAL_BEFORE_TRAIN"
echo "experiment=$EXPERIMENT_NAME print_trajectories=$SEARCH_R1_PRINT_TRAJECTORIES print_step_trajectory=$SEARCH_R1_PRINT_STEP_TRAJECTORY ray_dedup_logs=$RAY_DEDUP_LOGS"

python -m verl_tool.trainer.main_ppo \
  hydra/job_logging=disabled \
  hydra.job.chdir=false \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="$TRAIN_DATA" \
  data.val_files="[$VAL_DATA,$TEST_DATA_HOTPOTQA]" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-512}" \
  data.val_batch_size="${VAL_BATCH_SIZE:-8192}" \
  data.shuffle=true \
  data.validation_shuffle=false \
  data.seed="${DATA_SEED:-1}" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_TRAJECTORY_LENGTH" \
  +data.max_start_length=2048 \
  +data.max_obs_length="$MAX_OBS_LENGTH" \
  data.filter_overlong_prompts=true \
  data.truncation=left \
  data.return_raw_chat=true \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.optimizer=AdamW \
  actor_rollout_ref.actor.optim.optimizer_impl=torch.optim \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$LR_WARMUP_STEPS_RATIO" \
  actor_rollout_ref.actor.optim.weight_decay="${WEIGHT_DECAY:-0.01}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-512}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-16}" \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.checkpoint.save_contents="['hf_model']" \
  actor_rollout_ref.actor.checkpoint.load_contents="[]" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT_LENGTH" \
  actor_rollout_ref.rollout.response_length="$MAX_TRAJECTORY_LENGTH" \
  actor_rollout_ref.rollout.max_model_len="$((MAX_PROMPT_LENGTH + MAX_TRAJECTORY_LENGTH))" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$((MAX_PROMPT_LENGTH + MAX_TRAJECTORY_LENGTH))" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE:-1}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.6}" \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=0 \
  +actor_rollout_ref.rollout.multi_turn.max_response_length_per_turn="$MAX_RESPONSE_LENGTH" \
  +actor_rollout_ref.rollout.multi_turn.max_generated_response_length="$MAX_GENERATED_RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.multi_turn.format=search_r1 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_WORKERS:-8}" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=false \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.skip_tokenizer_init=true \
  critic.enable=false \
  reward_model.reward_manager=search_r1_qa_em \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.logger="['console','swanlab']" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.total_epochs="${TOTAL_EPOCHS:-3}" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.initial_global_step="$RESUME_FROM_STEP" \
  trainer.initial_data_steps="$RESUME_DATA_STEPS" \
  trainer.resume_mode=disable \
  trainer.max_actor_ckpt_to_keep=null \
  trainer.max_critic_ckpt_to_keep=null \
  trainer.validation_checkpoint_keep_limit=null \
  +trainer.validation_checkpoint_metrics_path="$CHECKPOINT_DIR/validation_checkpoint_metrics.jsonl" \
  +trainer.validation_trajectory_dir="$CHECKPOINT_DIR/validation_trajectories" \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  "$@"

echo "[$(date -Is)] Search-R1 finished"
