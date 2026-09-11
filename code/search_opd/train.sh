#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_TOOL_DIR="$(cd -- "$SCRIPT_DIR/../verl-tool" && pwd)"
PROJECT_DIR="$(cd -- "$VERL_TOOL_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="${MODEL_PATH:-/ssd2/llm_models/Qwen2.5-3B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_DIR/data/search_r1/training_data/train.parquet}"
VAL_DATA_NQ="${VAL_DATA_NQ:-${VAL_DATA:-$PROJECT_DIR/data/search_r1/test/nq.parquet}}"
VAL_DATA_HOTPOTQA="${VAL_DATA_HOTPOTQA:-$PROJECT_DIR/data/search_r1/test/hotpotqa.parquet}"
TEST_DATA_BAMBOOGLE="${TEST_DATA_BAMBOOGLE:-$PROJECT_DIR/data/search_r1/test/bamboogle.parquet}"
TEST_DATA_MUSIQUE="${TEST_DATA_MUSIQUE:-$PROJECT_DIR/data/search_r1/test/musique.parquet}"
TEST_DATA_TRIVIAQA="${TEST_DATA_TRIVIAQA:-${TEST_DATA_TRIVIALQA:-$PROJECT_DIR/data/search_r1/test/triviaqa.parquet}}"

RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8181/retrieve}"
TOOL_CONFIG="${TOOL_CONFIG:-$VERL_TOOL_DIR/scripts/Search-R1/search_tool_config.yaml}"
AGENT_LOOP_CONFIG="${AGENT_LOOP_CONFIG:-$VERL_TOOL_DIR/scripts/Search-R1/agent_loop_config.yaml}"
PROJECT_NAME="${PROJECT_NAME:-Search-OPD}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Qwen2.5-3B-Instruct-multi-teacher-opd}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/ssd1/tcbian/Search-OPD/Search-Qwen2.5-3B-Instruct}"
LOG_FILE="${LOG_FILE:-$PROJECT_DIR/Search-Qwen2.5-3B-Instruct.log}"
TEACHER_CHECKPOINT_ROOT="${TEACHER_CHECKPOINT_ROOT:-}"
TEACHER_TRAJECTORY_ROOT="${TEACHER_TRAJECTORY_ROOT:-${TEACHER_CHECKPOINT_ROOT:+$TEACHER_CHECKPOINT_ROOT/validation_trajectories}}"
TEACHER_VALIDATION_METRICS_PATH="${TEACHER_VALIDATION_METRICS_PATH:-${TEACHER_CHECKPOINT_ROOT:+$TEACHER_CHECKPOINT_ROOT/validation_checkpoint_metrics.jsonl}}"
OPD_SELECTION_STATE_PATH="${OPD_SELECTION_STATE_PATH:-$CHECKPOINT_DIR/adaptive_teacher_selection.json}"

OPD_TEACHER_MANIFEST="${OPD_TEACHER_MANIFEST:-${TEACHER_MANIFEST:-}}"
OPD_TEACHER_MODEL_PATHS="${OPD_TEACHER_MODEL_PATHS:-${TEACHER_MODEL_PATHS:-}}"
if [[ -z "$OPD_TEACHER_MANIFEST" && -z "$OPD_TEACHER_MODEL_PATHS" && -n "$TEACHER_CHECKPOINT_ROOT" ]]; then
  OPD_TEACHER_MODEL_PATHS="$(
    python - "$TEACHER_CHECKPOINT_ROOT" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
paths = []
for path in sorted(
    (item for item in root.glob("global_step_*") if item.is_dir()),
    key=lambda item: int(item.name.removeprefix("global_step_")),
):
    paths.append(str(path))
if not paths:
    raise SystemExit(f"No global_step_* teacher checkpoints found under {root}")
print(",".join(paths))
PY
  )"
fi
if [[ -z "$OPD_TEACHER_MANIFEST" && -z "$OPD_TEACHER_MODEL_PATHS" ]]; then
  cat >&2 <<'EOF'
Search-OPD requires frozen teacher checkpoints.
Set exactly one of:
  OPD_TEACHER_MANIFEST=/path/to/teacher_manifest.json
  OPD_TEACHER_MODEL_PATHS=/path/to/global_step_30,/path/to/global_step_105
EOF
  exit 2
fi
if [[ -n "$OPD_TEACHER_MANIFEST" && -n "$OPD_TEACHER_MODEL_PATHS" ]]; then
  echo "Set only one of OPD_TEACHER_MANIFEST and OPD_TEACHER_MODEL_PATHS" >&2
  exit 2
fi

N_GPUS="${N_GPUS:-8}"
ROLLOUT_N="${ROLLOUT_N:-4}"
LR="${LR:-1e-6}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.285}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-300}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-500}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-4}"
MAX_OBS_LENGTH="${MAX_OBS_LENGTH:-500}"
MAX_GENERATED_RESPONSE_LENGTH="${MAX_GENERATED_RESPONSE_LENGTH:-$((MAX_RESPONSE_LENGTH * MAX_ASSISTANT_TURNS))}"
MAX_TRAJECTORY_LENGTH="${MAX_TRAJECTORY_LENGTH:-4096}"
SAVE_FREQ="${SAVE_FREQ:-5}"
TEST_FREQ="${TEST_FREQ:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
OPD_LOSS_COEF="${OPD_LOSS_COEF:-1.0}"
OPD_TASK_LOSS_COEF="${OPD_TASK_LOSS_COEF:-0.0}"
OPD_MAX_ABS_LOG_RATIO="${OPD_MAX_ABS_LOG_RATIO:-null}"
OPD_TOP_K="${OPD_TOP_K:-16}"
# Long Search-R1 trajectories can exceed the per-GPU activation budget at 16.
# Keep the effective 512-example mini-batch through gradient accumulation.
ACTOR_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-4}"

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

export RETRIEVER_URL
export OPD_TEACHER_MANIFEST
export OPD_TEACHER_MODEL_PATHS
export OPD_SELECTION_STATE_PATH
export OPD_TOP_K
export OPD_TEACHER_TRAJECTORY_ROOT="$TEACHER_TRAJECTORY_ROOT"
export OPD_TEACHER_VALIDATION_METRICS_PATH="$TEACHER_VALIDATION_METRICS_PATH"
export OPD_TEACHER_SELECTION_INTERVAL="${OPD_TEACHER_SELECTION_INTERVAL:-5}"
export BASE_MODEL_REVISION="${BASE_MODEL_REVISION:-ba7dc0f38bdb4789f40899310796b84ac658fd40}"
export DATASET_REVISION="${DATASET_REVISION:-PeterJinGo/nq_hotpotqa_train@main}"
export REWARD_VERIFIER_REVISION="${REWARD_VERIFIER_REVISION:-search_r1_qa_em-working-tree}"
export TOOL_ENVIRONMENT_REVISION="${TOOL_ENVIRONMENT_REVISION:-search_tool_and_retriever-working-tree}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-$PROJECT_NAME}"
export SEARCH_R1_PRINT_TRAJECTORIES="${SEARCH_R1_PRINT_TRAJECTORIES:-1}"
export SEARCH_R1_PRINT_STEP_TRAJECTORY="${SEARCH_R1_PRINT_STEP_TRAJECTORY:-1}"
export RAY_DEDUP_LOGS=0
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,localhost"
export PYTHONPATH="$VERL_TOOL_DIR/verl:$VERL_TOOL_DIR:$PROJECT_DIR/code:${PYTHONPATH:-}"

if [[ -n "$OPD_TEACHER_MANIFEST" ]]; then
  OPD_EXPECTED_TEACHER_MANIFEST_SHA256="$(
    python - "$OPD_TEACHER_MANIFEST" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
print(digest.hexdigest())
PY
  )"
  export OPD_EXPECTED_TEACHER_MANIFEST_SHA256
fi

PREFLIGHT_ARGS=(
  --student "$MODEL_PATH"
  --output "$CHECKPOINT_DIR"
)
if [[ -n "$OPD_TEACHER_MANIFEST" ]]; then
  PREFLIGHT_ARGS+=(--teacher-manifest "$OPD_TEACHER_MANIFEST")
else
  PREFLIGHT_ARGS+=(--teacher-paths "$OPD_TEACHER_MODEL_PATHS")
fi
if [[ "${OPD_VERIFY_SAFETENSORS:-1}" == "0" ]]; then
  PREFLIGHT_ARGS+=(--no-verify-safetensors)
fi
if [[ "${OPD_COMPUTE_TEACHER_DIGEST:-0}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--compute-digest)
fi
python -m search_opd.preflight_opd "${PREFLIGHT_ARGS[@]}"

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

echo "[$(date -Is)] Search-OPD start"
echo "student=$MODEL_PATH teachers_manifest=${OPD_TEACHER_MANIFEST:-none} teachers_paths=${OPD_TEACHER_MODEL_PATHS:-none}"
echo "train=$TRAIN_DATA validation=[$VAL_DATA_NQ,$VAL_DATA_HOTPOTQA] test=[$TEST_DATA_BAMBOOGLE,$TEST_DATA_MUSIQUE,$TEST_DATA_TRIVIAQA] seed=${DATA_SEED:-1} rollout_n=$ROLLOUT_N"
echo "lr=$LR lr_warmup_steps_ratio=$LR_WARMUP_STEPS_RATIO total_training_steps=$TOTAL_TRAINING_STEPS"
echo "prompt_length=$MAX_PROMPT_LENGTH response_length_per_turn=$MAX_RESPONSE_LENGTH generated_response_length=$MAX_GENERATED_RESPONSE_LENGTH observation_length=$MAX_OBS_LENGTH trajectory_length=$MAX_TRAJECTORY_LENGTH retriever=$RETRIEVER_URL"
echo "opd_loss_coef=$OPD_LOSS_COEF opd_task_loss_coef=$OPD_TASK_LOSS_COEF max_abs_log_ratio=$OPD_MAX_ABS_LOG_RATIO"
echo "opd_top_k=$OPD_TOP_K"
echo "actor_micro_batch_size_per_gpu=$ACTOR_MICRO_BATCH_SIZE_PER_GPU"
echo "checkpoint_dir=$CHECKPOINT_DIR log=$LOG_FILE save_freq=$SAVE_FREQ test_freq=$TEST_FREQ"
echo "adaptive_teacher_state=$OPD_SELECTION_STATE_PATH teacher_trajectory_root=$TEACHER_TRAJECTORY_ROOT teacher_metrics=$TEACHER_VALIDATION_METRICS_PATH interval=$OPD_TEACHER_SELECTION_INTERVAL"

python -m search_opd.main_ppo_opd \
  hydra/job_logging=disabled \
  hydra.job.chdir=false \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="$TRAIN_DATA" \
  data.val_files="[$VAL_DATA_NQ,$VAL_DATA_HOTPOTQA]" \
  "++data.test_files={searchR1_bamboogle:$TEST_DATA_BAMBOOGLE,searchR1_musique:$TEST_DATA_MUSIQUE,searchR1_triviaqa:$TEST_DATA_TRIVIAQA}" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
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
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$ACTOR_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
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
  +actor_rollout_ref.opd.loss_coef="$OPD_LOSS_COEF" \
  +actor_rollout_ref.opd.task_loss_coef="$OPD_TASK_LOSS_COEF" \
  +actor_rollout_ref.opd.max_abs_log_ratio="$OPD_MAX_ABS_LOG_RATIO" \
  +actor_rollout_ref.opd.top_k="$OPD_TOP_K" \
  critic.enable=false \
  reward_model.reward_manager=search_r1_qa_em \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.nnodes=1 \
  trainer.logger="['console','swanlab']" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.test_eval_freq="$TEST_FREQ" \
  trainer.test_before_train=true \
  trainer.test_after_train=true \
  trainer.val_before_train=true \
  trainer.total_epochs="${TOTAL_EPOCHS:-3}" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.resume_mode=disable \
  trainer.max_actor_ckpt_to_keep=null \
  trainer.max_critic_ckpt_to_keep=null \
  trainer.validation_checkpoint_keep_limit=1 \
  +trainer.validation_checkpoint_metrics_path="$CHECKPOINT_DIR/validation_checkpoint_metrics.jsonl" \
  +trainer.validation_trajectory_dir="$CHECKPOINT_DIR/validation_trajectories" \
  +trainer.test_data_dir="$CHECKPOINT_DIR/test_trajectories" \
  trainer.default_local_dir="$CHECKPOINT_DIR" \
  "$@"

echo "[$(date -Is)] Search-OPD finished"
