import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN_SH = (ROOT / "train.sh").read_text(encoding="utf-8")


def test_search_r1_train_and_validation_files_match_reference_launcher():
    assert "data/search_r1/training_data/train.parquet" in TRAIN_SH
    assert "data/search_r1/test/nq.parquet" in TRAIN_SH
    assert "data/search_r1/test/hotpotqa.parquet" in TRAIN_SH
    assert "data/search_r1/test/bamboogle.parquet" in TRAIN_SH
    assert "data/search_r1/test/musique.parquet" in TRAIN_SH
    assert "data/search_r1/test/triviaqa.parquet" in TRAIN_SH
    assert 'data.val_files="[$VAL_DATA_NQ,$VAL_DATA_HOTPOTQA]"' in TRAIN_SH
    assert "++data.test_files={searchR1_bamboogle:" in TRAIN_SH
    assert "trainer.test_eval_freq" in TRAIN_SH


def test_opd_uses_one_active_teacher_without_data_source_routing():
    worker_text = (ROOT / "fsdp_workers_opd.py").read_text(encoding="utf-8")
    config_text = (ROOT / "config_opd.py").read_text(encoding="utf-8")
    actor_text = (ROOT / "dp_actor_opd.py").read_text(encoding="utf-8")
    assert "teacher_topk_log_probs" in worker_text
    assert "opd_topk_token_ids" in worker_text
    assert '"teacher_aggregation": "single_active_teacher"' in config_text
    assert "adaptive_teacher_selection" in config_text
    assert "teacher_topk_log_probs" in actor_text
    assert "topk_reverse_kl_per_teacher" in actor_text
    assert "active_teacher_index = data.meta_info.get" in actor_text
    # Teacher choice is validation-driven, not a data-source routing key.
    assert "routing_key" not in (ROOT / "fsdp_workers_opd.py").read_text(encoding="utf-8")


def test_opd_top_k_is_initialized_for_actor_and_reference_workers():
    worker_text = (ROOT / "fsdp_workers_opd.py").read_text(encoding="utf-8")
    top_k_assignment = worker_text.index("self.opd_top_k = int(opd_config.get")
    actor_branch = worker_text.index("if self._is_actor:")
    reference_guard = worker_text.index("if not self._is_ref:")
    assert top_k_assignment < actor_branch < reference_guard


def test_opd_update_does_not_mutate_locked_teacher_batch():
    actor_text = (ROOT / "dp_actor_opd.py").read_text(encoding="utf-8")
    assert 'data.batch["teacher_topk_log_probs"] =' not in actor_text


def test_train_script_requires_explicit_immutable_teachers():
    assert "OPD_TEACHER_MANIFEST" in TRAIN_SH
    assert "OPD_TEACHER_MODEL_PATHS" in TRAIN_SH
    assert "requires frozen teacher checkpoints" in TRAIN_SH


def test_train_script_uses_requested_rollout_and_topk_defaults():
    assert 'ROLLOUT_N="${ROLLOUT_N:-4}"' in TRAIN_SH
    assert 'OPD_TOP_K="${OPD_TOP_K:-16}"' in TRAIN_SH
    assert "actor_rollout_ref.rollout.n=\"$ROLLOUT_N\"" in TRAIN_SH
    assert '+actor_rollout_ref.opd.top_k="$OPD_TOP_K"' in TRAIN_SH
    assert "/ssd1/tcbian/Search-OPD/Search-Qwen2.5-3B-Instruct" in TRAIN_SH
    assert 'LOG_FILE="${LOG_FILE:-$PROJECT_DIR/Search-Qwen2.5-3B-Instruct.log}"' in TRAIN_SH


def test_selection_metrics_are_emitted_as_separate_logger_fields():
    trainer_text = (ROOT / "checkpoint_trainer_opd.py").read_text(encoding="utf-8")
    for field in (
        "adaptive_teacher/selected_teacher_indices",
        "adaptive_teacher/teacher_checkpoint",
        "adaptive_teacher/teacher_global_step",
        "adaptive_teacher/weighted_marginal_gain",
        "adaptive_teacher/marginal_question_count",
        "adaptive_teacher/teacher_switch_count",
    ):
        assert field in trainer_text


def test_swanlab_text_metadata_is_backend_local():
    tracking_path = ROOT.parent / "verl-tool" / "verl" / "verl" / "utils" / "tracking.py"
    spec = importlib.util.spec_from_file_location("tracking_under_test", tracking_path)
    tracking_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(tracking_module)
    Tracking = tracking_module.Tracking

    class FakeText:
        def __init__(self, value):
            self.value = value

    class CaptureLogger:
        def __init__(self):
            self.calls = []

        def log(self, *, data, step):
            self.calls.append((data, step))

        def finish(self, **kwargs):
            pass

    class FakeSwanLab(CaptureLogger):
        Text = FakeText

    tracker = Tracking.__new__(Tracking)
    console_logger = CaptureLogger()
    wandb_logger = CaptureLogger()
    swanlab_logger = FakeSwanLab()
    tracker.logger = {
        "console": console_logger,
        "wandb": wandb_logger,
        "swanlab": swanlab_logger,
    }
    data = {
        "adaptive_teacher/selected_teacher_indices": "[2]",
        "adaptive_teacher/teacher_checkpoint": "/checkpoints/global_step_20",
        "adaptive_teacher/selected_teacher_index": 2.0,
    }

    tracker.log(data=data, step=7)

    assert console_logger.calls[0] == (data, 7)
    assert wandb_logger.calls[0] == (data, 7)
    swanlab_data, swanlab_step = swanlab_logger.calls[0]
    assert swanlab_step == 7
    assert swanlab_data["adaptive_teacher/selected_teacher_indices"].value == "[2]"
    assert (
        swanlab_data["adaptive_teacher/teacher_checkpoint"].value
        == "/checkpoints/global_step_20"
    )
    assert swanlab_data["adaptive_teacher/selected_teacher_index"] == 2.0
    assert data["adaptive_teacher/teacher_checkpoint"] == "/checkpoints/global_step_20"
