import json

import pytest

omegaconf = pytest.importorskip("omegaconf")
from omegaconf import OmegaConf

from search_opd.config_opd import (
    configure_opd,
    load_teacher_manifest,
    parse_teacher_model_paths,
)


def _config(tmp_path):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "path": "/student",
                    "lora_rank": 0,
                    "lora_adapter_path": None,
                },
                "actor": {
                    "strategy": "fsdp2",
                    "ppo_epochs": 1,
                    "ppo_mini_batch_size": 4,
                    "use_dynamic_bsz": False,
                    "use_kl_loss": False,
                    "kl_loss_coef": 0.001,
                    "optim": {"lr": 1e-6},
                },
                "ref": {"model": None},
                "rollout": {
                    "n": 5,
                    "prompt_length": 4096,
                    "response_length": 4096,
                    "temperature": 1.0,
                    "multi_turn": {"max_assistant_turns": 4},
                },
            },
            "algorithm": {"use_kl_in_reward": False},
            "data": {
                "train_batch_size": 4,
                "train_files": ["/train.parquet"],
                "val_files": ["/nq.parquet", "/hotpotqa.parquet"],
                "seed": 1,
            },
            "reward_model": {"reward_manager": "search_r1_qa_em"},
            "trainer": {
                "default_local_dir": str(tmp_path / "run"),
                "total_training_steps": 10,
                "total_epochs": 3,
            },
        }
    )


def _write_model_fixture(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "architectures": ["Qwen2ForCausalLM"],
                "vocab_size": 32,
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "intermediate_size": 32,
                "max_position_embeddings": 128,
            }
        ),
        encoding="utf-8",
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.weight": "model-00001-of-00001.safetensors"
                }
            }
        ),
        encoding="utf-8",
    )
    (path / "model-00001-of-00001.safetensors").write_bytes(b"fixture")
    (path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer"}),
        encoding="utf-8",
    )
    (path / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_manifest_order_and_temporal_steps_are_preserved(tmp_path):
    manifest = tmp_path / "teachers.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "teachers": [
                    {"name": "late", "path": "/run/global_step_105", "step": 105},
                    {"name": "early", "path": "/run/global_step_30", "step": 30},
                ],
            }
        ),
        encoding="utf-8",
    )

    specs, _ = load_teacher_manifest(str(manifest))
    assert [spec.name for spec in specs] == ["late", "early"]
    assert [spec.step for spec in specs] == [105, 30]


def test_manifest_rejects_non_string_teacher_fields(tmp_path):
    manifest = tmp_path / "invalid_teachers.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "teachers": [{"name": None, "path": "/run/global_step_30"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="name"):
        load_teacher_manifest(str(manifest))


def test_path_parser_keeps_explicit_teacher_order():
    specs = parse_teacher_model_paths("/run/global_step_30,/run/global_step_105")
    assert [spec.path for spec in specs] == ["/run/global_step_30", "/run/global_step_105"]
    assert [spec.step for spec in specs] == [30, 105]


def test_duplicate_teacher_paths_are_rejected():
    with pytest.raises(ValueError, match="unique"):
        parse_teacher_model_paths("/run/global_step_30,/run/global_step_30")


def test_empty_teacher_source_is_rejected_as_configuration_error(tmp_path):
    with pytest.raises(ValueError, match="No OPD teachers"):
        parse_teacher_model_paths("")

    with pytest.raises(ValueError, match="No OPD teachers"):
        configure_opd(_config(tmp_path), environ={}, write_manifest=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loss_coef", float("nan")),
        ("task_loss_coef", float("inf")),
        ("max_abs_log_ratio", float("-inf")),
    ],
)
def test_non_finite_objective_values_are_rejected(tmp_path, field, value):
    config = _config(tmp_path)
    config.actor_rollout_ref.opd = {field: value}

    with pytest.raises(ValueError, match="finite"):
        configure_opd(
            config,
            environ={"OPD_TEACHER_MODEL_PATHS": "/run/global_step_30"},
            write_manifest=False,
        )


def test_configure_opd_injects_first_teacher_and_writes_repro_manifest(tmp_path):
    config = _config(tmp_path)
    student = tmp_path / "student"
    teacher_early = tmp_path / "teachers" / "global_step_30"
    teacher_late = tmp_path / "teachers" / "global_step_105"
    _write_model_fixture(student)
    _write_model_fixture(teacher_early)
    _write_model_fixture(teacher_late)
    config.actor_rollout_ref.model.path = str(student)
    configure_opd(
        config,
        environ={
            "OPD_TEACHER_MODEL_PATHS": f"{teacher_early},{teacher_late}",
            "BASE_MODEL_REVISION": "base-rev",
            "DATASET_REVISION": "data-rev",
            "OPD_VERIFY_SAFETENSORS": "0",
        },
    )

    assert config.actor_rollout_ref.ref.model.path == str(teacher_early)
    assert list(config.actor_rollout_ref.opd.teacher_model_paths) == [
        str(teacher_early),
        str(teacher_late),
    ]
    assert config.actor_rollout_ref.actor.use_kl_loss is True
    assert config.actor_rollout_ref.actor.kl_loss_coef == 0.0
    assert config.actor_rollout_ref.opd.top_k == 16
    saved = tmp_path / "run" / "opd_run_manifest.json"
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["teacher_aggregation"] == "single_active_teacher"
    assert payload["objective"]["top_k"] == 16
    assert payload["objective"]["direction"] == "reverse_kl_student_to_teacher"
    assert [item["step"] for item in payload["teachers"]] == [30, 105]
    assert payload["data"]["val_files"] == ["/nq.parquet", "/hotpotqa.parquet"]
