import hashlib
import json
from types import SimpleNamespace

import pytest

from search_opd.teacher_readiness_opd import (
    EXPECTED_MERGE_STEPS,
    directory_sha256,
    validate_model_checkpoint,
    validate_teacher_specs_ready,
)


def _write_model(path, *, model_type="qwen2"):
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
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
    (path / "model-00001-of-00001.safetensors").write_bytes(b"placeholder")
    (path / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer"}),
        encoding="utf-8",
    )
    (path / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_merge_manifest(path):
    (path / "merge_manifest.json").write_text(
        json.dumps(
            {
                "merge_method": "equal_weight_linear",
                "source_checkpoint_count": 100,
                "source_checkpoint_steps": list(EXPECTED_MERGE_STEPS),
                "source_coefficient": 0.01,
                "output_directory": str(path.resolve()),
                "source_index_sha256": _sha256(
                    path / "model.safetensors.index.json"
                ),
                "config_sha256": _sha256(path / "config.json"),
            }
        ),
        encoding="utf-8",
    )


def test_partial_or_missing_teacher_is_rejected(tmp_path):
    student = tmp_path / "student"
    _write_model(student)
    with pytest.raises(ValueError, match="partial/in-progress"):
        validate_teacher_specs_ready(
            [SimpleNamespace(path=str(tmp_path / ".merged.in_progress"), name="partial")],
            student_path=student,
            output_path=tmp_path / "run",
            verify_safetensors=False,
        )


def test_final_merged_teacher_requires_complete_manifest(tmp_path):
    merged = tmp_path / "merged"
    _write_model(merged)
    with pytest.raises(ValueError, match="merge completion manifest"):
        validate_model_checkpoint(
            merged,
            require_index=True,
            verify_safetensors=False,
        )


def test_complete_merged_teacher_is_eligible_and_digest_can_be_checked(tmp_path):
    student = tmp_path / "student"
    merged = tmp_path / "merged"
    _write_model(student)
    _write_model(merged)
    _write_merge_manifest(merged)
    expected_digest = directory_sha256(merged)

    infos = validate_teacher_specs_ready(
        [
            SimpleNamespace(
                path=str(merged),
                name="temporal_mean_global_step_5_to_500",
                sha256=expected_digest,
            )
        ],
        student_path=student,
        output_path=tmp_path / "run",
        verify_safetensors=False,
    )

    assert infos[0].path == merged.resolve()
    assert infos[0].merged_manifest["source_checkpoint_count"] == 100
    assert infos[0].directory_sha256 == expected_digest


def test_teacher_student_interface_and_output_overlap_are_rejected(tmp_path):
    student = tmp_path / "student"
    teacher = tmp_path / "global_step_30"
    _write_model(student)
    _write_model(teacher, model_type="different")

    with pytest.raises(ValueError, match="interface mismatch"):
        validate_teacher_specs_ready(
            [SimpleNamespace(path=str(teacher), name="teacher")],
            student_path=student,
            output_path=tmp_path / "run",
            verify_safetensors=False,
        )

    with pytest.raises(ValueError, match="overlap"):
        validate_teacher_specs_ready(
            [SimpleNamespace(path=str(teacher), name="teacher")],
            student_path=student,
            output_path=teacher / "new-run",
            verify_safetensors=False,
        )


def test_tokenizer_serialization_variants_with_equal_ids_are_accepted(tmp_path):
    student = tmp_path / "student"
    teacher = tmp_path / "global_step_30"
    _write_model(student)
    _write_model(teacher)

    base_tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": "Ġ",
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "vocab": {"a": 0, "b": 1},
            "merges": ["a b"],
        },
    }
    student_tokenizer = dict(base_tokenizer)
    teacher_tokenizer = dict(base_tokenizer)
    teacher_tokenizer["model"] = dict(base_tokenizer["model"])
    teacher_tokenizer["model"]["merges"] = [["a", "b"]]
    teacher_tokenizer["model"]["ignore_merges"] = False
    (student / "tokenizer.json").write_text(
        json.dumps(student_tokenizer),
        encoding="utf-8",
    )
    (teacher / "tokenizer.json").write_text(
        json.dumps(teacher_tokenizer),
        encoding="utf-8",
    )
    (student / "merges.txt").write_text("a b\n", encoding="utf-8")
    (teacher / "merges.txt").write_text(
        "#version: 0.2\na b\n",
        encoding="utf-8",
    )
    (student / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "chat_template": "student-only-formatting",
            }
        ),
        encoding="utf-8",
    )
    (teacher / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "extra_special_tokens": {},
            }
        ),
        encoding="utf-8",
    )

    infos = validate_teacher_specs_ready(
        [SimpleNamespace(path=str(teacher), name="teacher")],
        student_path=student,
        output_path=tmp_path / "run",
        verify_safetensors=False,
    )

    assert infos[0].path == teacher.resolve()


def test_tokenizer_semantic_mismatch_is_rejected(tmp_path):
    student = tmp_path / "student"
    teacher = tmp_path / "global_step_30"
    _write_model(student)
    _write_model(teacher)
    tokenizer = {
        "model": {
            "type": "BPE",
            "vocab": {"a": 0, "b": 1},
            "merges": ["a b"],
        }
    }
    (student / "tokenizer.json").write_text(
        json.dumps(tokenizer),
        encoding="utf-8",
    )
    changed = json.loads(json.dumps(tokenizer))
    changed["model"]["vocab"]["b"] = 2
    (teacher / "tokenizer.json").write_text(
        json.dumps(changed),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tokenizer interface mismatch"):
        validate_teacher_specs_ready(
            [SimpleNamespace(path=str(teacher), name="teacher")],
            student_path=student,
            output_path=tmp_path / "run",
            verify_safetensors=False,
        )
