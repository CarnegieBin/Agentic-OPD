"""Read-only validation for immutable Search-OPD teacher checkpoints.

The temporal merge publishes ``.merged.in_progress`` with an atomic rename to
``merged``.  A teacher is eligible only after that rename and after the final
directory passes the Hugging Face/safetensors inventory checks below.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


EXPECTED_MERGE_STEPS = tuple(range(5, 501, 5))
MERGE_MANIFEST_NAME = "merge_manifest.json"
MODEL_INDEX_NAME = "model.safetensors.index.json"
MODEL_CONFIG_NAME = "config.json"

_TRANSIENT_COMPONENTS = frozenset(
    {
        ".merged.in_progress",
        "merged.in_progress",
        ".part",
        ".ready",
        ".tmp",
    }
)
_TRANSIENT_SUFFIXES = (".part", ".ready", ".tmp")
_TOKENIZER_ARTIFACTS = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)
_MODEL_INTERFACE_KEYS = (
    "model_type",
    "architectures",
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "max_position_embeddings",
    "rope_theta",
)


@dataclass(frozen=True)
class CheckpointInfo:
    """Validated immutable model-directory facts used by startup guards."""

    path: Path
    config: Mapping[str, Any]
    weight_map: Mapping[str, str]
    shard_names: tuple[str, ...]
    tokenizer_files: tuple[str, ...]
    merged_manifest: Optional[Mapping[str, Any]]
    directory_sha256: Optional[str]


def canonical_path(value: str | Path) -> Path:
    """Return an absolute path without requiring it to exist."""

    text = str(value).strip()
    if not text:
        raise ValueError("Checkpoint path must be non-empty")
    return Path(text).expanduser().resolve(strict=False)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise ValueError(f"Missing {description}: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Empty {description}: {path}")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    _require_file(path, description)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return payload


def _reject_transient_path(path: Path) -> None:
    lowered_parts = {part.lower() for part in path.parts}
    if lowered_parts & _TRANSIENT_COMPONENTS:
        raise ValueError(
            f"Refusing a partial/in-progress checkpoint path: {path}"
        )
    lowered_name = path.name.lower()
    if any(lowered_name.endswith(suffix) for suffix in _TRANSIENT_SUFFIXES):
        raise ValueError(
            f"Refusing a partial/in-progress checkpoint path: {path}"
        )


def _reject_transient_files(path: Path) -> None:
    """Reject files that indicate a non-atomic or active merge."""

    try:
        children = tuple(path.iterdir())
    except OSError as exc:
        raise ValueError(f"Cannot inspect checkpoint directory: {path}") from exc
    for child in children:
        lowered = child.name.lower()
        if (
            lowered in _TRANSIENT_COMPONENTS
            or any(lowered.endswith(suffix) for suffix in _TRANSIENT_SUFFIXES)
            or "in_progress" in lowered
        ):
            raise ValueError(
                f"Checkpoint contains a partial/in-progress artifact: {child}"
            )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""

    return _file_sha256(canonical_path(path))


def directory_sha256(path: str | Path) -> str:
    """Hash stable file names and contents in deterministic path order.

    The digest is intentionally a directory digest rather than a single-shard
    digest.  It can be placed in a teacher manifest and checked on every
    subsequent startup.  Transient merge files are rejected before hashing.
    """

    root = canonical_path(path)
    if not root.is_dir():
        raise ValueError(f"Cannot hash non-directory checkpoint: {root}")
    _reject_transient_path(root)
    _reject_transient_files(root)

    digest = hashlib.sha256()
    files = sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    )
    for candidate in files:
        relative_name = candidate.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative_name).to_bytes(8, "big"))
        digest.update(relative_name)
        with candidate.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_model_config(path: Path) -> dict[str, Any]:
    config = _read_json(path / MODEL_CONFIG_NAME, "model config")
    if not isinstance(config.get("model_type"), str) or not config["model_type"]:
        raise ValueError(f"model config has no usable model_type: {path}")
    if "architectures" in config and not isinstance(config["architectures"], list):
        raise ValueError(f"model config architectures must be a list: {path}")
    return config


def _validate_weight_map(
    path: Path,
) -> tuple[dict[str, str], tuple[str, ...]]:
    index = _read_json(path / MODEL_INDEX_NAME, "safetensors index")
    raw_weight_map = index.get("weight_map")
    if not isinstance(raw_weight_map, dict) or not raw_weight_map:
        raise ValueError(f"safetensors index has no non-empty weight_map: {path}")

    weight_map: dict[str, str] = {}
    for tensor_name, shard_name in raw_weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ValueError(f"safetensors index contains an invalid tensor name: {path}")
        if not isinstance(shard_name, str) or not shard_name:
            raise ValueError(
                f"safetensors index contains an invalid shard name for {tensor_name}: {path}"
            )
        shard_path = Path(shard_name)
        if (
            shard_path.is_absolute()
            or shard_path.name != shard_name
            or ".." in shard_path.parts
            or not shard_name.endswith(".safetensors")
        ):
            raise ValueError(f"Unsafe or unsupported shard path in index: {shard_name}")
        weight_map[tensor_name] = shard_name

    shard_names = tuple(sorted(set(weight_map.values())))
    for shard_name in shard_names:
        shard_path = path / shard_name
        _require_file(shard_path, f"model shard {shard_name}")
    return weight_map, shard_names


def _validate_single_model_file(path: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    for filename in ("model.safetensors", "pytorch_model.bin"):
        candidate = path / filename
        if candidate.is_file():
            _require_file(candidate, "model weights")
            return {}, (filename,)
    raise ValueError(
        f"Checkpoint has neither {MODEL_INDEX_NAME} nor a supported model weight file: {path}"
    )


def _validate_safetensors_inventory(
    path: Path,
    *,
    weight_map: Mapping[str, str],
    shard_names: Iterable[str],
) -> None:
    """Parse every safetensors header and compare it with the index."""

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError(
            "safetensors is required for OPD teacher readiness validation"
        ) from exc

    expected_by_shard: dict[str, set[str]] = {name: set() for name in shard_names}
    for tensor_name, shard_name in weight_map.items():
        expected_by_shard[shard_name].add(tensor_name)

    for shard_name in shard_names:
        shard_path = path / shard_name
        try:
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                actual_keys = set(handle.keys())
        except Exception as exc:
            raise ValueError(f"Cannot read safetensors shard: {shard_path}") from exc
        expected_keys = expected_by_shard.get(shard_name, set())
        if weight_map and actual_keys != expected_keys:
            raise ValueError(
                f"Tensor inventory mismatch in {shard_path}: "
                f"index={len(expected_keys)} actual={len(actual_keys)}"
            )


def _validate_merge_manifest(
    path: Path,
    *,
    config_path: Path,
    index_path: Path,
) -> Mapping[str, Any]:
    manifest = _read_json(path / MERGE_MANIFEST_NAME, "merge completion manifest")
    if manifest.get("merge_method") != "equal_weight_linear":
        raise ValueError(
            f"Unsupported or incomplete merge method in {path / MERGE_MANIFEST_NAME}"
        )
    if manifest.get("source_checkpoint_count") != len(EXPECTED_MERGE_STEPS):
        raise ValueError(
            "Merged teacher must contain exactly 100 source checkpoints"
        )
    if tuple(manifest.get("source_checkpoint_steps", ())) != EXPECTED_MERGE_STEPS:
        raise ValueError(
            "Merged teacher source steps must be global_step_5..global_step_500 "
            "at interval 5"
        )
    source_coefficient = manifest.get("source_coefficient")
    if (
        isinstance(source_coefficient, bool)
        or not isinstance(source_coefficient, (int, float))
        or not math.isfinite(float(source_coefficient))
        or not math.isclose(
            float(source_coefficient),
            1.0 / len(EXPECTED_MERGE_STEPS),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Merged teacher must use a uniform 1/100 source coefficient")

    output_directory = manifest.get("output_directory")
    if output_directory is not None and canonical_path(output_directory) != path:
        raise ValueError(
            f"Merge manifest output_directory does not match checkpoint: {path}"
        )

    source_index_sha256 = manifest.get("source_index_sha256")
    if source_index_sha256 is not None:
        if not isinstance(source_index_sha256, str) or not re_fullmatch_hex(
            source_index_sha256
        ):
            raise ValueError("merge manifest source_index_sha256 is invalid")
        if source_index_sha256.lower() != _file_sha256(index_path):
            raise ValueError("Merged index does not match its recorded source digest")

    config_sha256 = manifest.get("config_sha256")
    if config_sha256 is not None:
        if not isinstance(config_sha256, str) or not re_fullmatch_hex(config_sha256):
            raise ValueError("merge manifest config_sha256 is invalid")
        if config_sha256.lower() != _file_sha256(config_path):
            raise ValueError("Merged config does not match its recorded source digest")
    return manifest


def re_fullmatch_hex(value: str) -> bool:
    """Avoid importing a regular-expression module for one fixed check."""

    return len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def validate_model_checkpoint(
    path: str | Path,
    *,
    require_index: bool,
    require_merge_manifest: bool = False,
    verify_safetensors: bool = True,
    compute_digest: bool = False,
) -> CheckpointInfo:
    """Validate one complete model directory without loading model weights."""

    root = canonical_path(path)
    _reject_transient_path(root)
    if not root.is_dir():
        raise ValueError(f"Teacher/student checkpoint directory is unavailable: {root}")
    _reject_transient_files(root)

    config_path = root / MODEL_CONFIG_NAME
    config = _validate_model_config(root)
    index_path = root / MODEL_INDEX_NAME
    if index_path.is_file():
        weight_map, shard_names = _validate_weight_map(root)
    elif require_index:
        raise ValueError(f"Teacher requires {MODEL_INDEX_NAME}: {root}")
    else:
        weight_map, shard_names = _validate_single_model_file(root)

    if require_index and not all(name.endswith(".safetensors") for name in shard_names):
        raise ValueError(f"Teacher weights must be safetensors shards: {root}")
    if verify_safetensors and all(name.endswith(".safetensors") for name in shard_names):
        _validate_safetensors_inventory(
            root,
            weight_map=weight_map,
            shard_names=shard_names,
        )

    manifest_path = root / MERGE_MANIFEST_NAME
    merged_manifest: Optional[Mapping[str, Any]] = None
    inferred_merged = manifest_path.is_file() or root.name.lower() == "merged"
    if require_merge_manifest or inferred_merged:
        if (root.parent / ".merged.in_progress").exists():
            raise ValueError(
                f"Final merged teacher is not stable while temporary merge exists: {root}"
            )
        merged_manifest = _validate_merge_manifest(
            root,
            config_path=config_path,
            index_path=index_path,
        )

    tokenizer_files = tuple(
        name for name in _TOKENIZER_ARTIFACTS if (root / name).is_file()
    )
    digest = directory_sha256(root) if compute_digest else None
    return CheckpointInfo(
        path=root,
        config=config,
        weight_map=weight_map,
        shard_names=shard_names,
        tokenizer_files=tokenizer_files,
        merged_manifest=merged_manifest,
        directory_sha256=digest,
    )


def _canonical_bpe_merges(
    value: Any,
    *,
    description: str,
) -> tuple[tuple[str, str], ...]:
    """Normalize legacy and modern tokenizer.json BPE merge encodings."""

    if not isinstance(value, list):
        raise ValueError(f"{description} must be a list")
    normalized: list[tuple[str, str]] = []
    for index, merge in enumerate(value):
        if isinstance(merge, list):
            if len(merge) != 2 or not all(
                isinstance(part, str) for part in merge
            ):
                raise ValueError(
                    f"{description}[{index}] must contain two string tokens"
                )
            normalized.append((merge[0], merge[1]))
            continue
        if isinstance(merge, str):
            parts = merge.split(" ", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"{description}[{index}] must contain two space-separated tokens"
                )
            normalized.append((parts[0], parts[1]))
            continue
        raise ValueError(
            f"{description}[{index}] must be a merge pair or encoded string"
        )
    return tuple(normalized)


def _canonical_tokenizer_json(path: Path) -> Mapping[str, Any]:
    """Return tokenizer.json semantics independent of serialization version."""

    payload = _read_json(path / "tokenizer.json", "tokenizer JSON")
    model = payload.get("model")
    if not isinstance(model, dict):
        return payload

    canonical_model = dict(model)
    if canonical_model.get("ignore_merges") is False:
        # New tokenizers serialize this default while older tokenizers omit it.
        canonical_model.pop("ignore_merges")
    if "merges" in canonical_model:
        canonical_model["merges"] = _canonical_bpe_merges(
            canonical_model["merges"],
            description="tokenizer.json model.merges",
        )

    canonical_payload = dict(payload)
    canonical_payload["model"] = canonical_model
    return canonical_payload


def _canonical_merges_file(path: Path) -> tuple[tuple[str, str], ...]:
    """Normalize merges.txt while ignoring its optional format-version header."""

    _require_file(path, "BPE merges")
    normalized: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#version:"):
            continue
        parts = line.split(" ", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid BPE merge line in {path}: {line!r}")
        normalized.append((parts[0], parts[1]))
    return tuple(normalized)


def _canonical_tokenizer_config(path: Path) -> Mapping[str, Any]:
    """Keep model-token semantics, excluding role-side formatting metadata."""

    payload = _read_json(path / "tokenizer_config.json", "tokenizer config")
    # The student tokenizer formats prompts and tool calls.  The frozen teacher
    # only scores the student's already-tokenized IDs, so these fields do not
    # define the teacher/student token-ID interface.
    return {
        key: value
        for key, value in payload.items()
        if key not in {"chat_template", "extra_special_tokens"}
    }


def _tokenizer_signature(
    path: Path,
    files: Sequence[str],
) -> dict[str, Any]:
    """Build a semantic tokenizer signature for teacher/student comparison."""

    signature: dict[str, Any] = {}
    for filename in files:
        artifact = path / filename
        if filename == "tokenizer.json":
            signature[filename] = _canonical_tokenizer_json(path)
        elif filename == "tokenizer_config.json":
            signature[filename] = _canonical_tokenizer_config(path)
        elif filename == "merges.txt":
            signature[filename] = _canonical_merges_file(artifact)
        elif filename.endswith(".json"):
            signature[filename] = _read_json(artifact, filename)
        else:
            signature[filename] = _file_sha256(artifact)
    return signature


def validate_model_interface(
    student: CheckpointInfo,
    teacher: CheckpointInfo,
) -> None:
    """Ensure teacher and student expose the same model/tokenizer interface."""

    for key in _MODEL_INTERFACE_KEYS:
        student_value = student.config.get(key)
        teacher_value = teacher.config.get(key)
        if student_value is not None and teacher_value is not None:
            if student_value != teacher_value:
                raise ValueError(
                    f"Teacher/student model interface mismatch for {key}: "
                    f"student={student_value!r} teacher={teacher_value!r}"
                )

    common_tokenizer_files = tuple(
        sorted(set(student.tokenizer_files) & set(teacher.tokenizer_files))
    )
    student_signature = _tokenizer_signature(student.path, common_tokenizer_files)
    teacher_signature = _tokenizer_signature(teacher.path, common_tokenizer_files)
    if student_signature != teacher_signature:
        mismatched = [
            filename
            for filename in common_tokenizer_files
            if student_signature[filename] != teacher_signature[filename]
        ]
        raise ValueError(
            "Teacher/student tokenizer interface mismatch in: "
            + ", ".join(mismatched)
        )


def validate_teacher_specs_ready(
    specs: Sequence[Any],
    *,
    student_path: str | Path,
    output_path: Optional[str | Path],
    verify_safetensors: bool = True,
    compute_digest: bool = False,
) -> tuple[CheckpointInfo, ...]:
    """Validate all ordered teachers and their relationship to the student."""

    if not specs:
        raise ValueError("At least one OPD teacher is required")
    student_info = validate_model_checkpoint(
        student_path,
        require_index=False,
        verify_safetensors=verify_safetensors,
    )
    output = canonical_path(output_path) if output_path is not None else None
    if output is not None and (
        output == student_info.path
        or _is_relative_to(output, student_info.path)
        or _is_relative_to(student_info.path, output)
    ):
        raise ValueError(
            f"Training output must not overlap the student model: "
            f"output={output} student={student_info.path}"
        )
    teacher_infos: list[CheckpointInfo] = []
    canonical_teachers: set[Path] = set()
    for spec in specs:
        teacher_path = canonical_path(getattr(spec, "path", ""))
        if teacher_path in canonical_teachers:
            raise ValueError(f"Teacher paths alias the same directory: {teacher_path}")
        canonical_teachers.add(teacher_path)
        if teacher_path == student_info.path:
            raise ValueError(
                f"Teacher must be distinct from the student model: {teacher_path}"
            )
        if output is not None and (
            output == teacher_path
            or _is_relative_to(output, teacher_path)
            or _is_relative_to(teacher_path, output)
        ):
            raise ValueError(
                f"Training output must not overlap teacher checkpoint: "
                f"output={output} teacher={teacher_path}"
            )

        is_merged_artifact = (
            teacher_path.name.lower() == "merged"
            or (teacher_path / MERGE_MANIFEST_NAME).is_file()
        )
        expected_digest = getattr(spec, "sha256", None)
        if is_merged_artifact and not expected_digest:
            raise ValueError(
                "Merged teacher requires a verified directory sha256 in its manifest"
            )
        info = validate_model_checkpoint(
            teacher_path,
            require_index=is_merged_artifact,
            require_merge_manifest=is_merged_artifact,
            verify_safetensors=verify_safetensors,
            compute_digest=compute_digest or bool(getattr(spec, "sha256", None)),
        )
        if expected_digest:
            actual_digest = info.directory_sha256 or directory_sha256(info.path)
            if actual_digest.lower() != str(expected_digest).lower():
                raise ValueError(
                    f"Teacher digest mismatch for {getattr(spec, 'name', teacher_path)}: "
                    f"expected={expected_digest} actual={actual_digest}"
                )
        validate_model_interface(student_info, info)
        teacher_infos.append(info)
    return tuple(teacher_infos)
