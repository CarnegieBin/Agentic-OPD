"""Configuration and immutable teacher-manifest handling for Search-OPD."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from omegaconf import DictConfig, OmegaConf, open_dict


_STEP_PATTERN = re.compile(r"(?:global_)?step[_-]?(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class TeacherSpec:
    """One immutable temporal teacher checkpoint."""

    name: str
    path: str
    step: Optional[int] = None
    sha256: Optional[str] = None


def _teacher_step(path: str) -> Optional[int]:
    match = _STEP_PATTERN.search(path)
    return int(match.group(1)) if match else None


def _validate_teacher_specs(specs: Sequence[TeacherSpec]) -> tuple[TeacherSpec, ...]:
    if not specs:
        raise ValueError(
            "No OPD teachers configured. Set OPD_TEACHER_MANIFEST or "
            "OPD_TEACHER_MODEL_PATHS."
        )
    names = [spec.name for spec in specs]
    paths = [spec.path for spec in specs]
    if any(not name.strip() for name in names):
        raise ValueError("Every teacher must have a non-empty name")
    if any(not path.strip() for path in paths):
        raise ValueError("Every teacher must have a non-empty checkpoint path")
    if len(set(names)) != len(names):
        raise ValueError(f"Teacher names must be unique: {names}")
    if len(set(paths)) != len(paths):
        raise ValueError(f"Teacher checkpoint paths must be unique: {paths}")
    for spec in specs:
        if spec.step is not None and spec.step < 0:
            raise ValueError(f"Teacher step must be non-negative: {spec}")
        if spec.sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", spec.sha256):
            raise ValueError(f"Teacher sha256 must contain exactly 64 hexadecimal characters: {spec.name}")
    return tuple(specs)


def _finite_float(value: Any, field_name: str) -> float:
    """Convert a configuration scalar and reject NaN or infinity."""

    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


def _validate_objective_values(
    opd_loss_coef: Any,
    task_loss_coef: Any,
    max_abs_log_ratio: Any,
) -> tuple[float, float, Optional[float]]:
    """Validate and normalize all scalar OPD objective settings."""

    opd_loss_coef = _finite_float(
        opd_loss_coef,
        "actor_rollout_ref.opd.loss_coef",
    )
    task_loss_coef = _finite_float(
        task_loss_coef,
        "actor_rollout_ref.opd.task_loss_coef",
    )
    if max_abs_log_ratio is not None:
        max_abs_log_ratio = _finite_float(
            max_abs_log_ratio,
            "actor_rollout_ref.opd.max_abs_log_ratio",
        )
    if opd_loss_coef <= 0:
        raise ValueError("actor_rollout_ref.opd.loss_coef must be positive")
    if task_loss_coef < 0:
        raise ValueError("actor_rollout_ref.opd.task_loss_coef must be non-negative")
    if max_abs_log_ratio is not None and max_abs_log_ratio <= 0:
        raise ValueError(
            "actor_rollout_ref.opd.max_abs_log_ratio must be positive or null"
        )
    return opd_loss_coef, task_loss_coef, max_abs_log_ratio


def load_teacher_manifest(manifest_path: str) -> tuple[tuple[TeacherSpec, ...], dict[str, Any]]:
    """Load an ordered, deterministic JSON teacher manifest."""

    path = Path(manifest_path).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Teacher manifest root must be a JSON object")
    if payload.get("version") != 1:
        raise ValueError("Teacher manifest version must be 1")
    raw_teachers = payload.get("teachers")
    if not isinstance(raw_teachers, list):
        raise ValueError("Teacher manifest field 'teachers' must be a list")

    specs = []
    for index, item in enumerate(raw_teachers):
        if not isinstance(item, dict):
            raise ValueError(f"Teacher entry {index} must be a JSON object")
        unknown = set(item) - {"name", "path", "step", "sha256"}
        if unknown:
            raise ValueError(f"Teacher entry {index} has unsupported fields: {sorted(unknown)}")
        if "name" not in item or "path" not in item:
            raise ValueError(f"Teacher entry {index} must define name and path")
        if not isinstance(item["name"], str) or not item["name"].strip():
            raise ValueError(f"Teacher entry {index} name must be a non-empty string")
        if not isinstance(item["path"], str) or not item["path"].strip():
            raise ValueError(f"Teacher entry {index} path must be a non-empty string")
        specs.append(
            TeacherSpec(
                name=item["name"],
                path=item["path"],
                step=None if item.get("step") is None else int(item["step"]),
                sha256=None if item.get("sha256") in (None, "") else str(item["sha256"]).lower(),
            )
        )
    return _validate_teacher_specs(specs), payload


def parse_teacher_model_paths(raw_paths: str) -> tuple[TeacherSpec, ...]:
    """Parse a JSON list or comma-separated ordered checkpoint list."""

    raw_paths = raw_paths.strip()
    if not raw_paths:
        return _validate_teacher_specs(())
    if raw_paths.startswith("["):
        values = json.loads(raw_paths)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError("OPD_TEACHER_MODEL_PATHS JSON form must be a list of strings")
    else:
        values = [value.strip() for value in raw_paths.split(",") if value.strip()]

    specs = [
        TeacherSpec(
            name=f"teacher_{index:02d}_{Path(value.rstrip('/')).name}",
            path=value,
            step=_teacher_step(value),
        )
        for index, value in enumerate(values)
    ]
    return _validate_teacher_specs(specs)


def resolve_teacher_specs(
    *,
    manifest_path: Optional[str],
    model_paths: Optional[str],
) -> tuple[tuple[TeacherSpec, ...], dict[str, Any]]:
    if manifest_path and model_paths:
        raise ValueError(
            "Configure teachers with exactly one source: "
            "OPD_TEACHER_MANIFEST or OPD_TEACHER_MODEL_PATHS"
        )
    if manifest_path:
        return load_teacher_manifest(manifest_path)
    specs = parse_teacher_model_paths(model_paths or "")
    return specs, {
        "version": 1,
        "selection": "explicit_ordered_paths",
        "teachers": [asdict(spec) for spec in specs],
    }


def _select(config: DictConfig, key: str, default: Any = None) -> Any:
    value = OmegaConf.select(config, key, default=default)
    return default if value is None else value


def validate_opd_config(config: DictConfig) -> None:
    """Reject settings that would silently change the requested OPD objective."""

    configured_teacher_paths = _select(
        config,
        "actor_rollout_ref.opd.teacher_model_paths",
        [],
    )
    configured_teacher_names = _select(
        config,
        "actor_rollout_ref.opd.teacher_names",
        [],
    )
    teacher_paths = (
        [] if configured_teacher_paths is None else list(configured_teacher_paths)
    )
    teacher_names = (
        [] if configured_teacher_names is None else list(configured_teacher_names)
    )
    if not teacher_paths or len(teacher_paths) != len(teacher_names):
        raise ValueError("OPD teacher paths and names must be non-empty and have equal lengths")
    if any(not isinstance(path, str) or not path.strip() for path in teacher_paths):
        raise ValueError("OPD teacher checkpoint paths must be non-empty strings")
    if any(not isinstance(name, str) or not name.strip() for name in teacher_names):
        raise ValueError("OPD teacher names must be non-empty strings")
    if len(set(teacher_paths)) != len(teacher_paths):
        raise ValueError("OPD teacher checkpoint paths must be unique")
    if len(set(teacher_names)) != len(teacher_names):
        raise ValueError("OPD teacher names must be unique")

    strategy = str(_select(config, "actor_rollout_ref.actor.strategy"))
    if strategy not in {"fsdp", "fsdp2"}:
        raise ValueError("Search-OPD currently supports FSDP/FSDP2 actors only")
    if int(_select(config, "actor_rollout_ref.actor.ppo_epochs", 1)) != 1:
        raise ValueError("PG-OPD requires actor_rollout_ref.actor.ppo_epochs=1")
    if bool(_select(config, "actor_rollout_ref.actor.use_dynamic_bsz", False)):
        raise ValueError("PG-OPD requires use_dynamic_bsz=false for exact teacher-loss averaging")

    train_batch_size = int(_select(config, "data.train_batch_size"))
    mini_batch_size = int(_select(config, "actor_rollout_ref.actor.ppo_mini_batch_size"))
    if mini_batch_size != train_batch_size:
        raise ValueError(
            "PG-OPD requires ppo_mini_batch_size == train_batch_size so every "
            "update is one on-policy mini-batch"
        )
    if int(_select(config, "actor_rollout_ref.model.lora_rank", 0)) != 0:
        raise ValueError("Multi-teacher OPD does not support ref-in-actor LoRA mode")
    if _select(config, "actor_rollout_ref.model.lora_adapter_path", None) is not None:
        raise ValueError("Multi-teacher OPD does not support ref-in-actor LoRA mode")
    if bool(_select(config, "algorithm.use_kl_in_reward", False)):
        raise ValueError("Disable algorithm.use_kl_in_reward; OPD already supplies the teacher objective")

    _validate_objective_values(
        _select(config, "actor_rollout_ref.opd.loss_coef", 1.0),
        _select(config, "actor_rollout_ref.opd.task_loss_coef", 0.0),
        _select(config, "actor_rollout_ref.opd.max_abs_log_ratio", None),
    )
    top_k = int(_select(config, "actor_rollout_ref.opd.top_k", 16))
    if top_k < 1:
        raise ValueError("actor_rollout_ref.opd.top_k must be positive")


def configure_opd(
    config: DictConfig,
    *,
    environ: Optional[Mapping[str, str]] = None,
    write_manifest: bool = True,
) -> tuple[TeacherSpec, ...]:
    """Resolve teachers, inject worker config, validate, and record the run."""

    environ = os.environ if environ is None else environ
    configured_manifest = _select(config, "actor_rollout_ref.opd.teacher_manifest", "")
    configured_paths = _select(config, "actor_rollout_ref.opd.teacher_model_paths_raw", "")
    manifest_path = environ.get("OPD_TEACHER_MANIFEST", configured_manifest) or None
    model_paths = environ.get("OPD_TEACHER_MODEL_PATHS", configured_paths) or None
    specs, source_manifest = resolve_teacher_specs(
        manifest_path=manifest_path,
        model_paths=model_paths,
    )

    existing_opd = OmegaConf.to_container(
        _select(config, "actor_rollout_ref.opd", OmegaConf.create({})),
        resolve=True,
    )
    if not isinstance(existing_opd, dict):
        raise ValueError("actor_rollout_ref.opd must be a mapping")
    opd_loss_coef, task_loss_coef, max_abs_log_ratio = _validate_objective_values(
        existing_opd.get("loss_coef", 1.0),
        existing_opd.get("task_loss_coef", 0.0),
        existing_opd.get("max_abs_log_ratio"),
    )
    top_k = int(existing_opd.get("top_k", 16))
    if top_k < 1:
        raise ValueError("actor_rollout_ref.opd.top_k must be positive")

    selection_state_path = environ.get(
        "OPD_SELECTION_STATE_PATH",
        existing_opd.get("teacher_selection_state_path"),
    )
    teacher_trajectory_root = environ.get(
        "OPD_TEACHER_TRAJECTORY_ROOT",
        existing_opd.get("teacher_trajectory_root"),
    )
    teacher_metrics_path = environ.get(
        "OPD_TEACHER_VALIDATION_METRICS_PATH",
        existing_opd.get("teacher_validation_metrics_path"),
    )

    # Verify that every immutable teacher has a capability record before
    # Hydra/Ray creates workers.  Do not permanently prune a teacher here:
    # every teacher must re-enter the competition after it is replaced.
    pruned_teachers: list[dict[str, object]] = []
    capability_check = "not_configured"
    if teacher_trajectory_root:
        from .adaptive_teacher_opd import load_teacher_solved_sets

        try:
            teacher_solved_sets = load_teacher_solved_sets(
                specs,
                teacher_trajectory_root=str(teacher_trajectory_root),
                teacher_metrics_path=(
                    None if not teacher_metrics_path else str(teacher_metrics_path)
                ),
            )
            capability_check = "complete"
            print(
                "Adaptive teacher capability preflight: "
                f"configured={len(teacher_solved_sets)}, active={len(specs)}, "
                "pruned=0 (all teachers remain eligible)",
                flush=True,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                "Unable to complete the required startup teacher capability "
                f"coverage check under {teacher_trajectory_root}"
            ) from exc

    existing_opd.update(
        {
            "enabled": True,
            "teacher_manifest": manifest_path,
            "teacher_model_paths": [spec.path for spec in specs],
            "teacher_names": [spec.name for spec in specs],
            "teacher_steps": [spec.step for spec in specs],
            "teacher_sha256": [spec.sha256 for spec in specs],
            "loss_coef": opd_loss_coef,
            "task_loss_coef": task_loss_coef,
            "max_abs_log_ratio": max_abs_log_ratio,
            "top_k": top_k,
            "pruned_teachers": pruned_teachers,
            "teacher_capability_check": capability_check,
        }
    )
    if selection_state_path:
        existing_opd["teacher_selection_state_path"] = str(selection_state_path)
    if teacher_trajectory_root:
        existing_opd["teacher_trajectory_root"] = str(teacher_trajectory_root)
    if teacher_metrics_path:
        existing_opd["teacher_validation_metrics_path"] = str(teacher_metrics_path)
    existing_opd["teacher_selection_interval"] = int(
        environ.get(
            "OPD_TEACHER_SELECTION_INTERVAL",
            existing_opd.get("teacher_selection_interval", 5),
        )
    )
    initial_teacher_index = 0
    if selection_state_path:
        state_path = Path(str(selection_state_path)).expanduser()
        if state_path.exists():
            try:
                selection_state = json.loads(state_path.read_text(encoding="utf-8"))
                selected = selection_state["selected"]
                selected_name = str(selected.get("name", "")) if isinstance(selected, dict) else ""
                selected_path = str(selected.get("path", "")) if isinstance(selected, dict) else ""
                matching_indices = [
                    index
                    for index, spec in enumerate(specs)
                    if (selected_name and spec.name == selected_name)
                    or (selected_path and spec.path == selected_path)
                ]
                if matching_indices:
                    initial_teacher_index = matching_indices[0]
                else:
                    initial_teacher_index = int(selected["index"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid adaptive teacher selection state: {state_path}"
                ) from exc
    if not 0 <= initial_teacher_index < len(specs):
        raise ValueError(
            f"Initial adaptive teacher index out of range: {initial_teacher_index}; "
            f"teacher_count={len(specs)}"
        )
    existing_opd["initial_teacher_index"] = initial_teacher_index

    if not specs:
        raise ValueError(
            "No OPD teachers configured. Set OPD_TEACHER_MANIFEST or "
            "OPD_TEACHER_MODEL_PATHS."
        )
    with open_dict(config):
        config.actor_rollout_ref.opd = OmegaConf.create(existing_opd)
        config.actor_rollout_ref.ref.model = OmegaConf.create(
            {"path": specs[initial_teacher_index].path}
        )
        # This flag makes the unchanged verl trainer allocate a RefPolicy role.
        # The OPD actor wrapper disables the legacy single-reference KL loss.
        config.actor_rollout_ref.actor.use_kl_loss = True
        config.actor_rollout_ref.actor.kl_loss_coef = 0.0
        config.algorithm.use_kl_in_reward = False

    validate_opd_config(config)
    if write_manifest:
        write_resolved_run_manifest(
            config,
            specs=specs,
            source_manifest=source_manifest,
            environ=environ,
        )
    return specs


def build_resolved_run_manifest(
    config: DictConfig,
    *,
    specs: Sequence[TeacherSpec],
    source_manifest: Mapping[str, Any],
    environ: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "version": 1,
        "method": "single_resident_temporal_teacher_student_topk_opd",
        "teacher_aggregation": "single_active_teacher",
        "student_model": str(_select(config, "actor_rollout_ref.model.path")),
        "base_model_revision": environ.get("BASE_MODEL_REVISION", "unknown"),
        "dataset_revision": environ.get("DATASET_REVISION", "unknown"),
        "teachers": [asdict(spec) for spec in specs],
        "teacher_source_manifest": dict(source_manifest),
        "data": {
            "train_files": OmegaConf.to_container(config.data.train_files, resolve=True)
            if OmegaConf.is_config(config.data.train_files)
            else config.data.train_files,
            "val_files": OmegaConf.to_container(config.data.val_files, resolve=True)
            if OmegaConf.is_config(config.data.val_files)
            else config.data.val_files,
            "test_files": OmegaConf.to_container(
                _select(config, "data.test_files", {})
            )
            if OmegaConf.is_config(_select(config, "data.test_files", {}))
            else _select(config, "data.test_files", {}),
            "seed": int(_select(config, "data.seed", 1)),
        },
        "rollout": {
            "count": int(_select(config, "actor_rollout_ref.rollout.n", 1)),
            "prompt_length": int(_select(config, "actor_rollout_ref.rollout.prompt_length")),
            "response_length": int(_select(config, "actor_rollout_ref.rollout.response_length")),
            "max_assistant_turns": int(
                _select(config, "actor_rollout_ref.rollout.multi_turn.max_assistant_turns")
            ),
            "temperature": float(_select(config, "actor_rollout_ref.rollout.temperature", 1.0)),
        },
        "objective": {
            "opd_loss_coef": float(_select(config, "actor_rollout_ref.opd.loss_coef", 1.0)),
            "task_loss_coef": float(_select(config, "actor_rollout_ref.opd.task_loss_coef", 0.0)),
            "max_abs_log_ratio": _select(config, "actor_rollout_ref.opd.max_abs_log_ratio", None),
            "top_k": int(_select(config, "actor_rollout_ref.opd.top_k", 16)),
            "direction": "reverse_kl_student_to_teacher",
            "support": "student_top_k_renormalized",
            "mask": "model_generated_tokens_only",
        },
        "adaptive_teacher_selection": {
            "enabled": bool(
                _select(config, "actor_rollout_ref.opd.teacher_selection_state_path", None)
                and _select(config, "actor_rollout_ref.opd.teacher_trajectory_root", None)
            ),
            "state_path": _select(
                config, "actor_rollout_ref.opd.teacher_selection_state_path", None
            ),
            "teacher_trajectory_root": _select(
                config, "actor_rollout_ref.opd.teacher_trajectory_root", None
            ),
            "teacher_validation_metrics_path": _select(
                config, "actor_rollout_ref.opd.teacher_validation_metrics_path", None
            ),
            "interval": int(
                _select(config, "actor_rollout_ref.opd.teacher_selection_interval", 5)
            ),
            "weight_scope": "independent_per_teacher",
            "marginal_set": "Q_i_minus_current_student_P_t",
            "selected_teacher_reset": "all_Q_i_to_one",
            "replaced_teacher_reenters_next_round": True,
            "question_weight_initial": 1,
            "question_weight_cap": 5,
            "teacher_capability_check": str(
                _select(config, "actor_rollout_ref.opd.teacher_capability_check", "not_configured")
            ),
            "pruned_teachers": OmegaConf.to_container(
                _select(config, "actor_rollout_ref.opd.pruned_teachers", [])
            )
            if OmegaConf.is_config(_select(config, "actor_rollout_ref.opd.pruned_teachers", []))
            else _select(config, "actor_rollout_ref.opd.pruned_teachers", []),
        },
        "reward_verifier": str(_select(config, "reward_model.reward_manager")),
        "reward_verifier_revision": environ.get("REWARD_VERIFIER_REVISION", "working-tree"),
        "tool_environment_revision": environ.get("TOOL_ENVIRONMENT_REVISION", "working-tree"),
        "training": {
            "total_steps": int(_select(config, "trainer.total_training_steps")),
            "epochs": int(_select(config, "trainer.total_epochs")),
            "learning_rate": float(_select(config, "actor_rollout_ref.actor.optim.lr")),
            "train_batch_size": int(_select(config, "data.train_batch_size")),
            "ppo_mini_batch_size": int(_select(config, "actor_rollout_ref.actor.ppo_mini_batch_size")),
        },
    }


def write_resolved_run_manifest(
    config: DictConfig,
    *,
    specs: Sequence[TeacherSpec],
    source_manifest: Mapping[str, Any],
    environ: Mapping[str, str],
) -> Path:
    root = Path(str(_select(config, "trainer.default_local_dir"))).expanduser().resolve()
    if root in {Path("/"), Path.home().resolve()}:
        raise ValueError(f"Refusing to write OPD artifacts to broad directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    output_path = root / "opd_run_manifest.json"
    payload = build_resolved_run_manifest(
        config,
        specs=specs,
        source_manifest=source_manifest,
        environ=environ,
    )
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=root,
            prefix=".opd_run_manifest.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return output_path
