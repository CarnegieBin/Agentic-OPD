"""Deterministic validation-coverage scheduling for temporal OPD teachers.

The student and every temporal teacher are evaluated on the same NQ and
HotpotQA question IDs.  A teacher is useful only for questions in ``Qi - P``:
questions solved by that teacher but not by the current student.  The
per-question priority starts at one and is increased once per validation round
when at least one active teacher covers that question, with a cap of five.

The state file is published with ``os.replace`` so reference workers either see
the old complete selection or the new complete selection, never a partial JSON
document.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config_opd import TeacherSpec


_DATASET_ALIASES = {
    "nq": frozenset({"nq", "searchr1_nq", "search_r1_nq"}),
    "hotpotqa": frozenset({"hotpotqa", "searchr1_hotpotqa", "search_r1_hotpotqa"}),
}
_VALIDATION_DATASETS = tuple(_DATASET_ALIASES)
_STEP_PATTERN = re.compile(r"(?:global_)?step[_-]?(\d+)", re.IGNORECASE)
_DEFAULT_QUESTION_WEIGHT = 1
_MAX_QUESTION_WEIGHT = 5


def _dataset_name(value: object, fallback: str = "") -> str | None:
    normalized = str(value or fallback).strip().lower().replace("-", "_")
    for dataset, aliases in _DATASET_ALIASES.items():
        if (
            normalized in aliases
            or normalized.endswith(f"_{dataset}")
            or normalized.startswith(f"{dataset}_")
        ):
            return dataset
    return None


def _uid(record: Mapping[str, object]) -> str:
    value = record.get("sample_uid", record.get("uid"))
    if value is None:
        raise ValueError("Validation trajectory record has no sample_uid/uid")
    return str(value)


def _question_key(dataset: str, uid: object) -> str:
    """Namespace UIDs so NQ and HotpotQA cannot collide."""

    return f"{dataset}:{uid}"


def _question_keys(solved_by_dataset: Mapping[str, Sequence[object] | set[str]]) -> set[str]:
    return {
        _question_key(dataset, uid)
        for dataset, uids in solved_by_dataset.items()
        if dataset in _DATASET_ALIASES
        for uid in uids
    }


def _normalized_solved_sets(
    solved_by_dataset: Mapping[str, Sequence[object] | set[str]] | None,
) -> dict[str, set[str]]:
    return {
        dataset: {str(uid) for uid in (solved_by_dataset or {}).get(dataset, ())}
        for dataset in _VALIDATION_DATASETS
    }


def load_validation_solved_sets(
    trajectory_dir: str | os.PathLike[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Return ``(all_uids, solved_uids)`` grouped by NQ and HotpotQA."""

    root = Path(trajectory_dir).expanduser()
    all_uids: dict[str, set[str]] = defaultdict(set)
    solved_uids: dict[str, set[str]] = defaultdict(set)
    # Historical runs also emit compact ``*_judgment.jsonl`` files.  Prefer
    # those: they contain exactly UID + EM and avoid rereading multi-gigabyte
    # rendered trajectories.  Live Search-OPD validation emits searchR1_*.jsonl
    # instead, so retain that fallback.
    files = sorted(root.glob("*_judgment.jsonl"))
    if not files:
        files = sorted(root.glob("searchR1_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No validation JSONL files found under {root}")

    for path in files:
        fallback = path.stem
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Trajectory record must be an object: {path}:{line_number}")
                dataset = _dataset_name(record.get("data_source"), fallback)
                if dataset is None:
                    continue
                sample_uid = _uid(record)
                all_uids[dataset].add(sample_uid)
                try:
                    solved = float(record.get("em", record.get("score", 0.0))) > 0.0
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid trajectory score at {path}:{line_number}: "
                        f"{record.get('em', record.get('score'))!r}"
                    ) from exc
                if solved:
                    solved_uids[dataset].add(sample_uid)

    missing = [dataset for dataset in _VALIDATION_DATASETS if not all_uids[dataset]]
    if missing:
        raise ValueError(f"Validation trajectories are missing datasets: {missing} ({root})")
    return (
        {dataset: set(all_uids[dataset]) for dataset in _VALIDATION_DATASETS},
        {dataset: set(solved_uids[dataset]) for dataset in _VALIDATION_DATASETS},
    )


def load_checkpoint_metrics_solved_sets(
    metrics_path: str | os.PathLike[str],
) -> dict[int, dict[str, set[str]]]:
    """Load compact per-checkpoint solved UID sets from validation metrics."""

    output: dict[int, dict[str, set[str]]] = {}
    path = Path(metrics_path).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                checkpoint = str(record["checkpoint"])
                match = _STEP_PATTERN.search(checkpoint)
                if match is None:
                    raise ValueError(f"checkpoint has no global step: {checkpoint!r}")
                step = int(match.group(1))
                datasets = record["datasets"]
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid checkpoint metrics record at {path}:{line_number}") from exc
            if not isinstance(datasets, dict):
                continue
            solved_by_dataset: dict[str, set[str]] = {}
            for source, info in datasets.items():
                dataset = _dataset_name(source)
                if dataset is None or not isinstance(info, dict):
                    continue
                solved_uids = info.get("solved_uids", [])
                if not isinstance(solved_uids, (list, tuple, set)):
                    raise ValueError(
                        f"Invalid solved_uids at {path}:{line_number} ({source})"
                    )
                solved_by_dataset[dataset] = {str(uid) for uid in solved_uids}
            if solved_by_dataset:
                output[step] = _normalized_solved_sets(solved_by_dataset)
    if not output:
        raise ValueError(f"No checkpoint solved UID records found in {path}")
    return output


def _teacher_step(spec: TeacherSpec) -> int:
    if spec.step is not None:
        return int(spec.step)
    match = _STEP_PATTERN.search(spec.path)
    return int(match.group(1)) if match else 2**31 - 1


def _teacher_trajectory_dir(
    spec: TeacherSpec,
    trajectory_root: Path,
) -> Path:
    step = _teacher_step(spec)
    if step != 2**31 - 1:
        return trajectory_root / f"global_step_{step}"
    return trajectory_root / Path(spec.path.rstrip("/")).name


def load_teacher_solved_sets(
    specs: Sequence[TeacherSpec],
    *,
    teacher_trajectory_root: str | os.PathLike[str],
    teacher_metrics_path: str | os.PathLike[str] | None = None,
) -> dict[int, dict[str, set[str]]]:
    """Load solved sets for every configured teacher.

    Compact checkpoint metrics are preferred because rendered trajectories can
    be very large.  If the metrics file is not available, the per-checkpoint
    trajectory directories are used.  Missing teacher records are an error:
    silently treating an unmeasured checkpoint as an incapable teacher would
    change the requested scheduling policy.
    """

    root = Path(teacher_trajectory_root).expanduser()
    metrics_solved: dict[int, dict[str, set[str]]] | None = None
    if teacher_metrics_path is not None:
        metrics_path = Path(teacher_metrics_path).expanduser()
        if metrics_path.exists():
            metrics_solved = load_checkpoint_metrics_solved_sets(metrics_path)

    solved_by_index: dict[int, dict[str, set[str]]] = {}
    missing: list[str] = []
    for index, spec in enumerate(specs):
        step = _teacher_step(spec)
        if metrics_solved is not None:
            solved = metrics_solved.get(step)
            if solved is None:
                missing.append(f"{spec.name} (step={step})")
                continue
            solved_by_index[index] = _normalized_solved_sets(solved)
            continue

        teacher_dir = _teacher_trajectory_dir(spec, root)
        try:
            _, solved = load_validation_solved_sets(teacher_dir)
        except (FileNotFoundError, ValueError) as exc:
            missing.append(f"{spec.name} ({teacher_dir}: {exc})")
            continue
        solved_by_index[index] = _normalized_solved_sets(solved)

    if missing:
        source = (
            str(teacher_metrics_path)
            if teacher_metrics_path is not None and metrics_solved is not None
            else str(root)
        )
        raise FileNotFoundError(
            f"Missing validation capability records for {len(missing)} teacher(s) "
            f"from {source}: {missing[:5]}"
        )
    if not solved_by_index:
        raise FileNotFoundError(f"No teacher validation capability records found under {root}")
    return solved_by_index


def _spec_order(index: int, spec: TeacherSpec) -> tuple[int, str, str, int]:
    return (_teacher_step(spec), spec.name, spec.path, index)


def prune_dominated_teachers(
    specs: Sequence[TeacherSpec],
    teacher_solved_sets: Mapping[int, Mapping[str, Sequence[object] | set[str]]],
) -> tuple[tuple[TeacherSpec, ...], list[dict[str, object]]]:
    """Drop checkpoints whose solved set is contained in another checkpoint.

    Equal solved sets are treated as coverage too; the deterministic
    lowest-step/name/path entry is retained.  The returned teacher tuple keeps
    the original manifest order, while state-file indices are assigned from
    that filtered tuple by the caller.
    """

    if not specs:
        raise ValueError("Cannot prune an empty teacher list")
    expected = set(range(len(specs)))
    observed = set(int(index) for index in teacher_solved_sets)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"Teacher solved-set indices must cover configured teachers; "
            f"missing={missing}, extra={extra}"
        )

    qualified = {
        index: _normalized_solved_sets(teacher_solved_sets[index])
        for index in range(len(specs))
    }
    qualified_keys = {
        index: _question_keys(solved)
        for index, solved in qualified.items()
    }
    pruned_indices: set[int] = set()
    pruned: list[dict[str, object]] = []

    for index, spec in enumerate(specs):
        covering = [
            other
            for other in range(len(specs))
            if other != index and qualified_keys[index] <= qualified_keys[other]
        ]
        if not covering:
            continue

        # Prefer the largest covering set; for equal coverage use a stable
        # temporal/name/path ordering.  This also gives a useful explanation
        # when a set is covered by several later checkpoints.
        covering.sort(
            key=lambda other: (
                -len(qualified_keys[other]),
                _spec_order(other, specs[other]),
            )
        )
        covering_index = covering[0]
        pruned_indices.add(index)
        pruned.append(
            {
                "index": index,
                "name": spec.name,
                "path": spec.path,
                "step": spec.step,
                "reason": "solved_set_covered_by_other_teacher",
                "covered_by": {
                    "index": covering_index,
                    "name": specs[covering_index].name,
                    "path": specs[covering_index].path,
                    "step": specs[covering_index].step,
                },
                "solved_count": len(qualified_keys[index]),
                "covering_solved_count": len(qualified_keys[covering_index]),
            }
        )

    active = tuple(spec for index, spec in enumerate(specs) if index not in pruned_indices)
    if not active:
        # This is only possible for pathological duplicate sets if the
        # covering relation is allowed to prune both sides.  Keep one stable
        # representative rather than starting with no teacher.
        keep_index = min(range(len(specs)), key=lambda index: _spec_order(index, specs[index]))
        active = (specs[keep_index],)
        pruned_indices.discard(keep_index)
        pruned = [
            item
            for item in pruned
            if int(item["index"]) != keep_index
        ]
    pruned.sort(key=lambda item: int(item["index"]))
    return active, pruned


def initialize_question_weights(
    all_uids: Mapping[str, Sequence[object] | set[str]],
    teacher_solved_sets: Mapping[int, Mapping[str, Sequence[object] | set[str]]] | None = None,
) -> dict[str, int]:
    """Initialize every observed validation question with weight one."""

    keys = _question_keys(_normalized_solved_sets(all_uids))
    for solved in (teacher_solved_sets or {}).values():
        keys.update(_question_keys(_normalized_solved_sets(solved)))
    return {key: _DEFAULT_QUESTION_WEIGHT for key in sorted(keys)}


def initialize_teacher_question_weights(
    teacher_solved_sets: Mapping[int, Mapping[str, Sequence[object] | set[str]]],
    *,
    raw_teacher_weights: Mapping[object, Mapping[str, object]] | None = None,
    max_weight: int = _MAX_QUESTION_WEIGHT,
) -> dict[int, dict[str, int]]:
    """Initialize an independent question-weight map for every teacher.

    A teacher's map contains only its own ``Q_i``.  This is intentionally
    different from the historical global question map: two teachers covering
    the same question accumulate independent priority and resetting one
    teacher never resets the other.
    """

    raw_teacher_weights = raw_teacher_weights or {}
    output: dict[int, dict[str, int]] = {}
    for raw_index, solved in teacher_solved_sets.items():
        index = int(raw_index)
        question_keys = sorted(_question_keys(_normalized_solved_sets(solved)))
        raw_weights = raw_teacher_weights.get(index, raw_teacher_weights.get(str(index), {}))
        if not isinstance(raw_weights, Mapping):
            raise ValueError(f"Teacher {index} question weights must be a mapping")
        validated = _validated_weights(raw_weights, max_weight=max_weight)
        output[index] = {
            key: int(validated.get(key, _DEFAULT_QUESTION_WEIGHT))
            for key in question_keys
        }
    return output


def update_teacher_question_weights(
    teacher_weights: Mapping[object, Mapping[str, object]] | None,
    *,
    student_solved: Mapping[str, Sequence[object] | set[str]],
    teacher_solved_sets: Mapping[int, Mapping[str, Sequence[object] | set[str]]],
    selected_teacher_index: int,
    max_weight: int = _MAX_QUESTION_WEIGHT,
) -> tuple[dict[int, dict[str, int]], dict[int, set[str]]]:
    """Advance independent teacher weights after a selection round.

    The ranking for the round is computed from the incoming weights.  After a
    teacher is selected, every question in its ``Q_i`` is immediately reset to
    one.  Every non-selected teacher increments only questions in
    ``Q_i - P_t``.  All configured teachers are processed on every call, so a
    teacher replaced in the previous round naturally competes again here.
    """

    if int(selected_teacher_index) not in {int(index) for index in teacher_solved_sets}:
        raise ValueError(
            "selected_teacher_index must identify a configured teacher: "
            f"{selected_teacher_index}"
        )
    student = _normalized_solved_sets(student_solved)
    current = initialize_teacher_question_weights(
        teacher_solved_sets,
        raw_teacher_weights=teacher_weights,
        max_weight=max_weight,
    )
    updated: dict[int, set[str]] = {}
    for raw_index, solved in teacher_solved_sets.items():
        index = int(raw_index)
        teacher = _normalized_solved_sets(solved)
        question_keys = _question_keys(teacher)
        if index == int(selected_teacher_index):
            for key in question_keys:
                current[index][key] = _DEFAULT_QUESTION_WEIGHT
            updated[index] = set(question_keys)
            continue

        marginal_keys = {
            _question_key(dataset, uid)
            for dataset in _VALIDATION_DATASETS
            for uid in teacher[dataset] - student[dataset]
        }
        for key in marginal_keys:
            current[index][key] = min(
                max_weight,
                current[index].get(key, _DEFAULT_QUESTION_WEIGHT) + 1,
            )
        updated[index] = marginal_keys
    return current, updated


def _validated_weights(
    raw_weights: Mapping[str, object] | None,
    *,
    max_weight: int = _MAX_QUESTION_WEIGHT,
) -> dict[str, int]:
    if max_weight < _DEFAULT_QUESTION_WEIGHT:
        raise ValueError("max_weight must be at least one")
    output: dict[str, int] = {}
    for raw_key, raw_value in (raw_weights or {}).items():
        try:
            value = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid question weight for {raw_key!r}: {raw_value!r}") from exc
        if not _DEFAULT_QUESTION_WEIGHT <= value <= max_weight:
            raise ValueError(
                f"Question weight must be in [{_DEFAULT_QUESTION_WEIGHT}, {max_weight}]: "
                f"{raw_key!r}={value}"
            )
        output[str(raw_key)] = value
    return output


def update_question_weights(
    question_weights: Mapping[str, object] | None,
    *,
    student_solved: Mapping[str, Sequence[object] | set[str]],
    teacher_solved_sets: Mapping[int, Mapping[str, Sequence[object] | set[str]]],
    all_uids: Mapping[str, Sequence[object] | set[str]] | None = None,
    max_weight: int = _MAX_QUESTION_WEIGHT,
) -> tuple[dict[str, int], set[str]]:
    """Increase weights for the union of all active teachers' ``Qi - P``.

    A question solved by multiple teachers is still one question, so it gains
    exactly one point in a validation round.  This follows the question-level
    wording of the scheduling policy and avoids candidate-order bias.
    """

    weights = _validated_weights(question_weights, max_weight=max_weight)
    initial = initialize_question_weights(
        all_uids or student_solved,
        teacher_solved_sets,
    )
    for key in initial:
        weights.setdefault(key, _DEFAULT_QUESTION_WEIGHT)

    student = _normalized_solved_sets(student_solved)
    uncovered: set[str] = set()
    for solved in teacher_solved_sets.values():
        teacher = _normalized_solved_sets(solved)
        for dataset in _VALIDATION_DATASETS:
            uncovered.update(
                _question_key(dataset, uid)
                for uid in teacher[dataset] - student[dataset]
            )

    for key in uncovered:
        weights[key] = min(max_weight, weights.get(key, _DEFAULT_QUESTION_WEIGHT) + 1)
    return weights, uncovered


def _rank_loaded_teachers(
    specs: Sequence[TeacherSpec],
    *,
    all_uids: Mapping[str, set[str]],
    student_solved: Mapping[str, set[str]],
    teacher_solved_sets: Mapping[int, Mapping[str, set[str]]],
    question_weights: Mapping[str, int] | None = None,
    teacher_weights: Mapping[object, Mapping[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Rank every teacher using only its own weighted ``Q_i - P_t``."""

    if teacher_weights is None:
        flat_weights = question_weights or {}
        teacher_weights = {
            index: {
                key: int(flat_weights.get(key, _DEFAULT_QUESTION_WEIGHT))
                for key in _question_keys(_normalized_solved_sets(solved))
            }
            for index, solved in teacher_solved_sets.items()
        }
    ranked: list[dict[str, object]] = []
    for index, spec in enumerate(specs):
        teacher_solved = teacher_solved_sets.get(index)
        if teacher_solved is None:
            continue
        teacher_solved = _normalized_solved_sets(teacher_solved)
        marginal_by_dataset = {
            dataset: teacher_solved[dataset] - student_solved.get(dataset, set())
            for dataset in _VALIDATION_DATASETS
        }
        marginal_questions = sorted(
            _question_key(dataset, uid)
            for dataset in _VALIDATION_DATASETS
            for uid in marginal_by_dataset[dataset]
        )
        weighted_gain = sum(
            int(
                teacher_weights.get(index, {}).get(
                    key,
                    _DEFAULT_QUESTION_WEIGHT,
                )
            )
            for key in marginal_questions
        )
        gains = {
            dataset: len(marginal_by_dataset[dataset])
            for dataset in _VALIDATION_DATASETS
        }
        normalized_gains = {
            dataset: gains[dataset] / max(len(all_uids.get(dataset, set())), 1)
            for dataset in _VALIDATION_DATASETS
        }
        ranked.append(
            {
                "index": index,
                "name": spec.name,
                "path": spec.path,
                "step": spec.step,
                "checkpoint": spec.path,
                "global_step": _teacher_step(spec),
                "trajectory_dir": None,
                "marginal_questions": marginal_questions,
                "marginal_gain": gains,
                "marginal_question_count": len(marginal_questions),
                "weighted_marginal_gain": weighted_gain,
                "normalized_marginal_gain": normalized_gains,
                "mean_normalized_marginal_gain": (
                    sum(normalized_gains.values()) / len(normalized_gains)
                ),
            }
        )

    if not ranked:
        raise FileNotFoundError("No teacher validation solved sets are available for ranking")
    ranked.sort(
        key=lambda item: (
            int(item["weighted_marginal_gain"]),
            int(item["marginal_question_count"]),
            float(item["mean_normalized_marginal_gain"]),
            # Stable ties favor the earlier temporal checkpoint, then the
            # original manifest order.
            -_teacher_step(specs[int(item["index"])]),
            -int(item["index"]),
        ),
        reverse=True,
    )
    return ranked


def rank_weighted_teachers(
    specs: Sequence[TeacherSpec],
    *,
    student_trajectory_dir: str | os.PathLike[str],
    teacher_trajectory_root: str | os.PathLike[str],
    teacher_metrics_path: str | os.PathLike[str] | None = None,
    question_weights: Mapping[str, object] | None = None,
    teacher_weights: Mapping[object, Mapping[str, object]] | None = None,
    update_weights: bool = False,
) -> list[dict[str, object]]:
    """Rank all teachers by independent weighted question-level coverage."""

    all_uids, student_solved = load_validation_solved_sets(student_trajectory_dir)
    teacher_solved_sets = load_teacher_solved_sets(
        specs,
        teacher_trajectory_root=teacher_trajectory_root,
        teacher_metrics_path=teacher_metrics_path,
    )
    if teacher_weights is None:
        flat_weights = initialize_question_weights(all_uids, teacher_solved_sets)
        flat_weights.update(_validated_weights(question_weights))
        teacher_weights = initialize_teacher_question_weights(
            teacher_solved_sets,
            raw_teacher_weights={
                index: flat_weights for index in teacher_solved_sets
            },
        )
    else:
        teacher_weights = initialize_teacher_question_weights(
            teacher_solved_sets,
            raw_teacher_weights=teacher_weights,
        )
    # ``update_weights`` is retained for callers of the historical API.  A
    # selection-aware update happens in ``select_and_publish_teacher`` where
    # the chosen teacher can be reset immediately.
    if update_weights:
        teacher_weights = {
            index: {
                key: min(
                    _MAX_QUESTION_WEIGHT,
                    value
                    + int(
                        key
                        in {
                            _question_key(dataset, uid)
                            for dataset in _VALIDATION_DATASETS
                            for uid in _normalized_solved_sets(
                                teacher_solved_sets[index]
                            )[dataset]
                            - _normalized_solved_sets(student_solved)[dataset]
                        }
                    ),
                )
                for key, value in weights.items()
            }
            for index, weights in teacher_weights.items()
        }
    return _rank_loaded_teachers(
        specs,
        all_uids=all_uids,
        student_solved=student_solved,
        teacher_solved_sets=teacher_solved_sets,
        teacher_weights=teacher_weights,
    )


def rank_temporal_teachers(
    specs: Sequence[TeacherSpec],
    *,
    student_trajectory_dir: str | os.PathLike[str],
    teacher_trajectory_root: str | os.PathLike[str],
    teacher_metrics_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, object]]:
    """Backward-compatible alias for a first-round unit-weight ranking."""

    return rank_weighted_teachers(
        specs,
        student_trajectory_dir=student_trajectory_dir,
        teacher_trajectory_root=teacher_trajectory_root,
        teacher_metrics_path=teacher_metrics_path,
        update_weights=False,
    )


def _read_selection_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid adaptive teacher selection state: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Adaptive teacher selection state must be an object: {path}")
    return payload


def _teacher_records(specs: Sequence[TeacherSpec]) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "name": spec.name,
            "path": spec.path,
            "step": spec.step,
            "checkpoint": spec.path,
            "global_step": _teacher_step(spec),
        }
        for index, spec in enumerate(specs)
    ]


def write_teacher_selection_state(
    path: str | os.PathLike[str],
    *,
    global_step: int,
    ranked_teachers: Sequence[Mapping[str, object]],
    question_weights: Mapping[str, object] | None = None,
    teacher_weights: Mapping[object, Mapping[str, object]] | None = None,
    active_teachers: Sequence[Mapping[str, object]] | None = None,
    pruned_teachers: Sequence[Mapping[str, object]] | None = None,
    updated_question_keys: Sequence[str] | None = None,
    updated_teacher_question_keys: Mapping[object, Sequence[str]] | None = None,
    switch_count: int | None = None,
) -> dict[str, object]:
    """Atomically publish independent teacher scheduling state."""

    if not ranked_teachers:
        raise ValueError("Cannot publish an empty teacher ranking")
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    output_path = Path(path).expanduser()
    previous = _read_selection_state(output_path)
    selected = dict(ranked_teachers[0])
    selected.setdefault("checkpoint", selected.get("path"))
    selected.setdefault("global_step", selected.get("step"))
    if teacher_weights is None:
        previous_teacher_weights = previous.get("teacher_weights", {})
        if isinstance(previous_teacher_weights, dict):
            teacher_weights = previous_teacher_weights
        else:
            legacy_weights = _validated_weights(
                question_weights
                if question_weights is not None
                else previous.get("weights", {})
            )
            teacher_weights = {
                item["index"]: legacy_weights for item in ranked_teachers
            }
    normalized_teacher_weights: dict[int, dict[str, int]] = {}
    for raw_index, raw_weights in teacher_weights.items():
        if not isinstance(raw_weights, Mapping):
            raise ValueError(f"Teacher {raw_index} weights must be a mapping")
        normalized_teacher_weights[int(raw_index)] = _validated_weights(raw_weights)

    previous_selected = previous.get("selected", {})
    previous_index = (
        int(previous_selected["index"])
        if isinstance(previous_selected, dict) and "index" in previous_selected
        else None
    )
    selected_index = int(selected["index"])
    if switch_count is None:
        previous_switch_count = int(previous.get("switch_count", 0))
        switch_count = previous_switch_count + int(
            previous_index is not None and previous_index != selected_index
        )
    if int(switch_count) < 0:
        raise ValueError("switch_count must be non-negative")

    if active_teachers is None:
        active_records = [
            {
                "index": item["index"],
                "name": item["name"],
                "path": item["path"],
                "step": item.get("step"),
                "checkpoint": item.get("checkpoint", item.get("path")),
                "global_step": item.get("global_step", item.get("step")),
            }
            for item in ranked_teachers
        ]
    else:
        active_records = [dict(item) for item in active_teachers]
    if pruned_teachers is None:
        retained_pruned = previous.get("pruned_teachers", [])
        pruned_records = (
            [dict(item) for item in retained_pruned if isinstance(item, dict)]
            if isinstance(retained_pruned, list)
            else []
        )
    else:
        pruned_records = [dict(item) for item in pruned_teachers]

    history_value = previous.get("history", [])
    history: list[dict[str, object]] = (
        [dict(item) for item in history_value if isinstance(item, dict)]
        if isinstance(history_value, list)
        else []
    )
    history.append(
        {
            "global_step": int(global_step),
            "selected": selected,
            "selected_teacher_index": selected_index,
            "teacher_checkpoint": selected.get("checkpoint", selected.get("path")),
            "teacher_global_step": selected.get("global_step", selected.get("step")),
            "weighted_marginal_gain": float(selected.get("weighted_marginal_gain", 0)),
            "marginal_question_count": int(selected.get("marginal_question_count", 0)),
            "teacher_switch_count": int(switch_count),
            "updated_question_count": len(set(updated_question_keys or ())),
            "updated_teacher_question_keys": {
                str(index): sorted({str(key) for key in keys})
                for index, keys in (updated_teacher_question_keys or {}).items()
            },
        }
    )
    legacy_weights: dict[str, int] = {}
    for weights in normalized_teacher_weights.values():
        for key, value in weights.items():
            legacy_weights[key] = max(legacy_weights.get(key, _DEFAULT_QUESTION_WEIGHT), value)
    payload: dict[str, object] = {
        "version": 3,
        "global_step": int(global_step),
        "selection_strategy": "per_teacher_weighted_qi_minus_pt_reset_selected",
        "question_weight_initial": _DEFAULT_QUESTION_WEIGHT,
        "question_weight_cap": _MAX_QUESTION_WEIGHT,
        # Kept as a migration/debug field.  Scheduling reads teacher_weights,
        # never this flattened legacy view.
        "weights": {key: legacy_weights[key] for key in sorted(legacy_weights)},
        "teacher_weights": {
            str(index): {key: weights[key] for key in sorted(weights)}
            for index, weights in sorted(normalized_teacher_weights.items())
        },
        "active_teachers": active_records,
        "pruned_teachers": pruned_records,
        "selected": selected,
        "selected_teacher_indices": [selected_index],
        "switch_count": int(switch_count),
        "ranking": [dict(item) for item in ranked_teachers],
        "updated_question_keys": sorted(set(updated_question_keys or ())),
        "updated_teacher_question_keys": {
            str(index): sorted({str(key) for key in keys})
            for index, keys in (updated_teacher_question_keys or {}).items()
        },
        "history": history,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
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
    return payload


def select_and_publish_teacher(
    specs: Sequence[TeacherSpec],
    *,
    student_trajectory_dir: str | os.PathLike[str],
    teacher_trajectory_root: str | os.PathLike[str],
    teacher_metrics_path: str | os.PathLike[str] | None = None,
    state_path: str | os.PathLike[str],
    global_step: int,
    pruned_teachers: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Select one teacher from all ``Q_i - P_t`` candidates and publish state."""

    state_file = Path(state_path).expanduser()
    previous = _read_selection_state(state_file)
    all_uids, student_solved = load_validation_solved_sets(student_trajectory_dir)
    teacher_solved_sets = load_teacher_solved_sets(
        specs,
        teacher_trajectory_root=teacher_trajectory_root,
        teacher_metrics_path=teacher_metrics_path,
    )
    # Never remove a teacher after it has been configured.  In particular, a
    # teacher replaced by a newly selected checkpoint must be eligible in the
    # next round.
    teacher_weights = initialize_teacher_question_weights(
        teacher_solved_sets,
        raw_teacher_weights=(
            previous.get("teacher_weights", {})
            if isinstance(previous.get("teacher_weights", {}), dict)
            else {}
        ),
    )
    ranked = _rank_loaded_teachers(
        specs,
        all_uids=all_uids,
        student_solved=student_solved,
        teacher_solved_sets=teacher_solved_sets,
        teacher_weights=teacher_weights,
    )
    selected_index = int(ranked[0]["index"])
    teacher_weights, updated_by_teacher = update_teacher_question_weights(
        teacher_weights,
        student_solved=student_solved,
        teacher_solved_sets=teacher_solved_sets,
        selected_teacher_index=selected_index,
    )
    updated_keys = set().union(*updated_by_teacher.values()) if updated_by_teacher else set()
    previous_selected = previous.get("selected", {})
    previous_index = (
        int(previous_selected["index"])
        if isinstance(previous_selected, dict) and "index" in previous_selected
        else None
    )
    switch_count = int(previous.get("switch_count", 0)) + int(
        previous_index is not None and previous_index != selected_index
    )
    active_records = _teacher_records(specs)
    supplied_pruned = list(pruned_teachers or ())
    # Preserve the field for old manifests, but it is informational only.
    merged_pruned = [dict(item) for item in supplied_pruned]
    return write_teacher_selection_state(
        state_file,
        global_step=global_step,
        ranked_teachers=ranked,
        teacher_weights=teacher_weights,
        active_teachers=active_records,
        pruned_teachers=merged_pruned,
        updated_question_keys=sorted(updated_keys),
        updated_teacher_question_keys=updated_by_teacher,
        switch_count=switch_count,
    )
