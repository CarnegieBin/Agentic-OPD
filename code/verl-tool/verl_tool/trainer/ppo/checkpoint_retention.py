"""Utilities for validation-aware checkpoint retention."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Set


def stable_validation_uid(data_source: object, extra_info: object, prompt_token_ids: object) -> str:
    """Build a deterministic question UID from dataset metadata.

    Validation batches do not always contain a ``uid`` column. Random UUIDs make
    solved-question coverage incomparable between validation runs, so use the
    stable source/index/question tuple and fall back to prompt tokens only when
    the source metadata is incomplete.
    """

    source = str(data_source)
    if isinstance(extra_info, Mapping):
        index = extra_info.get("index")
        question = extra_info.get("question")
    else:
        index = None
        question = None

    identity = {
        "data_source": source,
        "index": None if index is None else str(index),
        "question": None if question is None else str(question),
    }
    if identity["index"] is None and identity["question"] is None:
        if hasattr(prompt_token_ids, "tolist"):
            prompt_token_ids = prompt_token_ids.tolist()
        identity["prompt_token_ids"] = [int(token_id) for token_id in prompt_token_ids]

    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{source}:{digest}"


def solved_union_size(solved_uids_by_step: Mapping[int, Set[str]], steps: object | None = None) -> int:
    """Return the number of distinct solved UIDs covered by ``steps``."""

    selected_steps = solved_uids_by_step if steps is None else steps
    covered: set[str] = set()
    for step in selected_steps:
        covered.update(solved_uids_by_step[int(step)])
    return len(covered)


def select_max_union_steps(solved_uids_by_step: Mapping[int, Set[str]], keep_limit: int) -> list[int]:
    """Select a deterministic maximum-union subset.

    The online retention path calls this with at most ``keep_limit + 1``
    candidates: the currently retained checkpoints plus the new candidate.
    Enumerating all subsets is therefore exact and inexpensive. Ties first
    prefer checkpoints with a larger aggregate solved count, then newer steps.
    """

    if keep_limit < 1:
        raise ValueError(f"keep_limit must be positive, got {keep_limit}")

    normalized = {int(step): set(uids) for step, uids in solved_uids_by_step.items()}
    steps = sorted(normalized)
    if len(steps) <= keep_limit:
        return steps
    if len(steps) > keep_limit + 1:
        raise ValueError(
            "online max-union selection expects at most keep_limit + 1 candidates, "
            f"got {len(steps)} candidates for keep_limit={keep_limit}"
        )

    def score(candidate_steps: tuple[int, ...]) -> tuple[int, int, tuple[int, ...]]:
        return (
            solved_union_size(normalized, candidate_steps),
            sum(len(normalized[step]) for step in candidate_steps),
            tuple(sorted(candidate_steps, reverse=True)),
        )

    selected = max(itertools.combinations(steps, keep_limit), key=score)
    return sorted(selected)
