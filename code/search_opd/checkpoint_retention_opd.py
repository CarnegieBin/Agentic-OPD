"""Checkpoint selection helpers for Search-OPD validation runs.

The Search-R1 evaluation loader emits dataset-scoped exact-match metrics such
as ``test-core/searchR1_bamboogle/em/mean@1``.  OPD runs select checkpoints by
giving Bamboogle, Musique, and TriviaQA equal weight.  The legacy
NQ/HotpotQA validation helper remains available for compatibility, but is not
used by the adaptive trainer's checkpoint retention path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


_DATASET_ALIASES = {
    "nq": frozenset({"nq", "searchr1_nq", "search_r1_nq"}),
    "hotpotqa": frozenset({"hotpotqa", "searchr1_hotpotqa", "search_r1_hotpotqa"}),
    "bamboogle": frozenset({"bamboogle", "searchr1_bamboogle", "search_r1_bamboogle"}),
    "musique": frozenset({"musique", "searchr1_musique", "search_r1_musique"}),
    "triviaqa": frozenset(
        {"triviaqa", "trivia_qa", "searchr1_triviaqa", "search_r1_triviaqa"}
    ),
}


def _normalized_source_name(source: object) -> str:
    return str(source).strip().lower().replace("-", "_")


def _metric_candidates(
    metrics: Mapping[str, Any],
    dataset: str,
    section_prefixes: tuple[str, ...] = ("val-",),
) -> list[tuple[int, str, float]]:
    aliases = _DATASET_ALIASES[dataset]
    candidates: list[tuple[int, str, float]] = []
    for key, raw_value in metrics.items():
        parts = str(key).split("/")
        if len(parts) != 4:
            continue
        section, source, variable, metric_name = parts
        if not any(section.startswith(prefix) for prefix in section_prefixes) or variable != "em":
            continue
        if _normalized_source_name(source) not in aliases:
            continue
        if not metric_name.startswith("mean@"):
            continue
        try:
            response_count = int(metric_name.removeprefix("mean@"))
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid validation EM metric {key}={raw_value!r}") from exc
        if response_count < 1:
            raise ValueError(f"Validation EM metric has invalid response count: {key}")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Validation EM metric must be finite and in [0, 1]: {key}={value}")
        candidates.append((response_count, str(key), value))
    return candidates


def extract_validation_em(metrics: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    """Extract the most complete validation ``mean@N`` EM metric for a dataset."""

    if dataset not in _DATASET_ALIASES:
        raise ValueError(f"Unsupported checkpoint-selection dataset: {dataset}")
    candidates = _metric_candidates(metrics, dataset)
    if not candidates:
        aliases = ", ".join(sorted(_DATASET_ALIASES[dataset]))
        raise KeyError(f"Missing validation EM metric for {dataset}; expected one of {aliases}")

    response_count, metric_key, value = max(candidates, key=lambda item: (item[0], item[1]))
    return {
        "dataset": dataset,
        "metric_key": metric_key,
        "response_count": response_count,
        "em": value,
    }


def extract_test_em(metrics: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    """Extract the most complete held-out test ``mean@N`` EM metric."""

    if dataset not in _DATASET_ALIASES:
        raise ValueError(f"Unsupported checkpoint-selection dataset: {dataset}")
    candidates = _metric_candidates(metrics, dataset, ("test-",))
    if not candidates:
        aliases = ", ".join(sorted(_DATASET_ALIASES[dataset]))
        raise KeyError(f"Missing test EM metric for {dataset}; expected one of {aliases}")
    response_count, metric_key, value = max(candidates, key=lambda item: (item[0], item[1]))
    return {
        "dataset": dataset,
        "metric_key": metric_key,
        "response_count": response_count,
        "em": value,
    }


def average_nq_hotpotqa_em(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return the equal-weight NQ/HotpotQA validation EM selection score."""

    nq = extract_validation_em(metrics, "nq")
    hotpotqa = extract_validation_em(metrics, "hotpotqa")
    mean_em = (nq["em"] + hotpotqa["em"]) / 2.0
    if not math.isfinite(mean_em):
        raise ValueError(f"Non-finite NQ/HotpotQA average EM: {mean_em!r}")
    return {
        "nq_em": nq["em"],
        "hotpotqa_em": hotpotqa["em"],
        "mean_em": mean_em,
        "datasets": {
            "nq": nq,
            "hotpotqa": hotpotqa,
        },
    }


def average_bamboogle_musique_triviaqa_em(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return the equal-weight held-out Bamboogle/Musique/TriviaQA EM score."""

    datasets = {
        name: extract_test_em(metrics, name)
        for name in ("bamboogle", "musique", "triviaqa")
    }
    mean_em = sum(info["em"] for info in datasets.values()) / len(datasets)
    if not math.isfinite(mean_em):
        raise ValueError(f"Non-finite held-out test average EM: {mean_em!r}")
    return {
        "bamboogle_em": datasets["bamboogle"]["em"],
        "musique_em": datasets["musique"]["em"],
        "triviaqa_em": datasets["triviaqa"]["em"],
        "mean_em": mean_em,
        "datasets": datasets,
    }


def select_best_checkpoint_step(candidates: Mapping[int, Mapping[str, Any]]) -> int:
    """Select the highest average-EM checkpoint, breaking ties by newest step."""

    if not candidates:
        raise ValueError("Cannot select a checkpoint from an empty candidate set")

    normalized: dict[int, float] = {}
    for raw_step, candidate in candidates.items():
        step = int(raw_step)
        try:
            score = float(candidate["mean_em"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Checkpoint candidate {step} has no valid mean_em") from exc
        if not math.isfinite(score):
            raise ValueError(f"Checkpoint candidate {step} has non-finite mean_em: {score!r}")
        normalized[step] = score

    return max(normalized, key=lambda step: (normalized[step], step))
