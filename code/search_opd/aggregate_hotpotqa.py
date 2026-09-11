"""Aggregate HotpotQA solved-question coverage by validation checkpoint.

The input is the ``validation_trajectories`` directory written by Search-R1,
with one ``global_step_<N>`` directory per validation checkpoint.  Both the
full ``searchR1_hotpotqa.jsonl`` records and compact
``hotpotqa_judgment.jsonl`` records are supported.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


_STEP_RE = re.compile(r"(?:global_)?step[_-]?(\d+)", re.IGNORECASE)


def _step(path: Path, record: dict[str, Any]) -> int:
    value = record.get("global_step")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    match = _STEP_RE.search(path.parent.name)
    if match is None:
        raise ValueError(f"Cannot determine checkpoint step from {path}")
    return int(match.group(1))


def _is_hotpotqa(path: Path, record: dict[str, Any]) -> bool:
    source = str(record.get("data_source", "")).lower().replace("-", "_")
    name = path.stem.lower().replace("-", "_")
    return "hotpotqa" in source or "hotpotqa" in name


def _uid(record: dict[str, Any], path: Path, line_number: int) -> str:
    value = record.get("sample_uid", record.get("uid"))
    if value is None:
        raise ValueError(f"Missing sample_uid/uid at {path}:{line_number}")
    return str(value)


def _solved(record: dict[str, Any], path: Path, line_number: int) -> bool:
    value = record.get("em", record.get("score", 0.0))
    try:
        return float(value) > 0.0
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid EM/score at {path}:{line_number}: {value!r}") from exc


def collect_solved(input_dir: Path) -> dict[int, set[str]]:
    """Return one de-duplicated solved UID set for each checkpoint step."""

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    by_step: dict[int, set[str]] = {}
    # Judgment files contain the same UID/EM information without rendered
    # prompts and responses. Select them per checkpoint, falling back to the
    # full trajectory only for checkpoints that lack a judgment file.
    files: list[Path] = []
    for step_dir in input_dir.glob("global_step_*"):
        judgment = step_dir / "hotpotqa_judgment.jsonl"
        trajectory = step_dir / "searchR1_hotpotqa.jsonl"
        if judgment.is_file():
            files.append(judgment)
        elif trajectory.is_file():
            files.append(trajectory)
    if not files:
        raise FileNotFoundError(f"No HotpotQA JSONL files found under {input_dir}")

    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_number}")
                if not _is_hotpotqa(path, record) or not _solved(record, path, line_number):
                    continue
                step = _step(path, record)
                by_step.setdefault(step, set()).add(_uid(record, path, line_number))
    if not by_step:
        raise ValueError(f"No solved HotpotQA records found under {input_dir}")
    return by_step


def write_csv(by_step: dict[int, set[str]], output: Path) -> None:
    """Write current and prior-checkpoint solved-question counts."""

    output.parent.mkdir(parents=True, exist_ok=True)
    historical: set[str] = set()
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["step", "current_step_solved_count", "historical_solved_count"])
        for step in sorted(by_step):
            current = by_step[step]
            writer.writerow([step, len(current), len(historical)])
            historical |= current


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="validation_trajectories directory")
    parser.add_argument("output", type=Path, help="output CSV, e.g. data/preliminary/HotpotQA.csv")
    args = parser.parse_args()
    write_csv(collect_solved(args.input_dir), args.output)


if __name__ == "__main__":
    main()
