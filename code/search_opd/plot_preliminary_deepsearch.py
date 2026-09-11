"""Plot the preliminary DeepSearch capability curves.

The source CSV is the HotpotQA validation coverage exported from the
Search-R1 run.  Only positive steps through ``max_step`` are shown; the
historical series is the union of solved UIDs observed before the current
checkpoint, as defined by ``aggregate_hotpotqa.py``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


def read_rows(csv_path: Path, max_step: int) -> tuple[list[int], list[int], list[int]]:
    steps: list[int] = []
    current: list[int] = []
    historical: list[int] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {"step", "current_step_solved_count", "historical_solved_count"}
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(f"Unexpected CSV columns in {csv_path}: {reader.fieldnames}")
        for row in reader:
            step = int(row["step"])
            if step <= 0 or step > max_step:
                continue
            steps.append(step)
            current.append(int(row["current_step_solved_count"]))
            historical.append(int(row["historical_solved_count"]))
    if not steps:
        raise ValueError(f"No rows with 0 < step <= {max_step} in {csv_path}")
    return steps, current, historical


def plot(csv_path: Path, output_stem: Path, max_step: int) -> None:
    steps, current, historical = read_rows(csv_path, max_step)
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # A compact aspect ratio remains legible in the three-panel preliminary
    # figure while preserving vector text in the PDF output.
    fig, ax = plt.subplots(figsize=(3.25, 2.18), dpi=220)
    current_color = "#1B6E77"
    historical_color = "#C26A2E"
    gap_color = "#D9B44A"
    ax.fill_between(
        steps,
        current,
        historical,
        color=gap_color,
        alpha=0.16,
        linewidth=0,
        label="Historical gap",
        zorder=1,
    )
    ax.plot(
        steps,
        historical,
        color=historical_color,
        linewidth=1.8,
        label="Historical union",
        zorder=3,
    )
    ax.plot(
        steps,
        current,
        color=current_color,
        linewidth=1.8,
        label="Current checkpoint",
        zorder=4,
    )
    ax.set_title("DeepSearch (HotpotQA)", loc="left", fontweight="bold", pad=5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Solved questions")
    ax.set_xlim(min(steps), max_step)
    ax.set_xticks(range(0, max_step + 1, 50))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D8DDE3", linewidth=0.55, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.6, length=3)
    ax.legend(
        loc="upper left",
        frameon=False,
        ncol=1,
        handlelength=2.2,
        borderpad=0,
        labelspacing=0.35,
    )
    fig.tight_layout(pad=0.5)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", metadata={"Creator": "Agentic-OPD"})
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", metadata={"Software": "matplotlib"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("output_stem", type=Path)
    parser.add_argument("--max-step", type=int, default=300)
    args = parser.parse_args()
    plot(args.csv_path, args.output_stem, args.max_step)


if __name__ == "__main__":
    main()
