# Search-OPD

Search-OPD studies whether data-specialized Search-R1 reinforcement-learning
teachers can be consolidated into one deployable student with online policy
distillation (OPD).

## Current setup

- Runtime: `verlai/verl:vllm011.latest` Docker container
- Hardware: 8 x NVIDIA A800-SXM4 80 GB
- Default model: `/ssd2/llm_models/Qwen3-1.7B`
- Training entry point: `code/verl-tool/scripts/Search-R1/train.sh`
- Optimizer: Muon with automatic AdamW fallback for non-matrix parameters
- Tracking: console plus SwanLab
- Default quick evaluation: NQ (single-hop) and HotpotQA (multi-hop)

## Data

Data is kept under `data/search_r1/`. Training and test splits are separated and
the dataset-specific test files are preferred for evaluation. See
`data/search_r1/MANIFEST.md` for source revisions, hashes, row counts, and the
train/test overlap audit.

## Running

From the repository's `code/verl-tool/scripts/Search-R1` directory:

```bash
bash train.sh
```

The script validates the model and training paths, starts from its own directory,
and writes a timestamped `.log` file there while streaming the same output to the
console. Paths, batch sizes, rollout count, GPU count, SwanLab project, and other
budgets can be overridden with environment variables.

The model path is case-sensitive: the verified server path is
`/ssd2/llm_models/Qwen3-1.7B`.

## Research protocol

Motivation runs use NQ and HotpotQA to reduce turnaround time. Final claims must
restore all held-out datasets, include random-shard controls, preserve train/
validation/test separation, and report mean and variance across at least three
seeds with matched student and end-to-end compute budgets.
