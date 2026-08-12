# Agents.md

## Scope

Search-OPD is an experiment/research repository for data-sharded Search-R1 reinforcement learning and online policy distillation. Keep changes scoped to this project. Do not modify sibling projects or shared datasets in place.

## Reproducibility

- Record base model revision, dataset revision, shard assignment, random seed, rollout count, sequence limits, reward/verifier versions, and teacher/student checkpoints for every run.
- Keep train, validation, and test data strictly separated. Never use test data for routing, filtering, checkpoint selection, or hyperparameter tuning.
- Report both matched-student-budget and matched-end-to-end-compute results. Include teacher training, rollout, search/API, and OPD costs.
- Prefer deterministic shard manifests checked into the experiment config; do not silently reshuffle examples between runs.

## Method constraints

- The primary method is data-specialized RL teachers followed by OPD into one deployable student.
- Include random-shard controls before attributing gains to specialization.
- Preserve verified correct trajectories and document teacher-conflict resolution; do not average incompatible teacher policies without an explicit rationale.
- Keep a general-data replay/retention component when needed and measure per-shard retention to detect forgetting.

## Code and artifacts

- Use existing Search-R1 conventions where available; avoid unrelated refactors.
- Store configs, launch commands, metrics, and evaluation summaries alongside each experiment.
- Do not commit model checkpoints or generated corpora unless explicitly requested; reference their immutable locations instead.
- Add focused tests for shard assignment, data leakage checks, teacher routing, and OPD loss construction.

## Reporting

Headline claims require mean and variance across seeds. Separate accuracy gains from gains caused by extra compute, more trajectories, or test leakage. Update README status and experiment tables when a milestone is completed.
