# Agents.md

## Scope

Search-OPD is an experiment/research repository for temporal Search-R1
reinforcement learning and online policy distillation. Teachers are immutable
checkpoints from different steps of one training run. Keep changes scoped to
this project. Do not modify sibling projects or shared datasets in place.

## Reproducibility

- Record base model revision, dataset revision, deterministic checkpoint
  manifest, random seed, rollout count, sequence limits, reward/verifier
  versions, and teacher/student checkpoints for every run.
- Report both matched-student-budget and matched-end-to-end-compute results. Include teacher training, rollout, search/API, and OPD costs.
- Prefer deterministic temporal-checkpoint manifests checked into the experiment
  config; do not silently reshuffle checkpoints between runs.

## Method constraints

- The primary method is temporal RL teachers followed by OPD into one deployable
  student. Teachers must come from immutable checkpoints of one Search-R1 run
  with the same base model, data mixture, and tool environment.
- Include uniformly spaced and equal-count random temporal-checkpoint controls
  before attributing gains to the number or selection of teachers.
- Preserve verified correct trajectories and document teacher-conflict resolution; do not average incompatible teacher policies without an explicit rationale.
- Keep a general-data replay/retention component when needed and measure
  retention separately for capabilities acquired at earlier checkpoints.

## Code and artifacts

- Use existing Search-R1 conventions where available; avoid unrelated refactors.
- Store configs, launch commands, metrics, and evaluation summaries alongside each experiment.
- Do not commit model checkpoints or generated corpora unless explicitly requested; reference their immutable locations instead.
- Add focused tests for checkpoint assignment, data leakage checks, temporal
  teacher routing, and OPD loss construction.

## Reporting

Headline claims require mean and variance across seeds. Separate accuracy gains from gains caused by extra compute, more trajectories, or test leakage. Update README status and experiment tables when a milestone is completed.
