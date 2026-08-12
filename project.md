# Search-OPD

Search-OPD studies whether Search-R1-style reinforcement learning can use a much larger training corpus more effectively by training data-specialized search-policy teachers and consolidating them into one student with online policy distillation (OPD).

## Research question

Given a Search-R1 corpus of roughly 19M examples and a sub-7B base model, does

```text
partition data -> RL-train K specialized teachers -> OPD into one student
```

outperform direct RL on the complete corpus under matched model size and, ideally, matched end-to-end compute?

## Working name

**Search-OPD: Scaling Search Reinforcement Learning through Data-Specialized Online Policy Distillation**

## Planned comparisons

- Full-data Search-R1 RL (primary baseline)
- Full-data RL across multiple seeds
- Random data shards + OPD
- Task/ability shards + OPD
- Difficulty/search-behavior shards + OPD
- Single-teacher self-distillation and supervised trajectory distillation

The main sweep is `K in {2, 4, 8}`. All claims must report student size, rollout count, distilled tokens, teacher cost, search/API calls, and total GPU/FLOPs budget.

## Evaluation

Report answer accuracy (EM/F1), multi-hop and difficulty-stratified results, out-of-distribution generalization, evidence quality, search turns, invalid-query rate, and accuracy per search cost. Use a held-out validation split, a strictly untouched test split, and at least three random seeds for headline comparisons.

## Status

This directory currently contains the research specification. Training code, configs, checkpoints, and logs should be added only with reproducible commands and budget metadata.
