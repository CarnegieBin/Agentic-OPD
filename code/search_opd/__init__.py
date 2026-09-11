"""Search-R1 multi-teacher on-policy distillation."""

from search_opd.core_opd import (
    combine_opd_and_task_advantages,
    opd_advantages_per_teacher,
    sampled_reverse_kl_per_teacher,
    uniform_teacher_average,
)

__all__ = [
    "combine_opd_and_task_advantages",
    "opd_advantages_per_teacher",
    "sampled_reverse_kl_per_teacher",
    "uniform_teacher_average",
]
