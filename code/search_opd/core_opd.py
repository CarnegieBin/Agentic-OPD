"""Pure tensor operations for Search-OPD.

The configured objective is Student Top-k OPD.  At every model-generated
response position the student supplies its top-k token support.  Both the
student and each frozen teacher are then renormalized on that *same* support
and the objective is the reverse KL

    D_KL(p_student^k || p_teacher^k).

The older sampled-token helpers are retained below for compatibility with
existing experiment fixtures, but the actor uses :func:`topk_reverse_kl_per_teacher`
for the live objective.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


DEFAULT_OPD_TOP_K = 16


def _finite_float(value: float, field_name: str) -> float:
    """Convert a scalar objective setting and reject NaN or infinity."""

    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


def _validate_log_prob_shapes(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: Optional[torch.Tensor] = None,
) -> None:
    if student_log_probs.ndim != 2:
        raise ValueError(
            "student_log_probs must have shape [batch, response_length], "
            f"got {tuple(student_log_probs.shape)}"
        )
    if teacher_log_probs.ndim != 3:
        raise ValueError(
            "teacher_log_probs must have shape "
            "[batch, response_length, num_teachers], "
            f"got {tuple(teacher_log_probs.shape)}"
        )
    if teacher_log_probs.shape[:2] != student_log_probs.shape:
        raise ValueError(
            "Student and teacher log-probability shapes do not align: "
            f"student={tuple(student_log_probs.shape)}, "
            f"teachers={tuple(teacher_log_probs.shape)}"
        )
    if teacher_log_probs.shape[-1] < 1:
        raise ValueError("At least one teacher is required")
    if response_mask is not None and response_mask.shape != student_log_probs.shape:
        raise ValueError(
            "response_mask must match student_log_probs: "
            f"mask={tuple(response_mask.shape)}, "
            f"student={tuple(student_log_probs.shape)}"
        )


def sampled_reverse_kl_per_teacher(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    response_mask: Optional[torch.Tensor] = None,
    max_abs_log_ratio: Optional[float] = None,
) -> torch.Tensor:
    """Return detached sampled reverse-KL estimates for every teacher.

    Args:
        student_log_probs: Student rollout-policy log probabilities with shape
            ``[batch, response_length]``.
        teacher_log_probs: Frozen-teacher scores for the same sampled tokens
            with shape ``[batch, response_length, num_teachers]``.
        response_mask: Optional generated-token mask.  It is used for finite
            checks and inactive positions are zeroed for numerical safety; the
            PPO loss still applies the mask during aggregation.
        max_abs_log_ratio: Optional symmetric clamp applied independently to
            each teacher's sampled log-ratio before teachers are averaged.

    Returns:
        A stop-gradient tensor of shape
        ``[batch, response_length, num_teachers]``.
    """

    _validate_log_prob_shapes(student_log_probs, teacher_log_probs, response_mask)
    if max_abs_log_ratio is not None:
        max_abs_log_ratio = _finite_float(
            max_abs_log_ratio,
            "max_abs_log_ratio",
        )
        if max_abs_log_ratio <= 0:
            raise ValueError("max_abs_log_ratio must be positive or null")

    reverse_kl = student_log_probs.detach().unsqueeze(-1) - teacher_log_probs.detach()
    active = (
        response_mask.to(dtype=torch.bool).unsqueeze(-1).expand_as(reverse_kl)
        if response_mask is not None
        else torch.ones_like(reverse_kl, dtype=torch.bool)
    )
    if active.any() and not torch.isfinite(reverse_kl[active]).all():
        raise ValueError("Non-finite student/teacher log-probability ratio on active response tokens")
    reverse_kl = torch.where(
        active,
        reverse_kl,
        torch.zeros_like(reverse_kl),
    )
    if max_abs_log_ratio is not None:
        reverse_kl = reverse_kl.clamp(
            min=-float(max_abs_log_ratio),
            max=float(max_abs_log_ratio),
        )
    return reverse_kl


def _validate_topk_log_prob_shapes(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    response_mask: Optional[torch.Tensor] = None,
) -> None:
    """Validate the support-wise log-probability tensor contract."""

    if student_topk_log_probs.ndim != 3:
        raise ValueError(
            "student_topk_log_probs must have shape "
            "[batch, response_length, top_k], "
            f"got {tuple(student_topk_log_probs.shape)}"
        )
    if teacher_topk_log_probs.ndim != 4:
        raise ValueError(
            "teacher_topk_log_probs must have shape "
            "[batch, response_length, top_k, num_teachers], "
            f"got {tuple(teacher_topk_log_probs.shape)}"
        )
    if student_topk_log_probs.shape[:2] != teacher_topk_log_probs.shape[:2]:
        raise ValueError(
            "Student and teacher top-k log-probability batch/response shapes "
            "do not align: "
            f"student={tuple(student_topk_log_probs.shape)}, "
            f"teachers={tuple(teacher_topk_log_probs.shape)}"
        )
    if student_topk_log_probs.shape[-1] != teacher_topk_log_probs.shape[-2]:
        raise ValueError(
            "Student and teacher top-k supports do not have the same width: "
            f"student={student_topk_log_probs.shape[-1]}, "
            f"teacher={teacher_topk_log_probs.shape[-2]}"
        )
    if student_topk_log_probs.shape[-1] < 1:
        raise ValueError("The top-k support must contain at least one token")
    if teacher_topk_log_probs.shape[-1] < 1:
        raise ValueError("At least one teacher is required")
    if response_mask is not None and response_mask.shape != student_topk_log_probs.shape[:2]:
        raise ValueError(
            "response_mask must match the first two top-k dimensions: "
            f"mask={tuple(response_mask.shape)}, "
            f"student={tuple(student_topk_log_probs.shape)}"
        )


def topk_reverse_kl_per_teacher(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    *,
    response_mask: Optional[torch.Tensor] = None,
    max_abs_log_ratio: Optional[float] = None,
) -> torch.Tensor:
    """Compute support-renormalized reverse KL for every teacher.

    ``student_topk_log_probs`` and ``teacher_topk_log_probs`` contain the
    full-vocabulary log probabilities at the selected token IDs.  They are
    deliberately renormalized along the supplied support before calculating
    KL; using the full-vocabulary normalization constants here would not
    implement Student Top-k OPD.

    Args:
        student_topk_log_probs: Shape ``[batch, response_length, top_k]``.
            This tensor remains attached to the student graph.
        teacher_topk_log_probs: Shape
            ``[batch, response_length, top_k, num_teachers]``.  Teacher
            values are detached inside this function.
        response_mask: Optional mask selecting model-generated response
            tokens.  Inactive positions are returned as zero.
        max_abs_log_ratio: Optional symmetric clamp on the support log ratio.

    Returns:
        A differentiable tensor of shape
        ``[batch, response_length, num_teachers]``.
    """

    _validate_topk_log_prob_shapes(
        student_topk_log_probs,
        teacher_topk_log_probs,
        response_mask,
    )
    if max_abs_log_ratio is not None:
        max_abs_log_ratio = _finite_float(
            max_abs_log_ratio,
            "max_abs_log_ratio",
        )
        if max_abs_log_ratio <= 0:
            raise ValueError("max_abs_log_ratio must be positive or null")

    active = (
        response_mask.to(dtype=torch.bool)
        if response_mask is not None
        else torch.ones(
            student_topk_log_probs.shape[:2],
            dtype=torch.bool,
            device=student_topk_log_probs.device,
        )
    )
    student_active = active.unsqueeze(-1).expand_as(student_topk_log_probs)
    teacher_active = active.unsqueeze(-1).unsqueeze(-1).expand_as(teacher_topk_log_probs)
    if student_active.any() and not torch.isfinite(student_topk_log_probs[student_active]).all():
        raise ValueError("Non-finite student top-k log-probabilities on active response tokens")
    if teacher_active.any() and not torch.isfinite(teacher_topk_log_probs[teacher_active]).all():
        raise ValueError("Non-finite teacher top-k log-probabilities on active response tokens")

    # The teacher is frozen.  Keep the student support values attached so the
    # direct KL objective, rather than a sampled-token surrogate, supplies the
    # actor gradient.
    student_support_log_probs = torch.log_softmax(student_topk_log_probs, dim=-1)
    teacher_support_log_probs = torch.log_softmax(
        teacher_topk_log_probs.detach(),
        dim=-2,
    )
    log_ratio = (
        student_support_log_probs.unsqueeze(-1) - teacher_support_log_probs
    )
    if max_abs_log_ratio is not None:
        log_ratio = log_ratio.clamp(
            min=-float(max_abs_log_ratio),
            max=float(max_abs_log_ratio),
        )
    kl = (
        student_support_log_probs.exp().unsqueeze(-1) * log_ratio
    ).sum(dim=-2)
    return torch.where(
        active.unsqueeze(-1),
        kl,
        torch.zeros_like(kl),
    )


def uniform_teacher_average(per_teacher_values: torch.Tensor) -> torch.Tensor:
    """Sum independent teacher values and divide by the teacher count."""

    if per_teacher_values.ndim < 1 or per_teacher_values.shape[-1] < 1:
        raise ValueError("per_teacher_values must have a non-empty teacher dimension")
    return per_teacher_values.sum(dim=-1) / per_teacher_values.shape[-1]


def opd_advantages_per_teacher(
    reverse_kl_per_teacher: torch.Tensor,
    *,
    opd_loss_coef: float = 1.0,
) -> torch.Tensor:
    """Convert each teacher's reverse-KL estimate into its own PG advantage.

    The teacher dimension is intentionally retained.  The actor therefore
    computes one policy-gradient loss per teacher before applying the uniform
    arithmetic mean requested by the multi-teacher objective.
    """

    if reverse_kl_per_teacher.ndim != 3 or reverse_kl_per_teacher.shape[-1] < 1:
        raise ValueError(
            "reverse_kl_per_teacher must have shape "
            "[batch, response_length, num_teachers]"
        )
    opd_loss_coef = _finite_float(opd_loss_coef, "opd_loss_coef")
    if opd_loss_coef <= 0:
        raise ValueError("opd_loss_coef must be positive")
    return (-float(opd_loss_coef) * reverse_kl_per_teacher.detach()).detach()


def combine_opd_and_task_advantages(
    task_advantages: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    *,
    response_mask: torch.Tensor,
    opd_loss_coef: float = 1.0,
    task_loss_coef: float = 0.0,
    max_abs_log_ratio: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the PG-OPD advantage and combine it with an optional task signal.

    The OPD advantage for teacher ``i`` is
    ``log p_teacher_i - log p_student`` with stop-gradient.  The returned
    advantage averages those teacher-specific signals uniformly.  With the
    enforced one-epoch, one-mini-batch on-policy update, this is exactly the
    arithmetic mean of the per-teacher policy-gradient losses.

    Returns:
        ``(combined_advantages, reverse_kl_per_teacher)``.
    """

    if task_advantages.shape != student_log_probs.shape:
        raise ValueError(
            "task_advantages must match student_log_probs: "
            f"task={tuple(task_advantages.shape)}, "
            f"student={tuple(student_log_probs.shape)}"
        )
    opd_loss_coef = _finite_float(opd_loss_coef, "opd_loss_coef")
    task_loss_coef = _finite_float(task_loss_coef, "task_loss_coef")
    if opd_loss_coef <= 0:
        raise ValueError("opd_loss_coef must be positive")
    if task_loss_coef < 0:
        raise ValueError("task_loss_coef must be non-negative")

    reverse_kl = sampled_reverse_kl_per_teacher(
        student_log_probs,
        teacher_log_probs,
        response_mask=response_mask,
        max_abs_log_ratio=max_abs_log_ratio,
    )
    opd_advantages = opd_advantages_per_teacher(
        reverse_kl,
        opd_loss_coef=opd_loss_coef,
    )
    combined = uniform_teacher_average(opd_advantages)
    if task_loss_coef:
        combined = combined + task_loss_coef * task_advantages.detach()
    return combined.detach(), reverse_kl


def masked_teacher_means(
    per_teacher_values: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute one masked scalar mean per teacher for metrics."""

    if per_teacher_values.ndim != 3:
        raise ValueError("per_teacher_values must have shape [batch, response_length, num_teachers]")
    if response_mask.shape != per_teacher_values.shape[:2]:
        raise ValueError("response_mask must match the first two value dimensions")
    active = response_mask.to(dtype=torch.bool).unsqueeze(-1)
    weights = active.to(dtype=per_teacher_values.dtype)
    denominator = weights.sum().clamp_min(1.0)
    masked_values = torch.where(
        active,
        per_teacher_values,
        torch.zeros_like(per_teacher_values),
    )
    return masked_values.sum(dim=(0, 1)) / denominator


def masked_topk_teacher_means(
    per_teacher_values: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute one generated-token mean per teacher for top-k KL metrics."""

    if per_teacher_values.ndim != 3:
        raise ValueError(
            "per_teacher_values must have shape "
            "[batch, response_length, num_teachers]"
        )
    if response_mask.shape != per_teacher_values.shape[:2]:
        raise ValueError("response_mask must match the first two value dimensions")
    active = response_mask.to(dtype=torch.bool).unsqueeze(-1)
    weights = active.to(dtype=per_teacher_values.dtype)
    denominator = weights.sum().clamp_min(1.0)
    masked_values = torch.where(
        active,
        per_teacher_values,
        torch.zeros_like(per_teacher_values),
    )
    return masked_values.sum(dim=(0, 1)) / denominator
