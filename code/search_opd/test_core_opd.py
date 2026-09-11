import pytest
import torch

from search_opd.core_opd import (
    combine_opd_and_task_advantages,
    masked_teacher_means,
    masked_topk_teacher_means,
    opd_advantages_per_teacher,
    sampled_reverse_kl_per_teacher,
    topk_reverse_kl_per_teacher,
    uniform_teacher_average,
)


def test_reverse_kl_is_student_minus_each_teacher_and_detached():
    student = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    teachers = torch.tensor(
        [[[-2.0, -3.0], [-1.0, -4.0]]],
        requires_grad=True,
    )
    mask = torch.ones_like(student)

    values = sampled_reverse_kl_per_teacher(student, teachers, response_mask=mask)

    assert values.tolist() == [[[1.0, 2.0], [-1.0, 2.0]]]
    assert not values.requires_grad
    assert values.shape == (1, 2, 2)


def test_each_teacher_keeps_an_independent_stop_gradient_opd_advantage():
    reverse_kl = torch.tensor(
        [[[1.0, 3.0], [2.0, 4.0]]],
        requires_grad=True,
    )

    advantages = opd_advantages_per_teacher(
        reverse_kl,
        opd_loss_coef=2.0,
    )

    assert torch.equal(
        advantages,
        torch.tensor([[[-2.0, -6.0], [-4.0, -8.0]]]),
    )
    assert not advantages.requires_grad


def test_multiple_teachers_are_summed_then_uniformly_averaged():
    values = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])
    assert torch.equal(uniform_teacher_average(values), torch.tensor([[2.0, 6.0]]))

    mask = torch.tensor([[1, 0]])
    means = masked_teacher_means(values, mask)
    assert torch.equal(means, torch.tensor([1.0, 3.0]))
    assert torch.equal(uniform_teacher_average(means), torch.tensor(2.0))


def test_single_teacher_average_is_identity():
    values = torch.tensor([[[1.0], [5.0]]])

    assert torch.equal(uniform_teacher_average(values), torch.tensor([[1.0, 5.0]]))


def test_masked_teacher_means_ignore_non_finite_inactive_tokens():
    values = torch.tensor([[[1.0, 3.0], [float("nan"), float("nan")]]])
    mask = torch.tensor([[1, 0]])

    means = masked_teacher_means(values, mask)

    assert torch.equal(means, torch.tensor([1.0, 3.0]))


def test_opd_advantage_is_negative_reverse_kl_and_task_can_be_added():
    task = torch.tensor([[10.0, 20.0]])
    student = torch.tensor([[-1.0, -1.0]])
    teachers = torch.tensor([[[-2.0, -4.0], [-3.0, -5.0]]])
    mask = torch.ones_like(student)

    advantages, reverse_kl = combine_opd_and_task_advantages(
        task,
        student,
        teachers,
        response_mask=mask,
        opd_loss_coef=2.0,
        task_loss_coef=0.5,
    )

    # reverse KL per token: [[1, 3], [2, 4]], average then negate and scale.
    assert torch.equal(reverse_kl, torch.tensor([[[1.0, 3.0], [2.0, 4.0]]]))
    assert torch.equal(advantages, torch.tensor([[1.0, 4.0]]))


def test_invalid_teacher_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        sampled_reverse_kl_per_teacher(
            torch.zeros(2, 3),
            torch.zeros(2, 3),
            response_mask=torch.ones(2, 3),
        )


@pytest.mark.parametrize("max_abs_log_ratio", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_log_ratio_limit_is_rejected(max_abs_log_ratio):
    with pytest.raises(ValueError, match="finite"):
        sampled_reverse_kl_per_teacher(
            torch.zeros(1, 1),
            torch.zeros(1, 1, 1),
            response_mask=torch.ones(1, 1),
            max_abs_log_ratio=max_abs_log_ratio,
        )


def test_log_ratio_clamp_does_not_hide_non_finite_active_values():
    with pytest.raises(ValueError, match="Non-finite"):
        sampled_reverse_kl_per_teacher(
            torch.tensor([[float("inf")]]),
            torch.zeros(1, 1, 1),
            response_mask=torch.ones(1, 1),
            max_abs_log_ratio=10.0,
        )


def test_inactive_non_finite_log_ratios_are_zeroed():
    values = sampled_reverse_kl_per_teacher(
        torch.tensor([[1.0, float("nan")]]),
        torch.tensor([[[0.0, 0.0], [float("inf"), float("nan")]]]),
        response_mask=torch.tensor([[1, 0]]),
    )

    assert torch.equal(values, torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]))


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (opd_advantages_per_teacher, {"opd_loss_coef": float("nan")}),
        (
            combine_opd_and_task_advantages,
            {"opd_loss_coef": float("inf")},
        ),
        (
            combine_opd_and_task_advantages,
            {"task_loss_coef": float("nan")},
        ),
    ],
)
def test_non_finite_objective_coefficients_are_rejected(function, kwargs):
    reverse_kl = torch.zeros(1, 1, 1)
    if function is opd_advantages_per_teacher:
        arguments = (reverse_kl,)
    else:
        arguments = (
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            reverse_kl,
        )
        kwargs["response_mask"] = torch.ones(1, 1)

    with pytest.raises(ValueError, match="finite"):
        function(*arguments, **kwargs)


def test_student_topk_reverse_kl_renormalizes_both_distributions_on_support():
    student_support = torch.tensor([[[0.0, -1.0]]], requires_grad=True)
    teacher_support = torch.tensor([[[[-1.0], [0.0]]]])

    values = topk_reverse_kl_per_teacher(
        student_support,
        teacher_support,
        response_mask=torch.ones(1, 1),
    )

    expected = 0.4621171572600098
    assert values.shape == (1, 1, 1)
    assert values.item() == pytest.approx(expected)
    values.sum().backward()
    assert student_support.grad is not None
    assert torch.isfinite(student_support.grad).all()


def test_student_topk_reverse_kl_masks_observations_and_reports_teacher_means():
    student_support = torch.tensor(
        [[[0.0, -1.0], [0.0, -1.0]]],
        requires_grad=True,
    )
    teacher_support = torch.tensor(
        [[[[0.0, -1.0], [-1.0, 0.0]], [[0.0, -1.0], [-1.0, 0.0]]]]
    )
    mask = torch.tensor([[1, 0]])

    values = topk_reverse_kl_per_teacher(
        student_support,
        teacher_support,
        response_mask=mask,
    )

    assert values.shape == (1, 2, 2)
    assert torch.equal(values[:, 1], torch.zeros(1, 2))
    means = masked_topk_teacher_means(values, mask)
    assert means.shape == (2,)
    assert means[0].item() == pytest.approx(0.0)
    assert means[1].item() > 0.0
