import pytest
import torch

from search_opd.generated_token_mask_opd import (
    select_opd_response_mask,
    validate_generated_token_mask,
)


def test_multiturn_mask_keeps_actions_and_masks_observations_and_padding():
    # [assistant action, assistant action, tool observation, tool observation,
    #  final assistant EOS, right padding].  IDs are intentionally arbitrary:
    # observation text must never be inferred from token IDs.
    token_ids = torch.tensor([[101, 102, 7, 8, 103, 0]])
    mask = validate_generated_token_mask(
        torch.tensor([[1, 1, 0, 0, 1, 0]]),
        expected_shape=tuple(token_ids.shape),
    )

    assert mask.tolist() == [[True, True, False, False, True, False]]


def test_full_attention_mask_defensively_removes_response_padding():
    mask = validate_generated_token_mask(
        torch.tensor([[1, 1, 0, 1]]),
        response_attention_mask=torch.tensor([[1, 1, 1, 1, 0, 0]]),
    )

    assert mask.tolist() == [[True, True, False, False]]


def test_single_turn_without_observation_keeps_eos_and_all_generated_tokens():
    mask = validate_generated_token_mask(
        torch.tensor([[1, 1, 1]]),
        expected_shape=(1, 3),
    )

    assert mask.all()


def test_empty_generated_trajectory_is_valid_and_has_no_active_tokens():
    mask = validate_generated_token_mask(torch.zeros(2, 4, dtype=torch.long))

    assert mask.shape == (2, 4)
    assert not mask.any()


def test_explicit_opd_mask_takes_precedence_over_generic_response_mask():
    mask = select_opd_response_mask(
        {
            "response_mask": torch.ones(1, 3),
            "opd_response_mask": torch.tensor([[1, 0, 1]]),
        },
        expected_shape=(1, 3),
    )

    assert mask.tolist() == [[True, False, True]]


@pytest.mark.parametrize(
    "invalid_mask",
    [
        torch.tensor([[0, 2]]),
        torch.tensor([[0.0, float("nan")]]),
        torch.tensor([[0.0, float("inf")]]),
    ],
)
def test_non_binary_or_non_finite_masks_are_rejected(invalid_mask):
    with pytest.raises(ValueError, match="mask"):
        validate_generated_token_mask(invalid_mask)


def test_mask_shape_is_checked_before_actor_loss():
    with pytest.raises(ValueError, match="expected"):
        validate_generated_token_mask(
            torch.ones(1, 2),
            expected_shape=(1, 3),
        )
