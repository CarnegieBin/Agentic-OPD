import math

import pytest

from search_opd.checkpoint_retention_opd import (
    average_bamboogle_musique_triviaqa_em,
    average_nq_hotpotqa_em,
    extract_test_em,
    extract_validation_em,
    select_best_checkpoint_step,
)


def test_average_uses_equal_dataset_weight():
    selection = average_nq_hotpotqa_em(
        {
            "val-aux/searchR1_nq/em/mean@1": 0.8,
            "val-aux/searchR1_hotpotqa/em/mean@1": 0.2,
        }
    )

    assert selection["nq_em"] == pytest.approx(0.8)
    assert selection["hotpotqa_em"] == pytest.approx(0.2)
    assert selection["mean_em"] == pytest.approx(0.5)


def test_test_checkpoint_score_uses_three_equal_dataset_weights():
    selection = average_bamboogle_musique_triviaqa_em(
        {
            "test-core/searchR1_bamboogle/em/mean@1": 0.9,
            "test-core/searchR1_musique/em/mean@1": 0.3,
            "test-core/searchR1_triviaqa/em/mean@1": 0.6,
        }
    )

    assert selection["bamboogle_em"] == pytest.approx(0.9)
    assert selection["musique_em"] == pytest.approx(0.3)
    assert selection["triviaqa_em"] == pytest.approx(0.6)
    assert selection["mean_em"] == pytest.approx(0.6)


def test_extract_test_em_does_not_accept_validation_namespace():
    with pytest.raises(KeyError):
        extract_test_em({"val-aux/searchR1_bamboogle/em/mean@1": 0.9}, "bamboogle")


def test_extract_uses_largest_mean_response_count():
    extracted = extract_validation_em(
        {
            "val-aux/searchR1_nq/em/mean@1": 0.4,
            "val-core/searchR1_nq/em/mean@5": 0.6,
        },
        "nq",
    )

    assert extracted["response_count"] == 5
    assert extracted["em"] == pytest.approx(0.6)
    assert extracted["metric_key"] == "val-core/searchR1_nq/em/mean@5"


@pytest.mark.parametrize(
    "metrics",
    [
        {"val-aux/searchR1_nq/em/mean@1": 0.5},
        {
            "val-aux/searchR1_nq/em/mean@1": math.nan,
            "val-aux/searchR1_hotpotqa/em/mean@1": 0.5,
        },
        {
            "val-aux/searchR1_nq/em/mean@1": 0.5,
            "val-aux/searchR1_hotpotqa/em/mean@1": 1.1,
        },
    ],
)
def test_average_rejects_missing_or_invalid_metrics(metrics):
    with pytest.raises((KeyError, ValueError)):
        average_nq_hotpotqa_em(metrics)


def test_best_checkpoint_prefers_score_then_newest_step():
    candidates = {
        5: {"mean_em": 0.4},
        10: {"mean_em": 0.5},
        15: {"mean_em": 0.5},
    }

    assert select_best_checkpoint_step(candidates) == 15


def test_best_checkpoint_rejects_non_finite_score():
    with pytest.raises(ValueError, match="non-finite"):
        select_best_checkpoint_step({5: {"mean_em": float("inf")}})
