from verl_tool.trainer.ppo.checkpoint_retention import (
    select_max_union_steps,
    solved_union_size,
    stable_validation_uid,
)


def test_keeps_all_candidates_below_limit():
    candidates = {5: {"a"}, 10: {"b"}}

    assert select_max_union_steps(candidates, keep_limit=5) == [5, 10]


def test_prefers_maximum_union_over_largest_individual_checkpoint():
    candidates = {
        5: {"a", "b", "c", "d"},
        10: {"a", "b", "c"},
        15: {"e", "f", "g"},
    }

    selected = select_max_union_steps(candidates, keep_limit=2)

    assert selected == [5, 15]
    assert solved_union_size(candidates, selected) == 7


def test_union_ties_prefer_newer_steps():
    candidates = {
        5: {"a"},
        10: {"a"},
        15: {"a"},
    }

    assert select_max_union_steps(candidates, keep_limit=2) == [10, 15]


def test_validation_uid_is_stable_and_distinguishes_source_indices():
    first = stable_validation_uid(
        "searchR1_nq",
        {"index": 7, "question": "Who wrote Hamlet?"},
        [1, 2, 3],
    )
    repeated = stable_validation_uid(
        "searchR1_nq",
        {"question": "Who wrote Hamlet?", "index": 7},
        [9, 9, 9],
    )
    second = stable_validation_uid(
        "searchR1_nq",
        {"index": 8, "question": "Who wrote Hamlet?"},
        [1, 2, 3],
    )

    assert first == repeated
    assert first != second
