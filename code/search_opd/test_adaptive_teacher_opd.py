import json

import pytest

from search_opd.adaptive_teacher_opd import (
    initialize_teacher_question_weights,
    load_validation_solved_sets,
    prune_dominated_teachers,
    rank_temporal_teachers,
    select_and_publish_teacher,
    update_teacher_question_weights,
    update_question_weights,
)
from search_opd.config_opd import TeacherSpec


def _write_trajectories(root, step, rows):
    directory = root / f"global_step_{step}"
    directory.mkdir(parents=True)
    for dataset in ("nq", "hotpotqa"):
        with (directory / f"searchR1_{dataset}.jsonl").open("w", encoding="utf-8") as handle:
            for uid, score in rows[dataset]:
                handle.write(
                    json.dumps(
                        {
                            "data_source": dataset,
                            "sample_uid": uid,
                            "score": score,
                        }
                    )
                    + "\n"
                )


def test_rank_uses_question_level_marginal_gain(tmp_path):
    student_root = tmp_path / "student"
    teacher_root = tmp_path / "teachers"
    _write_trajectories(
        student_root,
        0,
        {
            "nq": [("nq-1", 1), ("nq-2", 0), ("nq-3", 0), ("nq-4", 0)],
            "hotpotqa": [("hp-1", 1), ("hp-2", 0)],
        },
    )
    _write_trajectories(
        teacher_root,
        10,
        {
            "nq": [("nq-1", 1), ("nq-2", 1), ("nq-3", 1), ("nq-4", 0)],
            "hotpotqa": [("hp-1", 1), ("hp-2", 0)],
        },
    )
    _write_trajectories(
        teacher_root,
        20,
        {
            "nq": [("nq-1", 1), ("nq-2", 0), ("nq-3", 0), ("nq-4", 0)],
            "hotpotqa": [("hp-1", 1), ("hp-2", 1)],
        },
    )

    ranked = rank_temporal_teachers(
        (
            TeacherSpec("t10", "/teachers/global_step_10", step=10),
            TeacherSpec("t20", "/teachers/global_step_20", step=20),
        ),
        student_trajectory_dir=student_root / "global_step_0",
        teacher_trajectory_root=teacher_root,
    )

    assert ranked[0]["name"] == "t10"
    assert ranked[0]["marginal_gain"] == {"nq": 2, "hotpotqa": 0}
    assert ranked[0]["weighted_marginal_gain"] == 2
    assert ranked[0]["mean_normalized_marginal_gain"] == pytest.approx(0.25)


def test_question_weights_increment_once_per_round_and_cap_at_five():
    teacher_sets = {
        0: {"nq": {"nq-1", "nq-2"}, "hotpotqa": {"hp-1"}},
        1: {"nq": {"nq-2"}, "hotpotqa": {"hp-1"}},
    }
    weights = {}
    for _ in range(10):
        weights, changed = update_question_weights(
            weights,
            student_solved={"nq": set(), "hotpotqa": set()},
            teacher_solved_sets=teacher_sets,
            all_uids={"nq": {"nq-1", "nq-2"}, "hotpotqa": {"hp-1"}},
        )
        assert changed == {"nq:nq-1", "nq:nq-2", "hotpotqa:hp-1"}

    assert weights == {
        "nq:nq-1": 5,
        "nq:nq-2": 5,
        "hotpotqa:hp-1": 5,
    }


def test_startup_prunes_fully_covered_teacher_sets():
    specs = (
        TeacherSpec("early", "/teachers/global_step_10", step=10),
        TeacherSpec("late", "/teachers/global_step_20", step=20),
        TeacherSpec("independent", "/teachers/global_step_30", step=30),
    )
    active, pruned = prune_dominated_teachers(
        specs,
        {
            0: {"nq": {"a"}, "hotpotqa": set()},
            1: {"nq": {"a", "b"}, "hotpotqa": {"h"}},
            2: {"nq": set(), "hotpotqa": {"x"}},
        },
    )

    assert [spec.name for spec in active] == ["late", "independent"]
    assert pruned[0]["name"] == "early"
    assert pruned[0]["covered_by"]["name"] == "late"


def test_selection_state_is_atomic_and_records_history(tmp_path):
    student_root = tmp_path / "student"
    teacher_root = tmp_path / "teachers"
    _write_trajectories(
        student_root,
        0,
        {"nq": [("nq-1", 0)], "hotpotqa": [("hp-1", 0)]},
    )
    _write_trajectories(
        teacher_root,
        5,
        {"nq": [("nq-1", 1)], "hotpotqa": [("hp-1", 0)]},
    )
    state_path = tmp_path / "state.json"
    payload = select_and_publish_teacher(
        (TeacherSpec("t5", "/teachers/global_step_5", step=5),),
        student_trajectory_dir=student_root / "global_step_0",
        teacher_trajectory_root=teacher_root,
        state_path=state_path,
        global_step=0,
    )

    assert payload["selected"]["name"] == "t5"
    assert payload["version"] == 3
    assert payload["teacher_weights"]["0"]["nq:nq-1"] == 1
    assert payload["selected_teacher_indices"] == [0]
    assert payload["switch_count"] == 0
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["history"][0]["global_step"] == 0
    assert load_validation_solved_sets(student_root / "global_step_0")[1]["nq"] == set()


def test_teacher_weights_are_independent_reset_selected_and_reenter():
    teacher_sets = {
        0: {"nq": {"q0"}, "hotpotqa": set()},
        1: {"nq": {"q1"}, "hotpotqa": set()},
    }
    weights = initialize_teacher_question_weights(teacher_sets)

    weights, updated = update_teacher_question_weights(
        weights,
        student_solved={"nq": set(), "hotpotqa": set()},
        teacher_solved_sets=teacher_sets,
        selected_teacher_index=0,
    )
    assert weights[0]["nq:q0"] == 1
    assert weights[1]["nq:q1"] == 2
    assert updated[0] == {"nq:q0"}
    assert updated[1] == {"nq:q1"}

    # The replaced teacher is not removed.  It can win the next round because
    # its independent marginal weight has accumulated.
    weights, _ = update_teacher_question_weights(
        weights,
        student_solved={"nq": set(), "hotpotqa": set()},
        teacher_solved_sets=teacher_sets,
        selected_teacher_index=1,
    )
    assert weights[1]["nq:q1"] == 1
    assert weights[0]["nq:q0"] == 2
