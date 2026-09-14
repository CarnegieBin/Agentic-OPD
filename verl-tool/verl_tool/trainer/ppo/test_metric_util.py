from verl_tool.trainer.ppo.metric_util import (
    flatten_evaluation_metrics,
    process_validation_metrics,
)


def test_test_metrics_keep_nq_and_hotpotqa_namespaces():
    grouped = process_validation_metrics(
        data_sources=["searchR1_nq", "searchR1_hotpotqa"],
        sample_uids=["nq-1", "hotpotqa-1"],
        infos_dict={"em": [1.0, 0.0], "f1": [1.0, 0.5]},
    )
    flattened = flatten_evaluation_metrics(grouped, split="test")

    assert flattened["test-aux/searchR1_nq/em/mean@1"] == 1.0
    assert flattened["test-aux/searchR1_hotpotqa/em/mean@1"] == 0.0
    assert flattened["test-aux/searchR1_nq/f1/mean@1"] == 1.0
    assert flattened["test-aux/searchR1_hotpotqa/f1/mean@1"] == 0.5
    assert not any("/reward/" in key for key in flattened)
    assert not any(key.startswith("val-") for key in flattened)
