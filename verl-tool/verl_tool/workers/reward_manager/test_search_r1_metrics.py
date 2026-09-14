import numpy as np
import torch

from verl import DataProto
from verl_tool.workers.reward_manager.search_r1_qa_em import (
    SearchR1QAEMRewardManager,
    em_check,
    token_f1,
)


def test_search_r1_em_and_token_f1_use_normalized_references():
    references = ["The red fox"]

    assert em_check("a RED fox", references) == 1
    assert token_f1("red fox jumps", references) == 0.8
    assert token_f1("blue whale", references) == 0.0


class _ResponseOnlyTokenizer:
    def decode(self, token_ids, **_kwargs):
        token_ids = tuple(token_ids.tolist())
        if token_ids == ():
            return ""
        if token_ids == (20, 21):
            return "unfinished response"
        if token_ids == (30, 31):
            return "<answer> Correct </answer>"
        raise AssertionError(f"unexpected token ids decoded: {token_ids}")


def _reward_batch(response_ids):
    return DataProto.from_single_dict(
        {
            "prompts": torch.tensor([[10, 11]]),
            "responses": torch.tensor([response_ids]),
            "attention_mask": torch.tensor([[1, 1] + [1] * len(response_ids)]),
            "reward_model": np.asarray(
                [{"ground_truth": {"target": ["Correct"]}}],
                dtype=object,
            ),
        }
    )


def test_reward_does_not_extract_answer_from_prompt_example():
    manager = SearchR1QAEMRewardManager(tokenizer=_ResponseOnlyTokenizer())

    result = manager(_reward_batch([20, 21]), return_dict=True)

    assert result["reward_tensor"].sum().item() == 0.0
    assert result["reward_extra_info"]["em"] == [0.0]


def test_empty_response_does_not_extract_answer_from_prompt_example():
    manager = SearchR1QAEMRewardManager(tokenizer=_ResponseOnlyTokenizer())

    result = manager(_reward_batch([]), return_dict=True)

    assert result["reward_tensor"].numel() == 0
    assert result["reward_extra_info"]["em"] == [0.0]


def test_reward_extracts_answer_from_response():
    manager = SearchR1QAEMRewardManager(tokenizer=_ResponseOnlyTokenizer())

    result = manager(_reward_batch([30, 31]), return_dict=True)

    assert result["reward_tensor"].sum().item() == 1.0
    assert result["reward_extra_info"]["em"] == [1.0]
