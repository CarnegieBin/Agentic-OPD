import torch

from verl_tool.agent_loop.agent_loop import _pad_single_sequence


class _Tokenizer:
    padding_side = "right"
    pad_token_id = 0
    eos_token_id = 2

    def pad(self, *_args, **_kwargs):
        return {
            "input_ids": [4, 5, 0, 0],
            "attention_mask": [1, 1, 0, 0],
        }


def test_empty_sequence_is_padded_to_tensor_shape():
    output = _pad_single_sequence(
        _Tokenizer(),
        [],
        max_length=4,
        padding_side="right",
        pad_value=0,
        return_attention_mask=True,
    )

    assert torch.equal(output["input_ids"], torch.zeros((1, 4), dtype=torch.long))
    assert torch.equal(output["attention_mask"], torch.zeros((1, 4), dtype=torch.long))


def test_list_tokenizer_output_is_normalized_to_tensor():
    output = _pad_single_sequence(
        _Tokenizer(),
        [4, 5],
        max_length=4,
        padding_side="right",
        pad_value=0,
        return_attention_mask=True,
    )

    assert output["input_ids"].shape == (1, 4)
    assert output["input_ids"].dtype == torch.long
    assert output["attention_mask"].shape == (1, 4)
