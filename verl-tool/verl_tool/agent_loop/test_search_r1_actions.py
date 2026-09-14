import asyncio

from verl.experimental.agent_loop.tool_agent_loop import AgentData, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import SearchR1ToolParser


def _loop_with_markers():
    loop = object.__new__(ToolAgentLoop)
    loop.search_r1_format = True
    loop.search_r1_stop_token_ids = {
        "</search>": [9, 10],
        "</answer>": [11, 12],
        "<information>": [13, 14],
    }
    return loop


def test_search_action_keeps_only_first_action_and_its_logprobs():
    loop = _loop_with_markers()
    token_ids, log_probs, action = loop._truncate_search_r1_generation(
        [1, 2, 9, 10, 13, 14],
        [0.1] * 6,
    )

    assert token_ids == [1, 2, 9, 10]
    assert log_probs == [0.1] * 4
    assert action == "search"


def test_information_guard_removes_model_generated_information():
    loop = _loop_with_markers()
    token_ids, log_probs, action = loop._truncate_search_r1_generation(
        [1, 13, 14, 2],
        [0.1] * 4,
    )

    assert token_ids == [1]
    assert log_probs == [0.1]
    assert action == "invalid_information"


def test_answer_is_an_action_and_keeps_no_suffix():
    loop = _loop_with_markers()
    token_ids, _, action = loop._truncate_search_r1_generation(
        [1, 11, 12, 2],
        [0.1] * 4,
    )

    assert token_ids == [1, 11, 12]
    assert action == "answer"


def test_search_parser_executes_only_first_search():
    class Tokenizer:
        def decode(self, _):
            return "<search>a</search><search>b</search>"

    parser = SearchR1ToolParser(Tokenizer())
    _, calls = asyncio.run(parser.extract_tool_calls([1]))

    assert len(calls) == 1
    assert '"a"' in calls[0].arguments
    assert '"b"' not in calls[0].arguments


def test_observation_uses_official_plain_continuation_protocol():
    observation = ToolAgentLoop._format_search_r1_observation("retrieved passage")

    assert observation == "\n\n<information>retrieved passage</information>\n\n"
    assert "<|im_start|>" not in observation
    assert "<|im_end|>" not in observation


def test_observation_tokens_do_not_consume_generated_budget():
    agent_data = AgentData(
        messages=[],
        image_data=None,
        metrics={},
        request_id="test",
        tools_kwargs={},
    )
    agent_data.response_mask = [1, 1, 0, 0, 0, 1]

    assert ToolAgentLoop._generated_response_length(agent_data) == 3


def test_context_sensitive_stop_tokens_fall_back_to_decoded_text():
    class Tokenizer:
        texts = {
            (1,): "<think>reason</think> ",
            (1, 2): "<think>reason</think> <search>query",
            (1, 2, 3): "<think>reason</think> <search>query</search>",
            (1, 2, 3, 4): "<think>reason</think> <search>query</search>ignored",
        }

        def decode(self, token_ids, **_):
            return self.texts[tuple(token_ids)]

        def encode(self, _text, **_):
            return [99]

    loop = _loop_with_markers()
    loop.tokenizer = Tokenizer()
    loop.search_r1_stop_strings = ["</search>", "</answer>", "<information>"]

    token_ids, log_probs, action = loop._truncate_search_r1_generation(
        [1, 2, 3, 4],
        [0.1, 0.2, 0.3, 0.4],
    )

    assert token_ids == [1, 2, 3]
    assert log_probs == [0.1, 0.2, 0.3]
    assert action == "search"


def test_empty_generation_keeps_prompt_out_of_response():
    prompt_ids, response_ids = ToolAgentLoop._split_prompt_and_response([1, 2, 3], [])

    assert prompt_ids == [1, 2, 3]
    assert response_ids == []


def test_generation_split_uses_response_mask_length():
    prompt_ids, response_ids = ToolAgentLoop._split_prompt_and_response([1, 2, 3, 4], [1, 0])

    assert prompt_ids == [1, 2]
    assert response_ids == [3, 4]
