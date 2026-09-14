# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import copy
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import add_generation_prompt_for_gpt_oss, format_gpt_oss_tool_response_manually
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


SEARCH_R1_ACTION_ENDINGS = ("</search>", "</answer>")
SEARCH_R1_FORBIDDEN_OUTPUT = "<information>"


def _find_token_subsequence(tokens: list[int], needle: list[int]) -> int:
    """Return the first token offset of ``needle`` in ``tokens``."""
    if not needle or len(needle) > len(tokens):
        return -1
    width = len(needle)
    for start in range(len(tokens) - width + 1):
        if tokens[start : start + width] == needle:
            return start
    return -1


class AgentData:
    """Encapsulates all state variables for the agent loop."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: Any,
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0
        # Search-R1 counts search and answer as actions.  The count is
        # cumulative across incremental generations; tool observations never
        # increment it.
        self.action_count = 0
        self.last_action = None

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level ToolAgentLoop initialization")

        # Initialize tools from config file
        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.max_user_turns = config.actor_rollout_ref.rollout.multi_turn.max_user_turns
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns
        cls.max_generated_response_length = (
            config.actor_rollout_ref.rollout.multi_turn.max_generated_response_length
            or config.actor_rollout_ref.rollout.response_length
        )
        cls.max_response_length_per_turn = (
            config.actor_rollout_ref.rollout.multi_turn.max_response_length_per_turn
            or cls.max_generated_response_length
        )
        cls.max_parallel_calls = config.actor_rollout_ref.rollout.multi_turn.max_parallel_calls
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.multi_turn.max_tool_response_length
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side
        tool_config_path = config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        cls.tools = {tool.name: tool for tool in tool_list}
        cls.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        cls.tool_parser = ToolParser.get_tool_parser(config.actor_rollout_ref.rollout.multi_turn.format, cls.tokenizer)
        cls.tool_parser_name = config.actor_rollout_ref.rollout.multi_turn.format
        cls.search_r1_format = cls.tool_parser_name == "search_r1"
        cls.search_r1_max_obs_length = int(config.data.get("max_obs_length", 500))
        multi_turn_config = config.actor_rollout_ref.rollout.multi_turn
        configured_stop_tokens = multi_turn_config.get("stop_tokens", None)
        cls.search_r1_stop_strings = list(
            configured_stop_tokens or list(SEARCH_R1_ACTION_ENDINGS) + [SEARCH_R1_FORBIDDEN_OUTPUT]
        )
        cls.search_r1_include_stop_str_in_output = bool(
            multi_turn_config.get("include_stop_str_in_output", True)
        )
        cls.search_r1_stop_token_ids = {
            marker: cls.tokenizer.encode(marker, add_special_tokens=False)
            for marker in cls.search_r1_stop_strings
        }
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.system_prompt = tokenizer.apply_chat_template(
            [{}], add_generation_prompt=False, tokenize=True, **cls.apply_chat_template_kwargs
        )
        print(
            "Initialized Search-R1 ToolAgentLoop: "
            f"trajectory_length={cls.response_length}, "
            f"max_response_length_per_turn={cls.max_response_length_per_turn}, "
            f"max_generated_response_length={cls.max_generated_response_length}, "
            f"max_obs_length={cls.search_r1_max_obs_length}, "
            f"max_assistant_turns={cls.max_assistant_turns}, "
            f"tools={list(cls.tools)}"
        )
        # Initialize interactions from config file
        cls.interaction_config_file = config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        if cls.interaction_config_file:
            cls.interaction_map: dict[str, BaseInteraction] = cls._initialize_interactions(cls.interaction_config_file)

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy(kwargs.get("multi_modal_data", {}).get("image", None))
        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output.  Keep the prompt separate even when the model
        # returned an empty completion (``-0`` would otherwise duplicate it).
        prompt_ids, response_ids = self._split_prompt_and_response(
            agent_data.prompt_ids,
            agent_data.response_mask,
        )
        multi_modal_data = {"image": agent_data.image_data} if agent_data.image_data is not None else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields={
                "action_count": agent_data.action_count,
                "last_action": agent_data.last_action,
            },
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        template_kwargs = {} if self.search_r1_format else {"tools": self.tool_schemas}
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    agent_data.messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **template_kwargs,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            agent_data.prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            agent_data.prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    agent_data.messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    **template_kwargs,
                    **self.apply_chat_template_kwargs,
                ),
            )
        return AgentState.GENERATING

    def _truncate_search_r1_generation(
        self, token_ids: list[int], log_probs: Optional[list[float]]
    ) -> tuple[list[int], Optional[list[float]], Optional[str]]:
        """Keep the first complete action from this incremental generation.

        ``output.token_ids`` contains only the newly generated tokens.  The
        previous actions and observations remain in ``prompt_ids``, so the
        first action in this chunk is the next cumulative action (the k-th
        action on the k-th action-producing generation).
        """
        if not self.search_r1_format or not token_ids:
            return token_ids, log_probs, None

        first_match: tuple[int, str] | None = None
        for marker, marker_ids in self.search_r1_stop_token_ids.items():
            start = _find_token_subsequence(token_ids, marker_ids)
            if start < 0:
                continue
            if first_match is None or start < first_match[0]:
                first_match = (start, marker)

        if first_match is None:
            # BPE tokenization is context-sensitive. For example, Qwen2.5
            # encodes standalone ``</search>`` differently from the same text
            # after a space. Fall back to decoded text and map the first marker
            # back to an original token boundary.
            decoded = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            decoded_matches = [
                (decoded.find(marker), marker)
                for marker in self.search_r1_stop_strings
                if decoded.find(marker) >= 0
            ]
            if not decoded_matches:
                return token_ids, log_probs, None

            char_start, marker = min(decoded_matches, key=lambda item: item[0])
            if marker == SEARCH_R1_FORBIDDEN_OUTPUT:
                cutoff = 0
                for end in range(1, len(token_ids) + 1):
                    if len(self.tokenizer.decode(token_ids[:end], skip_special_tokens=False)) > char_start:
                        break
                    cutoff = end
                return (
                    token_ids[:cutoff],
                    log_probs[:cutoff] if log_probs is not None else None,
                    "invalid_information",
                )

            char_end = char_start + len(marker)
            decoded_prefix_ids = self.tokenizer.encode(decoded[:char_end], add_special_tokens=False)
            if token_ids[: len(decoded_prefix_ids)] == decoded_prefix_ids:
                cutoff = len(decoded_prefix_ids)
            else:
                cutoff = len(token_ids)
                for end in range(1, len(token_ids) + 1):
                    prefix = self.tokenizer.decode(token_ids[:end], skip_special_tokens=False)
                    if len(prefix) >= char_end and marker in prefix:
                        cutoff = end
                        break
            action = "search" if marker == "</search>" else "answer"
            return (
                token_ids[:cutoff],
                log_probs[:cutoff] if log_probs is not None else None,
                action,
            )

        start, marker = first_match
        if marker == SEARCH_R1_FORBIDDEN_OUTPUT:
            end = start
            action = "invalid_information"
        else:
            end = start + len(self.search_r1_stop_token_ids[marker])
            action = "search" if marker == "</search>" else "answer"

        return token_ids[:end], log_probs[:end] if log_probs is not None else None, action

    @staticmethod
    def _generated_response_length(agent_data: AgentData) -> int:
        """Count model-generated tokens; observations carry mask value zero."""
        return sum(agent_data.response_mask)

    @staticmethod
    def _split_prompt_and_response(
        prompt_ids: list[int], response_mask: list[int]
    ) -> tuple[list[int], list[int]]:
        """Split accumulated ids without treating ``-0`` as a slice.

        An empty model generation is valid, for example when the inference
        server returns an empty completion.  ``prompt_ids[-0:]`` would return
        the entire prompt and contaminate the response.
        """
        response_length = len(response_mask)
        if response_length > len(prompt_ids):
            raise ValueError(
                "Response mask is longer than the accumulated prompt: "
                f"{response_length} > {len(prompt_ids)}"
            )
        split_index = len(prompt_ids) - response_length
        return prompt_ids[:split_index], prompt_ids[split_index:]

    @staticmethod
    def _format_search_r1_observation(observation: str) -> str:
        """Format an official Search-R1 continuation without model-specific chat tokens."""
        return f"\n\n<information>{observation}</information>\n\n"

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []

        # Keep this Search-R1-specific.  The manager can share the base
        # sampling dictionary across concurrent requests.
        generation_params = dict(sampling_params)
        if self.search_r1_format:
            remaining_generated_tokens = (
                self.max_generated_response_length - self._generated_response_length(agent_data)
            )
            remaining_trajectory_tokens = self.response_length - len(agent_data.response_mask)
            if remaining_generated_tokens <= 0 or remaining_trajectory_tokens <= 0:
                return AgentState.TERMINATED
            configured_max_tokens = generation_params.get("max_tokens")
            if configured_max_tokens is None:
                configured_max_tokens = remaining_generated_tokens
            generation_params["max_tokens"] = min(
                configured_max_tokens,
                self.max_response_length_per_turn,
                remaining_generated_tokens,
                remaining_trajectory_tokens,
            )
            generation_params["stop"] = list(self.search_r1_stop_strings)
            generation_params["include_stop_str_in_output"] = self.search_r1_include_stop_str_in_output

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=generation_params,
                image_data=agent_data.image_data,
            )

        response_ids, response_logprobs, action = self._truncate_search_r1_generation(
            output.token_ids,
            output.log_probs,
        )
        agent_data.assistant_turns += 1
        agent_data.response_ids = response_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if response_logprobs:
            agent_data.response_logprobs += response_logprobs
        agent_data.last_action = action
        if action in {"search", "answer"}:
            agent_data.action_count += 1

        # Check termination conditions
        if not ignore_termination and (
            len(agent_data.response_mask) >= self.response_length
            or self._generated_response_length(agent_data) >= self.max_generated_response_length
        ):
            return AgentState.TERMINATED
        if action in {"answer", "invalid_information"}:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Extract tool calls
        if self.search_r1_format and action == "search":
            _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
        elif not self.search_r1_format:
            _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids)
        else:
            agent_data.tool_calls = []

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, tool_metrics in responses:
            # Create message from tool response
            if self.search_r1_format:
                if tool_metrics.get("invalid_search_query") or tool_metrics.get("repeated_search_query"):
                    agent_data.metrics["search_r1/invalid_search_query"] = (
                        agent_data.metrics.get("search_r1/invalid_search_query", 0) + 1
                    )
                    agent_data.tool_calls = []
                    return AgentState.TERMINATED
                # Search-R1 consumes plain passages. The generic tool text is a
                # JSON envelope and may already be truncated by character count.
                observation_text = (getattr(tool_response, "metadata", None) or {}).get("formatted_result")
                if not observation_text:
                    observation_text = tool_response.text or ""
                    try:
                        decoded_response = json.loads(observation_text)
                        observation_text = decoded_response.get("result", observation_text)
                    except (json.JSONDecodeError, AttributeError):
                        pass
                observation_ids = self.tokenizer.encode(observation_text, add_special_tokens=False)
                observation = self.tokenizer.decode(observation_ids[: self.search_r1_max_obs_length])
                message = {
                    "role": "user",
                    "content": self._format_search_r1_observation(observation),
                }
            elif tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                if agent_data.image_data is None:
                    agent_data.image_data = []
                elif not isinstance(agent_data.image_data, list):
                    agent_data.image_data = [agent_data.image_data]

                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            agent_data.image_data.append(img)
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        agent_data.image_data.append(tool_response.image)
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)
        # Update prompt with tool responses
        if self.search_r1_format:
            information = "".join(message["content"] for message in add_messages)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(information, add_special_tokens=False)
            )
        elif self.processor is not None:
            raw_tool_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # Use only the new images from this turn for processing tool responses
            current_images = new_images_this_turn if new_images_this_turn else None  # Using local variable
            model_inputs = self.processor(text=[raw_tool_response], images=current_images, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if self.tool_parser_name == "gpt-oss":
                logger.info("manually format tool responses for gpt-oss")
                # Format tool responses manually
                tool_response_texts = []
                for i, tool_msg in enumerate(add_messages):
                    actual_tool_name = tool_call_names[i]
                    formatted = format_gpt_oss_tool_response_manually(tool_msg["content"], actual_tool_name)
                    tool_response_texts.append(formatted)

                tool_response_text = add_generation_prompt_for_gpt_oss("".join(tool_response_texts))
                response_ids = await self.loop.run_in_executor(
                    None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
                )
            else:
                response_ids = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
                )
                response_ids = response_ids[len(self.system_prompt) :]
        remaining_response_tokens = self.response_length - len(agent_data.response_mask)
        if remaining_response_tokens <= 0:
            return AgentState.TERMINATED
        if len(response_ids) > remaining_response_tokens:
            response_ids = response_ids[:remaining_response_tokens]
            agent_data.metrics["search_r1/observation_truncated"] = (
                agent_data.metrics.get("search_r1/observation_truncated", 0) + 1
            )
        agent_data.metrics["search_r1/observation_tokens"] = (
            agent_data.metrics.get("search_r1/observation_tokens", 0) + len(response_ids)
        )
        if not response_ids:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        if self.processor is not None:
            raw_user_response = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    add_messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_user_response], images=None, return_tensors="pt")
            response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            response_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(add_messages, add_generation_prompt=True, tokenize=True),
            )
        response_ids = response_ids[len(self.system_prompt) :]

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any]
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(instance_id, tool_args)
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if (
            tool_response_text
            and self.max_tool_response_length > 0
            and len(tool_response_text) > self.max_tool_response_length
        ):
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    @classmethod
    def _initialize_interactions(cls, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        logger.info(f"Initialize interactions from configuration: interaction_map: {list(interaction_map.keys())}")
        return interaction_map
