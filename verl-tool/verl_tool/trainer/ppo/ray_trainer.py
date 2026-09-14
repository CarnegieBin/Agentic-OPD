import ray
import torch
import os
import json
import random
import shutil
import numpy as np
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
from typing import Optional
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    pad_dataproto_to_divisor,
    unpad_dataproto,
    process_validation_metrics,
) # for train and validate
from verl.trainer.ppo.ray_trainer import (
    DataProto,
) # for init
from verl.utils.debug import marked_timer
from verl_tool.workers.reward_manager.search_r1_qa_em import extract_solution


##############################################################################
#### Replace the original classes/functions with verl-tool customized ones ####
import verl.experimental.agent_loop
from verl_tool.agent_loop import AgentLoopManager
import verl.trainer.ppo.ray_trainer
from .reward import compute_reward, compute_reward_async
from .checkpoint_retention import (
    select_max_union_steps,
    solved_union_size,
    stable_validation_uid,
)
from verl_tool.workers.rollout.vllm_rollout.vllm_async_server import VerlToolvLLMHttpServer
import verl.workers.rollout.vllm_rollout.vllm_async_server
from .metric_util import (
    compute_data_metrics,
    flatten_evaluation_metrics,
    process_validation_metrics,
)
verl.experimental.agent_loop.AgentLoopManager = AgentLoopManager
verl.trainer.ppo.ray_trainer.compute_reward = compute_reward
verl.trainer.ppo.ray_trainer.compute_reward_async = compute_reward_async
verl.trainer.ppo.ray_trainer.compute_data_metrics = compute_data_metrics
verl.trainer.ppo.ray_trainer.process_validation_metrics = process_validation_metrics
verl.workers.rollout.vllm_rollout.vllm_async_server.vLLMHttpServer = VerlToolvLLMHttpServer
##############################################################################

class AgentRayPPOTrainer(RayPPOTrainer):
    def __init__(self, *args, test_datasets=None, collate_fn=None, **kwargs):
        super().__init__(*args, collate_fn=collate_fn, **kwargs)

        self.test_datasets = dict(test_datasets or {})
        self.test_dataloaders = {}
        if not self.test_datasets:
            return

        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn
        test_batch_size = self.config.data.get("test_batch_size", self.config.data.val_batch_size)
        for data_source, dataset in self.test_datasets.items():
            batch_size = test_batch_size if test_batch_size is not None else len(dataset)
            dataloader = StatefulDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=self.config.data["dataloader_num_workers"],
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )
            if len(dataloader) < 1:
                raise ValueError(f"Test dataloader for {data_source} is empty")
            self.test_dataloaders[data_source] = dataloader
            print(
                f"Size of test dataloader {data_source}: {len(dataloader)} "
                f"(dataset={len(dataset)}, batch_size={batch_size})"
            )

    def _update_validation_solve_history(self, data_sources, uids, scores):
        """Update stable question-ID coverage per data_source and retain the current solved set."""
        if not hasattr(self, "_validation_solved_history_by_source"):
            self._validation_solved_history_by_source = defaultdict(set)
            self._validation_previous_solved_by_source = defaultdict(set)

        current_by_source: dict[str, set[str]] = defaultdict(set)
        for data_source, uid, score in zip(data_sources, uids, scores, strict=True):
            if float(score) > 0:
                current_by_source[str(data_source)].add(str(uid))

        all_sources = sorted(set(current_by_source) | set(self._validation_solved_history_by_source))
        metrics: dict[str, float] = {}
        current_all: set[str] = set()
        for source in all_sources:
            current = current_by_source.get(source, set())
            previous_solved = self._validation_previous_solved_by_source[source]
            historical = self._validation_solved_history_by_source[source] | current
            newly = current - self._validation_solved_history_by_source[source]
            forgotten = previous_solved - current
            retention = (
                len(current & previous_solved) / len(previous_solved)
                if previous_solved else 1.0
            )
            self._validation_solved_history_by_source[source] = historical
            self._validation_previous_solved_by_source[source] = current
            current_all |= current

            metrics[f"validation/{source}/current_solved"] = len(current)
            metrics[f"validation/{source}/historical_solved"] = len(historical)
            metrics[f"validation/{source}/newly_solved"] = len(newly)
            metrics[f"validation/{source}/forgotten"] = len(forgotten)
            metrics[f"validation/{source}/retention"] = retention

        self._last_validation_step = int(self.global_steps)
        self._last_validation_solved_uids = current_all
        return metrics

    @staticmethod
    def _validation_batch_uids(test_batch):
        data_sources = test_batch.non_tensor_batch.get("data_source")
        extra_infos = test_batch.non_tensor_batch.get("extra_info")
        prompt_token_ids = test_batch.batch["input_ids"]
        batch_size = len(prompt_token_ids)

        if data_sources is None:
            data_sources = ["unknown"] * batch_size
        if extra_infos is None:
            extra_infos = [None] * batch_size

        return np.array(
            [
                stable_validation_uid(data_sources[index], extra_infos[index], prompt_token_ids[index])
                for index in range(batch_size)
            ],
            dtype=object,
        )

    def _checkpoint_root(self) -> Path:
        root = Path(self.config.trainer.default_local_dir).expanduser().resolve()
        if root == Path("/") or root == Path.home().resolve():
            raise ValueError(f"Refusing to use broad checkpoint root: {root}")
        return root

    def _write_checkpoint_selection_manifest(self) -> None:
        root = self._checkpoint_root()
        root.mkdir(parents=True, exist_ok=True)
        retained = getattr(self, "_retained_checkpoint_solved_uids", {})
        history = getattr(self, "_checkpoint_selection_history", [])
        selected_steps = sorted(retained)
        payload = {
            "version": 1,
            "strategy": "online_exact_max_solved_uid_union",
            "keep_limit": self.config.trainer.get("validation_checkpoint_keep_limit", 5),
            "retained_union_solved_count": solved_union_size(retained),
            "retained": [
                {
                    "global_step": step,
                    "path": f"global_step_{step}",
                    "solved_count": len(retained[step]),
                    "solved_uids": sorted(retained[step]),
                }
                for step in selected_steps
            ],
            "history": history,
        }
        manifest_path = root / "checkpoint_selection.json"
        temporary_path = root / ".checkpoint_selection.json.tmp"
        temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_path, manifest_path)

    def _append_checkpoint_validation_record(self, step: int) -> None:
        """Append per-dataset validation coverage for a retained checkpoint."""
        configured_path = self.config.trainer.get("validation_checkpoint_metrics_path", None)
        if not configured_path:
            return
        metrics_path = Path(str(configured_path)).expanduser()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "global_step": int(step),
            "checkpoint": f"global_step_{int(step)}",
            "datasets": getattr(self, "_last_validation_by_dataset", {}),
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _trajectory_json_value(value):
        if isinstance(value, torch.Tensor):
            return AgentRayPPOTrainer._trajectory_json_value(value.detach().cpu().tolist())
        if isinstance(value, np.ndarray):
            return [AgentRayPPOTrainer._trajectory_json_value(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): AgentRayPPOTrainer._trajectory_json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [AgentRayPPOTrainer._trajectory_json_value(item) for item in value]
        return value

    def _write_validation_trajectories(
        self,
        *,
        data_sources,
        sample_uids,
        raw_prompts,
        rendered_prompts,
        responses,
        ground_truths,
        scores,
        num_turns,
        reward_extra_infos,
    ) -> None:
        configured_dir = self.config.trainer.get("validation_trajectory_dir", None)
        if not configured_dir:
            return

        step_dir = Path(str(configured_dir)).expanduser() / f"global_step_{int(self.global_steps)}"
        step_dir.mkdir(parents=True, exist_ok=True)
        source_indices: dict[str, list[int]] = defaultdict(list)
        for index, data_source in enumerate(data_sources):
            source_indices[str(data_source)].append(index)

        for data_source, indices in sorted(source_indices.items()):
            safe_source = "".join(char if char.isalnum() or char in "._-" else "_" for char in data_source)
            output_path = step_dir / f"{safe_source}.jsonl"
            temporary_path = output_path.with_suffix(".jsonl.tmp")
            with temporary_path.open("w", encoding="utf-8") as handle:
                for index in indices:
                    extra_info = {
                        key: self._trajectory_json_value(values[index]) if index < len(values) else None
                        for key, values in reward_extra_infos.items()
                    }
                    record = {
                        "global_step": int(self.global_steps),
                        "data_source": data_source,
                        "sample_uid": str(sample_uids[index]),
                        "prompt_messages": self._trajectory_json_value(raw_prompts[index]),
                        "rendered_prompt": rendered_prompts[index],
                        "response": responses[index],
                        "trajectory": rendered_prompts[index] + responses[index],
                        "ground_truth": self._trajectory_json_value(ground_truths[index]),
                        "score": float(scores[index]),
                        "num_turns": self._trajectory_json_value(num_turns[index]),
                        "search_count": responses[index].count("<search>"),
                        "reward_extra_info": extra_info,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            os.replace(temporary_path, output_path)

    def _print_random_validation_trajectory(
        self,
        *,
        data_sources,
        sample_uids,
        raw_prompts,
        rendered_prompts,
        responses,
        ground_truths,
        scores,
        num_turns,
        reward_extra_infos,
    ) -> None:
        """Print one complete, randomly selected trajectory after each validation."""
        if not responses:
            print("[validation trajectory] no validation samples were generated", flush=True)
            return

        index = random.randrange(len(responses))
        response = str(responses[index])
        extra_info = {
            key: self._trajectory_json_value(values[index])
            for key, values in reward_extra_infos.items()
            if len(values) == len(responses)
        }
        trajectory = {
            "global_step": int(self.global_steps),
            "sample_index": index,
            "data_source": self._trajectory_json_value(data_sources[index]),
            "uid": self._trajectory_json_value(sample_uids[index]),
            "raw_prompt": self._trajectory_json_value(raw_prompts[index]),
            "rendered_prompt": str(rendered_prompts[index]),
            "response": response,
            "extracted_answer": extract_solution(response),
            "ground_truth": self._trajectory_json_value(ground_truths[index]),
            "score": self._trajectory_json_value(scores[index]),
            "num_turns": self._trajectory_json_value(num_turns[index]),
            "reward_extra_info": extra_info,
        }
        print(
            "\n========== RANDOM COMPLETE VALIDATION TRAJECTORY ==========\n"
            + json.dumps(trajectory, ensure_ascii=False, indent=2)
            + "\n========== END RANDOM VALIDATION TRAJECTORY ==============\n",
            flush=True,
        )

    def _save_checkpoint(self):
        """Export and retain validation-selected HuggingFace actor weights only."""
        save_freq = int(self.config.trainer.save_freq)
        if save_freq <= 0 or self.global_steps % save_freq != 0:
            print(
                f"Skipping non-periodic checkpoint request at step {self.global_steps}; "
                f"HF checkpoints are evaluated only every {save_freq} steps."
            )
            return

        if getattr(self, "_last_validation_step", None) != self.global_steps:
            raise RuntimeError(
                f"Checkpoint step {self.global_steps} has no matching validation result; "
                "validation must run immediately before checkpoint selection."
            )
        if self.config.trainer.default_hdfs_dir is not None:
            raise ValueError("Validation-selected HF-only checkpoints support local storage only")

        keep_limit_config = self.config.trainer.get("validation_checkpoint_keep_limit", 5)
        keep_limit = None if keep_limit_config is None else int(keep_limit_config)
        if keep_limit is not None and keep_limit < 1:
            raise ValueError("trainer.validation_checkpoint_keep_limit must be positive or null")

        if not hasattr(self, "_retained_checkpoint_solved_uids"):
            self._retained_checkpoint_solved_uids = {}
            self._checkpoint_selection_history = []

        step = int(self.global_steps)
        current_solved = set(self._last_validation_solved_uids)
        candidates = dict(self._retained_checkpoint_solved_uids)
        candidates[step] = current_solved
        selected_steps = (
            sorted(candidates)
            if keep_limit is None
            else select_max_union_steps(candidates, keep_limit)
        )
        selected_candidates = {selected_step: candidates[selected_step] for selected_step in selected_steps}
        candidate_selected = step in selected_candidates
        evicted_steps = sorted(set(self._retained_checkpoint_solved_uids) - set(selected_steps))
        union_count = solved_union_size(selected_candidates)

        decision = {
            "global_step": step,
            "candidate_solved_count": len(current_solved),
            "candidate_selected": candidate_selected,
            "evicted_steps": evicted_steps,
            "selected_steps": selected_steps,
            "selected_union_solved_count": union_count,
        }

        if not candidate_selected:
            self._checkpoint_selection_history.append(decision)
            self._write_checkpoint_selection_manifest()
            print(
                f"Checkpoint candidate step {step} rejected by solved-UID union retention: "
                f"selected_steps={selected_steps}, union={union_count}"
            )
            return

        root = self._checkpoint_root()
        root.mkdir(parents=True, exist_ok=True)
        staging_root = root / f".global_step_{step}.staging"
        actor_staging_path = staging_root / "actor"
        final_path = root / f"global_step_{step}"
        if staging_root.exists():
            shutil.rmtree(staging_root)

        try:
            self.actor_rollout_wg.save_checkpoint(
                str(actor_staging_path),
                None,
                step,
                max_ckpt_to_keep=None,
            )
            hf_staging_path = actor_staging_path / "huggingface"
            weight_files = list(hf_staging_path.glob("*.safetensors")) + list(hf_staging_path.glob("*.bin"))
            if not hf_staging_path.is_dir() or not weight_files:
                raise RuntimeError(f"HuggingFace checkpoint export is incomplete: {hf_staging_path}")

            if final_path.exists():
                shutil.rmtree(final_path)
            os.replace(hf_staging_path, final_path)

            for evicted_step in evicted_steps:
                evicted_path = root / f"global_step_{evicted_step}"
                if evicted_path.exists():
                    shutil.rmtree(evicted_path)
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root)

        self._retained_checkpoint_solved_uids = selected_candidates
        self._checkpoint_selection_history.append(decision)
        self._write_checkpoint_selection_manifest()
        self._append_checkpoint_validation_record(step)
        print(
            f"Saved HF-only checkpoint step {step} to {final_path}; "
            f"selected_steps={selected_steps}, solved_uid_union={union_count}"
        )

    def _evaluate_loader(
        self,
        dataloader,
        split: str,
        reward_fn,
        expected_data_source: Optional[str] = None,
        update_validation_history: bool = False,
        log_generations: bool = False,
        dump_path: Optional[str] = None,
    ):
        if split not in {"val", "test"}:
            raise ValueError(f"Unsupported evaluation split: {split}")

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        raw_prompts = []

        for test_data in dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            original_uids = test_batch.non_tensor_batch.get("uid")
            if original_uids is None:
                original_uids = self._validation_batch_uids(test_batch)
            test_batch.non_tensor_batch["uid"] = np.asarray(original_uids, dtype=object)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )
            batch_raw_prompts = test_batch.non_tensor_batch.get("raw_prompt", None)
            if batch_raw_prompts is None:
                raw_prompts.extend([None] * len(test_batch.batch["input_ids"]))
            elif isinstance(batch_raw_prompts, np.ndarray):
                raw_prompts.extend(batch_raw_prompts.tolist())
            else:
                raw_prompts.extend(batch_raw_prompts)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_attention_mask = test_batch.batch["attention_mask"]
            input_texts = [
                self.tokenizer.decode(ids[input_attention_mask[index] == 1], skip_special_tokens=False)
                for index, ids in enumerate(input_ids)
            ]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_attention_mask = test_output_gen_batch.batch["attention_mask"][:, test_output_gen_batch.batch["prompts"].shape[1]:]
            output_texts = [self.tokenizer.decode(ids[output_attention_mask[i]==1], skip_special_tokens=False) for i, ids in enumerate(output_ids)]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if reward_fn is None:
                raise ValueError(f"reward_fn must be provided for {split} evaluation.")
            result = reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    
            tool_interact_info = test_batch.non_tensor_batch.get('tool_interact_info', None)
            if isinstance(tool_interact_info, np.ndarray):
                tool_interact_info = tool_interact_info.tolist()
            if tool_interact_info:
                for tool_interact in tool_interact_info:
                    if "image" in tool_interact:
                        if isinstance(tool_interact['image'], list):
                            tool_interact['image'] = [x[:50] for x in tool_interact['image']]  # crop the image to first 50 characters
                        elif isinstance(tool_interact['image'], str):
                            tool_interact['image'] = tool_interact['image'][:50] # for debug
                if "tool_interact_info" not in reward_extra_infos_dict:
                    reward_extra_infos_dict["tool_interact_info"] = []
                if "traj_stop_reason" not in reward_extra_infos_dict:
                    reward_extra_infos_dict["traj_stop_reason"] = []
                reward_extra_infos_dict["tool_interact_info"].extend(tool_interact_info)
                reward_extra_infos_dict["traj_stop_reason"].extend(
                    test_batch.non_tensor_batch.get("traj_stop_reason", [None] * reward_tensor.shape[0])
                )
                reward_extra_infos_dict["verl_tool_metrics"].extend(
                    test_batch.non_tensor_batch.get("verl_tool_metrics", [None] * reward_tensor.shape[0])
                )

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        if log_generations:
            self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        if dump_path:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=dump_path,
            )
        trajectory_extra_infos = deepcopy(reward_extra_infos_dict)
        if "tool_interact_info" in reward_extra_infos_dict:
            # remove if after dump
            reward_extra_infos_dict.pop("tool_interact_info")
        if "traj_stop_reason" in reward_extra_infos_dict:
            reward_extra_infos_dict.pop("traj_stop_reason")
        if "verl_tool_metrics" in reward_extra_infos_dict:
            reward_extra_infos_dict.pop("verl_tool_metrics")

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)
        flattened_turns = np.concatenate(sample_turns) if sample_turns else [None] * len(sample_scores)

        per_dataset: dict[str, dict[str, object]] = {}
        for data_source, uid, score in zip(data_sources, sample_uids, sample_scores, strict=True):
            source = str(data_source)
            record = per_dataset.setdefault(source, {"total": set(), "solved": set()})
            record["total"].add(str(uid))
            if float(score) > 0:
                record["solved"].add(str(uid))
        self._last_validation_by_dataset = {
            source: {
                "total": len(record["total"]),
                "solved": len(record["solved"]),
                "accuracy": len(record["solved"]) / len(record["total"]) if record["total"] else 0.0,
                "solved_uids": sorted(record["solved"]),
            }
            for source, record in sorted(per_dataset.items())
        }
        self._write_validation_trajectories(
            data_sources=data_sources,
            sample_uids=sample_uids,
            raw_prompts=raw_prompts,
            rendered_prompts=sample_inputs,
            responses=sample_outputs,
            ground_truths=sample_gts,
            scores=sample_scores,
            num_turns=flattened_turns,
            reward_extra_infos=trajectory_extra_infos,
        )
        if split == "val":
            self._print_random_validation_trajectories_by_dataset(
                data_sources=data_sources,
                sample_uids=sample_uids,
                raw_prompts=raw_prompts,
                rendered_prompts=sample_inputs,
                responses=sample_outputs,
                ground_truths=sample_gts,
                scores=sample_scores,
                num_turns=flattened_turns,
                reward_extra_infos=trajectory_extra_infos,
            )

        if expected_data_source is not None:
            observed_data_sources = {str(data_source) for data_source in data_sources}
            if observed_data_sources != {expected_data_source}:
                raise ValueError(
                    f"Test dataset {expected_data_source} contains unexpected data sources: "
                    f"{sorted(observed_data_sources)}"
                )

        core_metric_infos = {
            "em": reward_extra_infos_dict["em"]
        } if "em" in reward_extra_infos_dict and len(reward_extra_infos_dict["em"]) == len(sample_scores) else {}
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, core_metric_infos)
        metric_dict = flatten_evaluation_metrics(data_src2var2metric2val, split=split)
        metric_dict = {
            key: value
            for key, value in metric_dict.items()
            if "/em/" in key or not ("/f1/" in key or "/subem/" in key)
        }

        if update_validation_history:
            metric_dict.update(
                self._update_validation_solve_history(data_sources, sample_uids, sample_scores)
            )

        if len(sample_turns) > 0:
            sample_turns = flattened_turns
            turns_prefix = f"{split}-aux"
            if split == "test" and expected_data_source is not None:
                turns_prefix = f"{turns_prefix}/{expected_data_source}"
            metric_dict[f"{turns_prefix}/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _print_random_validation_trajectories_by_dataset(
        self,
        *,
        data_sources,
        sample_uids,
        raw_prompts,
        rendered_prompts,
        responses,
        ground_truths,
        scores,
        num_turns,
        reward_extra_infos,
    ) -> None:
        """Print one complete Search-R1 trajectory for every validation dataset."""
        source_indices: dict[str, list[int]] = defaultdict(list)
        for index, data_source in enumerate(data_sources):
            source_indices[str(data_source)].append(index)

        for data_source, indices in sorted(source_indices.items()):
            index = random.choice(indices)
            response = str(responses[index])
            print(
                "\n"
                "[Search-R1 validation trajectory]\n"
                f"data_source: {data_source}\n"
                f"global_step: {int(self.global_steps)}\n"
                f"sample_index: {index}\n"
                f"uid: {self._trajectory_json_value(sample_uids[index])}\n"
                f"score: {float(scores[index]):.6f}\n"
                f"num_turns: {self._trajectory_json_value(num_turns[index])}\n"
                f"ground_truth: {self._trajectory_json_value(ground_truths[index])}\n"
                "prompt:\n"
                f"{rendered_prompts[index]}\n"
                "response:\n"
                f"{response}\n"
                "[/Search-R1 validation trajectory]\n",
                flush=True,
            )

    def _should_run_test_evaluation(self) -> bool:
        if not self.test_dataloaders:
            return False
        step = int(self.global_steps)
        if step == 0:
            return bool(self.config.trainer.get("test_before_train", False))
        if step >= int(self.total_training_steps):
            return bool(self.config.trainer.get("test_after_train", True))
        test_frequency = int(self.config.trainer.get("test_eval_freq", -1))
        return test_frequency > 0 and step % test_frequency == 0

    def _evaluate_tests(self) -> dict[str, float]:
        """Evaluate NQ/HotpotQA without touching validation checkpoint state."""

        validation_state_before = (
            getattr(self, "_last_validation_step", None),
            frozenset(getattr(self, "_last_validation_solved_uids", set())),
        )
        metrics = {}
        test_data_dir = self.config.trainer.get("test_data_dir", None)
        for data_source, dataloader in self.test_dataloaders.items():
            dump_path = None
            if test_data_dir:
                dump_path = str(Path(test_data_dir) / str(data_source))
            source_metrics = self._evaluate_loader(
                dataloader=dataloader,
                split="test",
                reward_fn=self.val_reward_fn,
                expected_data_source=str(data_source),
                update_validation_history=False,
                log_generations=False,
                dump_path=dump_path,
            )
            overlap = set(metrics).intersection(source_metrics)
            if overlap:
                raise RuntimeError(f"Test metric namespaces overlap: {sorted(overlap)}")
            metrics.update(source_metrics)

            core_metrics = {
                key: value
                for key, value in source_metrics.items()
                if key.startswith(f"test-core/{data_source}/")
                and ("/reward/mean@" in key or "/acc/mean@" in key)
            }
            print(f"Search-R1 test results step {self.global_steps} ({data_source}): {core_metrics}")

        validation_state_after = (
            getattr(self, "_last_validation_step", None),
            frozenset(getattr(self, "_last_validation_solved_uids", set())),
        )
        if validation_state_after != validation_state_before:
            raise RuntimeError("Test evaluation mutated validation checkpoint-selection state")
        return metrics

    def _validate(self):
        """Run configured validation, and optionally an isolated benchmark test pass."""

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        metrics = self._evaluate_loader(
            dataloader=self.val_dataloader,
            split="val",
            reward_fn=self.val_reward_fn,
            update_validation_history=True,
            log_generations=True,
            dump_path=val_data_dir,
        )
        if self._should_run_test_evaluation():
            metrics.update(self._evaluate_tests())
        return metrics

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs_attention_masks = batch.batch['attention_mask'][:, :batch.batch['prompts'].shape[1]]
            outputs_attention_masks = batch.batch['attention_mask'][:, batch.batch['prompts'].shape[1]:]
            inputs = [self.tokenizer.decode(batch.batch["prompts"][i][inputs_attention_masks[i]==1], skip_special_tokens=False) for i in range(batch.batch["prompts"].shape[0])]
            outputs = [self.tokenizer.decode(batch.batch["responses"][i][outputs_attention_masks[i]==1], skip_special_tokens=False) for i in range(batch.batch["responses"].shape[0])]
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )
            
            tool_interact_info = batch.non_tensor_batch.get('tool_interact_info', None)
            if isinstance(tool_interact_info, np.ndarray):
                tool_interact_info = tool_interact_info.tolist()
            if tool_interact_info:
                for tool_interact in tool_interact_info:
                    if "image" in tool_interact:
                        if isinstance(tool_interact['image'], list):
                            tool_interact['image'] = [x[:50] for x in tool_interact['image']]  # crop the image to first 50 characters
                        elif isinstance(tool_interact['image'], str):
                            tool_interact['image'] = tool_interact['image'][:50] # for debug
                reward_extra_infos_to_dump.update({
                    "tool_interact_info": tool_interact_info,
                    "traj_stop_reason": batch.non_tensor_batch.get("traj_stop_reason", None),
                    "verl_tool_metrics": batch.non_tensor_batch.get("verl_tool_metrics", None),
                })

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )
