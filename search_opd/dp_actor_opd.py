"""Multi-teacher OPD update built on verl's data-parallel PPO actor."""

from __future__ import annotations

import math
import re
from typing import Any

import torch

from search_opd.core_opd import (
    masked_topk_teacher_means,
    topk_reverse_kl_per_teacher,
)
from search_opd.generated_token_mask_opd import select_opd_response_mask
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn
from verl.utils.device import get_device_id
from verl.utils.py_functional import append_to_dict
from verl.workers.actor.dp_actor import DataParallelPPOActor


def _metric_name(index: int, name: str) -> str:
    """Return a deterministic logger-safe teacher metric component."""

    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")
    return f"{index:02d}_{normalized or 'teacher'}"


class DataParallelPPOActorOPD(DataParallelPPOActor):
    """Average independently computed frozen-teacher PG-OPD losses."""

    def __init__(
        self,
        *args: Any,
        opd_config: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.opd_config = opd_config
        self.teacher_names = tuple(str(name) for name in opd_config.teacher_names)
        if not self.teacher_names:
            raise ValueError("OPD actor requires at least one teacher")
        self.teacher_metric_names = tuple(
            _metric_name(index, name)
            for index, name in enumerate(self.teacher_names)
        )
        if self.config.use_dynamic_bsz:
            raise ValueError("PG-OPD actor requires use_dynamic_bsz=false")
        if int(self.config.ppo_epochs) != 1:
            raise ValueError("PG-OPD actor requires exactly one PPO epoch")
        if self.config.use_kl_loss:
            raise ValueError("Legacy actor KL loss must be disabled inside the OPD actor")

    def update_policy(self, data: DataProto):
        """Apply direct Student Top-k reverse-KL distillation.

        The student top-k IDs were selected before the teacher call and are
        fixed for this on-policy update.  The student is scored again on those
        IDs with gradients enabled; the frozen teacher scores the same support.
        Both distributions are renormalized on the support and the per-teacher
        KL losses are averaged only after each scalar loss is formed.
        """

        required = {
            "advantages",
            "opd_topk_token_ids",
            "teacher_topk_log_probs",
        }
        mask_key = (
            "opd_response_mask"
            if "opd_response_mask" in data.batch.keys()
            else "response_mask"
        )
        required.add(mask_key)
        missing = required - set(data.batch.keys())
        if missing:
            raise KeyError(f"OPD actor batch is missing keys: {sorted(missing)}")

        teacher_topk_log_probs = data.batch["teacher_topk_log_probs"]
        if teacher_topk_log_probs.ndim == 3:
            teacher_topk_log_probs = teacher_topk_log_probs.unsqueeze(-1)
        if teacher_topk_log_probs.ndim != 4:
            raise ValueError(
                "teacher_topk_log_probs must have shape "
                "[batch, response_length, top_k, num_teachers], "
                f"got {tuple(teacher_topk_log_probs.shape)}"
            )
        topk_ids = data.batch["opd_topk_token_ids"]
        if topk_ids.ndim != 3:
            raise ValueError(
                "opd_topk_token_ids must have shape "
                "[batch, response_length, top_k], "
                f"got {tuple(topk_ids.shape)}"
            )
        configured_top_k = int(self.opd_config.get("top_k", topk_ids.shape[-1]))
        if topk_ids.shape[-1] != configured_top_k:
            raise ValueError(
                "Student top-k tensor/config mismatch: "
                f"tensor={topk_ids.shape[-1]}, config={configured_top_k}"
            )
        if teacher_topk_log_probs.shape[-2] != configured_top_k:
            raise ValueError(
                "Teacher top-k tensor/config mismatch: "
                f"tensor={teacher_topk_log_probs.shape[-2]}, config={configured_top_k}"
            )
        teacher_count = teacher_topk_log_probs.shape[-1]
        if teacher_count not in {1, len(self.teacher_names)}:
            raise ValueError(
                "Teacher tensor/config mismatch: "
                f"tensor={teacher_count}, names={len(self.teacher_names)}; "
                "adaptive mode expects one active teacher or the complete teacher set"
            )
        active_teacher_index = data.meta_info.get("opd_active_teacher_index", None)
        if teacher_count == 1:
            active_teacher_index = 0 if active_teacher_index is None else int(active_teacher_index)
            if not 0 <= active_teacher_index < len(self.teacher_names):
                raise ValueError(
                    f"Adaptive teacher index out of range: {active_teacher_index}; "
                    f"teacher_count={len(self.teacher_names)}"
                )
            active_teacher_names = (self.teacher_names[active_teacher_index],)
        else:
            active_teacher_names = self.teacher_names
            active_teacher_index = None

        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        opd_loss_coef = float(self.opd_config.get("loss_coef", 1.0))
        task_loss_coef = float(self.opd_config.get("task_loss_coef", 0.0))
        max_abs_log_ratio = self.opd_config.get("max_abs_log_ratio", None)
        max_abs_log_ratio = (
            None if max_abs_log_ratio is None else float(max_abs_log_ratio)
        )
        if not math.isfinite(opd_loss_coef) or opd_loss_coef <= 0:
            raise ValueError("OPD loss coefficient must be positive and finite")
        if not math.isfinite(task_loss_coef) or task_loss_coef < 0:
            raise ValueError("OPD task loss coefficient must be non-negative and finite")
        if max_abs_log_ratio is not None and (
            not math.isfinite(max_abs_log_ratio) or max_abs_log_ratio <= 0
        ):
            raise ValueError("OPD max_abs_log_ratio must be positive and finite")

        select_keys = [
            "responses",
            mask_key,
            "input_ids",
            "attention_mask",
            "position_ids",
            "advantages",
            "opd_topk_token_ids",
            "teacher_topk_log_probs",
        ]
        if "old_log_probs" in data.batch.keys():
            select_keys.append("old_log_probs")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(
            batch_keys=select_keys,
            non_tensor_batch_keys=non_tensor_select_keys,
        )

        mini_batches = data.split(self.config.ppo_mini_batch_size)
        if len(mini_batches) != 1:
            raise ValueError(
                "PG-OPD requires one on-policy mini-batch per actor update; "
                f"received {len(mini_batches)}"
            )
        mini_batch = mini_batches[0]
        micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
        if not micro_batches:
            raise ValueError("PG-OPD actor received an empty mini-batch")
        self.gradient_accumulation = len(micro_batches)

        metrics: dict[str, list[float]] = {}
        self.actor_optimizer.zero_grad()
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            response_mask = select_opd_response_mask(
                model_inputs,
                expected_shape=tuple(model_inputs["responses"].shape),
                response_attention_mask=model_inputs.get("attention_mask"),
            )
            # Keep the normalized mask under the conventional key for any
            # inherited forward helpers; observations remain in input_ids.
            model_inputs["response_mask"] = response_mask
            task_advantages = model_inputs["advantages"].detach()
            student_topk_ids = model_inputs["opd_topk_token_ids"]
            teacher_topk_log_probs = model_inputs["teacher_topk_log_probs"]
            if teacher_topk_log_probs.ndim == 3:
                teacher_topk_log_probs = teacher_topk_log_probs.unsqueeze(-1)
            rollout_is_weights = model_inputs.get("rollout_is_weights", None)
            loss_scale_factor = 1.0 / self.gradient_accumulation

            calculate_entropy = self.config.entropy_coeff != 0
            entropy, log_prob, student_topk_log_probs, _ = self._forward_micro_batch(
                model_inputs,
                temperature=temperature,
                calculate_entropy=calculate_entropy,
                opd_topk_token_ids=student_topk_ids,
            )
            loss_agg_mode = self.config.loss_agg_mode

            topk_kl_per_teacher = topk_reverse_kl_per_teacher(
                student_topk_log_probs,
                teacher_topk_log_probs,
                response_mask=response_mask,
                max_abs_log_ratio=max_abs_log_ratio,
            )
            teacher_kl_losses = [
                agg_loss(
                    loss_mat=topk_kl_per_teacher[..., teacher_index],
                    loss_mask=response_mask,
                    loss_agg_mode=loss_agg_mode,
                )
                for teacher_index in range(teacher_count)
            ]
            opd_kl_loss = torch.stack(teacher_kl_losses).mean()
            total_loss = float(opd_loss_coef) * opd_kl_loss

            task_pg_loss = None
            task_clipfrac = None
            task_ppo_kl = None
            task_clipfrac_lower = None
            if task_loss_coef:
                policy_loss_fn = get_policy_loss_fn(
                    self.config.policy_loss.get("loss_mode", "vanilla")
                )
                policy_old_log_prob = log_prob.detach()
                (
                    task_pg_loss,
                    task_clipfrac,
                    task_ppo_kl,
                    task_clipfrac_lower,
                ) = policy_loss_fn(
                    old_log_prob=policy_old_log_prob,
                    log_prob=log_prob,
                    advantages=task_loss_coef * task_advantages,
                    response_mask=response_mask,
                    loss_agg_mode=loss_agg_mode,
                    config=self.config,
                    rollout_is_weights=rollout_is_weights,
                )
                total_loss = total_loss + task_pg_loss

            entropy_loss = None
            if calculate_entropy:
                entropy_loss = agg_loss(
                    loss_mat=entropy,
                    loss_mask=response_mask,
                    loss_agg_mode=loss_agg_mode,
                )
                total_loss = total_loss - self.config.entropy_coeff * entropy_loss

            (total_loss * loss_scale_factor).backward()

            teacher_means = masked_topk_teacher_means(
                topk_kl_per_teacher.detach(),
                response_mask,
            )
            mean_reverse_kl = teacher_means.mean()
            metric_device = log_prob.device
            micro_batch_metrics = {
                "actor/pg_loss": total_loss.detach().item() * loss_scale_factor,
                "actor/opd/loss": opd_kl_loss.detach().item()
                * float(opd_loss_coef)
                * loss_scale_factor,
                "actor/opd/reverse_kl": mean_reverse_kl.detach().item(),
                "actor/opd/top_k": float(configured_top_k),
                "actor/opd/teacher_count": float(teacher_count),
                "actor/opd/loss_coef": opd_loss_coef,
                "actor/opd/task_loss_coef": task_loss_coef,
            }
            if entropy_loss is not None:
                micro_batch_metrics["actor/entropy_loss"] = entropy_loss.detach().item()
            if task_pg_loss is not None:
                micro_batch_metrics.update(
                    {
                        "actor/opd/task_pg_loss": (
                            task_pg_loss.detach().item() * loss_scale_factor
                        ),
                        "actor/opd/task_pg_clipfrac": task_clipfrac.detach().item(),
                        "actor/opd/task_ppo_kl": task_ppo_kl.detach().item(),
                        "actor/opd/task_pg_clipfrac_lower": (
                            task_clipfrac_lower.detach().item()
                        ),
                    }
                )
            for index, teacher_name in enumerate(active_teacher_names):
                metric_name = _metric_name(
                    active_teacher_index if active_teacher_index is not None else index,
                    teacher_name,
                )
                prefix = f"actor/opd/teacher/{metric_name}"
                teacher_metric_index = (
                    active_teacher_index
                    if active_teacher_index is not None
                    else index
                )
                micro_batch_metrics[f"{prefix}/loss"] = (
                    teacher_kl_losses[index].detach().item()
                    * float(opd_loss_coef)
                    * loss_scale_factor
                )
                micro_batch_metrics[f"{prefix}/reverse_kl"] = (
                    teacher_means[index].detach().item()
                )

            active_mask = response_mask.to(dtype=torch.bool).unsqueeze(-1).expand_as(
                topk_kl_per_teacher
            )
            active_values = topk_kl_per_teacher.detach()[active_mask]
            if active_values.numel():
                micro_batch_metrics.update(
                    {
                        "actor/opd/reverse_kl_abs": (
                            active_values.abs().mean().detach().item()
                        ),
                        "actor/opd/reverse_kl_min": (
                            active_values.min().detach().item()
                        ),
                        "actor/opd/reverse_kl_max": (
                            active_values.max().detach().item()
                        ),
                    }
                )
            append_to_dict(metrics, micro_batch_metrics)

        grad_norm = self._optimizer_step()
        append_to_dict(
            metrics,
            {"actor/grad_norm": grad_norm.detach().item()},
        )
        self.actor_optimizer.zero_grad()
        return metrics
