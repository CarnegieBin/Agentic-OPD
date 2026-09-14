"""FSDP workers with one dynamically resident temporal OPD teacher."""

from __future__ import annotations

import gc
import json
from dataclasses import replace
from pathlib import Path

import torch
from omegaconf import OmegaConf

from search_opd.dp_actor_opd import DataParallelPPOActorOPD
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import fsdp_version
from verl.utils.profiler import DistProfiler
from verl.workers.actor import DataParallelPPOActor
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
    AsyncActorRolloutRefWorker,
)


class _MultiTeacherOPDMixin:
    """Shared overrides for synchronous and asynchronous FSDP workers."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self) -> None:
        super().init_model()
        opd_config = self.config.get("opd", None)
        if opd_config is None or not opd_config.get("enabled", False):
            raise ValueError("The OPD worker requires actor_rollout_ref.opd.enabled=true")

        # Both actor and reference workers use this value.  In particular, the
        # async actor calls ``compute_log_prob`` through this mixin, so it must
        # be initialized before the reference-only setup below.
        self.opd_top_k = int(opd_config.get("top_k", 16))
        if self.opd_top_k < 1:
            raise ValueError(f"OPD top_k must be positive, got {self.opd_top_k}")

        if self._is_actor:
            actor_config = replace(
                omega_conf_to_dataclass(self.config.actor),
                use_kl_loss=False,
            )
            # actor.use_kl_loss is enabled in the outer config only to make the
            # unchanged trainer allocate a RefPolicy worker.  The legacy
            # single-reference KL term must not be added to the OPD objective.
            # ``BaseConfig`` freezes fields after construction, so create a
            # separate immutable config instead of mutating the converted one.
            self.actor = DataParallelPPOActorOPD(
                config=actor_config,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
                opd_config=opd_config,
            )

        if not self._is_ref:
            return
        if self._is_lora:
            raise ValueError("Multi-teacher OPD cannot use ref-in-actor LoRA mode")

        teacher_paths = tuple(str(path) for path in opd_config.teacher_model_paths)
        teacher_names = tuple(str(name) for name in opd_config.teacher_names)
        if len(teacher_paths) != len(teacher_names) or not teacher_paths:
            raise ValueError("OPD teacher paths/names must be non-empty and aligned")

        initial_teacher_index = int(opd_config.get("initial_teacher_index", 0))
        if not 0 <= initial_teacher_index < len(teacher_paths):
            raise ValueError(
                f"Initial adaptive teacher index out of range: {initial_teacher_index}"
            )
        configured_first_path = str(self.config.ref.model.path)
        if configured_first_path != teacher_paths[initial_teacher_index]:
            raise ValueError(
                "The reference model must match the selected initial teacher: "
                f"ref={configured_first_path}, "
                f"teacher={teacher_paths[initial_teacher_index]}"
            )

        # Keep the complete immutable manifest, but only one 3B teacher on GPU.
        # The driver publishes the active index after each validation pass.
        self.opd_teacher_paths = teacher_paths
        self.opd_teacher_names = teacher_names
        self.opd_teacher_selection_state_path = opd_config.get(
            "teacher_selection_state_path", None
        )
        self.opd_active_teacher_index = initial_teacher_index
        self.opd_teacher_module = self.ref_module_fsdp
        self.opd_teacher_policy = self.ref_policy

        override_model_config = OmegaConf.to_container(
            OmegaConf.create(self.config.model.get("override_config", {}))
        )
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self.rank == 0:
            print(
                "Initialized adaptive OPD teacher pool with "
                f"{len(teacher_paths)} immutable checkpoints; resident_teacher="
                f"{teacher_names[initial_teacher_index]}="
                f"{teacher_paths[initial_teacher_index]}",
                flush=True,
            )

        self._opd_teacher_model_config = {
            "override_model_config": override_model_config,
            "use_remove_padding": use_remove_padding,
            "use_shm": use_shm,
            "use_fused_kernels": use_fused_kernels,
        }

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob_topk")
    def compute_log_prob(self, data: DataProto) -> DataProto:
        """Recompute student scores and publish its fixed top-k support."""

        if not self._is_actor:
            raise RuntimeError("compute_log_prob must run on the actor worker")
        if self._is_offload_param:
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu

            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = (
            self.actor.actor_module.disable_adapter()
            if is_lora
            else nullcontext()
        )
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            with adapter_ctx:
                log_probs, entropys, _, topk_ids = self.actor.compute_log_prob(
                    data=data,
                    calculate_entropy=True,
                    opd_top_k=self.opd_top_k,
                )
            if topk_ids is None:
                raise RuntimeError("Student Top-k support was not returned by the actor")
            output = DataProto.from_dict(
                tensors={
                    "old_log_probs": log_probs,
                    "entropys": entropys,
                    "opd_topk_token_ids": topk_ids,
                },
                meta_info={"temperature": self.config.rollout.temperature},
            )

        output = output.to("cpu")
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)
        if self._is_offload_param:
            from verl.utils.fsdp_utils import offload_fsdp_model_to_cpu

            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        return output

    def _read_active_teacher_index(self) -> int:
        path = self.opd_teacher_selection_state_path
        if not path:
            return self.opd_active_teacher_index
        try:
            with Path(str(path)).expanduser().open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            index = int(payload["selected"]["index"])
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            # Keep the configured/preflight-selected resident teacher until a
            # complete state file is atomically published by the driver.
            return self.opd_active_teacher_index
        if not 0 <= index < len(self.opd_teacher_paths):
            raise ValueError(
                f"Adaptive teacher index out of range: {index}; "
                f"teacher_count={len(self.opd_teacher_paths)}"
            )
        return index

    def _activate_teacher(self, teacher_index: int) -> None:
        if teacher_index == self.opd_active_teacher_index:
            return
        teacher_path = self.opd_teacher_paths[teacher_index]
        if self.rank == 0:
            print(
                f"Switching adaptive OPD teacher: "
                f"{self.opd_teacher_names[self.opd_active_teacher_index]} -> "
                f"{self.opd_teacher_names[teacher_index]} ({teacher_path})",
                flush=True,
            )

        config = self._opd_teacher_model_config
        del self.opd_teacher_policy
        del self.opd_teacher_module
        self.ref_policy = None
        self.ref_module_fsdp = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        local_path = copy_to_local(teacher_path, use_shm=config["use_shm"])
        teacher_module = self._build_model_optimizer(
            model_path=local_path,
            fsdp_config=omega_conf_to_dataclass(self.config.ref.fsdp_config),
            optim_config=None,
            override_model_config=config["override_model_config"],
            use_remove_padding=config["use_remove_padding"],
            use_fused_kernels=config["use_fused_kernels"],
            trust_remote_code=self.config.model.get("trust_remote_code", False),
            use_liger=self.config.model.get("use_liger", False),
            role="ref",
        )[0]
        teacher_policy = DataParallelPPOActor(
            config=self.config.ref,
            actor_module=teacher_module,
        )
        self.opd_teacher_module = teacher_module
        self.opd_teacher_policy = teacher_policy
        self.opd_active_teacher_index = teacher_index
        # Keep base-worker aliases coherent for cleanup/profiling utilities.
        self.ref_module_fsdp = teacher_module
        self.ref_policy = teacher_policy

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="olive", role="multi_teacher_opd_log_prob")
    def compute_ref_log_prob(self, data: DataProto) -> DataProto:
        if self._is_lora:
            raise ValueError("Multi-teacher OPD cannot use ref-in-actor LoRA mode")
        if not self._is_ref:
            raise RuntimeError("compute_ref_log_prob must run on the frozen teacher worker")

        data.meta_info["micro_batch_size"] = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        teacher_index = self._read_active_teacher_index()
        self._activate_teacher(teacher_index)
        teacher_policy = self.opd_teacher_policy
        if "opd_topk_token_ids" not in data.batch.keys():
            raise KeyError("Student Top-k token IDs are required before teacher scoring")
        try:
            with self.ulysses_sharding_manager:
                teacher_data = data.to("cpu")
                output, _, teacher_topk_log_probs, _ = teacher_policy.compute_log_prob(
                    data=teacher_data,
                    calculate_entropy=False,
                )
            output = output.to("cpu")
            if teacher_topk_log_probs is None:
                raise RuntimeError("Teacher Top-k support scores were not returned")
            teacher_topk_log_probs = teacher_topk_log_probs.to("cpu")
        finally:
            self._reshard_teacher(teacher_policy)
        if output.ndim != 2:
            raise ValueError(
                "Adaptive OPD teacher log-probability output must have shape "
                f"[batch, response_length], got {tuple(output.shape)}"
            )
        output = DataProto.from_dict(
            tensors={
                "ref_log_prob": output,
                "teacher_topk_log_probs": teacher_topk_log_probs,
            },
            meta_info={
                "opd_teacher_names": [self.opd_teacher_names[teacher_index]],
                "opd_teacher_count": 1,
                "opd_active_teacher_index": teacher_index,
                "opd_top_k": self.opd_top_k,
            },
        )
        return output

    def _reshard_teacher(self, teacher_policy: DataParallelPPOActor) -> None:
        if self.world_size <= 1:
            return
        module = teacher_policy.actor_module
        version = fsdp_version(module)
        if version == 1:
            module._handle.reshard(True)
        elif version == 2:
            module.reshard()


class ActorRolloutRefWorkerOPD(_MultiTeacherOPDMixin, ActorRolloutRefWorker):
    """Synchronous actor/rollout/reference worker with multi-teacher OPD."""


class AsyncActorRolloutRefWorkerOPD(_MultiTeacherOPDMixin, AsyncActorRolloutRefWorker):
    """Asynchronous actor/rollout/reference worker with multi-teacher OPD."""
