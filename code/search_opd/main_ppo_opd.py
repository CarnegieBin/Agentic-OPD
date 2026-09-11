"""Independent Search-R1 multi-teacher OPD training entry point."""

from __future__ import annotations

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from search_opd.config_opd import configure_opd
from search_opd.fsdp_workers_opd import (
    ActorRolloutRefWorkerOPD,
    AsyncActorRolloutRefWorkerOPD,
)
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role, need_critic
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local
from verl_tool.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from search_opd.checkpoint_trainer_opd import AgentRayPPOTrainerOPD
from verl_tool.trainer.ppo.reward import load_reward_manager


@hydra.main(
    config_path="../verl-tool/verl_tool/trainer/config",
    config_name="ppo_trainer",
    version_base=None,
)
def main(config: OmegaConf) -> None:
    configure_opd(config)
    run_ppo_opd(config)


def run_ppo_opd(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        if not is_nvtx_available():
            raise RuntimeError("nvtx is required for nsys profiling")
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = TaskRunnerOPD.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunnerOPD.remote()
    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class TaskRunnerOPD:
    """Build the unchanged Search-R1 stack with OPD-specific actor/ref workers."""

    def __init__(self) -> None:
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup

        if config.actor_rollout_ref.actor.strategy not in {"fsdp", "fsdp2"}:
            raise NotImplementedError("Search-OPD currently supports FSDP/FSDP2 only")
        actor_rollout_cls = (
            AsyncActorRolloutRefWorkerOPD
            if config.actor_rollout_ref.rollout.mode == "async"
            else ActorRolloutRefWorkerOPD
        )
        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        return actor_rollout_cls, RayWorkerGroup

    def add_critic_worker(self, config) -> None:
        if not need_critic(config):
            return
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import CriticWorker
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker
        else:
            raise NotImplementedError(f"Unsupported critic strategy: {config.critic.strategy}")
        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)

    def add_reward_model_worker(self, config) -> None:
        if not config.reward_model.enable:
            return
        if config.reward_model.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError(f"Unsupported reward strategy: {config.reward_model.strategy}")
        self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        self.mapping[Role.RewardModel] = (
            "reward_pool" if config.reward_model.enable_resource_pool else "global_pool"
        )

    def add_teacher_worker(self, actor_rollout_cls) -> None:
        self.role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
        self.mapping[Role.RefPolicy] = "global_pool"

    def init_resource_pool_mgr(self, config) -> ResourcePoolManager:
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0 or config.reward_model.nnodes <= 0:
                raise ValueError("Reward-model resource pool dimensions must be positive")
            resource_pool_spec["reward_pool"] = [
                config.reward_model.n_gpus_per_node
            ] * config.reward_model.nnodes

        self.mapping[Role.ActorRollout] = global_pool_id
        if need_critic(config):
            self.mapping[Role.Critic] = global_pool_id
        return ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
        )

    def run(self, config) -> None:
        from pprint import pprint

        print(f"Search-OPD TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_teacher_worker(actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=True,
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(
            local_path,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )
        val_reward_fn = load_reward_manager(
            config,
            tokenizer,
            num_examine=1,
            **config.reward_model.get("reward_kwargs", {}),
        )
        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
        )
        test_datasets = {}
        test_files = config.data.get("test_files", {})
        if test_files:
            test_files = OmegaConf.to_container(test_files, resolve=True)
            if isinstance(test_files, (str, list)):
                test_files = {"test": test_files}
            for data_source, data_paths in test_files.items():
                if data_paths in (None, "", []):
                    continue
                test_datasets[str(data_source)] = create_rl_dataset(
                    data_paths,
                    config.data,
                    tokenizer,
                    processor,
                    is_train=False,
                    max_samples=config.data.get("test_max_samples", -1),
                )

        trainer = AgentRayPPOTrainerOPD(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_datasets=test_datasets,
            collate_fn=collate_fn,
            train_sampler=create_rl_sampler(config.data, train_dataset),
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
