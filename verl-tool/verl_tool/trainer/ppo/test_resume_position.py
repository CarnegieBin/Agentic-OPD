from types import SimpleNamespace

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def _loader():
    dataset = list(range(20))
    sampler = torch.utils.data.RandomSampler(
        dataset,
        generator=torch.Generator().manual_seed(1),
    )
    return StatefulDataLoader(
        dataset=dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )


def test_initial_data_steps_resumes_at_next_deterministic_batch():
    expected_loader = _loader()
    expected_iterator = iter(expected_loader)
    for _ in range(2):
        next(expected_iterator)
    expected_batch = next(expected_iterator).tolist()

    resumed_loader = _loader()
    trainer = SimpleNamespace(
        train_dataloader=resumed_loader,
        global_steps=2,
    )
    RayPPOTrainer._skip_train_dataloader_steps(trainer, 2)

    resumed_iterator = trainer._resume_train_iterator
    assert next(resumed_iterator).tolist() == expected_batch
