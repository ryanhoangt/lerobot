#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

from lerobot.scripts.lerobot_train import evaluate_on_dataset


class _DummyEvalDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Return 1-indexed values to avoid zero means
        return {"input": torch.tensor(idx + 1, dtype=torch.float32)}


class _DummyPolicy(torch.nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:  # type: ignore[override]
        values = batch["input"].float()
        mean = values.mean()
        metrics = {"accuracy": mean + 1.0}
        return mean, metrics


def _make_preprocessor(device: torch.device):
    def _preprocess(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: tensor.to(device) for key, tensor in batch.items()}

    return _preprocess


def test_evaluate_on_dataset_accumulates_metrics():
    accelerator = Accelerator(cpu=True)
    try:
        dataset = _DummyEvalDataset()
        dataloader = DataLoader(dataset, batch_size=2)
        policy = _DummyPolicy()
        preprocessor = _make_preprocessor(accelerator.device)

        metrics = evaluate_on_dataset(policy, dataloader, preprocessor, accelerator)
        assert set(metrics) == {"val/loss", "val/accuracy"}
        assert metrics["val/loss"] == pytest.approx(2.5)
        assert metrics["val/accuracy"] == pytest.approx(3.5)

        limited_metrics = evaluate_on_dataset(policy, dataloader, preprocessor, accelerator, max_batches=1)
        assert set(limited_metrics) == {"val/loss", "val/accuracy"}
        assert limited_metrics["val/loss"] == pytest.approx(1.5)
        assert limited_metrics["val/accuracy"] == pytest.approx(2.5)
    finally:
        accelerator.free_memory()
