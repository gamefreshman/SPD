import multiprocessing
import random

import torch
from torch_geometric.data import Batch

from shepherd.dpo_dataset import collate_dpo_batch


class MixedDPODataset(torch.utils.data.Dataset):
    def __init__(self, standard_dataset, dpo_dataset, real_data_ratio=0.5):
        self.standard_dataset = standard_dataset
        self.dpo_dataset = dpo_dataset
        self.real_data_ratio = real_data_ratio

    def __len__(self):
        dpo_length = len(self.dpo_dataset) if self.dpo_dataset is not None else 0
        return max(len(self.standard_dataset), dpo_length, 1)

    def __getitem__(self, idx):
        dpo_length = len(self.dpo_dataset) if self.dpo_dataset is not None else 0
        use_standard = dpo_length == 0 or random.random() < self.real_data_ratio
        if use_standard:
            return self.standard_dataset[idx % len(self.standard_dataset)]
        return self.dpo_dataset[idx % dpo_length]


def collate_mixed_batch(batch_list):
    dpo_items = [
        batch
        for batch in batch_list
        if isinstance(batch, dict) and batch.get("batch_type") == "dpo"
    ]
    standard_items = [
        batch
        for batch in batch_list
        if not (isinstance(batch, dict) and batch.get("batch_type") == "dpo")
    ]

    if len(dpo_items) == len(batch_list):
        return collate_dpo_batch(dpo_items)
    if len(standard_items) == len(batch_list):
        return Batch.from_data_list(standard_items)
    if len(dpo_items) >= len(standard_items):
        return collate_dpo_batch(dpo_items)
    return Batch.from_data_list(standard_items)


def create_mixed_dataloader(
    standard_dataset,
    dpo_dataset,
    batch_size,
    num_workers,
    real_data_ratio=0.5,
    shuffle=True,
    multiprocessing_spawn=False,
    worker_init_fn=None,
):
    mixed_dataset = MixedDPODataset(
        standard_dataset=standard_dataset,
        dpo_dataset=dpo_dataset,
        real_data_ratio=real_data_ratio,
    )

    kwargs = {
        "dataset": mixed_dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_mixed_batch,
        "worker_init_fn": worker_init_fn,
    }
    if multiprocessing_spawn:
        kwargs["multiprocessing_context"] = multiprocessing.get_context("spawn")
    return torch.utils.data.DataLoader(**kwargs)
