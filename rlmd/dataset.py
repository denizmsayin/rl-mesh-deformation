import omegaconf 
import os
import json
import bisect
from collections import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence







class ShapeDiskDataset(Dataset):
    def __init__(
        self,
        dataset_folder,
        max_samples=None,
        shape_names=None,
        cache_size=8,
        dtype=torch.float32,
        map_location="cpu",
    ):
        self.dataset_folder = dataset_folder
        self.max_samples = max_samples
        self.cache_size = cache_size
        self.dtype = dtype
        self.map_location = map_location

        manifest_path = os.path.join(dataset_folder, "manifest.json")

        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)

        self.specs = {
            int(spec["spec_idx"]): spec
            for spec in self.manifest["specs"]
        }

        if shape_names is not None:
            shape_names = set(shape_names)

        self.shards = []

        for shard in self.manifest["shards"]:
            spec_idx = int(shard["spec_idx"])
            spec = self.specs[spec_idx]

            instance_name = spec["instance_name"]
            shape_name = spec["shape"]

            if shape_names is not None:
                if instance_name not in shape_names and shape_name not in shape_names:
                    continue

            self.shards.append(
                {
                    "path": os.path.join(dataset_folder, shard["path"]),
                    "spec_idx": spec_idx,
                    "num_samples": int(shard["num_samples"]),
                }
            )

        self.shards = self._truncate_shards(self.shards, max_samples)

        self.cumulative_sizes = []
        total = 0

        for shard in self.shards:
            total += shard["num_samples"]
            self.cumulative_sizes.append(total)

        self.total_samples = total

        self._shard_cache = OrderedDict()
        self._base_cache = OrderedDict()

    def _truncate_shards(self, shards, max_samples):
        if max_samples is None:
            return shards

        max_samples = int(max_samples)
        truncated = []
        remaining = max_samples

        for shard in shards:
            if remaining <= 0:
                break

            n = min(shard["num_samples"], remaining)

            new_shard = dict(shard)
            new_shard["num_samples"] = n
            truncated.append(new_shard)

            remaining -= n

        return truncated

    def __len__(self):
        return self.total_samples

    def _resolve_index(self, index):
        if index < 0:
            index = len(self) + index

        if index < 0 or index >= len(self):
            raise IndexError("Index out of range.")

        shard_idx = bisect.bisect_right(self.cumulative_sizes, index)
        previous_total = 0 if shard_idx == 0 else self.cumulative_sizes[shard_idx - 1]
        local_idx = index - previous_total

        return shard_idx, local_idx

    def _load_shard(self, shard_idx):
        if shard_idx in self._shard_cache:
            self._shard_cache.move_to_end(shard_idx)
            return self._shard_cache[shard_idx]

        shard_path = self.shards[shard_idx]["path"]
        payload = torch.load(shard_path, map_location=self.map_location)

        self._shard_cache[shard_idx] = payload

        if len(self._shard_cache) > self.cache_size:
            self._shard_cache.popitem(last=False)

        return payload

    def _load_base_shape(self, spec_idx):
        if spec_idx in self._base_cache:
            self._base_cache.move_to_end(spec_idx)
            return self._base_cache[spec_idx]

        spec = self.specs[spec_idx]
        base_path = os.path.join(self.dataset_folder, spec["base_shape_file"])

        base = torch.load(base_path, map_location=self.map_location)

        base_points = base["base_points"].to(dtype=self.dtype)
        edges = base["edges"].long()

        self._base_cache[spec_idx] = (base_points, edges)

        if len(self._base_cache) > self.cache_size:
            self._base_cache.popitem(last=False)

        return base_points, edges

    def __getitem__(self, index):
        shard_idx, local_idx = self._resolve_index(index)

        shard_info = self.shards[shard_idx]
        spec_idx = shard_info["spec_idx"]

        spec = self.specs[spec_idx]
        shape = spec["instance_name"]

        payload = self._load_shard(shard_idx)
        base_points, edges = self._load_base_shape(spec_idx)

        if "points" in payload:
            vertices = payload["points"][local_idx].to(dtype=self.dtype)
        else:
            translation = payload["translation"][local_idx].to(dtype=self.dtype)
            linear_matrix = payload["linear_matrix"][local_idx].to(dtype=self.dtype)

            vertices = base_points @ linear_matrix.T + translation

        length = torch.tensor(vertices.shape[0], dtype=torch.long)

        return vertices, edges, length, shape


def shape_collate_fn(batch):
    vertices_list, edges_list, lengths_list, shapes_list = zip(*batch)

    vertices = pad_sequence(
        vertices_list,
        batch_first=True,
        padding_value=0.0,
    )

    edges = pad_sequence(
        edges_list,
        batch_first=True,
        padding_value=-1,
    ).long()

    lengths = torch.stack(lengths_list).long()

    shapes = list(shapes_list)

    return vertices, edges, lengths, shapes


def make_shape_dataloader(
    dataset_folder,
    batch_size=4086,
    max_samples=None,
    shape_names=None,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=False,
    cache_size=8,
    dtype=torch.float32,
):
    dataset = ShapeDiskDataset(
        dataset_folder=dataset_folder,
        max_samples=max_samples,
        shape_names=shape_names,
        cache_size=cache_size,
        dtype=dtype,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=shape_collate_fn,
    )

    return dataset, dataloader