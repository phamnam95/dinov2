# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar, Tuple, Union, Iterable

import torch
from torch.utils.data import Sampler, IterableDataset, get_worker_info

from .datasets import ImageNet, ImageNet22k

try:
    import xarray as xr  # type: ignore
    from xbatcher import BatchGenerator  # type: ignore
    _XR_AVAILABLE = True
except Exception:
    _XR_AVAILABLE = False
from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler


logger = logging.getLogger("dinov2")


class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4


def _make_bool_str(b: bool) -> str:
    return "yes" if b else "no"


def _make_sample_transform(image_transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
    def transform(sample):
        image, target = sample
        if image_transform is not None:
            image = image_transform(image)
        if target_transform is not None:
            target = target_transform(target)
        return image, target

    return transform


def _parse_dataset_str(dataset_str: str):
    tokens = dataset_str.split(":")

    name = tokens[0]
    kwargs = {}

    for token in tokens[1:]:
        key, value = token.split("=")
        assert key in ("root", "extra", "split")
        kwargs[key] = value

    if name == "ImageNet":
        class_ = ImageNet
        if "split" in kwargs:
            kwargs["split"] = ImageNet.Split[kwargs["split"]]
    elif name == "ImageNet22k":
        class_ = ImageNet22k
    else:
        raise ValueError(f'Unsupported dataset "{name}"')

    return class_, kwargs


def make_dataset(
    *,
    dataset_str: str,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
):
    """
    Creates a dataset with the specified parameters.

    Args:
        dataset_str: A dataset string description (e.g. ImageNet:split=TRAIN).
        transform: A transform to apply to images.
        target_transform: A transform to apply to targets.

    Returns:
        The created dataset.
    """
    logger.info(f'using dataset: "{dataset_str}"')

    class_, kwargs = _parse_dataset_str(dataset_str)
    dataset = class_(transform=transform, target_transform=target_transform, **kwargs)

    logger.info(f"# of dataset samples: {len(dataset):,d}")

    # Aggregated datasets do not expose (yet) these attributes, so add them.
    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform)
    if not hasattr(dataset, "target_transform"):
        setattr(dataset, "target_transform", target_transform)

    return dataset


def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        # TODO: Remove support for old shuffling
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
):
    """
    Creates a data loader with the specified parameters.

    Args:
        dataset: A dataset (third party, LaViDa or WebDataset).
        batch_size: The size of batches to generate.
        num_workers: The number of workers to use.
        shuffle: Whether to shuffle samples.
        seed: The random seed to use.
        sampler_type: Which sampler to use: EPOCH, INFINITE, SHARDED_INFINITE, SHARDED_INFINITE_NEW, DISTRIBUTED or None.
        sampler_size: The number of images per epoch (when applicable) or -1 for the entire dataset.
        sampler_advance: How many samples to skip (when applicable).
        drop_last: Whether the last non-full batch of data should be dropped.
        persistent_workers: maintain the workers Dataset instances alive after a dataset has been consumed once.
        collate_fn: Function that performs batch collation
    """

    sampler = _make_sampler(
        dataset=dataset,
        type=sampler_type,
        shuffle=shuffle,
        seed=seed,
        size=sampler_size,
        advance=sampler_advance,
    )

    logger.info("using PyTorch data loader")
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader


# ---------------------- Optional tiling for large inputs ----------------------
try:
    from torchvision.transforms.functional import crop as tv_crop
except Exception:
    tv_crop = None  # type: ignore


def _compute_starts(size: int, tile: int, stride: int, drop_last: bool) -> List[int]:
    if tile >= size:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if not drop_last:
        last_start = size - tile
        if starts[-1] != last_start:
            starts.append(last_start)
    return starts


def _iter_tiles_2d(img, tile_hw: Tuple[int, int], stride_hw: Tuple[int, int], drop_last: bool):
    # img can be PIL Image or Tensor [C,H,W]
    if torch.is_tensor(img):
        _, H, W = img.shape
    else:
        W, H = img.size  # PIL
    th, tw = tile_hw
    sh, sw = stride_hw
    hs = _compute_starts(H, th, sh, drop_last)
    ws = _compute_starts(W, tw, sw, drop_last)
    for top in hs:
        for left in ws:
            if torch.is_tensor(img):
                yield img[:, top : top + th, left : left + tw], (top, left)
            else:
                assert tv_crop is not None, "torchvision is required for 2D PIL tiling"
                tile = tv_crop(img, top, left, th, tw)
                yield tile, (top, left)


def _iter_tiles_3d(vol: torch.Tensor, tile_dhw: Tuple[int, int, int], stride_dhw: Tuple[int, int, int], drop_last: bool):
    # vol: Tensor [C,D,H,W]
    _, D, H, W = vol.shape
    td, th, tw = tile_dhw
    sd, sh, sw = stride_dhw
    ds = _compute_starts(D, td, sd, drop_last)
    hs = _compute_starts(H, th, sh, drop_last)
    ws = _compute_starts(W, tw, sw, drop_last)
    for z in ds:
        for y in hs:
            for x in ws:
                yield vol[:, z : z + td, y : y + th, x : x + tw], (z, y, x)


class TiledIterableDataset(IterableDataset):
    def __init__(
        self,
        base_dataset,
        *,
        is_3d: bool,
        tile_size: Union[Tuple[int, int], Tuple[int, int, int]],
        stride: Union[Tuple[int, int], Tuple[int, int, int]],
        drop_last_tiles: bool = True,
        transform=None,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.is_3d = is_3d
        self.tile_size = tile_size
        self.stride = stride
        self.drop_last_tiles = drop_last_tiles
        self.transform = transform

    def _sharded_indices(self) -> Iterable[int]:
        # Shard base sample indices across distributed processes and dataloader workers
        world_size = distributed.get_global_size()
        global_rank = distributed.get_global_rank()

        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        step = max(1, world_size * num_workers)
        offset = global_rank * num_workers + worker_id
        for idx in range(offset, len(self.base_dataset), step):
            yield idx

    def __iter__(self):
        for idx in self._sharded_indices():
            sample = self.base_dataset[idx]
            if isinstance(sample, tuple):
                img, target = sample
            else:
                img, target = sample, None

            if self.is_3d:
                assert torch.is_tensor(img) and img.dim() == 4, "3D tiling expects Tensor [C,D,H,W]"
                for tile, offs in _iter_tiles_3d(img, self.tile_size, self.stride, self.drop_last_tiles):
                    out = self.transform(tile) if self.transform is not None else tile
                    if isinstance(out, dict):
                        out["tile_origin"] = offs
                    yield (out, target)
            else:
                for tile, offs in _iter_tiles_2d(img, self.tile_size, self.stride, self.drop_last_tiles):
                    out = self.transform(tile) if self.transform is not None else tile
                    if isinstance(out, dict):
                        out["tile_origin"] = offs
                    yield (out, target)


def make_tiled_iterable_dataset(
    *,
    dataset,
    is_3d: bool,
    tile_size: Union[Tuple[int, int], Tuple[int, int, int]],
    stride: Union[Tuple[int, int], Tuple[int, int, int]],
    drop_last_tiles: bool,
    transform=None,
):
    return TiledIterableDataset(
        dataset,
        is_3d=is_3d,
        tile_size=tuple(tile_size),
        stride=tuple(stride),
        drop_last_tiles=drop_last_tiles,
        transform=transform,
    )


def make_xarray_loader(
    *,
    ds_path: str,
    image_var: str,
    target_var: Optional[str] = None,
    batch_size: int,
    num_workers: int = 0,
    to_chw: bool = True,
    normalize: bool = True,
    chunks: Optional[dict] = None,
    tiling: Optional[dict] = None,
):
    if not _XR_AVAILABLE:
        raise RuntimeError("xarray/xbatcher not available; please install xarray and xbatcher.")

    ds = xr.open_zarr(ds_path) if ds_path.endswith(".zarr") else xr.open_dataset(ds_path)
    if chunks:
        ds = ds.chunk(chunks)

    img = ds[image_var]
    tgt = ds[target_var] if target_var and target_var in ds else None

    # Expect a leading sample dimension; name may vary
    sample_dim = img.dims[0]
    if tiling and tiling.get("enabled", False):
        # Determine spatial dims for tiling
        is_3d = bool(tiling.get("is_3d", False))
        tile_size = tuple(tiling.get("tile_size"))
        stride = tuple(tiling.get("stride"))

        # Build dimension dict for xbatcher: batch over samples and window over spatial dims
        if is_3d:
            # assume dims: (sample, D, H, W, C) or (sample, C, D, H, W)
            spatial_dims = [d for d in img.dims if d not in (sample_dim,)]
            # pick last 3 as D,H,W
            dhw = spatial_dims[-3:]
            input_dims = {
                image_var: {
                    sample_dim: batch_size,
                    dhw[0]: tile_size[0],
                    dhw[1]: tile_size[1],
                    dhw[2]: tile_size[2],
                }
            }
            bg = BatchGenerator(ds, input_dims=input_dims, steps={dhw[0]: stride[0], dhw[1]: stride[1], dhw[2]: stride[2]}, shuffle=True)
        else:
            # 2D: pick last 2 spatial dims as H,W
            spatial_dims = [d for d in img.dims if d not in (sample_dim,)]
            hw = spatial_dims[-2:]
            input_dims = {
                image_var: {sample_dim: batch_size, hw[0]: tile_size[0], hw[1]: tile_size[1]}
            }
            bg = BatchGenerator(ds, input_dims=input_dims, steps={hw[0]: stride[0], hw[1]: stride[1]}, shuffle=True)
    else:
        bg = BatchGenerator(ds, input_dims={image_var: {sample_dim: batch_size}}, shuffle=True)

    def _to_tensor(npx):
        import numpy as np
        import torch

        arr = npx.values  # NumPy array
        if to_chw and arr.shape[-1] in (1, 3):
            arr = arr.transpose(0, 3, 1, 2)
        t = torch.from_numpy(arr).float()
        if normalize:
            t = t / 255.0
        return t

    def _iter():
        for batch in bg:
            images = _to_tensor(batch[image_var])
            targets = None
            if tgt is not None:
                targets = torch.from_numpy(batch[target_var].values)
            yield [(dict(global_crops=[img for img in images[:2]], local_crops=list(images[2:])), targets)]

    return _iter()
