# /usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from hydra.utils import to_absolute_path
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.functional as TF
from torchvision.io import read_image
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import glob
import os
from . import tools
from .plasma import PlasmaStorage


log = logging.getLogger(__name__)


class PhotoDataset(Dataset):

    def __init__(
        self, path, transform, quantity=-1,
        store_transformed=False, preload=False
    ):
        assert store_transformed or not preload
        self.store_transformed = store_transformed
        self.filenames = glob.glob(
            to_absolute_path(os.path.join(path, "*.jpg"))
        )
        self.filenames.sort()
        if 0 < quantity <= len(self.filenames):
            self.filenames = self.filenames[:quantity]

        self.transform = transform

        if preload:
            log.info(f"Preloading data ({len(self.filenames)} files)")
            self.files = self.preload()
            log.info(f"{len(self.filenames)} files have been preloaded!")
        else:
            self.files = PlasmaStorage(autocuda=True)

    def preload(self):
        files = PlasmaStorage(autocuda=True)
        for i, filename in enumerate(self.filenames):
            if (i - tools.rank) % tools.size == 0:
                files[i] = self.load_image(filename)
        dist.barrier()

        # pooling across all tasks
        return files.merge()

    def load_image(self, filename):
        image = read_image(filename)
        image = TF.to_pil_image(image)
        return self.transform(image).cuda()

    def get_image_from_filename(self, filename):
        return self.get_image_from_idx(self.filenames.index(filename))

    def get_image_from_idx(self, idx):
        img = self.files[idx]
        if img is None:
            img = self.load_image(self.filenames[idx])
            if self.store_transformed:
                self.files[idx] = img
        return img

    def __len__(self):
        return len(self.filenames)

    def get_image(self, fileId):
        if isinstance(fileId, int):  # that's an index
            return self.get_image_from_idx(fileId)
        elif isinstance(fileId, str):  # that's a filename
            return self.get_image_from_filename(fileId)

    def get_names(self, indices):
        return [
            os.path.splitext(
                os.path.basename(self.filenames[idx])
            )[0]
            for idx in indices
        ]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return idx, self.get_image(idx)
        return idx, torch.stack([self.get_image(i) for i in idx])


class TrainingDataset(Dataset):

    def __init__(self, cfg, content_dataset, style_dataset):
        self.cfg = cfg
        self.content_dataset = content_dataset
        self.style_dataset = style_dataset

    def __len__(self):
        return self.cfg.training.repeat * len(self.style_dataset)

    def __getitem__(self, idx):
        return (
            self.content_dataset[np.random.randint(len(self.content_dataset))],
            self.style_dataset[idx % len(self.style_dataset)]
        )


class Resize(object):
    """
    Resize with aspect ratio preserved.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        m = min(img.size)
        new_size = (
            int(img.size[0] / m * self.size),
            int(img.size[1] / m * self.size)
        )
        return img.resize(new_size, resample=Image.BILINEAR)


class DataManager:

    def __init__(self, cfg):
        self.cfg = cfg
        self.transform = transforms.Compose([
            Resize(513),
            transforms.CenterCrop([513, 513]),
            transforms.ToTensor(),
        ])
        self.load_datasets()
        if self.cfg.training.train:
            self.make_training_dataloader()

    def load_datasets(self):
        log.info("Loading real pictures dataset")
        self.content_dataset = PhotoDataset(
            path=self.cfg.data.photo,
            transform=self.transform,
            store_transformed=self.cfg.data.store_transformed,
            preload=self.cfg.data.preload_transformed
        )

        log.info(
            f"Real pictures dataset has {len(self.content_dataset)} samples"
        )

        log.info("Loading monet paintings dataset")
        self.style_dataset = PhotoDataset(
            path=self.cfg.data.monet,
            transform=self.transform,
            quantity=self.cfg.data.style_quantity,
            store_transformed=self.cfg.data.store_transformed,
            preload=self.cfg.data.preload_transformed
        )
        log.info(f"Paintings dataset has {len(self.style_dataset)} samples")

    def _distributed_sampler(self, dataset, **kwargs):
        sampler = DistributedSampler(
            dataset,
            num_replicas=tools.size,
            rank=tools.rank,
            shuffle=False,
            **kwargs
        )
        return sampler

    def _dataloader(self, dataset, sampler=None, **kwargs):
        if sampler is None:
            sampler = self._distributed_sampler(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.training.batch_size,
            sampler=sampler,
            **kwargs
        )
        return dataloader

    def make_training_dataloader(self):
        training_dataset = TrainingDataset(
            self.cfg, self.content_dataset, self.style_dataset
        )
        return self._dataloader(training_dataset)

    def make_preload_dataloaders(self):
        content_dataloader = self._dataloader(self.content_dataset)
        style_dataloader = self._dataloader(self.style_dataset)
        return content_dataloader, style_dataloader

    def cycle(self, iterable):
        # This version of cycle shuffles the dataset between
        # each epoch unlike itertools' version
        while True:
            for x in iterable:
                yield x

    def make_generation_dataloader(self):
        combined_dataset = TrainingDataset(
            self.cfg, self.content_dataset, self.style_dataset
        )
        dataloader = self._dataloader(combined_dataset)
        return self.cycle(dataloader)
