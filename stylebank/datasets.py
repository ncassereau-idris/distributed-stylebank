# /usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from hydra.utils import to_absolute_path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.functional as TF
from torchvision.io import read_image
import torchvision.transforms as transforms
import horovod.torch as hvd
from PIL import Image
import numpy as np
import glob
import os
import multiprocessing


log = logging.getLogger(__name__)


class PhotoDataset(Dataset):

    def __init__(self, path, transform, quantity=-1, preload=True):
        self.filenames = glob.glob(
            to_absolute_path(os.path.join(path, "*.jpg"))
        )
        self.filenames.sort()
        if quantity > 0:
            self.filenames = self.filenames[:quantity]

        self.transform = transform
        self.files = [None] * len(self.filenames)
        if preload:
            log.info("Preloading data")
            self.preload()
            log.info("Data preloaded!")

    def preload(self, num_workers=10):
        with multiprocessing.Pool(num_workers) as pool:
            result = pool.map(self.load_image, self.filenames)
        self.files = result

    def load_image(self, filename):
        image = read_image(filename)
        image = TF.to_pil_image(image)
        return self.transform(image)

    def get_image_from_filename(self, filename):
        return self.get_image_from_idx(self.filenames.index(filename))

    def get_image_from_idx(self, idx):
        img = self.files[idx]
        if img is None:
            img = self.load_image(self.filenames[idx])
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
            return self.get_image(idx)
        return torch.stack([self.get_image(i) for i in idx])


class PaintingsDataset(PhotoDataset):

    def __getitem__(self, idx):
        # return both the indices and the paintings
        return idx, super().__getitem__(idx)


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
            transforms.RandomCrop([513, 513]),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.load_datasets()
        if self.cfg.training.train:
            self.make_training_dataloader()

    def load_datasets(self):
        log.info("Loading real pictures dataset")
        self.content_dataset = PhotoDataset(
            path=self.cfg.data.photo,
            transform=self.transform,
            preload=True
        )
        log.info(
            f"Real pictures dataset has {len(self.content_dataset)} samples"
        )

        log.info("Loading monet paintings dataset")
        self.style_dataset = PaintingsDataset(
            path=self.cfg.data.monet,
            transform=self.transform,
            quantity=self.cfg.data.style_quantity,
            preload=True
        )
        log.info(f"Paintings dataset has {len(self.style_dataset)} samples")

    def make_training_dataloader(self):
        self.training_dataset = TrainingDataset(
            self.cfg, self.content_dataset, self.style_dataset
        )
        self.training_sampler = DistributedSampler(
            self.training_dataset,
            num_replicas=hvd.size(),
            rank=hvd.rank(),
            shuffle=True
        )
        self.training_dataloader = DataLoader(
            self.training_dataset,
            batch_size=self.cfg.training.batch_size,
            sampler=self.training_sampler,
            num_workers=10
        )
