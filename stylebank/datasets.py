# /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.io import read_image
import numpy as np

class PhotoDataset(Dataset):

    def __init__(self, path, transform, quantity=None):
        self.filenames = glob.glob(os.path.join(path, "*.jpg"))
        self.filenames.sort()
        if quantity is not None:
            self.filenames = self.filenames[:quantity]

        self.files = self._cache_files(self.filenames)

        self.transform = transform

    def _cache_files(self, filenames):
        files = dict()
        for filename in filenames:
            files[filename] = self.load_image(filename)
        return files

    def load_image(self, filename):
        image = read_image(filename)
        image = TF.to_pil_image(image)
        return self.transform(image)

    def get_image_from_filename(self, filename):
        return self.files[filename]

    def get_image_from_idx(self, idx):
        return self.get_image_from_filename(self.filenames[idx])

    def __len__(self):
        return len(self.filenames)

    def get_image(self, fileId):
        if isinstance(fileId, int):  # that's an index
            return self.get_image_from_idx(fileId)
        elif isinstance(fileId, str):  # that's a filename
            return self.get_image_from_filename(fileId)

    def get_names(self, indices):
        return [os.path.splitext(os.path.basename(self.filenames[idx]))[0] for idx in indices]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.get_image(idx)
        return torch.stack([self.get_image(i) for i in idx])


class PaintingsDataset(PhotoDataset):

    def __getitem__(self, idx):
        # return both the indices and the paintings
        return np.atleast_1d(idx), super().__getitem__(idx)
