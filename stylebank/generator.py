# /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import math
import logging
from PIL import Image
import zipfile
import os
from . import tools


log = logging.getLogger(__name__)


class Generator:

    def __init__(self, cfg, data_manager, network_manager):
        self.cfg = cfg
        self.data_manager = data_manager
        self.network_manager = network_manager

    def generate(self):
        log.info("Generating images")
        batches = self.cfg.generation.nb_images // self.cfg.training.batch_size
        iterator = enumerate(self.data_manager.make_generation_dataloader())
        output_pictures = []
        content_pictures = []
        style_pictures = []
        for i, ((_, content), (style_id, style)) in iterator:
            log.info(f"Batch {i + 1} / {batches}")
            if i % tools.size != tools.rank:
                continue
            elif i >= batches:
                break

            with torch.no_grad():
                output = self.network_manager.model(content, style_id)
            content_pictures.extend(tools.prepare_imgs(content))
            output_pictures.extend(tools.prepare_imgs(output))
            style_pictures.extend(tools.prepare_imgs(style))
        log.info("Saving on disk...")

        with tools.Lock(0):
            self.save_output(output_pictures)
        with tools.Lock(1):
            self.save_side_by_side(
                content_pictures, style_pictures, output_pictures
            )

    def format_idx(self, idx):
        # avoiding collision between processes
        idx = idx * tools.size + tools.rank

        nb_images = self.cfg.generation.nb_images
        n = math.ceil(math.log10(nb_images))
        S = str(idx)
        while len(S) < n:
            S = "0" + S
        return S

    def save_side_by_side(self, content, style, output):
        with zipfile.ZipFile("images_sbs.zip", "a") as myzip:
            assert len(content) == len(style) == len(output)
            for i in range(len(content)):
                img = np.concatenate((content[i], style[i], output[i]), axis=1)
                filename = f"img{self.format_idx(i)}_sbs.jpg"
                self.save_in_zip(img, myzip, filename)

    def save_output(self, pictures):
        with zipfile.ZipFile("images.zip", "a") as myzip:
            for i, img in enumerate(pictures):
                filename = f"img{self.format_idx(i)}.jpg"
                self.save_in_zip(img, myzip, filename)

    def save_in_zip(self, img, zip, filename):
        im = Image.fromarray(img)
        im.save(filename)
        zip.write(filename)
        os.remove(filename)
