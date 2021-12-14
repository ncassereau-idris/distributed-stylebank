# /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.distributed as dist
import numpy as np
import math
import logging
from PIL import Image
import os
from . import tools


log = logging.getLogger(__name__)


class Generator:

    def __init__(self, cfg, data_manager, network_manager):
        self.cfg = cfg
        self.data_manager = data_manager
        self.network_manager = network_manager

        self.idx_1 = tools.rank
        self.idx_2 = tools.rank
        self.length_idx = math.ceil(math.log10(self.cfg.generation.nb_images))

    def generate(self):
        if tools.rank == 0:
            tools.mkdir("images")
            tools.mkdir("images_sbs")
        dist.barrier()
        log.info("Generating images")
        q, r = divmod(
            self.cfg.generation.nb_images,
            self.cfg.training.batch_size
        )
        batches = q if r == 0 else q + 1
        iterator = enumerate(self.data_manager.make_generation_dataloader())
        for i, ((_, content), (style_id, style)) in iterator:
            if i >= batches:
                break
            log.info(f"Batch {i + 1} / {batches}")
            if i % tools.size != tools.rank:
                continue

            with torch.no_grad():
                output = self.network_manager.model(content, style_id)
                content = tools.prepare_imgs(content)
                output = tools.prepare_imgs(output)
                style = tools.prepare_imgs(style)

            self.save_side_by_side(content, style, output)
            self.save_output(output)
        dist.barrier()
        log.info("Paintings generation done!")

    def format_idx(self, idx):
        S = str(idx)
        while len(S) < self.length_idx:
            S = "0" + S
        return S

    def save_side_by_side(self, content, style, output):
        assert len(content) == len(style) == len(output)
        for i in range(len(content)):
            img = np.concatenate((content[i], style[i], output[i]), axis=1)
            filename = f"img{self.format_idx(self.idx_1)}_sbs.jpg"
            self.save(img, "images_sbs", filename, resize=False)
            self.idx_1 += tools.size

    def save_output(self, pictures):
        for i, img in enumerate(pictures):
            filename = f"img{self.format_idx(self.idx_2)}.jpg"
            self.save(img, "images", filename, resize=True)
            self.idx_2 += tools.size

    def save(self, img, folder, filename, resize=False):
        path = os.path.join(folder, filename)
        im = Image.fromarray(img)
        if resize:
            im = im.resize((256, 256))
        im.save(path)
