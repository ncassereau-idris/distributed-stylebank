# /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.distributed as dist
from torchvision import transforms
import logging
import cv2
import os
from PIL import Image
import numpy as np
from hydra.utils import to_absolute_path
import time
import pathlib
import subprocess
import mlflow
from . import tools


log = logging.getLogger(__name__)


class ImageConverter:

    def __init__(self, cfg, network_manager):
        self.network_manager = network_manager
        self.cfg = cfg

    def __call__(self, image, style_id):
        return self.forward(image, style_id)

    def separate(self, image):
        h, w = image.shape[-2:]
        height_padding = h % 513
        width_padding = w % 513
        qh = height_padding // 2
        qw = width_padding // 2
        cropped = transforms.functional.crop(
            img=image, top=qh, left=qw,
            height=h - height_padding,
            width=w - width_padding
        )
        separated_image = []
        for tensor in torch.split(cropped, 513, dim=2):
            separated_image.extend(torch.split(tensor, 513, dim=1))
        separated_image = torch.stack(separated_image, dim=0)
        cropped = (cropped.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        return cropped, separated_image, (h // 513, w // 513)

    def unite(self, image, layout):
        assert image.shape[0] == (layout[0] * layout[1])
        splitted = np.split(image, layout[1], axis=0)
        united = np.concatenate(
            [np.concatenate(img, axis=0) for img in splitted],
            axis=1
        )
        return united

    def forward(self, image, style_id):
        cropped, content, layout = self.separate(image)
        bsz = content.shape[0]
        with torch.no_grad():
            output = self.network_manager.model(content.cuda(), [style_id] * bsz)
        output = tools.prepare_imgs(output)
        output = np.stack(output)
        output = self.unite(output, layout)
        return cropped, output

    def merge(self, im1, im2):
        # im1 is assumed to be the larger pic
        w1 = im1.shape[1]
        w2 = im2.shape[1]
        diff = w1 - w2
        if diff != 0:
            q_pad, r_pad = divmod(diff, 2)
            im2 = np.pad(im2, [[0, 0], [q_pad, q_pad + r_pad], [0, 0]])
        im = np.concatenate([im1, im2], axis=0)
        return im


class VideoGenerator:

    def __init__(self, cfg, data_manager, network_manager):
        self.cfg = cfg
        self.data_manager = data_manager
        self.network_manager = network_manager
        self.converter = ImageConverter(cfg, network_manager)

    def _frames_generator(self, filename):
        path = to_absolute_path(os.path.join(
            self.cfg.data.folder,
            filename
        ))
        cam = cv2.VideoCapture(path)
        fps = cam.get(cv2.CAP_PROP_FPS)
        n_frames = cam.get(cv2.CAP_PROP_FRAME_COUNT)
        yield int(fps), int(n_frames)

        while True:
            ret, frame = cam.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                yield frame
            else:
                break

        cam.release()
        cv2.destroyAllWindows()

    def _generate(self, filename, style_id):
        root_filename = pathlib.Path(filename).stem
        if tools.rank == 0:
            tools.mkdir("video_" + root_filename)
            time.sleep(1)
        dist.barrier()
        path = os.path.join("video_" + root_filename, "pic%09d.jpg")
        frames_gen = self._frames_generator(filename)
        fps, n_frames = next(frames_gen)
        log.info(f"Video at {fps} fps has {n_frames} frames")

        for i, frame in enumerate(frames_gen):
            if i % 10 == 0:
                log.info(f"{filename} - Frame {i} / {n_frames}")
            if i % tools.size != tools.rank:
                continue
            tensor = transforms.functional.to_tensor(frame)
            cropped, converted = self.converter(tensor, style_id)
            merged_im = self.converter.merge(cropped, converted)
            image = Image.fromarray(merged_im)
            image.save(path % i)

        if tools.rank == 0:
            log.info("Concatenating frames...")
            directory = pathlib.Path().resolve()
            command = (
                f"ffmpeg -framerate {fps} -i {directory / path} -vf "
                f"\"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p {filename}"
            )
            subprocess.run(command, shell=True)
        dist.barrier()
        if tools.rank == 0:
            mlflow.log_artifact(
                filename, artifact_path=os.path.join("videos", filename)
            )

    def generate(self):
        for filename, style_id in self.cfg.generation.files.items():
            log.info(f"Converting {filename}")
            self._generate(filename, style_id)
            log.info(f"{filename} has been converted!")
        log.info("Video generation done!")
        dist.barrier()
        if tools.rank == 0:
            mlflow.log_artifact("main.log")
