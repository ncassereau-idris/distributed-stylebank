# /usr/bin/env python3
# -*- coding: utf-8 -*-

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import logging
import numpy as np
import subprocess
import os
from pyarrow import plasma
from .datasets import DataManager
from .networks import NetworkManager
from .trainer import Trainer
from . import tools


log = logging.getLogger(__name__)
server = None


class Rank0Filter(logging.Filter):
    def filter(self, record):
        """
        Only allows the rank 0 to log
        """
        return tools.rank == 0


def init(cfg):
    dist.init_process_group(
        backend='nccl',
        world_size=tools.size, 
        rank=tools.rank
    )

    for handler in logging.root.handlers:
        handler.addFilter(Rank0Filter())

    log.info("\n" + OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(tools.local_rank)
        torch.cuda.manual_seed(cfg.seed)

    log.info(f"Torch initialized | World size: {tools.size}")

    global server
    if tools.rank == 0:
        GB100 = 100 * (1024 ** 3)
        server = subprocess.Popen(
            ["plasma_store", "-m", str(GB100), "-s", "/tmp/plasma"]
        )
    log.info(f"Plasma store initialized")


def launch(cfg):
    data_manager = DataManager(cfg)
    network_manager = NetworkManager(cfg, len(data_manager.style_dataset))

    if cfg.training.train:
        Trainer(cfg, data_manager, network_manager).train()


def cleanup(cfg):
    global server
    # assert (tools.rank == 0) == (server is not None)
    if server is not None:
        server.kill()