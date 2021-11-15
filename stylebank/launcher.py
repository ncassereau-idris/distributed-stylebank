# /usr/bin/env python3
# -*- coding: utf-8 -*-

from omegaconf import OmegaConf
import torch
import horovod.torch as hvd
import logging
import numpy as np
from .datasets import DataManager
from .networks import NetworkManager
from .trainer import Trainer

log = logging.getLogger(__name__)

class Rank0Filter(logging.Filter):
    def filter(self, record):
        """
        Only allows the rank 0 to log
        """
        return hvd.rank() == 0


def init(cfg):
    hvd.init()

    for handler in logging.root.handlers:
        handler.addFilter(Rank0Filter())

    log.info("\n" + OmegaConf.to_yaml(cfg) + "\n")
    log.info(" | ".join([
        "Horovod initialized",
        f"World size: {hvd.size()}",
        f"GPUs per node: {hvd.local_size()}"
    ]))

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(cfg.seed)


def launch(cfg):
    data_manager = DataManager(cfg)
    network_manager = NetworkManager(cfg, len(data_manager.style_dataset))

    if cfg.training.train:
        Trainer(cfg, data_manager, network_manager).train()
