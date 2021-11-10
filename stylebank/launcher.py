# /usr/bin/env python3
# -*- coding: utf-8 -*-

from datasets import DataManager
from networks import NetworkManager
from trainer import Trainer


def main(cfg):
    data_manager = DataManager(cfg)
    network_manager = NetworkManager(cfg)

    if cfg.training.train:
        Trainer(cfg, data_manager, network_manager).train()
