# /usr/bin/env python3
# -*- coding: utf-8 -*-

from omegaconf import OmegaConf
import torch
import torch.distributed as dist
import logging
import numpy as np
import mlflow
from .datasets import DataManager
from .generator import Generator
from .networks import NetworkManager
from .trainer import Trainer
from . import plasma
from . import tools
from .video import VideoGenerator


log = logging.getLogger(__name__)


class Rank0Filter(logging.Filter):
    def filter(self, record):
        """
        Only allows the rank 0 to log
        """
        return tools.rank == 0


def init_torch(cfg):
    dist.init_process_group(
        backend='nccl',
        world_size=tools.size,
        rank=tools.rank
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(tools.local_rank)
        torch.cuda.manual_seed(cfg.seed)

    log.info(f"Torch initialized | World size: {tools.size}")


def init_mlflow(cfg):
    mlflow.set_tracking_uri(cfg.data.mlflow_uri)
    mlflow.set_experiment(cfg.data.experiment_name)

    # Find the correct name. If it is not given, it could be a datetime and
    # every processes may not agree on this one so, let's just keep the name
    # of the master node
    name_list = [None] * tools.size
    dist.all_gather_object(
        name_list, (tools.rank, cfg.data.run_name)
    )
    name_list = [name for rank, name in name_list if rank == 0]
    assert len(name_list) == 1
    cfg.data.run_name = name_list[0]

    log.info("MLFlow initialized")


def init(cfg):
    for handler in logging.root.handlers:
        handler.addFilter(Rank0Filter())

    log.info("\n" + OmegaConf.to_yaml(cfg))

    init_torch(cfg)
    init_mlflow(cfg)

    plasma.plasma_server.connect()
    log.info("Plasma store initialized")


def mlflow_run_setup(cfg):
    params = {
        "Batch size": cfg.training.batch_size,
        "Learning rate": cfg.training.learning_rate,
        "Content weight": cfg.training.content_weight,
        "Style weight": cfg.training.style_weight,
        "Regularization weight": cfg.training.reg_weight,
        "Consecutive style step": cfg.training.consecutive_style_step
    }
    if tools.rank == 0:
        mlflow.log_params(params)
        mlflow.log_artifacts(".hydra")


def launch(cfg):
    with mlflow.start_run(run_name=cfg.data.run_name):
        mlflow_run_setup(cfg)
        data_manager = DataManager(cfg)
        network_manager = NetworkManager(cfg, len(data_manager.style_dataset))
        if tools.rank == 0:
            mlflow.log_artifact("main.log")

        if cfg.training.train:
            Trainer(cfg, data_manager, network_manager).train()

        if cfg.generation.generate_images:
            Generator(cfg, data_manager, network_manager).generate()

        if cfg.generation.generate_videos:
            VideoGenerator(cfg, data_manager, network_manager).generate()

    if cfg.generation.generate_videos:
        VideoGenerator(cfg, data_manager, network_manager).generate()



def cleanup(cfg):
    plasma.plasma_server.kill()
