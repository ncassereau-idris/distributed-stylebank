# /usr/bin/env python3
# -*- coding: utf-8 -*-

from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
import stylebank.dataclasses as dc
from stylebank.networks import NetworkManager
from stylebank.datasets import DataManager
from stylebank.trainer import Trainer


cs = ConfigStore.instance()
cs.store(name="base_config", node=dc.Configuration)
cs.store(group="training", name="base_training_conf", node=dc.TrainingConf)
cs.store(group="vgg_layers", name="base_vgg_conf", node=dc.VGGConf)
cs.store(group="data", name="base_data_conf", node=dc.DataConf)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: dc.Configuration) -> None:
    print(OmegaConf.to_yaml(cfg))

    Trainer(cfg, DataManager(cfg), NetworkManager(cfg)).train()


if __name__ == "__main__":
    main()
