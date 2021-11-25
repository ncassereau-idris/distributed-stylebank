# /usr/bin/env python3
# -*- coding: utf-8 -*-

import hydra
from hydra.core.config_store import ConfigStore
from stylebank import (
    init,
    launch,
    cleanup,
    Configuration,
    TrainingConf,
    VGGConf,
    DataConf
)


cs = ConfigStore.instance()
cs.store(name="base_config", node=Configuration)
cs.store(group="training", name="base_training_conf", node=TrainingConf)
cs.store(group="vgg_layers", name="base_vgg_conf", node=VGGConf)
cs.store(group="data", name="base_data_conf", node=DataConf)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: Configuration) -> None:
    init(cfg)
    launch(cfg)
    cleanup(cfg)


if __name__ == "__main__":
    main()
