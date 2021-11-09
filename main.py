# /usr/bin/env python3
# -*- coding: utf-8 -*-

from omegaconf import OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
from stylebank.dataclasses import Configuration


cs = ConfigStore.instance()
cs.store(name="base_config", node=Configuration)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: Configuration) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
