# /usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class TrainingConf:

    train: bool = field(default=False, metadata={
        "help": "Whether or not the model should train"
    })

    consecutive_style_step: int = field(default=1, metadata={
        "help": (
            "Stylebank uses a (T+1)-steps training strategy where the style branch is trained T"
            "consecutive steps before the autoencoder branch is trained for one step."
        )
    })

    batch_size: int = field(default=4, metadata={
        "help": "Batch size used during training"
    })

    learning_rate: float = field(default=0.001, metadata={
        "help": "Learning rate applied during training"
    })

    content_weight: float = field(default=1., metadata={
        "help": "Weight of content losses used during training"
    })

    style_weight: float = field(default=1., metadata={
        "help": "Weight of style losses used during training"
    })

    reg_weight: float = field(default=1., metadata={
        "help": "Weight of regularization loss used during training"
    })


@dataclass
class VGGConf:

    content: Dict[str, float] = field(default_factory=dict, metadata={
        "help": (
            "Dictionary where keys refer to VGG layers names to consider for content loss "
            "and values the weight for the aforementioned loss"
        )
    })

    style: Dict[str, float] = field(default_factory=dict, metadata={
        "help": (
            "Dictionary where keys refer to VGG layers names to consider for style loss "
            "and values the weight for the aforementioned loss"
        )
    })


@dataclass
class DataConf:

    folder: str = field(default="data", metadata={
        "help": "name of the folder where all the data is stored"
    })

    vgg_file: str = field(default="data/vgg.pth", metadata={
        "help": "name of the file in the data folder where the pretrained VGG weights are stored"
    })

    monet: str = field(default="data/monet_jpg", metadata={
        "help": "name of the folder where real monet paintings are stored"
    })

    style_quantity: int = field(default=-1, metadata={
        "help": "number of monet paintings to consider. -1 uses all of them."
    })

    photo: str = field(default="data/photo_jpg", metadata={
        "help": "name of the folder where real pictures (to be monet-fied) are stored"
    })

    load_model: bool = field(default=False, metadata={
        "help": "whether or not we should load pretrained weights in the data folder"
    })

    weights_subfolder: str = field(default="weights", metadata={
        "help": "name of the directory of model weights"
    })

    bank_weight_filename: str = field(default="weights/bank_{0}.pth", metadata={
        "help": "filename for each style bank"
    })

    model_weight_filename: str = field(default="weights/model.pth", metadata={
        "help": "filename for model weights"
    })

    encoder_weight_filename: str = field(default="weights/encoder.pth", metadata={
        "help": "filename for encoder weights"
    })

    decoder_weight_filename: str = field(default="weights/decoder.pth", metadata={
        "help": "filename for decoder weights"
    })


@dataclass
class Configuration:

    training: TrainingConf = TrainingConf()
    vgg_layers: VGGConf = VGGConf()
    data: DataConf = DataConf()
