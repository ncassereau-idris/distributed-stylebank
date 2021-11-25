# /usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict
from dataclasses import dataclass, field
import functools
import torch.distributed as dist
from . import tools


@dataclass
class TrainingConf:

    train: bool = field(default=False, metadata={
        "help": "Whether or not the model should train"
    })

    consecutive_style_step: int = field(default=1, metadata={
        "help": (
            "Stylebank uses a (T+1)-steps training strategy where the "
            "style branch is trained T consecutive steps before the "
            "autoencoder branch is trained for one step."
        )
    })

    batch_size: int = field(default=4, metadata={
        "help": "Batch size used during training per GPU"
    })

    repeat: int = field(default=1, metadata={
        "help": (
            "number of times the dataset is used in a single epoch."
            "It is equivalent to using more epochs, but in the case "
            "where you have a small dataset and a big batch size which "
            "does not divide the size of the dataset, it smoothes the batches"
        )
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

    epochs: int = field(default=1, metadata={
        "help": "Number of epochs for the training session"
    })

    log_interval: int = field(default=100, metadata={
        "help": "Trainer will log every n steps"
    })

    save_interval: int = field(default=1000, metadata={
        "help": "Trainer will save model every n steps"
    })

    adjust_learning_rate_interval: int = field(default=1000, metadata={
        "help": "Trainer will decay learning rate every n steps"
    })


@dataclass
class VGGConf:

    store: bool = field(default=False, metadata={
        "help": (
            "whether or not content losses and style losses should"
            "store targets for faster feed-forward"
        )
    })

    content: Dict[str, float] = field(default_factory=dict, metadata={
        "help": (
            "Dictionary where keys refer to VGG layers names to consider "
            "for content loss and values the weight for the "
            "aforementioned loss"
        )
    })

    style: Dict[str, float] = field(default_factory=dict, metadata={
        "help": (
            "Dictionary where keys refer to VGG layers names to consider "
            "for style loss and values the weight for the "
            "aforementioned loss"
        )
    })


@dataclass
class DataConf:

    folder: str = field(default="data", metadata={
        "help": "name of the folder where all the data is stored"
    })

    vgg_file: str = field(default="data/vgg.pth", metadata={
        "help": (
            "name of the file in the data folder where the pretrained "
            "VGG weights are stored"
        )
    })

    monet: str = field(default="data/monet_jpg", metadata={
        "help": "name of the folder where real monet paintings are stored"
    })

    style_quantity: int = field(default=-1, metadata={
        "help": "number of monet paintings to consider. -1 uses all of them."
    })

    photo: str = field(default="data/photo_jpg", metadata={
        "help": (
            "name of the folder where real pictures (to be monet-fied)"
            " are stored"
        )
    })

    store_transformed: bool = field(default=False, metadata={
        "help": "whether or not to store transformed data for faster training"
    })

    preload_transformed: bool = field(default=False, metadata={
        "help": (
            "whether or not to preload transformed. "
            "Should only be True if store_transformed is True as well"
        )
    })

    load_model: bool = field(default=False, metadata={
        "help": (
            "whether or not we should load pretrained weights in the "
            "data folder"
        )
    })

    weights_subfolder: str = field(default="weights", metadata={
        "help": "name of the directory of model weights"
    })

    bank_weight_filename: str = field(
        default="weights/bank_{0}.pth",
        metadata={
            "help": "filename for each style bank"
        }
    )

    model_weight_filename: str = field(
        default="weights/model.pth",
        metadata={
            "help": "filename for model weights"
        }
    )

    encoder_weight_filename: str = field(
        default="weights/encoder.pth",
        metadata={
            "help": "filename for encoder weights"
        }
    )

    decoder_weight_filename: str = field(
        default="weights/decoder.pth",
        metadata={
            "help": "filename for decoder weights"
        }
    )


@dataclass
class Configuration:

    training: TrainingConf = TrainingConf()
    vgg_layers: VGGConf = VGGConf()
    data: DataConf = DataConf()

    seed: int = field(default=4, metadata={
        "help": "seed"
    })


class MovingAverage:

    def __init__(self):
        self.reset()

    def reset(self):
        self.data = 0.
        self.n = 0

    def update(self, new_data):
        self.data = self.n * self.data + new_data
        self.n += 1
        self.data /= self.n
        return self.data

    def __str__(self):
        return f"{self.data:.6f}"


@dataclass
class TrainingData:

    style_loss: float = 0.
    content_loss: float = 0.
    total_loss: float = 0.
    reconstruction_loss: float = 0.
    regularizer_loss: float = 0.

    epoch_style_loss = MovingAverage()
    epoch_content_loss = MovingAverage()
    epoch_total_loss = MovingAverage()
    epoch_reconstruction_loss = MovingAverage()
    epoch_regularizer_loss = MovingAverage()

    def reset(self, reset_epoch_average=False):
        if reset_epoch_average:
            self.epoch_style_loss.reset()
            self.epoch_content_loss.reset()
            self.epoch_total_loss.reset()
            self.epoch_reconstruction_loss.reset()
            self.epoch_regularizer_loss.reset()
        else:
            self.epoch_style_loss.update(self.style_loss)
            self.epoch_content_loss.update(self.content_loss)
            self.epoch_total_loss.update(self.total_loss)
            self.epoch_reconstruction_loss.update(self.reconstruction_loss)
            self.epoch_regularizer_loss.update(self.regularizer_loss)

        self.style_loss = 0.
        self.content_loss = 0.
        self.total_loss = 0.
        self.reconstruction_loss = 0.
        self.regularizer_loss = 0.

    def update(
        self,
        style_loss=None,
        content_loss=None,
        total_loss=None,
        reconstruction_loss=None,
        regularizer_loss=None
    ):
        if style_loss is not None:
            self.style_loss += style_loss.item()
        if content_loss is not None:
            self.content_loss += content_loss.item()
        if total_loss is not None:
            self.total_loss += total_loss.item()
        if reconstruction_loss is not None:
            self.reconstruction_loss += reconstruction_loss.item()
        if regularizer_loss is not None:
            self.regularizer_loss += regularizer_loss.item()

    def log(self):
        losses = [
            f"Total loss: {self.total_loss:.6f}",
            f"Content loss: {self.content_loss:.6f}",
            f"Style loss: {self.style_loss:.6f}",
            f"Regularizer loss: {self.regularizer_loss:.6f}",
            f"Reconstruction loss: {self.reconstruction_loss:.6f}"
        ]
        return " | ".join(losses)

    def log_epoch(self):
        losses = [
            f"Total loss: {str(self.epoch_total_loss)}",
            f"Content loss: {str(self.epoch_content_loss)}",
            f"Style loss: {str(self.epoch_style_loss)}",
            f"Regularizer loss: {str(self.epoch_regularizer_loss)}",
            f"Reconstruction loss: {str(self.epoch_reconstruction_loss)}",
        ]
        return " | ".join(losses)

    def __iadd__(self, other):
        self.style_loss += other.style_loss
        self.content_loss += other.content_loss
        self.total_loss += other.total_loss
        self.reconstruction_loss += other.reconstruction_loss
        self.regularizer_loss += other.regularizer_loss
        return self

    def merge(self):
        data = [None] * tools.size
        dist.all_gather_object(data, self)
        return functools.reduce(TrainingData.__iadd__, data)
