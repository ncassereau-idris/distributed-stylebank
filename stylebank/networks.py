# /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchvision.models as models
from copy import deepcopy
from hydra.utils import to_absolute_path
import os
import logging
import pickle
from . import tools
from .plasma import PlasmaStorage


log = logging.getLogger(__name__)


class ContentLoss(nn.Module):

    def __init__(self, weight, store=False, name=None):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        # self.target = target
        self.target = None
        self.mode = 'learn'
        self.weight = weight
        self.name = name

        self.store = store
        self.target_ids = None
        if store:
            self.storage = PlasmaStorage(autocuda=True, name=name)

    def forward(self, input):
        if self.mode == 'loss':
            target = (
                self.storage[self.target_ids]
                if self.store and self.target_ids is not None
                else self.target
            )
            self.loss = self.weight * F.mse_loss(input, target)
        elif self.mode == 'learn':
            self.target = input.detach()
            if self.store and self.target_ids is not None:
                self.storage[self.target_ids] = self.target
        return input


@autocast(enabled=False)
def gram_matrix(input):
    if input.dtype is torch.half:
        input = input.float()
    bsz, channels, height, width = input.size()
    features = input.view(bsz * channels, height * width)
    G = torch.mm(features, features.t())
    return G.div(bsz * channels * height * width)


class StyleLoss(nn.Module):

    def __init__(self, weight, store=False, name=None):
        super(StyleLoss, self).__init__()
        self.target = None
        self.mode = 'learn'
        self.weight = weight
        self.name = name

        self.store = store
        self.target_ids = None
        if store:
            self.storage = PlasmaStorage(autocuda=True, name=name)

    def forward(self, input):
        if self.mode == 'loss':
            G = gram_matrix(input)
            target = (
                self.storage[self.target_ids]
                if self.store and self.target_ids is not None
                else self.target
            )
            G_target = gram_matrix(target)
            self.loss = self.weight * F.mse_loss(G, G_target)
        elif self.mode == 'learn':
            if self.store and self.target_ids is not None:
                self.storage[self.target_ids] = input.detach()
            else:
                self.target = input.detach()
        return input


def init_vgg(cfg):
    vgg = models.vgg16(pretrained=False)
    path = to_absolute_path(cfg.data.vgg_file)
    vgg.load_state_dict(torch.load(path))
    vgg = vgg.features
    vgg = vgg.cuda()
    vgg = vgg.eval()
    return vgg


class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        mean = torch.tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.tensor([0.229, 0.224, 0.225]).cuda()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class LossNetwork(nn.Module):

    def __init__(self, cfg, cnn):
        super(LossNetwork, self).__init__()
        self.cfg = cfg
        cnn = deepcopy(cnn)
        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(Normalization())

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError(
                    'Unrecognized layer: {}'.format(layer.__class__.__name__)
                )

            model.add_module(name, layer)

            if name in cfg.vgg_layers.content.keys():
                content_loss = ContentLoss(
                    weight=cfg.vgg_layers.content[name],
                    store=cfg.vgg_layers.store,
                    name="content_"+name
                )
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in cfg.vgg_layers.style.keys():
                style_loss = StyleLoss(
                    weight=cfg.vgg_layers.style[name],
                    store=cfg.vgg_layers.store,
                    name="style_"+name
                )
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], (ContentLoss, StyleLoss)):
                break

        model = model[:(i + 1)]

        self.model = model
        self.style_losses = style_losses
        self.content_losses = content_losses

        self.known_contents = set()
        self.known_styles = set()

    def learn_content(self, input, target_ids=None):
        if (
            self.cfg.vgg_layers.store and
            target_ids is not None and
            set(target_ids).issubset(self.known_contents)
        ):
            # it has already been computed
            return

        for cl in self.content_losses:
            cl.mode = 'learn'
            cl.target_ids = target_ids
        if target_ids is not None:
            self.known_contents.update(target_ids)
        for sl in self.style_losses:
            sl.mode = 'nop'
        self.model(input)

    def learn_style(self, input, target_ids=None):
        if (
            self.cfg.vgg_layers.store and
            target_ids is not None and
            set(target_ids).issubset(self.known_styles)
        ):
            # it has already been computed
            return

        for cl in self.content_losses:
            cl.mode = 'nop'
        for sl in self.style_losses:
            sl.mode = 'learn'
            sl.target_ids = target_ids
        if target_ids is not None:
            self.known_styles.update(target_ids)
        self.model(input)

    def _forward_ae_branch(self, input, content):
        return F.mse_loss(input, content)

    def _forward_style_bank_branch(
        self, input, content, style=None, content_ids=None, style_ids=None
    ):
        if isinstance(content_ids, int):
            content_ids = [content_ids]
        if isinstance(style_ids, int):
            style_ids = [style_ids]

        self.learn_content(content, content_ids)
        self.learn_style(style, style_ids)

        for cl in self.content_losses:
            cl.mode = 'loss'
            cl.target_ids = content_ids
        for sl in self.style_losses:
            sl.mode = 'loss'
            sl.target_ids = style_ids
        self.model(input)

        content_loss = sum([cl.loss for cl in self.content_losses])
        style_loss = sum([sl.loss for sl in self.style_losses])
        return content_loss, style_loss

    def _forward_reg_loss(self, input):
        diff_i = torch.sum(
            torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        )
        diff_j = torch.sum(
            torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        )
        return diff_i + diff_j

    def forward(
        self, input, content, style=None, content_ids=None, style_ids=None
    ):
        if style is None:  # auto encoder branch
            return self._forward_ae_branch(input, content)
        else:  # style bank branch
            content_loss, style_loss = self._forward_style_bank_branch(
                input, content, style, content_ids, style_ids
            )
            tv_loss = self._forward_reg_loss(input)
            content_loss *= self.cfg.training.content_weight
            style_loss *= self.cfg.training.style_weight
            tv_loss *= self.cfg.training.reg_weight

            return content_loss, style_loss, tv_loss

    def preload(self, content_dataloader, style_dataloader):
        for content_ids, content in content_dataloader:
            self.learn_content(content, content_ids)
        dist.barrier()
        for style_ids, style in style_dataloader:
            self.learn_style(style, style_ids)

        for cl in self.content_losses:
            cl.storage.merge()
        for sl in self.style_losses:
            sl.storage.merge()

        self.known_contents.update(range(len(content_dataloader.dataset)))
        self.known_styles.update(range(len(style_dataloader.dataset)))


class StyleBankNet(nn.Module):
    def __init__(self, total_style):
        super(StyleBankNet, self).__init__()
        self.total_style = total_style

        self.encoder_net = nn.Sequential(
            nn.Conv2d(
                3, 32, kernel_size=(9, 9), stride=2,
                padding=(4, 4), bias=False
            ),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                32, 64, kernel_size=(3, 3), stride=2,
                padding=(1, 1), bias=False
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                64, 128, kernel_size=(3, 3), stride=1,
                padding=(1, 1), bias=False
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                128, 256, kernel_size=(3, 3), stride=1,
                padding=(1, 1), bias=False
            ),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=(3, 3), stride=1,
                padding=(1, 1), bias=False
            ),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                128, 64, kernel_size=(3, 3), stride=1,
                padding=(1, 1), bias=False
            ),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                64, 32, kernel_size=(3, 3), stride=2,
                padding=(1, 1), bias=False
            ),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                32, 3, kernel_size=(9, 9), stride=2,
                padding=(4, 4), bias=False
            ),
        )

        self.style_bank = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    256, 256, kernel_size=(3, 3), stride=(1, 1),
                    padding=(1, 1), bias=False
                ),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    256, 256, kernel_size=(3, 3), stride=(1, 1),
                    padding=(1, 1), bias=False
                ),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True)
            ) for i in range(total_style)
        ])

    def forward(self, X, style_id=None):
        z = self.encoder_net(X)
        if style_id is not None:
            new_z = []
            for idx, i in enumerate(style_id):
                zs = self.style_bank[i](z[idx].view(1, *z[idx].shape))
                new_z.append(zs)
            z = torch.cat(new_z, dim=0)
        return self.decoder_net(z)


class NetworkManager:

    def __init__(self, cfg, style_quantity):
        self.cfg = cfg
        self.model = StyleBankNet(style_quantity).cuda()

        if cfg.data.load_model:
            self.load_model()
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[tools.local_rank],
            find_unused_parameters=True
        )

        if cfg.training.train:
            cnn = init_vgg(cfg)
            self.loss_network = LossNetwork(cfg, cnn).cuda()

    def save_models(self, epoch, training_data):
        if tools.rank != 0:
            return
        path = tools.mkdir(self.cfg.data.weights_subfolder, f"epoch_{epoch}")
        log.info("Storing model weights...")

        torch.save(
            self.model.module.state_dict(),
            path / self.cfg.data.model_weight_filename
        )
        torch.save(
            self.model.module.encoder_net.state_dict(),
            path / self.cfg.data.encoder_weight_filename
        )
        torch.save(
            self.model.module.decoder_net.state_dict(),
            path / self.cfg.data.decoder_weight_filename
        )
        for i in range(len(self.model.module.style_bank)):
            torch.save(
                self.model.module.style_bank[i].state_dict(),
                path / self.cfg.data.bank_weight_filename.format(i)
            )

        losses_file = os.path.join(self.cfg.data.weights_subfolder, "losses")
        if os.path.exists(losses_file):
            with open(losses_file, "rb") as file_:
                D = pickle.load(file_)
        else:
            D = list()
        D.append({
            "Total loss": training_data.epoch_total_loss,
            "Content loss": training_data.epoch_content_loss,
            "Style loss": training_data.epoch_style_loss,
            "Regularizer loss": training_data.epoch_regularizer_loss,
            "Reconstruction loss": training_data.epoch_reconstruction_loss
        })
        with open(losses_file, "wb") as file_:
            pickle.dump(D, file_)

        log.info("Model saved!")

    def load_model(self):
        dist.barrier()
        log.info("Loading model...")
        map_location = {'cuda:%d' % 0: 'cuda:%d' % tools.local_rank}
        self.model.load_state_dict(torch.load(
            to_absolute_path(os.path.join(
                self.cfg.data.folder,
                self.cfg.data.weights_subfolder,
                self.cfg.data.model_weight_filename
            )),
            map_location=map_location
        ))
        log.info("Model loaded!")
