# /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import horovod.torch as hvd
from copy import deepcopy
from hydra.utils import to_absolute_path
import os
import logging


log = logging.getLogger(__name__)


class ContentLoss(nn.Module):

    def __init__(self, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        # self.target = target
        self.target = None
        self.mode = 'learn'
        self.weight = weight

    def forward(self, input):
        if self.mode == 'loss':
            self.loss = self.weight * F.mse_loss(input, self.target)
        elif self.mode == 'learn':
            self.target = input.detach()
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, weight):
        super(StyleLoss, self).__init__()
        self.target = None
        self.mode = 'learn'
        self.weight = weight

    def forward(self, input):
        if self.mode == 'loss':
            G = gram_matrix(input)
            self.loss = self.weight * F.mse_loss(G, self.target)
        elif self.mode == 'learn':
            G = gram_matrix(input)
            self.target = G.detach()
        return input


def init_vgg(cfg):
    vgg = models.vgg16(pretrained=False)
    path = to_absolute_path(cfg.data.vgg_file)
    vgg.load_state_dict(torch.load(path))
    vgg = vgg.features
    vgg = vgg.cuda()
    vgg = vgg.eval()
    return vgg


class LossNetwork(nn.Module):

    def __init__(self, cfg, cnn):
        super(LossNetwork, self).__init__()
        cnn = deepcopy(cnn)
        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential()

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in cfg.vgg_layers.content.keys():
                content_loss = ContentLoss(weight=cfg.vgg_layers.content[name])
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in cfg.vgg_layers.style.keys():
                style_loss = StyleLoss(weight=cfg.vgg_layers.style[name])
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        self.model = model
        self.style_losses = style_losses
        self.content_losses = content_losses

    def learn_content(self, input):
        for cl in self.content_losses:
            cl.mode = 'learn'
        for sl in self.style_losses:
            sl.mode = 'nop'
        self.model(input)

    def learn_style(self, input):
        for cl in self.content_losses:
            cl.mode = 'nop'
        for sl in self.style_losses:
            sl.mode = 'learn'
        self.model(input)

    def forward(self, input, content, style=None):
        if style is None:  # auto encoder branch
            return F.mse_loss(input, content)

        # style bank branch
        self.learn_content(content)
        self.learn_style(style)

        for cl in self.content_losses:
            cl.mode = 'loss'
        for sl in self.style_losses:
            sl.mode = 'loss'
        self.model(input)

        content_loss = 0
        style_loss = 0

        for cl in self.content_losses:
            content_loss += cl.loss
        for sl in self.style_losses:
            style_loss += sl.loss

        return content_loss, style_loss


class StyleBankNet(nn.Module):
    def __init__(self, total_style):
        super(StyleBankNet, self).__init__()
        self.total_style = total_style

        self.encoder_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(9, 9), stride=2, padding=(4, 4), bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=(1, 1), bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=1, padding=(1, 1), bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, padding=(1, 1), bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=(9, 9), stride=2, padding=(4, 4), bias=False),
        )

        self.style_bank = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.InstanceNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
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

    def __init__(self, cfg):
        self.cfg = cfg

        self.model = StyleBankNet(cfg.data.style_quantity).cuda()
        if cfg.data.load_model and hvd.rank() == 0:
            self.load_model()

        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)

        if cfg.training.train:
            cnn = init_vgg(cfg)
            self.loss_network = LossNetwork(cfg, cnn).cuda()

    def save_models(self):
        if hvd.rank() != 0:
            log.info("Not rank 0, model not saved")
        try:
            os.mkdir(self.cfg.data.weights_subfolder)
        except FileExistsError:  # folder already exists
            pass
        else:
            log.info("Created a weights subfolder to store model weights")
        log.info("Storing model weights...")
        torch.save(self.model.state_dict(), self.cfg.data.model_weight_filename)
        torch.save(self.model.encoder_net.state_dict(), self.cfg.data.encoder_weight_filename)
        torch.save(self.model.decoder_net.state_dict(), self.cfg.data.decoder_weight_filename)
        for i in range(len(self.model.style_bank)):
            torch.save(
                self.model.style_bank[i].state_dict(),
                self.cfg.data.bank_weight_filename.format(i)
            )
        log.info("Model saved!")

    def load_model(self):
        log.info("Loading model...")
        self.model.load_state_dict(torch.load(
            to_absolute_path(os.path.join(self.cfg.data.folder, self.cfg.data.model_weight_filename))
        ))
        log.info("Model loaded!")
