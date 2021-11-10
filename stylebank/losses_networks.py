# /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from copy import deepcopy
from hydra.utils import to_absolute_path


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


class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def init_vgg(cfg):
    vgg = models.vgg16(pretrained=False)
    path = to_absolute_path(cfg.data.vgg_file)
    vgg.load_state_dict(torch.load(path))
    vgg = vgg.features
    vgg = vgg.eval()
    return vgg


class LossNetwork(nn.Module):

    def __init__(self, cfg, cnn):
        super(LossNetwork, self).__init__()
        cnn = deepcopy(cnn)
        normalization = Normalization()
        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

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

    def forward(self, input, content, style):
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
