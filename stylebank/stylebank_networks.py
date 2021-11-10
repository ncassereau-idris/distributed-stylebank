# /usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

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
            Sequential(
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
