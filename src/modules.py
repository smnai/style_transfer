# Standard Python
import os
import copy
import logging

# Thirdparty
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import nn
from torchvision.models import vgg19
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize
from torchvision.transforms import functional as F

logging.basicConfig(level=logging.INFO, format='\n%(message)s\n')
logging.getLogger().setLevel(logging.INFO)


class StyleTransfer(torch.nn.Module):
    def __init__(self, cfg):
        super(StyleTransfer, self).__init__()
        self.cfg = cfg
        self.params = cfg.params
        self.style_features = []
        self.set_images()
        self.set_net()
        self.set_content_image_features()
        self.set_style_image_gram_matrices()

    def set_images(self):
        trfs = [
            CenterCrop(self.cfg.image_size),
            ToTensor(),
        ]
        if self.cfg.normalize:
            trfs.append(
                Normalize(
                    mean=self.cfg.norm_mean,
                    std=self.cfg.norm_std,
                ))
        transforms = Compose(trfs)
        self.content_image = transforms(Image.open(
            self.cfg.content_image_path)).unsqueeze(0).requires_grad_(
                False).to(self.cfg.device)
        self.style_image = transforms(Image.open(
            self.cfg.style_image_path)).unsqueeze(0).requires_grad_(False).to(
                self.cfg.device)

    def set_net(self):
        self.net = nn.Sequential(*[
            l if type(l) is not torch.nn.modules.pooling.MaxPool2d else nn.
            AvgPool2d(kernel_size=2, stride=2, padding=0)
            for l in vgg19(pretrained=True).features.children()
        ]).to(self.cfg.device).eval()

        for p in self.net.parameters():
            p.requires_grad = False

        self.net._modules[str(
            self.cfg.content_layer_idx)].register_forward_hook(
                self.get_content_features)

        c = 0
        for idx, layer in self.net._modules.items():
            if type(layer) is torch.nn.modules.conv.Conv2d:
                layer.register_forward_hook(self.get_style_features)
                c += 1
                if c == self.cfg.num_style_layers:
                    break

    def get_style_features(self, module, input, output):
        bs, fs, h, w = output.shape
        layer_style_features = output.view(fs, h * w)
        style_features = layer_style_features.unsqueeze(0)
        self.style_features.append(style_features)

    def get_content_features(self, module, input, output):
        bs, fs, h, w = output.shape
        layer_content_features = output.view(fs, h * w)
        self.content_features = layer_content_features.unsqueeze(0)

    def set_content_image_features(self):
        self.net(self.content_image)
        self.content_image_features = self.content_features

    def set_style_image_gram_matrices(self):
        self.style_features.clear()
        self.net(self.style_image)
        self.style_image_gram_matrices = [
            torch.mm(l.squeeze(0),
                     l.squeeze(0).t()).div(
                         l.squeeze(0).shape[0] *
                         l.squeeze(0).shape[1]).clone().detach().to(
                             self.cfg.device)
            for idx, l in enumerate(self.style_features)
        ]

    def forward(self, x):
        del self.content_features
        self.style_features.clear()
        self.net(x)
