'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
import models

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, act):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            act,
            nn.Dropout(),
            nn.Linear(512, 512),
            act,
            nn.Linear(512, 10),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, act, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), act]
            else:
                layers += [conv2d, act]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11(activation, beta, train=False, fix_act=False, fix_act_val=None, **kwargs):
    """VGG 11-layer model (configuration "A")"""
    act = models.__dict__[activation](beta=beta)
    if train and fix_act:
       act.alpha = nn.Parameter(torch.tensor([fix_act_val]))
       act.alpha.requires_grad=False
    return VGG(make_layers(cfg['A'], act), act)


def vgg11_bn(activation, beta, train=False, fix_act=False, fix_act_val=None, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    act = models.__dict__[activation](beta=beta)
    if train and fix_act:
       act.alpha = nn.Parameter(torch.tensor([fix_act_val]))
       act.alpha.requires_grad=False
    return VGG(make_layers(cfg['A'], act, batch_norm=True), act)


def vgg13(activation, beta, train=False, fix_act=False, fix_act_val=None, **kwargs):
    """VGG 13-layer model (configuration "B")"""
    act = models.__dict__[activation](beta=beta)
    if train and fix_act:
       act.alpha = nn.Parameter(torch.tensor([fix_act_val]))
       act.alpha.requires_grad=False
    return VGG(make_layers(cfg['B'], act), act)


def vgg13_bn(activation, beta, train=False, fix_act=False, fix_act_val=None, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    act = models.__dict__[activation](beta=beta)
    if train and fix_act:
       act.alpha = nn.Parameter(torch.tensor([fix_act_val]))
       act.alpha.requires_grad=False
    return VGG(make_layers(cfg['B'], act, batch_norm=True), act)


def vgg16(activation, beta, train=False, fix_act=False, fix_act_val=None, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    act = models.__dict__[activation](beta=beta)
    if train and fix_act:
       act.alpha = nn.Parameter(torch.tensor([fix_act_val]))
       act.alpha.requires_grad=False
    return VGG(make_layers(cfg['D'], act), act)


def vgg16_bn(activation, beta, train=False, fix_act=False, fix_act_val=None, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    act = models.__dict__[activation](beta=beta)
    if train and fix_act:
       act.alpha = nn.Parameter(torch.tensor([fix_act_val]))
       act.alpha.requires_grad=False
    return VGG(make_layers(cfg['D'], act, batch_norm=True), act)


def vgg19(activation, beta, train=False, fix_act=False, fix_act_val=None, **kwargs):
    """VGG 19-layer model (configuration "E")"""
    act = models.__dict__[activation](beta=beta)
    if train and fix_act:
       act.alpha = nn.Parameter(torch.tensor([fix_act_val]))
       act.alpha.requires_grad=False
    return VGG(make_layers(cfg['E'], act), act)


def vgg19_bn(activation, beta, train=False, fix_act=False, fix_act_val=None, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    act = models.__dict__[activation](beta=beta)
    if train and fix_act:
       act.alpha = nn.Parameter(torch.tensor([fix_act_val]))
       act.alpha.requires_grad=False
    return VGG(make_layers(cfg['E'], act, batch_norm=True), act)
