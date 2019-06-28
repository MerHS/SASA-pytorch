import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models.resnet as resnet

from sa_layer import SelfAttentionConv2d, SelfAttentionBottleneck

model_dict = {
    'resnet26': (bottleneck_resnet, resnet.Bottleneck, [1, 2, 4, 1]),
    'resnet38': (bottleneck_resnet, resnet.Bottleneck, [2, 3, 5, 2]),
    'sa_resnet26': (sa_resnet, SelfAttentionBottleneck, [1, 2, 4, 1]),
    'sa_resnet38': (sa_resnet, SelfAttentionBottleneck, [2, 3, 5, 2]),
    'sa_resnet50': (sa_resnet, SelfAttentionBottleneck, [3, 4, 6, 3]),
    'sa_resnet101': (sa_resnet, SelfAttentionBottleneck, [3, 4, 23, 3]),
    'sa_resnet152': (sa_resnet, SelfAttentionBottleneck, [3, 8, 36, 3]),
    'cstem_sa_resnet26': (cstem_sa_resnet, SelfAttentionBottleneck, [1, 2, 4, 1]),
    'cstem_sa_resnet38': (cstem_sa_resnet, SelfAttentionBottleneck, [2, 3, 5, 2]),
    'cstem_sa_resnet50': (cstem_sa_resnet, SelfAttentionBottleneck, [3, 4, 6, 3]),
    'cstem_sa_resnet101': (cstem_sa_resnet, SelfAttentionBottleneck, [3, 4, 23, 3]),
    'cstem_sa_resnet152': (cstem_sa_resnet, SelfAttentionBottleneck, [3, 8, 36, 3]),
}

model_names = list(model_dict.keys())

def get_model(arch, **kwargs):
    model_fn, block, layers = model_dict[arch]
    return model_fn(block, layers, **kwargs)

def bottleneck_resnet(block, layers, **kwargs):
    return resnet.ResNet(block, layers, **kwargs)

def sa_resnet(block, layers, **kwargs):
    return resnet.ResNet(block, layers, **kwargs)

def cstem_sa_resnet(block, layers, **kwargs):
    return resnet.ResNet(block, layers, **kwargs)