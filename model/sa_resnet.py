import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models.resnet as resnet

from sa_layer import SelfAttentionConv2d, SelfAttentionBlock

model_dict = {
    'sa_resnet18': (sa_resnet, [2, 2, 2, 2]),
    'sa_resnet34': (sa_resnet, [2, 2, 2, 2]),
    'sa_resnet50': (sa_resnet, [2, 2, 2, 2]),
    'sa_resnet101': (sa_resnet, [2, 2, 2, 2]),
    'sa_resnet152': (sa_resnet, [2, 2, 2, 2]),
    'cstem_sa_resnet18': (cstem_sa_resnet, [2, 2, 2, 2]),
    'cstem_sa_resnet34': (cstem_sa_resnet, [2, 2, 2, 2]),
    'cstem_sa_resnet50': (cstem_sa_resnet, [2, 2, 2, 2]),
    'cstem_sa_resnet101': (cstem_sa_resnet, [2, 2, 2, 2]),
    'cstem_sa_resnet152': (cstem_sa_resnet, [2, 2, 2, 2]),
}

model_names = list(model_dict.keys())

def get_model(arch, **kwargs):
    model_fn, layers = model_dict[arch]
    return model_fn(SelfAttentionBlock, layers, **kwargs)

def sa_resnet(block, layers, **kwargs):
    model = resnet.ResNet(block, layers, **kwargs)
    return model

def cstem_sa_resnet(block, layers, **kwargs):
    model = resnet.ResNet(block, layers, **kwargs)
    return model