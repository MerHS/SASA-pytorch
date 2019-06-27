import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def _double(num):
    if type(num) == int:
        return (num, num)
    elif type(num) == tuple and len(num) == 1:
        return (num[0], num[0])
    else:
        return num

class SelfAttentionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(SelfAttentionConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _double(kernel_size)
        self.stride = _double(stride)
        self.padding = _double(padding)
        self.dilation = _double(dilation)
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.rel_size = out_channels // 2
        self.relative_x = nn.Parameter(torch.Tensor(self.rel_size))
        self.relative_y = nn.Parameter(torch.Tensor(out_channels - self.rel_size))

        self.weight_query = [nn.Parameter(torch.Tensor(out_channels // groups, in_channels // groups)) for _ in range(groups)]
        self.weight_key = [nn.Parameter(torch.Tensor(out_channels // groups, in_channels // groups)) for _ in range(groups)]
        self.weight_value = [nn.Parameter(torch.Tensor(out_channels // groups, in_channels // groups)) for _ in range(groups)]

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_query, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_key, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_value, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

        init.uniform_(self.relative_x, 1 / math.sqrt(self.rel_size))
        init.uniform_(self.relative_y, 1 / math.sqrt(self.out_channels - self.rel_size))

    def forward(self, x):
        return x

class SelfAttentionBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(SelfAttentionBlock, self).__init__()

    def forward(self, x):
        return x