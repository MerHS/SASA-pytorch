import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.utils import _pair

class SelfAttentionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1, bias=True):
        super(SelfAttentionConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.groups = groups

        self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.rel_size = out_channels // 2
        self.relative_x = [nn.Parameter(torch.Tensor(self.rel_size)) for _ in self.kernel_size[1]]
        self.relative_y = [nn.Parameter(torch.Tensor(out_channels - self.rel_size)) for _ in self.kernel_size[0]]

        self.weight_query = nn.Parameter(torch.Tensor(groups, in_channels // groups, out_channels // groups))
        self.weight_key = nn.Parameter(torch.Tensor(groups, in_channels // groups, out_channels // groups))
        self.weight_value = nn.Parameter(torch.Tensor(groups, in_channels // groups, out_channels // groups))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_query, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_key, a=math.sqrt(5))
        init.kaiming_uniform_(self.weight_value, a=math.sqrt(5))

        if self.bias is not None:
            bound = 1 / math.sqrt(self.out_channels)
            init.uniform_(self.bias, -bound, bound)

        rel_bound = 1 / math.sqrt(self.rel_size)
        for rel_x in self.relative_x:
            init.uniform_(rel_x, -rel_bound, rel_bound)
        for rel_y in self.relative_y:
            init.uniform_(rel_y, -rel_bound, rel_bound)

    def forward(self, x):
        b, c, h, w = x.size()
        kh, kw = self.kernel

        fh = (h + self.padding[0] * 2 - self.kernel[0]) // self.stride[0] + 1
        fw = (w + self.padding[1] * 2 - self.kernel[1]) // self.stride[1] + 1
        fc = self.out_channels

        # TODO: check this could be moved to init
        relative_pos = []
        for rel_x in self.relative_x:
            for rel_y in self.relative_y:
                relative_pos.append(torch.cat(rel_x, rel_y))
        relative_pos = torch.stack(relative_pos, dim=0).view(self.kernel_size[0]*self.kernel_size[1], self.out_channels)

        win = F.unfold(x, self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        win = win.transpose(2, 1).view(b, fh*fw, c, kh*kw).transpose(3, 2).view(b, fh*fw, kh, kw, c)
        win_query = win[:, :, (kh-1)//2, (kw-1)//2, :]

        v_q = win_query.matmul(self.weight_query).view(b, fh*fw, 1, fc)
        v_k = win.matmul(self.weight_key).view(b, fh*fw, kh*kw, fc)
        v_v = win.matmul(self.weight_value).view(b, fh*fw, kh*kw, fc)

        v_ab = v_q * (v_k + relative_pos)
        v_x = F.softmax(v_ab, dim=2)

        v = (v_x * v_v).sum(dim=2)
        v = v.transpose(2, 1).view(b, fc, fh, fw)

        if self.bias is not None:
            v += self.bias

        return v


class SelfAttentionBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(SelfAttentionBottleneck, self).__init__()

    def forward(self, x):
        return x

class SelfAttentionStem(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(SelfAttentionStem, self).__init__()

    def forward(self, x):
        return x