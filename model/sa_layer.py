import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.utils import _pair
from torchvision.models.resnet import conv1x1


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
        self.groups = groups # multi-head count

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # relative position offsets are shared between multi-heads
        self.rel_size = (out_channels // groups) // 2
        self.relative_x = nn.Parameter(torch.Tensor(self.rel_size, 1, self.kernel_size[1]))
        self.relative_y = nn.Parameter(torch.Tensor((out_channels // groups) - self.rel_size, self.kernel_size[0], 1))

        self.weight_query = nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False)
        self.weight_key = nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False)
        self.weight_value = nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight_query.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.weight_key.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.weight_value.weight, mode='fan_out', nonlinearity='relu')

        if self.bias is not None:
            bound = 1 / math.sqrt(self.out_channels)
            init.uniform_(self.bias, -bound, bound)

        rel_bound = 1 / math.sqrt(self.rel_size)
        init.uniform_(self.relative_x, -rel_bound, rel_bound)
        init.uniform_(self.relative_y, -rel_bound, rel_bound)

    def forward(self, x):
        b, c, h, w = x.size()
        kh, kw = self.kernel_size
        ph, pw = h + self.padding[0] * 2, w + self.padding[1] * 2

        fh = (ph - kh) // self.stride[0] + 1
        fw = (pw - kw) // self.stride[1] + 1
        fc = self.out_channels

        # TODO: check this could be moved to init
        rel_x = self.relative_x.repeat(1, kh, 1)
        rel_y = self.relative_y.repeat(1, 1, kw)
        relative_pos = torch.cat([rel_x, rel_y], dim=0).repeat(self.groups, 1, 1).view(fc, kh*kw, 1)

        px, py = self.padding
        x = F.pad(x, (py, py, px, px))

        vq = self.weight_query(x)
        vk = self.weight_key(x)
        vv = self.weight_value(x)

        win_k = F.unfold(vk, (kh, kw), stride=self.stride).view(b, fc, kh*kw, fh*fw)
        win_q = F.unfold(vq, (kh, kw), stride=self.stride).view(b, fc, kh, kw, fh*fw)
        win_v = F.unfold(vv, (kh, kw), stride=self.stride).view(b, fc, kh*kw, fh*fw)

        win_q = win_q[:, :, (kh-1)//2, (kw-1)//2, :].view(b, fc, 1, fh*fw)
        vx = (win_q * (win_k + relative_pos)).sum(dim=1) # (b, kh*kw, fh*fw)

        vx = F.softmax(vx, dim=1).unsqueeze(1) # (b, 1, kh*kw, fh*fw)

        v = (vx * win_v).sum(dim=2) # (b, c2, kh*kw, fh*fw) -> (b, c2, fh*fw)

        if self.bias is not None:
            v += self.bias.view(1, -1, 1)

        fin_v = v.view(b, fc, fh, fw)

        return fin_v

class SAMixtureConv2d(nn.Module):
    """ spatially-aware SA / multiple value transformation for stem layer """
    def __init__(self, in_height, in_width, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1, mix=4, bias=True):
        super(SAMixtureConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.in_height = in_height
        self.in_width = in_width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.groups = groups # multi-head count
        self.mix = mix # weight mixture

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        # relative position offsets are shared between multi-heads
        self.rel_size = (out_channels // groups) // 2
        self.relative_x = nn.Parameter(torch.Tensor(self.rel_size, 1, self.kernel_size[1]))
        self.relative_y = nn.Parameter(torch.Tensor((out_channels // groups) - self.rel_size, self.kernel_size[0], 1))

        self.weight_query = nn.Parameter(torch.Tensor(groups, in_channels // groups, out_channels // groups))
        self.weight_key = nn.Parameter(torch.Tensor(groups, in_channels // groups, out_channels // groups))
        self.weight_value = [nn.Parameter(torch.Tensor(groups, in_channels // groups, out_channels // groups)) for _ in range(mix)]

        self.emb_x = nn.Parameter(torch.Tensor(out_channels // groups, 1,                          in_width + 2 * padding[1]))
        self.emb_y = nn.Parameter(torch.Tensor(out_channels // groups, in_height + 2 * padding[0], 1                        ))
        self.emb_m = nn.Parameter(torch.Tensor(mix, out_channels // groups, 1, 1)) # m, fc/g, 1, 1

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight_query, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.weight_key, mode='fan_out', nonlinearity='relu')
        for wv in self.weight_value:
            init.kaiming_normal_(wv, mode='fan_out', nonlinearity='relu')

        if self.bias is not None:
            bound = 1 / math.sqrt(self.out_channels)
            init.uniform_(self.bias, -bound, bound)

        rel_bound = 1 / math.sqrt(self.rel_size)
        init.uniform_(self.relative_x, -rel_bound, rel_bound)
        init.uniform_(self.relative_y, -rel_bound, rel_bound)

        emb_bound = 1 / math.sqrt(self.out_channels // self.gropus)
        init.uniform_(self.emb_x, -emb_bound, emb_bound)
        init.uniform_(self.emb_y, -emb_bound, emb_bound)
        init.uniform_(self.emb_m, -emb_bound, emb_bound)

    def forward(self, x):
        b, c, h, w = x.size()
        kh, kw = self.kernel_size
        ph, pw = h + self.padding[0] * 2, w + self.padding[1] * 2

        fh = (ph - kh) // self.stride[0] + 1
        fw = (pw - kw) // self.stride[1] + 1
        fc = self.out_channels

        # TODO: check this could be moved to init
        rel_x = self.relative_x.repeat(1, kh, 1)
        rel_y = self.relative_y.repeat(1, 1, kw)
        relative_pos = torch.cat([rel_x, rel_y], dim=0).repeat(self.groups, 1, 1).view(fc, kh*kw, 1)

        px, py = self.padding
        x = F.pad(x, (py, py, px, px))

        x_ij = x.permute(0, 2, 3, 1).view(b, ph*pw, self.groups, 1, c // self.groups) # b, ph*pw, g, 1, c/g

        vq = x_ij.matmul(self.weight_query).permute(0, 2, 3, 4, 1).view(b, fc, ph, pw)
        vk = x_ij.matmul(self.weight_key).permute(0, 2, 3, 4, 1).view(b, fc, ph, pw)

        # spatially aware mixture embedding
        p_ab = (self.emb_x.repeat(1, ph, 1) + self.emb_y.repeat(1, 1, pw)).unsqueeze(0) # 1, fc/g, ph, pw
        p_abm = (p_ab * self.emb_m).sum(dim=1) # m, fc/g, ph, pw, -> ßm, ph, pw
        p_abm = F.softmax(p_abm, dim=0).unsqueeze(1).unsqueeze(1) # m, 1, 1, ph, pw

        vv = []
        for w_v in self.weight_value:
            vv_x = x_ij.matmul(w_v).permute(0, 2, 3, 4, 1).view(b, fc, ph, pw)
            vv.append(vv_x)
        vv = torch.stack(vv, dim=0) * p_abm # m, b, fc, ph, pw
        vv = vv.sum(dim=0) # b, fc, ph, pw

        # window
        win_k = F.unfold(vk, (kh, kw), stride=self.stride).view(b, fc, kh*kw, fh*fw)
        win_q = F.unfold(vq, (kh, kw), stride=self.stride).view(b, fc, kh, kw, fh*fw)
        win_v = F.unfold(vv, (kh, kw), stride=self.stride).view(b, fc, kh*kw, fh*fw)

        win_q = win_q[:, :, (kh-1)//2, (kw-1)//2, :].view(b, fc, 1, fh*fw)
        vx = (win_q * (win_k + relative_pos)).sum(dim=1) # (b, kh*kw, fh*fw)

        vx = F.softmax(vx, dim=2).unsqueeze(1) # (b, 1, kh*kw, fh*fw)

        v = (vx * win_v).sum(dim=2) # (b,  c2, kh*kw, fh*fw) -> (b, c2, fh*fw)

        if self.bias is not None:
            v += self.bias.view(1, -1, 1)

        fin_v = v.view(b, fc, fh, fw)

        return fin_v