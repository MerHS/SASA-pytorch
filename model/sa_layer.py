import math
import time

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
            self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)

        # relative position offsets are shared between multi-heads
        self.rel_size = (out_channels // groups) // 2
        self.relative_x = nn.Parameter(torch.Tensor(self.rel_size, self.kernel_size[1]))
        self.relative_y = nn.Parameter(torch.Tensor((out_channels // groups) - self.rel_size, self.kernel_size[0]))

        self.weight_query = nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False)
        self.weight_key = nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False)
        self.weight_value = nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False)

        self.softmax = nn.Softmax(dim=3)

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

        px, py = self.padding
        x = F.pad(x, (py, py, px, px))

        vq = self.weight_query(x)
        vk = self.weight_key(x)
        vv = self.weight_value(x) # b, fc, ph, pw

        # b, fc, fh, fw
        win_q = vq[:, :, (kh-1)//2:ph-(kh//2):self.stride[0], (kw-1)//2:pw-(kw//2):self.stride[1]]

        win_q_b = win_q.view(b, self.groups, -1, fh, fw) # b, g, fc/g, fh, fw

        win_q_x, win_q_y = win_q_b.split(self.rel_size, dim=2) # (b, g, x, fh, fw), (b, g, y, fh, fw)
        win_q_x = torch.einsum('bgxhw,xk->bhwk', (win_q_x, self.relative_x)) # b, fh, fw, kw
        win_q_y = torch.einsum('bgyhw,yk->bhwk', (win_q_y, self.relative_y)) # b, fh, fw, kh

        win_k = vk.unfold(2, kh, self.stride[0]).unfold(3, kw, self.stride[1]) # b, fc, fh, fw, kh, kw

        vx = (win_q.unsqueeze(4).unsqueeze(4) * win_k).sum(dim=1)  # b, fh, fw, kh, kw
        vx = vx + win_q_x.unsqueeze(3) + win_q_y.unsqueeze(4) # add rel_x, rel_y
        vx = self.softmax(vx.view(b, fh, fw, -1)).view(b, 1, fh, fw, kh, kw)

        win_v = vv.unfold(2, kh, self.stride[0]).unfold(3, kw, self.stride[1])
        fin_v = torch.einsum('bchwkl->bchw', (vx * win_v, )) # (b, fc, fh, fw, kh, kw) -> (b, fc, fh, fw)

        if self.bias is not None:
            fin_v += self.bias

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
            self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
        else:
            self.register_parameter('bias', None)

        # relative position offsets are shared between multi-heads
        self.rel_size = (out_channels // groups) // 2
        self.relative_x = nn.Parameter(torch.Tensor(self.rel_size, self.kernel_size[1]))
        self.relative_y = nn.Parameter(torch.Tensor((out_channels // groups) - self.rel_size, self.kernel_size[0]))

        self.weight_query = nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False)
        self.weight_key = nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False)
        self.weight_values = nn.ModuleList([nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.groups, bias=False) for _ in range(mix)])

        self.emb_x = nn.Parameter(torch.Tensor(out_channels // groups, in_width + 2 * self.padding[1])) # fc/g, pw
        self.emb_y = nn.Parameter(torch.Tensor(out_channels // groups, in_height + 2 * self.padding[0])) # fc/g, ph
        self.emb_m = nn.Parameter(torch.Tensor(mix, out_channels // groups)) # m, fc/g

        self.softmax = nn.Softmax(dim=3)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.weight_query.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.weight_key.weight, mode='fan_out', nonlinearity='relu')
        for wv in self.weight_values:
            init.kaiming_normal_(wv.weight, mode='fan_out', nonlinearity='relu')

        if self.bias is not None:
            bound = 1 / math.sqrt(self.out_channels)
            init.uniform_(self.bias, -bound, bound)

        rel_bound = 1 / math.sqrt(self.rel_size)
        init.uniform_(self.relative_x, -rel_bound, rel_bound)
        init.uniform_(self.relative_y, -rel_bound, rel_bound)

        emb_bound = 1 / math.sqrt(self.out_channels // self.groups)
        init.uniform_(self.emb_x, -emb_bound, emb_bound)
        init.uniform_(self.emb_y, -emb_bound, emb_bound)
        init.uniform_(self.emb_m, -emb_bound, emb_bound)

    def forward(self, x):
        b, c, h, w = x.size()
        kh, kw = self.kernel_size
        ph, pw = h + self.padding[0] * 2, w + self.padding[1] * 2

        fh = (ph - kh) // self.stride[0] + 1
        fw = (pw - kw) // self.stride[1] + 1

        px, py = self.padding
        x = F.pad(x, (py, py, px, px))

        vq = self.weight_query(x)
        vk = self.weight_key(x) # b, fc, fh, fw

        # b, fc, fh, fw
        win_q = vq[:, :, (kh-1)//2:ph-(kh//2):self.stride[0], (kw-1)//2:pw-(kw//2):self.stride[1]]

        win_q_b = win_q.view(b, self.groups, -1, fh, fw) # b, g, fc/g, fh, fw

        win_q_x, win_q_y = win_q_b.split(self.rel_size, dim=2) # (b, g, x, fh, fw), (b, g, y, fh, fw)
        win_q_x = torch.einsum('bgxhw,xk->bhwk', (win_q_x, self.relative_x)) # b, fh, fw, kw
        win_q_y = torch.einsum('bgyhw,yk->bhwk', (win_q_y, self.relative_y)) # b, fh, fw, kh

        win_k = vk.unfold(2, kh, self.stride[0]).unfold(3, kw, self.stride[1]) # b, fc, fh, fw, kh, kw

        vx = (win_q.unsqueeze(4).unsqueeze(4) * win_k).sum(dim=1)  # b, fh, fw, kh, kw
        vx = vx + win_q_x.unsqueeze(3) + win_q_y.unsqueeze(4) # add rel_x, rel_y
        vx = self.softmax(vx.view(b, fh, fw, -1)).view(b, 1, fh, fw, kh, kw)

        # spatially aware mixture embedding
        p_abm_x = torch.einsum('mc,cw->mw', (self.emb_m, self.emb_x)).unsqueeze(1) # m, 1, pw
        p_abm_y = torch.einsum('mc,ch->mh', (self.emb_m, self.emb_y)).unsqueeze(2) # m, ph, 1
        p_abm = F.softmax(p_abm_x + p_abm_y, dim=0) # m, ph, pw

        vv = torch.stack([weight_value(x) for weight_value in self.weight_values], dim=0) # m, b, fc, ph, pw
        vv = torch.einsum('mbchw,mhw->bchw', (vv, p_abm)) # b, fc, ph, pw

        win_v = vv.unfold(2, kh, self.stride[0]).unfold(3, kw, self.stride[1])
        fin_v = torch.einsum('bchwkl->bchw', (vx * win_v, )) # (b, fc, fh, fw, kh, kw) -> (b, fc, fh, fw)

        if self.bias is not None:
            fin_v += self.bias

        return fin_v