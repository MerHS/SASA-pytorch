import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.resnet import Bottleneck, ResNet, conv1x1

from .sa_layer import SelfAttentionConv2d, SAMixtureConv2d

def sa_conv7x7(in_planes, out_planes, stride=1, groups=1, padding=3):
    """ 7x7 SA Convolution with padding """
    return SelfAttentionConv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                               padding=padding, groups=groups, bias=False)

def sa_stem4x4(in_height, in_width, in_planes, out_planes, stride=1, groups=1, padding=2, mix=4):
    """ 4x4 mixed SA Convolution for stem """
    return SAMixtureConv2d(in_height, in_width, in_planes, out_planes, kernel_size=4, stride=stride,
                           padding=padding, groups=groups, mix=mix, bias=False)

class SelfAttentionBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64, groups=1):
        super(SelfAttentionBottleneck, self).__init__()

        width = int(planes * (base_width / 64.))

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = sa_conv7x7(width, width, groups=groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if stride >= 2:
            self.avg_pool = nn.AvgPool2d(stride, stride)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self.stride >= 2:
            out = self.avg_pool(out)

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SAResNet(nn.Module):
    """ simpler version of torchvision official ResNet """
    def __init__(self, block, layers, num_classes=1000, use_conv_stem=False, **kwargs):
        super(SAResNet, self).__init__()

        self.inplanes = 64
        self.head_count = 8

        if use_conv_stem:
            self.conv1 = sa_stem4x4(kwargs['in_height'], kwargs['in_width'], 
                                    3, self.in_planes, groups=4, mix=4)
            self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                                   padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.head_count))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.head_count))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


model_dict = {
    'resnet26': (ResNet, Bottleneck, [1, 2, 4, 1]),
    'resnet38': (ResNet, Bottleneck, [2, 3, 5, 2]),
    'sa_resnet26': (SAResNet, SelfAttentionBottleneck, [1, 2, 4, 1]),
    'sa_resnet38': (SAResNet, SelfAttentionBottleneck, [2, 3, 5, 2]),
    'sa_resnet50': (SAResNet, SelfAttentionBottleneck, [3, 4, 6, 3]),
    'sa_resnet101': (SAResNet, SelfAttentionBottleneck, [3, 4, 23, 3]),
    'sa_resnet152': (SAResNet, SelfAttentionBottleneck, [3, 8, 36, 3]),
    'cstem_sa_resnet26': (SAResNet, SelfAttentionBottleneck, [1, 2, 4, 1]),
    'cstem_sa_resnet38': (SAResNet, SelfAttentionBottleneck, [2, 3, 5, 2]),
    'cstem_sa_resnet50': (SAResNet, SelfAttentionBottleneck, [3, 4, 6, 3]),
    'cstem_sa_resnet101': (SAResNet, SelfAttentionBottleneck, [3, 4, 23, 3]),
    'cstem_sa_resnet152': (SAResNet, SelfAttentionBottleneck, [3, 8, 36, 3]),
}

model_names = list(model_dict.keys())

def get_model(args, **kwargs):
    arch = args.arch
    width = args.width

    model_fn, block, layers = model_dict[arch]
    use_conv_stem = arch.startswith('cstem')
    return model_fn(block, layers, use_conv_stem=use_conv_stem, in_height=width, in_width=width, **kwargs)
