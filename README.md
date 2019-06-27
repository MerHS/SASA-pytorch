# PyTorch Implementation of [Stand-Alone Self-Attention in Vision Models](https://arxiv.org/pdf/1906.05909.pdf)

**This is NOT an official implementation**. Please let me know whether this implementation contains any misreadings of the original paper.

## Prerequisites
 * Python +3.6
 * pytorch +1.0
 * torchvision

## Benchmark

Trained with CIFAR-10, CIFAR-100, ImageNet.

Backbone network and parameters are based on the official torchvision ResNet and trainer example.

Trained 90 epochs / batch 64 on a single NVIDIA 1080Ti GPU, with SGD optimizer with a learning rate of 0.1 which is linearly warmed up for 10 epochs followed by cosine decay. (according to the SASA paper)

