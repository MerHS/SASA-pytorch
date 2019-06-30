# PyTorch Implementation of [Stand-Alone Self-Attention in Vision Models](https://arxiv.org/pdf/1906.05909.pdf)

**This is NOT an official implementation**. Please let me know whether this implementation contains any misreadings of the original paper.

## Prerequisites
 * Python +3.6
 * pytorch +1.1.0
 * scipy
 * Pillow
 * torchvision

## Benchmark (WIP)

Trained with ImageNet. (WIP: CIFAR-10, CIFAR-100)

Backbone network and parameters are based on the official torchvision ResNet and trainer example.

Trained up to 90 epochs / batch 64 on a single NVIDIA 1080Ti GPU, with SGD optimizer with a learning rate of 0.1 which is linearly warmed up for 10 epochs followed by cosine decay. (according to the SASA paper)

