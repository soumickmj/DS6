import numpy as np
import torch
import torch.nn as nn
from Models.prob_unet.utils import init_weights
from torch.autograd import Variable


class DownConvBlock(nn.Module):
    """
    A block of three convolutional layers where each layer is followed by a non-linear activation function
    Between each block we add a pooling operation.
    """
    def __init__(self, input_dim, output_dim, initializers, padding, pool=True):
        super(DownConvBlock, self).__init__()
        layers = []

        if pool:
            layers.append(nn.AvgPool3d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

        layers.extend(
            (
                nn.Conv3d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=1,
                    padding=int(padding),
                ),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    output_dim,
                    output_dim,
                    kernel_size=3,
                    stride=1,
                    padding=int(padding),
                ),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    output_dim,
                    output_dim,
                    kernel_size=3,
                    stride=1,
                    padding=int(padding),
                ),
                nn.ReLU(inplace=True),
            )
        )

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, patch):
        return self.layers(patch)


class UpConvBlock(nn.Module):
    """
    A block consists of an upsampling layer followed by a convolutional layer to reduce the amount of channels and then a DownConvBlock
    If bilinear is set to false, we do a transposed convolution instead of upsampling
    """
    def __init__(self, input_dim, output_dim, initializers, padding, bilinear=True):
        super(UpConvBlock, self).__init__()
        self.bilinear = bilinear

        if not self.bilinear:
            self.upconv_layer = nn.ConvTranspose3d(input_dim, output_dim, kernel_size=2, stride=2)
            self.upconv_layer.apply(init_weights)

        self.conv_block = DownConvBlock(input_dim, output_dim, initializers, padding, pool=False)

    def forward(self, x, bridge):
        if self.bilinear:
            up = nn.functional.interpolate(x, mode='trilinear', scale_factor=2, align_corners=True) #chaneged to trilinear for 3D
        else:
            up = self.upconv_layer(x)

        assert up.shape[3] == bridge.shape[3]
        out = torch.cat([up, bridge], 1)
        out =  self.conv_block(out)

        return out
