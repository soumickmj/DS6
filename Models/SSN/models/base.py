from abc import ABC
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import warnings


# output = (input * 2 * padding - kernel_size - (kernel_size - 1) (dilation - 1)) / stride + 1


def calculate_convolution_output_size(input_size, conv):
    assert (isinstance(conv, nn.Conv3d) or isinstance(conv, nn.Conv2d))
    output_size = tuple((i + 2 * p - k - (k - 1) * (d - 1)) // s + 1 for i, p, k, d, s
                        in zip(input_size, conv.padding, conv.kernel_size, conv.dilation, conv.stride))
    return output_size


def calculate_convolution_input_size(output_size, conv):
    assert (isinstance(conv, nn.Conv3d) or isinstance(conv, nn.Conv2d))
    input_size = tuple((o - 1) * s + (k - 1) * (d - 1) + k - 2 * p for o, p, k, d, s
                       in zip(output_size, conv.padding, conv.kernel_size, conv.dilation, conv.stride))
    return input_size


def crop_center(x, size):
    if tuple(x.shape[2:]) == size:
        return x
    crop = tuple((slice(0, x.shape[0], 1), slice(0, x.shape[1], 1)))
    crop += tuple(slice(c // 2 - s // 2, c // 2 + s // 2 + s % 2, 1) for c, s in zip(x.shape[2:], size))
    return x[crop]


class BiomedicalModule(nn.Module, ABC):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def calculate_input_size(self, output_size):
        raise NotImplementedError

    def calculate_output_size(self, input_size):
        raise NotImplementedError

    def update_fov_and_scale_factor(self, fov, scale_factor):
        raise NotImplementedError


class BiomedicalBlock(BiomedicalModule, ABC):
    def __init__(self, dim):
        super().__init__(dim)
        self.output_sizes = {}
        self.forward_layers = None

    def get_output_size(self, input_size):
        if input_size not in self.output_sizes:
            output_size = self.calculate_output_size(input_size)
            assert all(o > 0 for o in output_size)

            correct_input_size = self.calculate_input_size(output_size)
            if input_size != correct_input_size:
                warnings.warn(f'The input size {str(input_size):s} wastes memory. Consider using '
                              f'{str(correct_input_size):s} instead for an output size of {str(output_size):s}.')
            self.output_sizes[input_size] = output_size
        return self.output_sizes[input_size]

    def calculate_input_size(self, output_size):
        input_size = output_size
        for layer in reversed(self.layers):
            input_size = layer.calculate_input_size(input_size)
        return input_size

    # note that the calc size function in not invertible due to the downsampling and upsamping
    def calculate_output_size(self, input_size):
        output_size = input_size
        for layer in self.layers:
            output_size = layer.calculate_output_size(output_size)
        return output_size

    def update_fov_and_scale_factor(self, fov, scale_factor):
        for layer in self.layers:
            fov, scale_factor = layer.update_fov_and_scale_factor(fov, scale_factor)
        return fov, scale_factor

    def calculate_fov(self):
        fov, _ = self.update_fov_and_scale_factor((1,) * self.dim, (1,) * self.dim)
        return fov


class SqueezeAndExciteBlock(BiomedicalModule):
    def __init__(self, in_planes, dim=2):
        super().__init__(dim)
        self.dim = dim
        if self.dim == 2:
            self.fc1 = nn.Conv2d(in_planes, in_planes // 2, kernel_size=1)
            self.fc2 = nn.Conv2d(in_planes // 2, in_planes, kernel_size=1)
        if self.dim == 3:
            self.fc1 = nn.Conv3d(in_planes, in_planes // 2, kernel_size=1)
            self.fc2 = nn.Conv3d(in_planes // 2, in_planes, kernel_size=1)
        else:
            raise ValueError('The spatial dimensionality of the kernel ({0:d}) is not supported.'.format(self.dim))

    def forward(self, x):
        if self.dim == 2:
            w = F.avg_pool2d(x, x.size(2))
        elif self.dim == 3:
            w = F.avg_pool3d(x, x.size(2))
        else:
            raise ValueError('The spatial dimensionality of the kernel ({0:d}) is not supported.'.format(self.dim))
        w = F.relu(self.fc1(w))
        w = self.fc2(w).sigmoid()
        out = x * w
        return out

    def calculate_input_size(self, output_size):
        return output_size

    def calculate_output_size(self, input_size):
        return input_size

    def update_fov_and_scale_factor(self, fov, scale_factor):
        return fov, scale_factor


class PreActBlock(BiomedicalModule):
    """Pre-activation version of the BasicBlock."""

    def __init__(self, in_planes, planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), dropout_prob=0.0,
                 is_first_block=False, se=False):
        super(PreActBlock, self).__init__(len(kernel_size))
        assert len(stride) == self.dim
        if self.dim == 2:
            self.norm = nn.BatchNorm2d(in_planes)
            self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, bias=False)
        elif self.dim == 3:
            self.norm = nn.BatchNorm3d(in_planes)
            self.conv = nn.Conv3d(in_planes, planes, kernel_size=kernel_size, stride=stride, bias=False)
        else:
            raise ValueError('The spatial dimensionality of the kernel ({0:d}) is not supported.'.format(self.dim))

        self.dropout = nn.Dropout(dropout_prob)
        self.prelu = nn.PReLU()
        self.shortcut_slices = 2 * (slice(0, None, 1),) + \
                               tuple(slice(k // 2, None if k == 1 else (-k // 2 + 1), s) for k, s in
                                     zip(kernel_size, stride))
        self.is_first_block = is_first_block
        self.se = SqueezeAndExciteBlock(planes, self.dim) if se else None

    def calculate_input_size(self, output_size):
        return calculate_convolution_input_size(output_size, self.conv)

    def calculate_output_size(self, input_size):
        return calculate_convolution_output_size(input_size, self.conv)

    def forward(self, x):
        out = x if self.is_first_block else self.prelu(self.dropout(self.norm(x)))
        out = self.conv(out)
        out = self.se(out) if self.se is not None else out
        if x.shape[1] == out.shape[1]:
            out += x[self.shortcut_slices]

        return out

    def update_fov_and_scale_factor(self, fov, scale_factor):
        fov = tuple(f + s * (k - 1) for f, s, k in zip(fov, scale_factor, self.conv.kernel_size))
        return fov, scale_factor


class DownSample(BiomedicalModule):

    def __init__(self, scale_factor):
        super().__init__(len(scale_factor))
        self.scale_factor = scale_factor
        self.slices = 2 * (slice(0, None, 1),) + tuple(slice(s // 2, None, s) for s in self.scale_factor)

    def calculate_input_size(self, output_size):
        return tuple(o * s for o, s in zip(output_size, self.scale_factor))

    def calculate_output_size(self, input_size):
        return tuple((i - s) // s + 1 for i, s in zip(input_size, self.scale_factor))

    def forward(self, x):
        return x[self.slices]

    def update_fov_and_scale_factor(self, fov, scale_factor):
        # fov = tuple(f * s for f, s in zip(fov, self.scale_factor)) # under debate
        scale_factor = tuple(s0 * s1 for s0, s1 in zip(scale_factor, self.scale_factor))
        return fov, scale_factor


class UpSample(BiomedicalModule):
    def __init__(self, scale_factor):
        super().__init__(len(scale_factor))
        self.scale_factor = scale_factor

    def calculate_input_size(self, output_size):
        return tuple(o // s + bool(o % s) for o, s in zip(output_size, self.scale_factor))

    def calculate_output_size(self, input_size):
        return tuple(i * s for i, s in zip(input_size, self.scale_factor))

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')

    def update_fov_and_scale_factor(self, fov, scale_factor):
        scale_factor = tuple(s0 / float(s1) for s0, s1 in zip(scale_factor, self.scale_factor))
        return fov, scale_factor


class UpConv(BiomedicalModule):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__(len(scale_factor))
        self.scale_factor = scale_factor
        padding = ()
        for k in scale_factor:
            padding += ((k - 1) // 2 + bool((k - 1) % 2), (k - 1) // 2)

        if self.dim == 2:
            self.padding = nn.ReplicationPad2d(padding)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=scale_factor)
        elif self.dim == 3:
            self.padding = nn.ReplicationPad3d(padding)
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=scale_factor)
        else:
            raise ValueError('The spatial dimensionality ({0:d}) is not supported.'.format(self.dim))

    def calculate_input_size(self, output_size):
        return tuple(o // s + bool(o % s) for o, s in zip(output_size, self.scale_factor))

    def calculate_output_size(self, input_size):
        return tuple(i * s for i, s in zip(input_size, self.scale_factor))

    def forward(self, x):
        out = self.conv(self.padding(nn.functional.interpolate(x, scale_factor=self.scale_factor, mode='nearest')))
        assert all([i == j * 2 for i, j in zip(out.shape[2:], x.shape[2:])])
        return out

    def update_fov_and_scale_factor(self, fov, scale_factor):
        scale_factor = tuple(s0 / float(s1) for s0, s1 in zip(scale_factor, self.scale_factor))
        return fov, scale_factor


class TrainableTanhNormalization(nn.Module):
    def __init__(self, num_filters, initial_slope_std=.01, initial_bias_std=0.01):
        super().__init__()
        # self.m = nn.Parameter(torch.tensor(np.random.normal(0, initial_slope_std, num_filters), dtype=torch.float32))
        # self.b = nn.Parameter(torch.tensor(np.random.normal(0, initial_bias_std, num_filters), dtype=torch.float32))
        self.m = nn.Parameter(torch.Tensor(np.ones(num_filters) * 0.001, dtype=torch.float32))
        self.b = nn.Parameter(torch.Tensor(np.zeros(num_filters), dtype=torch.float32))

    def forward(self, x):
        shape = (1, len(self.m)) + (1,) * len(x.shape[2:])
        return (torch.tanh(self.m.view(shape) * x + self.b.view(shape)) + 1.) / 2.
