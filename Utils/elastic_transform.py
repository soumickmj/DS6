#!/usr/bin/env python
'''

Purpose : 

'''


from numbers import Number
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

__author__ = "Kartik Prabhu, Mahantesh Pattadkal, and Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Kartik Prabhu", "Mahantesh Pattadkal", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

from torchio.utils import to_tuple

try:
    from torchio import RandomElasticDeformation
except:
    from torchio.transforms.augmentation import RandomElasticDeformation

from airlab import utils as tu
from airlab.transformation.pairwise import _KernelTransformation
from airlab.transformation.utils import compute_grid
from airlab.utils import kernelFunction as utils

SPLINE_ORDER = 3

"""
    Warp image with displacement
    * input(tensor) : input of shape (N, C, H_\text{in}, W_\text{in})(N,C,H,W) (4-D case)(N,C,D,H ,W) (5-D case)
    * grid(tensor): flow-field of shape (N, H_\text{out}, W_\text{out}, 2)(N,H ,W,2) (4-D case) or (N, D, H, W, 3)(N,D,H,W,3) (5-D case)
    * mult = true if batched input
"""
def warp_image(image, displacement, multi=False):
    image_size = image.size() #[B, D, H, W]
    batch_size  = image_size[0]
    if multi:
        image_size = image_size[2:]#[D, H, W]

    grid = compute_grid(image_size, dtype=image.dtype, device=image.device)
    grid = displacement + grid
    grid = torch.cat([grid] * batch_size, dim=0)  # batch number of times

    # warp image
    if multi:
        warped_image = F.grid_sample(image, grid)  #[B, C, D, H, W]
    else:
        warped_image = F.grid_sample(image.unsqueeze(0).unsqueeze(0), grid) #[B, C, D, H, W], unsqueeze to give batch and channel dimension

    return warped_image  #[B, C, D, H, W]

"""
    Base class for kernel transformations
"""
class _ParameterizedKernelTransformation(_KernelTransformation):
    def __init__(self, image_size, rnd_grid_params=None, diffeomorphic=False, dtype=th.float32, device='cpu'):
        super(_ParameterizedKernelTransformation, self).__init__(image_size, diffeomorphic, dtype, device)
        self.rnd_grid_params = rnd_grid_params

    def get_coarse_field(self,
                        grid_shape,
                        max_displacement,
                        num_locked_borders,
                        ):
        coarse_field = th.rand(self._dim, *grid_shape)  # [0, 1)
        coarse_field -= 0.5  # [-0.5, 0.5)
        coarse_field *= 2  # [-1, 1]
        for dimension in range(3):
            # [-max_displacement, max_displacement)
            coarse_field[dimension, ...] *= max_displacement[dimension]

        # Set displacement to 0 at the borders
        for i in range(num_locked_borders):
            coarse_field[:, i, :] = 0
            coarse_field[:, -1 - i, :] = 0
            coarse_field[:, :, i] = 0
            coarse_field[:, :, -1 - i] = 0

        return coarse_field.unsqueeze(0)

    def _initialize(self):
        cp_grid = np.ceil(np.divide(self._image_size, self._stride)).astype(dtype=int)

        # new image size after convolution
        inner_image_size = np.multiply(self._stride, cp_grid) - (self._stride - 1)

        # add one control point at each side
        cp_grid = cp_grid + 2

        # image size with additional control points
        new_image_size = np.multiply(self._stride, cp_grid) - (self._stride - 1)

        # center image between control points
        image_size_diff = inner_image_size - self._image_size
        image_size_diff_floor = np.floor((np.abs(image_size_diff)/2))*np.sign(image_size_diff)

        self._crop_start = image_size_diff_floor + np.remainder(image_size_diff, 2)*np.sign(image_size_diff)
        self._crop_end = image_size_diff_floor

        # create transformation parameters
        if self.rnd_grid_params is None:
            cp_grid = [1, self._dim] + cp_grid.tolist()
            self.trans_parameters = Parameter(th.Tensor(*cp_grid))
            self.trans_parameters.data.fill_(0)
        else:
            self.trans_parameters = Parameter(self.get_coarse_field(cp_grid, self.rnd_grid_params['max_displacement'], self.rnd_grid_params['num_locked_borders']))

        # copy to gpu if needed
        self.to(dtype=self._dtype, device=self._device)

        # convert to integer
        self._padding = self._padding.astype(dtype=int).tolist()
        self._stride = self._stride.astype(dtype=int).tolist()

        self._crop_start = self._crop_start.astype(dtype=int)
        self._crop_end = self._crop_end.astype(dtype=int)

        size = [1, 1] + new_image_size.astype(dtype=int).tolist()
        self._displacement_tmp = th.empty(*size, dtype=self._dtype, device=self._device)

        size = [1, 1] + self._image_size.astype(dtype=int).tolist()
        self._displacement = th.empty(*size, dtype=self._dtype, device=self._device)
"""
    bspline kernel transformation
"""
class ParameterizedBsplineTransformation(_ParameterizedKernelTransformation):
    def __init__(self, image_size, sigma, rnd_grid_params=None, diffeomorphic=False, order=2, dtype=th.float32, device='cpu'):
        super(ParameterizedBsplineTransformation, self).__init__(image_size, rnd_grid_params, diffeomorphic, dtype, device)

        self._stride = np.array(sigma)

        # compute bspline kernel
        self._kernel = utils.bspline_kernel(sigma, dim=self._dim, order=order, asTensor=True, dtype=dtype)

        self._padding = (np.array(self._kernel.size()) - 1) / 2

        self._kernel.unsqueeze_(0).unsqueeze_(0)
        self._kernel = self._kernel.expand(self._dim, *((np.ones(self._dim + 1, dtype=int)*-1).tolist()))
        self._kernel = self._kernel.to(dtype=dtype, device=self._device)

        self._initialize()

class RandomElasticDeformation(nn.Module):
    def __init__(
            self,
            num_control_points: Union[int, Tuple[int, int, int]] = 7,
            max_displacement: Union[float, Tuple[float, float, float]] = 7.5,
            locked_borders: int = 2,
            ):
        super().__init__()
        self.num_control_points = to_tuple(num_control_points, length=3)
        self.parse_control_points(self.num_control_points)
        self.max_displacement = to_tuple(max_displacement, length=3)
        self.parse_max_displacement(self.max_displacement)
        self.num_locked_borders = locked_borders
        if locked_borders not in (0, 1, 2):
            raise ValueError('locked_borders must be 0, 1, or 2')
        if locked_borders == 2 and 4 in self.num_control_points:
            message = (
                'Setting locked_borders to 2 and using less than 5 control'
                'points results in an identity transform. Lock fewer borders'
                ' or use more control points.'
            )
            raise ValueError(message)
        self.bspline_params = {'max_displacement':self.max_displacement, 'num_locked_borders':self.num_locked_borders}

    @staticmethod
    def parse_control_points(
            num_control_points: Tuple[int, int, int],
            ) -> None:
        for axis, number in enumerate(num_control_points):
            if not isinstance(number, int) or number < 4:
                message = (
                    f'The number of control points for axis {axis} must be'
                    f' an integer greater than 3, not {number}'
                )
                raise ValueError(message)

    @staticmethod
    def parse_max_displacement(
            max_displacement: Tuple[float, float, float],
            ) -> None:
        for axis, number in enumerate(max_displacement):
            if not isinstance(number, Number) or number < 0:
                message = (
                    'The maximum displacement at each control point'
                    f' for axis {axis} must be'
                    f' a number greater or equal to 0, not {number}'
                )
                raise ValueError(message)

    """
        Images: shape of [N,D,H,W] or [N,H,W]
    """
    def forward(self, images):
        bspline_transform = ParameterizedBsplineTransformation(images.size()[2:], #ignore batch and channel dim
                                                                sigma=self.num_control_points,
                                                                rnd_grid_params=self.bspline_params,
                                                                diffeomorphic=True,
                                                                order=SPLINE_ORDER,
                                                                device=images.device)
        displacement = bspline_transform.get_displacement()
        inv_displacement = bspline_transform.get_inverse_displacement()

        warped_images = warp_image(images, displacement, multi=True)

        return warped_images, displacement, inv_displacement
