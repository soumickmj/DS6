import os
import torch
import numpy as np
import SimpleITK as sitk
from .datasets import FullImageToOverlappingPatchesNiftiDataset
from .patch_samplers import get_patch_and_padding


def save_image(output_array, input_image, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    is_vector = True if output_array.ndim > input_image.GetDimension() else False
    image = sitk.GetImageFromArray(output_array, isVector=is_vector)
    image.CopyInformation(input_image)
    sitk.WriteImage(image, path)


def get_num_maps(dim, shape):
    if len(shape) == dim:
        return 1
    elif len(shape) == dim + 1:
        return shape[0]
    else:
        raise ValueError('Trying to save a tensor with incorrect dimensionality.')


def reconstruct_image(patches, image_shape, center_points, patch_shape):
    num_maps = get_num_maps(len(center_points[0]), patches[0].shape)
    assert len(patches) == len(center_points)
    padded_shape = tuple(s - s % ps + ps for s, ps in zip(image_shape, patch_shape))
    reconstruction = np.zeros(shape=(num_maps,) + padded_shape)
    for center, patch in zip(center_points, patches):
        slices, _ = get_patch_and_padding(padded_shape, patch_shape, center)
        reconstruction[(slice(0, num_maps, 1),) + tuple(slices)] = patch
    reconstruction = reconstruction[(slice(0, num_maps, 1),) + tuple(slice(0, s, 1) for s in image_shape)]
    reconstruction = reconstruction.transpose(tuple(range(1, reconstruction.ndim)) + (0,))
    return reconstruction


class NiftiPatchSaver(object):
    def __init__(self, job_dir, dataloader, write_prob_maps=True, extra_output_names=None):
        assert isinstance(dataloader.dataset, FullImageToOverlappingPatchesNiftiDataset)
        self.prediction_dir = os.path.join(job_dir, 'predictions')
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        self.write_prob_maps = write_prob_maps
        self.patches = []
        self.extra_output_patches = {key: [] for key in extra_output_names} if extra_output_names is not None else {}
        self.image_index = 0
        self.data_index = self.dataset.data_index.copy()

    def reset(self):
        self.image_index = 0
        self.patches = []
        if self.extra_output_patches is not None:
            self.extra_output_patches = {key: [] for key in self.extra_output_patches}

    def append(self, state):
        if self.write_prob_maps:
            prob = state['prob'].cpu()
            self.patches += list(prob.cpu())
        else:
            pred = state['pred']
            self.patches += list(pred.cpu())
        for name in self.extra_output_patches:
            tensor = state[name].cpu()
            self.extra_output_patches[name] += list(tensor.cpu())

    def __call__(self, state):
        self.append(state)

        while self.image_index < len(self.dataset.image_mapping):
            target_shape, center_points = self.dataset.image_mapping[self.image_index]
            target_patch_shape = self.dataset.patch_sampler.target_patch_size
            patches_in_image = len(center_points)
            if len(self.patches) < patches_in_image:
                return

            to_write = {}
            case_id = str(self.dataset.data_index.loc[self.image_index]['id'])
            input_image = sitk.ReadImage(self.dataset.data_index.loc[self.image_index][self.dataset.channels[0]])
            patches = list(torch.stack(self.patches[0:patches_in_image]).numpy())
            self.patches = self.patches[patches_in_image:]
            reconstruction = reconstruct_image(patches, target_shape, center_points, target_patch_shape)

            if self.write_prob_maps:
                to_write['prob_maps'] = reconstruction
                to_write['prediction'] = np.argmax(reconstruction, axis=-1).astype(np.float64)
            else:
                to_write['prediction'] = reconstruction

            for name in self.extra_output_patches:
                patches = list(torch.stack(self.extra_output_patches[name][0:patches_in_image]).numpy())
                self.extra_output_patches[name] = self.extra_output_patches[name][patches_in_image:]
                images = reconstruct_image(patches, target_shape, center_points, target_patch_shape)
                to_write[name] = images

            for name, array in to_write.items():
                path = os.path.join(self.prediction_dir, f'{case_id:s}_{name:s}.nii.gz')
                self.data_index.loc[self.data_index['id'] == case_id, name] = path
                save_image(array, input_image, path)
            self.image_index += 1

        self.data_index.to_csv(os.path.join(self.prediction_dir, 'prediction.csv'), index=False)
        self.reset()
