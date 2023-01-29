import itertools
import pandas as pd
import torch
import SimpleITK as sitk
import numpy as np
import math
import random
import torch.utils.data as data
from .patch_samplers import PatchSampler
from .transformation import Transformation
from .augmention import RandomAugmentation


# numpy.random state is discarded at the end of a worker process and does not propagate between workers or to the
# parent process, random state does and hence we need to reseed numpy to ensure each worker process is actually random
def worker_init_fn(worker_id):
    np.random.seed(random.randint(0, 2 ** 32 - 1))


class NiftiDataset(data.Dataset):
    """ nifti dataset for medical imaging segmentation.

    Args:
        data_csv_path: path to csv file containing paths of channels, targets and sampling masks
        target (string): The name of the target column in the csv.
        sampling_mask (string): The name of the sampling mask column in the csv.
        transformation (callable, optional): A list of transformations that are always applied.
        augmentation (callable, optional): A list of random augmentations which are applied with a given probability.
    """

    def __init__(self,
                 data_csv_path,
                 channels,
                 target=None,
                 sampling_mask=None,
                 sample_weight=None,
                 transformation=None,
                 augmentation=None,
                 max_cases_in_memory=0,
                 task='segmentation'):
        self.data_index = pd.read_csv(data_csv_path)
        if 'id' not in self.data_index:
            raise ValueError('id column no provided in csv file.')
        if len(self.data_index.id) != len(set(self.data_index.id)):
            raise ValueError('There are repeated ids in the dataset')
        assert task in ['segmentation', 'classification', 'regression']

        self.channels = channels
        self.target = target
        self.sampling_mask = sampling_mask
        self.sample_weight = sample_weight
        self.transformation = [] if transformation is None else transformation
        self.augmentation = [] if augmentation is None else augmentation
        self.max_cases_in_memory = max_cases_in_memory
        self.case_memory = {}
        self.task = task

        for el in self.transformation:
            assert isinstance(el, Transformation)
        assert isinstance(self.augmentation, list)
        for el in self.augmentation:
            assert isinstance(el, RandomAugmentation)

    def get_array_from_dataset(self, index, name):
        if name in self.data_index:
            return sitk.GetArrayFromImage(sitk.ReadImage(self.data_index.loc[index][name])).astype(np.float32)
        return None

    def get_case_from_disk(self, index):
        target = self.get_array_from_dataset(index, self.target)
        sampling_mask = self.get_array_from_dataset(index, self.sampling_mask)

        stack = list()
        for channel in self.channels:
            stack.append(self.get_array_from_dataset(index, channel))
        image = np.stack(stack)

        if sampling_mask is None:
            sampling_mask = np.ones_like(stack[0])

        for transformation in self.transformation:
            image, target, sampling_mask = transformation(image, target, sampling_mask)

        # image level augmentation is only performed once - assuming images are discarded at the end of the epoch (num_workers>0)
        for augmentation in self.augmentation:
            image, target, sampling_mask = augmentation(image, None if self.task == 'classification' else target,
                                                        sampling_mask)

        sample_weight = np.array(self.data_index[self.sample_weight][index],
                                 dtype=np.float32) if self.sample_weight is not None else None

        # make sure these arrays are read-only so that if they are stored they won't change
        for array in [image, target, sampling_mask, sample_weight]:
            if array is not None:
                array.flags.writeable = False

        return image, target, sampling_mask, sample_weight

    def get_case(self, index):
        if index in self.case_memory:
            image, target, sampling_mask, sample_weight = self.case_memory[index]
        else:
            image, target, sampling_mask, sample_weight = self.get_case_from_disk(index)
            if self.max_cases_in_memory > 0:
                if len(self.case_memory) >= self.max_cases_in_memory:
                    self.case_memory.pop(list(self.case_memory.keys())[0])
                self.case_memory[index] = [image, target, sampling_mask, sample_weight]

        return image, target, sampling_mask, sample_weight

    def to_tensors(self, image, target, sampling_mask=None, sample_weight=None):
        image = torch.tensor(image, dtype=torch.float32)
        target_type = torch.int64 if self.task in ['segmentation', 'classification'] else torch.float64
        nan_tensor = torch.tensor(float('nan'))
        target = torch.tensor(target, dtype=target_type) if target is not None else nan_tensor
        sampling_mask = torch.tensor(sampling_mask, dtype=torch.float32) if sampling_mask is not None else nan_tensor
        sample_weight = torch.tensor(sample_weight, dtype=torch.float32) if sample_weight is not None else nan_tensor
        return {'image': image, 'target': target, 'sampling_mask': sampling_mask, 'sampling_weight': sample_weight}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target)
        """
        image, target, sampling_mask, sample_weight = self.get_case(index)
        return self.to_tensors(image, target, sampling_mask, sample_weight)

    def __len__(self):
        return len(self.data_index)

    def __repr__(self):
        return 'Nifti Dataset containing {} cases with {} channels.\n'.format(len(self), len(self.channels))


class PatchWiseNiftiDataset(NiftiDataset, data.IterableDataset):
    def __init__(self,
                 patch_sampler,
                 patches_per_image,
                 images_per_epoch,
                 data_csv_path,
                 channels,
                 target=None,
                 sampling_mask=None,
                 sample_weight=None,
                 transformation=None,
                 augmentation=None,
                 max_cases_in_memory=0,
                 sequential=False):

        # save RAM since when num_workers = 0 dataset state is kept
        max_cases_in_memory = images_per_epoch if max_cases_in_memory > images_per_epoch else max_cases_in_memory
        super().__init__(data_csv_path,
                         channels,
                         target, sampling_mask,
                         sample_weight,
                         transformation,
                         augmentation,
                         max_cases_in_memory,
                         task='segmentation')

        assert isinstance(patch_sampler, PatchSampler)
        self.patch_sampler = patch_sampler
        self.patches_per_image = patches_per_image
        self.images_per_epoch = images_per_epoch
        self.images_in_epoch = None
        self.max_count = None
        self.patch_count = None
        self.sequential = sequential

    def __iter__(self):
        worker_info = data.get_worker_info()
        start = 0
        end = self.images_per_epoch
        if worker_info is not None:  # more than one process data-loading
            per_worker = int(math.ceil((end - start) / float(worker_info.num_workers)))
            start = start + worker_info.id * per_worker
            end = min(start + per_worker, end)
            # save RAM - likely not necessary since dataset state isn't kept when num_workers > 0
            if self.max_cases_in_memory > per_worker:
                self.max_cases_in_memory = per_worker

        # At the start of every epoch sample which images are in this epoch, assuring each worker has the same images
        replace = False if len(self.data_index) >= self.images_per_epoch else True
        self.images_in_epoch = np.random.choice(range(len(self.data_index)), self.images_per_epoch, replace=replace)
        self.images_in_epoch = self.images_in_epoch[start:end]
        self.max_count = len(self.images_in_epoch) * self.patches_per_image
        self.patch_count = 0
        return self

    def __next__(self):
        if self.patch_count >= self.max_count:
            raise StopIteration
        idx = self.patch_count % len(
            self.images_in_epoch) if not self.sequential else self.patch_count // self.patches_per_image
        image_index = self.images_in_epoch[idx]
        image, target, sampling_mask, sample_weight = self.get_case(image_index)
        image_patch, target_patch, sampling_mask_patch = self.patch_sampler(image, target, sampling_mask)
        self.patch_count += 1
        return self.to_tensors(image_patch, target_patch, sampling_mask_patch, sample_weight)

    def __len__(self):
        return int(self.images_per_epoch * self.patches_per_image)


class FullImageToOverlappingPatchesNiftiDataset(NiftiDataset, data.IterableDataset):
    def __init__(self,
                 image_patch_shape,
                 target_patch_shape,
                 data_csv_path,
                 channels,
                 target=None,
                 sampling_mask=None,
                 sample_weight=None,
                 transformation=None,
                 augmentation=None,
                 use_bbox=False):

        super().__init__(data_csv_path,
                         channels,
                         target,
                         sampling_mask,
                         sample_weight,
                         transformation,
                         augmentation,
                         max_cases_in_memory=1,
                         task='segmentation')
        self.patch_sampler = PatchSampler(image_patch_shape, target_patch_shape)
        self.target_patch_shape = target_patch_shape
        self.index_mapping = []
        self.image_mapping = {}
        for image_index, row in self.data_index.iterrows():
            target_shape = sitk.ReadImage(self.data_index.loc[image_index][self.channels[0]]).GetSize()[::-1]
            center_points = self.get_center_points(target_shape, target_patch_shape)
            self.image_mapping[image_index] = (target_shape, center_points)
            for patch_index in range(len(center_points)):
                self.index_mapping.append((image_index, patch_index))
        self.patch_index = None

    @staticmethod
    def get_center_points(shape, patch_shape):
        return list(itertools.product(*((ps * i + ps // 2 for i in range(s // ps + 1)) for s, ps in
                                        zip(shape, patch_shape))))

    @staticmethod
    def get_bbox(sampling_mask):
        return tuple((np.min(arr), np.max(arr)) for arr in np.where(sampling_mask))

    @staticmethod
    def get_center_points_bbox(shape, patch_shape, bbox):
        c = tuple(tuple(
            ps * i + ps // 2 + bbox[dim][0] for i in range((bbox[dim][1] - bbox[dim][0]) // ps + 1))
                  for dim, (s, ps) in enumerate(zip(shape, patch_shape)))
        center_points = list(itertools.product(*c))
        return center_points

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is not None:
            if worker_info.num_workers > 1:
                raise ValueError(
                    'Patches must be sequential for the saver to reconstruct the image hence num_workers must be 0 or 1')
        self.patch_index = 0
        return self

    def __next__(self):
        if self.patch_index >= len(self.index_mapping):
            raise StopIteration
        image_index, patch_index = self.index_mapping[self.patch_index]
        target_shape, center_points = self.image_mapping[image_index]

        image, target, sampling_mask, _ = self.get_case(image_index)

        center = center_points[patch_index]
        image_patch, target_patch, sampling_mask_patch = \
            self.patch_sampler.get_patches(center, image, target, sampling_mask)

        self.patch_index += 1
        return self.to_tensors(image_patch, target_patch, sampling_mask_patch, None)

    def __len__(self):
        return len(self.index_mapping)
