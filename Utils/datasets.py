# from __future__ import self.logger.debug_function, division


import glob
import os
import sys
from random import randint, random, seed

import nibabel
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchio as tio
from PIL import Image

from Utils.customutils import createCenterRatioMask, performUndersampling

__author__ = "Soumick Chatterjee, Chompunuch Sarasaen"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee", "Chompunuch Sarasaen"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

torch.manual_seed(2020)
np.random.seed(2020)
seed(2020)


class SRDataset(Dataset):

    def __init__(self, logger, patch_size, dir_path, label_dir_path, stride_depth=16, stride_length=32, stride_width=32,
                 Size=None, fly_under_percent=None, patch_size_us=None, return_coords=False, pad_patch=True,
                 pre_interpolate=None, norm_data=True, pre_load=False):
        self.patch_size = patch_size  # -1 = full vol
        self.stride_depth = stride_depth
        self.stride_length = stride_length
        self.stride_width = stride_width
        self.size = Size
        self.logger = logger
        self.fly_under_percent = fly_under_percent  # if None, then use already undersampled data. Gets priority over patch_size_us. They are both mutually exclusive
        self.return_coords = return_coords
        self.pad_patch = pad_patch
        self.pre_interpolate = pre_interpolate
        if patch_size == patch_size_us:
            patch_size_us = None
        if patch_size != -1 and patch_size_us is not None:
            stride_length_us = stride_length // (patch_size // patch_size_us)
            stride_width_us = stride_width // (patch_size // patch_size_us)
            self.stride_length_us = stride_length_us
            self.stride_width_us = stride_width_us
        elif patch_size == -1:
            patch_size_us = None
        if self.fly_under_percent is not None:
            patch_size_us = None
        self.patch_size_us = patch_size_us  # If already downsampled data is supplied, then this can be used. Calculate already based on the downsampling size.
        self.norm_data = norm_data
        self.pre_load = pre_load
        self.pre_loaded_data = pd.DataFrame(columns=["pre_loaded_img", "pre_loaded_lbl", "pre_loaded_lbl_mip"])
        pre_loaded_lbl = np.empty([1], dtype=object)
        pre_loaded_img = np.empty([1], dtype=object)
        pre_loaded_lbl_mip = np.empty([1], dtype=object)
        pre_loaded_lbl = np.delete(pre_loaded_lbl, 0)
        pre_loaded_img = np.delete(pre_loaded_img, 0)
        pre_loaded_lbl_mip = np.delete(pre_loaded_lbl_mip, 0)

        if not self.norm_data:
            print("No Norm")  # TODO remove

        # Constants
        self.IMAGE_FILE_NAME = "imageFilename"
        self.IMAGE_FILE_SHAPE = "imageFileShape"
        self.IMAGE_FILE_MAXVAL = "imageFileMaxVal"
        self.LABEL_FILE_NAME = "labelFilename"
        self.LABEL_FILE_SHAPE = "labelFileShape"
        self.LABEL_FILE_MAXVAL = "labelFileMaxVal"
        self.STARTINDEX_DEPTH = "startIndex_depth"
        self.STARTINDEX_LENGTH = "startIndex_length"
        self.STARTINDEX_WIDTH = "startIndex_width"
        self.STARTINDEX_DEPTH_US = "startIndex_depth_us"
        self.STARTINDEX_LENGTH_US = "startIndex_length_us"
        self.STARTINDEX_WIDTH_US = "startIndex_width_us"

        self.trans = transforms.ToTensor()  # used to convert tiffimagefile to tensor
        dataDict = {self.IMAGE_FILE_NAME: [], self.IMAGE_FILE_SHAPE: [], self.IMAGE_FILE_MAXVAL: [],
                    self.LABEL_FILE_NAME: [], self.LABEL_FILE_SHAPE: [], self.LABEL_FILE_MAXVAL: [],
                    self.STARTINDEX_DEPTH: [], self.STARTINDEX_LENGTH: [], self.STARTINDEX_WIDTH: [],
                    self.STARTINDEX_DEPTH_US: [], self.STARTINDEX_LENGTH_US: [], self.STARTINDEX_WIDTH_US: []}

        column_names = [self.IMAGE_FILE_NAME, self.IMAGE_FILE_SHAPE, self.IMAGE_FILE_MAXVAL, self.LABEL_FILE_NAME,
                        self.LABEL_FILE_SHAPE, self.LABEL_FILE_MAXVAL, self.STARTINDEX_DEPTH, self.STARTINDEX_LENGTH,
                        self.STARTINDEX_WIDTH,
                        self.STARTINDEX_DEPTH_US, self.STARTINDEX_LENGTH_US, self.STARTINDEX_WIDTH_US]
        self.data = pd.DataFrame(columns=column_names)

        files_us = glob.glob(dir_path + '/**/*.nii', recursive=True)
        files_us += glob.glob(dir_path + '/**/*.nii.gz', recursive=True)
        for imageFileName in files_us:
            labelFileName = imageFileName.replace(dir_path[:-1], label_dir_path[
                                                                 :-1])  # [:-1] is needed to remove the trailing slash for shitty windows

            if imageFileName == labelFileName:
                sys.exit('Input and Output save file')

            if not (os.path.isfile(imageFileName) and os.path.isfile(labelFileName)):
                # trick to include the other file extension
                if labelFileName.endswith('.nii.nii.gz'):
                    labelFileName = labelFileName.replace('.nii.nii.gz', '.nii.gz')
                elif labelFileName.endswith('.nii.gz'):
                    labelFileName = labelFileName.replace('.nii.gz', '.nii')
                else:
                    labelFileName = labelFileName.replace('.nii', '.nii.gz')

                # check again, after replacing the file extension
                if not (os.path.isfile(imageFileName) and os.path.isfile(labelFileName)):
                    self.logger.debug(
                        "skipping file as label for the corresponding image doesn't exist :" + str(imageFileName))
                    continue

            imageFile = tio.ScalarImage(imageFileName)  # shape (Length X Width X Depth X Channels)
            header_shape_us = imageFile.data.shape
            imageFile_max = imageFile.data.max()
            labelFile = tio.LabelMap(
                labelFileName)  # shape (Length X Width X Depth X Channels) - changed to label file name as input image can have different (lower) size
            labelFile_data = nibabel.load(labelFileName).get_data()
            labelFile_mip = torch.from_numpy(labelFile_data).float()
            labelFile_mip = torch.amax(labelFile_mip, -1)
            header_shape = labelFile.data.shape
            labelFile_max = labelFile_data.max()
            label_filename_trimmed = labelFileName.split("\\")
            if label_filename_trimmed:
                label_filename_trimmed = label_filename_trimmed[len(label_filename_trimmed) - 1]

            self.logger.debug(header_shape)
            n_depth, n_length, n_width = header_shape[3], header_shape[2], header_shape[
                1]  # gives depth which is no. of slices
            n_depth_us, n_length_us, n_width_us = header_shape_us[3], header_shape_us[2], header_shape_us[
                1]  # gives depth which is no. of slices

            if self.pre_load:
                pre_loaded_img = np.append(pre_loaded_img,
                                           {'subjectname': imageFileName, 'data': imageFile.data})
                pre_loaded_lbl = np.append(pre_loaded_lbl,
                                           {'subjectname': labelFileName, 'data': labelFile.data})
                pre_loaded_lbl_mip = np.append(pre_loaded_lbl_mip,
                                                {'subjectname': label_filename_trimmed, 'data': labelFile_mip})

            if patch_size != 1 and (n_depth < patch_size or n_length < patch_size or n_width < patch_size):
                self.logger.debug(
                    "skipping file because of its size being less than the patch size :" + str(imageFileName))
                continue

            ############ Following the fully sampled size
            if patch_size != -1:
                depth_i = 0
                ranger_depth = int((n_depth - patch_size) / stride_depth) + 1
                for depth_index in range(
                        ranger_depth if n_depth % patch_size == 0 else ranger_depth + 1):  # iterate through the whole image voxel, and extract patch
                    length_i = 0
                    # self.logger.debug("depth")
                    # self.logger.debug(depth_i)
                    ranger_length = int((n_length - patch_size) / stride_length) + 1
                    for length_index in range(ranger_length if n_length % patch_size == 0 else ranger_length + 1):
                        width_i = 0
                        # self.logger.debug("length")
                        # self.logger.debug(length_i)

                        ranger_width = int((n_width - patch_size) / stride_width) + 1
                        for width_index in range(ranger_width if n_width % patch_size == 0 else ranger_width + 1):
                            # self.logger.debug("width")
                            # self.logger.debug(width_i)
                            dataDict[self.IMAGE_FILE_NAME].append(imageFileName)
                            dataDict[self.IMAGE_FILE_SHAPE].append(header_shape_us)
                            dataDict[self.IMAGE_FILE_MAXVAL].append(imageFile_max)
                            dataDict[self.LABEL_FILE_NAME].append(labelFileName)
                            dataDict[self.LABEL_FILE_SHAPE].append(header_shape)
                            dataDict[self.LABEL_FILE_MAXVAL].append(labelFile_max)
                            dataDict[self.STARTINDEX_DEPTH].append(depth_i)
                            dataDict[self.STARTINDEX_LENGTH].append(length_i)
                            dataDict[self.STARTINDEX_WIDTH].append(width_i)

                            if patch_size_us is None:  # data is zero padded
                                dataDict[self.STARTINDEX_DEPTH_US].append(depth_i)
                                dataDict[self.STARTINDEX_LENGTH_US].append(length_i)
                                dataDict[self.STARTINDEX_WIDTH_US].append(width_i)

                            width_i += stride_width
                        length_i += stride_length
                    depth_i += stride_depth
            else:
                dataDict[self.IMAGE_FILE_NAME].append(imageFileName)
                dataDict[self.IMAGE_FILE_SHAPE].append(header_shape_us)
                dataDict[self.IMAGE_FILE_MAXVAL].append(imageFile_max)
                dataDict[self.LABEL_FILE_NAME].append(labelFileName)
                dataDict[self.LABEL_FILE_SHAPE].append(header_shape)
                dataDict[self.LABEL_FILE_MAXVAL].append(labelFile_max)
                dataDict[self.STARTINDEX_DEPTH].append(0)
                dataDict[self.STARTINDEX_LENGTH].append(0)
                dataDict[self.STARTINDEX_WIDTH].append(0)
                dataDict[self.STARTINDEX_DEPTH_US].append(0)
                dataDict[self.STARTINDEX_LENGTH_US].append(0)
                dataDict[self.STARTINDEX_WIDTH_US].append(0)

            ############ Following the undersampled size, only if patch_size_us has been provied
            if patch_size_us is not None:
                depth_i = 0
                ranger_depth = int((n_depth_us - patch_size_us) / stride_depth) + 1
                for depth_index in range(
                        ranger_depth if n_depth_us % patch_size_us == 0 else ranger_depth + 1):  # iterate through the whole image voxel, and extract patch
                    length_i = 0
                    # self.logger.debug("depth")
                    # self.logger.debug(depth_i)
                    ranger_length = int((n_length_us - patch_size_us) / stride_length_us) + 1
                    for length_index in range(ranger_length if n_length_us % patch_size_us == 0 else ranger_length + 1):
                        width_i = 0
                        # self.logger.debug("length")
                        # self.logger.debug(length_i)
                        ranger_width = int((n_width_us - patch_size_us) / stride_width_us) + 1
                        for width_index in range(ranger_width if n_width_us % patch_size_us == 0 else ranger_width + 1):
                            # self.logger.debug("width")
                            # self.logger.debug(width_i)
                            dataDict[self.STARTINDEX_DEPTH_US].append(depth_i)
                            dataDict[self.STARTINDEX_LENGTH_US].append(length_i)
                            dataDict[self.STARTINDEX_WIDTH_US].append(width_i)
                            width_i += stride_width_us
                        length_i += stride_length_us
                    depth_i += stride_depth

        self.data = pd.DataFrame.from_dict(dataDict)
        self.pre_loaded_data = pd.DataFrame.from_dict(
            {'pre_loaded_img': pre_loaded_img, 'pre_loaded_lbl': pre_loaded_lbl,
             'pre_loaded_lbl_mip': pre_loaded_lbl_mip})
        self.logger.debug(len(self.data))

        if Size is not None and len(self.data) > Size:
            self.logger.debug('Dataset is larger tham supplied size. Choosing s subset randomly of size ' + str(Size))
            self.data = self.data.sample(n=Size, replace=False, random_state=2020)

        if patch_size != -1 and fly_under_percent is not None:
            self.mask = createCenterRatioMask(np.zeros((patch_size, patch_size, patch_size)), fly_under_percent)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''
        imageFilename: 0
        imageFileShape: 1
        imageFileMaxVal: 2
        labelFilename: 3
        labelFileShape: 4
        labelFileMaxVal: 5
        startIndex_depth : 6
        startIndex_length : 7
        startIndex_width : 8
        startIndex_depth_us : 9
        startIndex_length_us : 10
        startIndex_width_us : 11

        '''
        imageFile_max = self.data.iloc[index, 2]
        labelFile_max = self.data.iloc[index, 5]

        if self.pre_load:
            groundTruthImages = [lbl for lbl in self.pre_loaded_data['pre_loaded_lbl'] if
                                 lbl['subjectname'] == self.data.iloc[index, 3]]
            groundTruthImages = groundTruthImages[0]['data']
        else:
            groundTruthImages = tio.LabelMap(self.data.iloc[index, 3])  # TODO: Update this to tio.ScalarImage
            groundTruthImages = groundTruthImages.data

        startIndex_depth = self.data.iloc[index, 6]
        startIndex_length = self.data.iloc[index, 7]
        startIndex_width = self.data.iloc[index, 8]
        start_coords = np.array([(startIndex_width, startIndex_length, startIndex_depth)])

        if self.patch_size_us is not None:
            startIndex_depth_us = self.data.iloc[index, 9]
            startIndex_length_us = self.data.iloc[index, 10]
            startIndex_width_us = self.data.iloc[index, 11]
            start_coords = start_coords + [(startIndex_width_us, startIndex_length_us, startIndex_depth_us)]

        if self.patch_size != -1:
            if len(groundTruthImages.shape) == 4:  # don't know why, but an additional dim is noticed in some of the fully-sampled NIFTIs
                targetPatch = groundTruthImages[:, startIndex_width:startIndex_width + self.patch_size,
                              startIndex_length:startIndex_length + self.patch_size,
                              startIndex_depth:startIndex_depth + self.patch_size]  # .squeeze()
            else:
                targetPatch = groundTruthImages[startIndex_width:startIndex_width + self.patch_size,
                              startIndex_length:startIndex_length + self.patch_size,
                              startIndex_depth:startIndex_depth + self.patch_size]  # .squeeze()
        else:
            if len(groundTruthImages.shape) == 4:  # don't know why, but an additional dim is noticed in some of the fully-sampled NIFTIs
                targetPatch = groundTruthImages[:, :, :, :]  # .squeeze()
            else:
                targetPatch = groundTruthImages[...]  # .squeeze()

        if self.fly_under_percent is not None:
            if self.patch_size != -1:
                patch = abs(performUndersampling(np.array(targetPatch).copy(), mask=self.mask, zeropad=False))
                patch = patch[..., ::2]  # 2 for 25% - harcoded. TODO fix it
            else:
                mask = createCenterRatioMask(targetPatch, self.fly_under_percent)
                patch = abs(performUndersampling(np.array(targetPatch).copy(), mask=mask, zeropad=False))
                patch = patch[..., ::2]  # 2 for 25% - harcoded. TODO fix it
        else:
            if self.pre_load:
                image_us = [img for img in self.pre_loaded_data['pre_loaded_img'] if
                            img['subjectname'] == self.data.iloc[index, 0]]
                image_us = image_us[0]['data']
            else:
                image_us = tio.ScalarImage(self.data.iloc[index, 0])  # TODO change this to use tio.ScalarImage

            # images = nibabel.load(self.data.iloc[index, 0])
            if self.patch_size_us is not None:
                patch = image_us[:, startIndex_width_us:startIndex_width_us + self.patch_size_us,
                        startIndex_length_us:startIndex_length_us + self.patch_size_us,
                        startIndex_depth_us:startIndex_depth_us + self.patch_size]  # .squeeze()
            else:
                if self.patch_size != -1 and self.pre_interpolate is None:
                    patch = image_us[:, startIndex_width:startIndex_width + self.patch_size,
                            startIndex_length:startIndex_length + self.patch_size,
                            startIndex_depth:startIndex_depth + self.patch_size]  # .squeeze()
                else:
                    patch = image_us[...]

        # target_slices = np.array(target_voxel).astype(
        #     np.float32)  # get slices in range, convert to array, change axis of depth (because nibabel gives LXWXD, but we need in DXLXW) TODO Now it is WXLXD
        # slices = np.array(voxel).astype(
        #     np.float32)  # get slices in range, convert to array, change axis of depth (because nibabel gives LXWXD, but we need in DXLXW) TODO Now it is WXLXD

        # patch = torch.from_numpy(voxel)
        # patch = patch/torch.max(patch)# normalisation
        if self.pre_interpolate:
            patch = F.interpolate(patch.unsqueeze(0).unsqueeze(0), size=tuple(np.roll(groundTruthImages.shape, 1)),
                                  mode=self.pre_interpolate, align_corners=False).squeeze()
            patch = patch[startIndex_width:startIndex_width + self.patch_size,
                    startIndex_length:startIndex_length + self.patch_size,
                    startIndex_depth:startIndex_depth + self.patch_size]
        if self.norm_data:
            patch = patch / imageFile_max  # normalisation

        # targetPatch = torch.from_numpy(target_voxel)
        # targetPatch = targetPatch/torch.max(targetPatch)
        if self.norm_data:
            targetPatch = targetPatch / labelFile_max

        # to deal the patches which has smaller size
        if self.pad_patch:
            pad = ()
            for dim in range(len(targetPatch.shape)):
                target_shape = targetPatch.shape[::-1]
                pad_needed = self.patch_size - target_shape[dim]
                pad_dim = (pad_needed // 2, pad_needed - (pad_needed // 2))
                pad += pad_dim
            if self.patch_size_us is None and self.fly_under_percent is None:
                pad_us = pad
            else:
                pad_us = ()
                if self.patch_size_us is None and self.fly_under_percent is not None:
                    real_patch_us = int(self.patch_size * (
                            self.fly_under_percent * 2))  # TODO: works for 25%, but not sure about others. Need to fix the logic
                else:
                    real_patch_us = self.patch_size_us
                for dim in range(len(patch.shape)):
                    patch_shape = patch.shape[::-1]
                    pad_needed = real_patch_us - patch_shape[dim]
                    pad_dim = (pad_needed // 2, pad_needed - (pad_needed // 2))
                    pad_us += pad_dim
            patch = F.pad(patch, pad_us[:6], value=np.finfo(
                np.float).eps)  # tuple has to be reveresed before using it for padding. As the tuple contains in DHW manner, and input is needed as WHD mannger TODO input already in WXLXD
            targetPatch = F.pad(targetPatch, pad[:6], value=np.finfo(np.float).eps)

        if self.return_coords is True:
            trimmed_label_filename = (self.data.iloc[index, 3]).split("\\")
            trimmed_label_filename = trimmed_label_filename[len(trimmed_label_filename) - 1]
            ground_truth_mip = [lbl for lbl in self.pre_loaded_data['pre_loaded_lbl_mip'] if
                                 lbl['subjectname'] == trimmed_label_filename][0]['data']
            ground_truth_mip_patch = ground_truth_mip[startIndex_width:startIndex_width + self.patch_size,
                                     startIndex_length:startIndex_length + self.patch_size]
            pad = ()
            for dim in range(len(ground_truth_mip_patch.shape)):
                target_shape = ground_truth_mip_patch.shape[::-1]
                pad_needed = self.patch_size - target_shape[dim]
                pad_dim = (pad_needed // 2, pad_needed - (pad_needed // 2))
                pad += pad_dim
            ground_truth_mip_patch = torch.nn.functional.pad(ground_truth_mip_patch, pad[:6], value=np.finfo(np.float).eps)

            subject = tio.Subject(
                img=tio.ScalarImage(tensor=patch),
                label=tio.LabelMap(tensor=targetPatch),
                subjectname=trimmed_label_filename[len(trimmed_label_filename) - 1],
                ground_truth_mip_patch=ground_truth_mip_patch,
                start_coords=start_coords
            )
            return subject
        else:
            return patch, targetPatch

# DATASET_FOLDER = "/nfs1/schatter/Chimp/data_3D_sr/"
# DATASET_FOLDER = r"S:\MEMoRIAL_SharedStorage_M1.2+4+7\Data\Skyra\unet_3D_sr"
# US_Folder = 'Center25Mask'
# patch_size=64

# import logging
# logger = logging.getLogger('x')
# traindataset = SRDataset(logger, patch_size, DATASET_FOLDER + '/usVal/' + US_Folder + '/', DATASET_FOLDER + '/hrVal/', stride_depth =64,
#                                    stride_length=64, stride_width=64,Size =10, patch_size_us=None, return_coords=True)

# train_loader = torch.utils.data.DataLoader(traindataset, batch_size=8, shuffle=True)

# for epoch in range(3):

#    for batch_index, (local_batch, local_labels) in enumerate(train_loader):
#        self.logger.debug(str(epoch) + "  "+ str(batch_index))
