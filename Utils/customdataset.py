#!/usr/bin/env python

# from __future__ import print_function, division
'''

Purpose : 

'''


import os

import pandas as pd
import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np
import nibabel
from random import seed
from random import randint

__author__ = "Kartik Prabhu, Mahantesh Pattadkal, and Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Kartik Prabhu", "Mahantesh Pattadkal", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

torch.manual_seed(2020)
np.random.seed(2020)
seed(2020)

def _check_label_filename(label_dir_path, imageFileName):
    labelfile = os.path.join(label_dir_path, imageFileName)
    if os.path.isfile(labelfile):
        return labelfile
    else:
        if ".gz" in labelfile:
            labelfile = labelfile.replace(".gz", "")
        else:
            labelfile = labelfile + ".gz"
        return labelfile 

class VesselDataset(Dataset):
    """
    VesselDataset - Custom dataset used to get images and labels

    patch_size : Size of a 3d voxel to be used for training (Note: Ideal selection would have a number which is iteratively divisible by 2, to avoid upsampling problems)
    dir_path : Folder path for  images
    label_dir_path : Folder path for the ground truth images
    """

    def __init__(self,logger, patch_size, dir_path, label_dir_path, stride_depth=16, stride_length=32, stride_width=32,Size=4000, crossvalidation_set = None):
        self.patch_size = patch_size
        self.stride_depth = stride_depth
        self.stride_length = stride_length
        self.stride_width = stride_width
        self.image_names = os.listdir(dir_path) if crossvalidation_set == None else crossvalidation_set
        self.size = Size
        self.logger = logger

        # Constants
        self.IMAGE_FILE_NAME = "imageFilename"
        self.LABEL_FILE_NAME = "labelFilename"
        self.STARTINDEX_DEPTH = "startIndex_depth"
        self.STARTINDEX_LENGTH = "startIndex_length"
        self.STARTINDEX_WIDTH = "startIndex_width"

        self.trans = transforms.ToTensor()  # used to convert tiffimagefile to tensor
        dataDict = { self.IMAGE_FILE_NAME: [], self.LABEL_FILE_NAME: [], self.STARTINDEX_DEPTH: [],self.STARTINDEX_LENGTH: [],self.STARTINDEX_WIDTH: []}

        column_names = [ self.IMAGE_FILE_NAME, self.LABEL_FILE_NAME, self.STARTINDEX_DEPTH, self.STARTINDEX_LENGTH,self.STARTINDEX_WIDTH]
        self.data = pd.DataFrame(columns=column_names)

        for imageFileName in self.image_names:  # Parallelly read image file and groundtruth
            if not imageFileName or 'nfs' in imageFileName:
                continue

            imageFile = nibabel.load(os.path.join(dir_path, imageFileName)) # shape (Length X Width X Depth X Channels)
            header_shape = imageFile.header.get_data_shape()
            n_depth,n_length,n_width = header_shape[2],header_shape[0],header_shape[1] # gives depth which is no. of slices

            depth_i =0
            for depth_index in range(int((n_depth-patch_size)/stride_depth)+1):  # iterate through the whole image voxel, and extract patch
                length_i = 0
                # print("depth")
                # print(depth_i)
                for length_index in range(int((n_length-patch_size)/stride_length)+1):
                    width_i = 0
                    # print("length")
                    # print(length_i)

                    for width_index in range(int((n_width - patch_size)/stride_width)+1):
                        # print("width")
                        # print(width_i)
                        dataDict[self.IMAGE_FILE_NAME].append(os.path.join(dir_path, imageFileName))
                        dataDict[self.LABEL_FILE_NAME].append(_check_label_filename(label_dir_path, imageFileName))
                        dataDict[self.STARTINDEX_DEPTH].append(depth_i)
                        dataDict[self.STARTINDEX_LENGTH].append(length_i)
                        dataDict[self.STARTINDEX_WIDTH].append(width_i)
                        width_i += stride_width          
                    length_i += stride_length                
                depth_i += stride_depth
            
        self.data = pd.DataFrame.from_dict(dataDict)
        print(len(self.data))               

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        '''
        imageFilename: 0
        labelFilename: 1
        startIndex_depth : 2
        startIndex_length : 3
        startIndex_width : 4

        '''
        
        index  = randint(0,len(self.data)-1)

        images = nibabel.load(self.data.iloc[index, 0])
        groundTruthImages = nibabel.load(self.data.iloc[index, 1])
        
        startIndex_depth = self.data.iloc[index, 2]
        startIndex_length = self.data.iloc[index, 3]
        startIndex_width = self.data.iloc[index, 4]

        # voxel = images.dataobj[startIndex_length:startIndex_length+self.patch_size, startIndex_width:startIndex_width+self.patch_size, startIndex_depth:startIndex_depth+self.patch_size,0]
        voxel = images.dataobj[startIndex_length:startIndex_length+self.patch_size, startIndex_width:startIndex_width+self.patch_size, startIndex_depth:startIndex_depth+self.patch_size].squeeze()
        slices = np.moveaxis(np.array(voxel),-1, 0).astype(np.float32) #get slices in range, convert to array, change axis of depth (because nibabel gives LXWXD, but we need in DXLXW)
        patch =  torch.from_numpy(slices)
        patch = patch/torch.max(patch)# normalisation
        
        # target_voxel = groundTruthImages.dataobj[startIndex_length:startIndex_length+self.patch_size, startIndex_width:startIndex_width+self.patch_size, startIndex_depth:startIndex_depth+self.patch_size,0]
        target_voxel = groundTruthImages.dataobj[startIndex_length:startIndex_length+self.patch_size, startIndex_width:startIndex_width+self.patch_size, startIndex_depth:startIndex_depth+self.patch_size].squeeze()
        target_slices = np.moveaxis(np.array(target_voxel), -1, 0).astype( np.float32)  # get slices in range, convert to array, change axis of depth (because nibabel gives LXWXD, but we need in DXLXW)
        targetPatch = torch.from_numpy(target_slices)
        # targetPatch = targetPatch/255.0
        # targetPatch = torch.where(torch.eq(targetPatch, 2), 0 * torch.ones_like(targetPatch), targetPatch) #Ilastik Fix: convert all 2's to 0's (2 means background, so make it 0). Soumick's Advice: Don't do it here. Pre-process before supplying

        #Checking if the entire input voxel or the segmentation mask is black
        while (torch.sum(patch)==0 or torch.sum(targetPatch)==0 ):
            index  = randint(0,len(self.data)-1)
            images = nibabel.load(self.data.iloc[index, 0])
            groundTruthImages = nibabel.load(self.data.iloc[index, 1])
            
            startIndex_depth = self.data.iloc[index, 2]
            startIndex_length = self.data.iloc[index, 3]
            startIndex_width = self.data.iloc[index, 4]
    
            voxel = images.dataobj[startIndex_length:startIndex_length+self.patch_size, startIndex_width:startIndex_width+self.patch_size, startIndex_depth:startIndex_depth+self.patch_size].squeeze()
            slices = np.moveaxis(np.array(voxel),-1, 0).astype(np.float32) #get slices in range, convert to array, change axis of depth (because nibabel gives LXWXD, but we need in DXLXW)
            patch =  torch.from_numpy(slices)
            patch = patch/torch.max(patch)# normalisation

        #no need of padding with voxel size is 64x64x64

            target_voxel = groundTruthImages.dataobj[startIndex_length:startIndex_length+self.patch_size, startIndex_width:startIndex_width+self.patch_size, startIndex_depth:startIndex_depth+self.patch_size].squeeze()
            target_slices = np.moveaxis(np.array(target_voxel), -1, 0).astype( np.float32)  # get slices in range, convert to array, change axis of depth (because nibabel gives LXWXD, but we need in DXLXW)
            targetPatch = torch.from_numpy(target_slices)
            # targetPatch = targetPatch/255.0
            # targetPatch = torch.where(torch.eq(targetPatch, 2), 0 * torch.ones_like(targetPatch), targetPatch)#convert all 2's to 0's (2 means background, so make it 0)

        return patch, targetPatch


class validation_VesselDataset(Dataset):
    """
    VesselDataset - Custom dataset used to get images and labels

    patch_size : Size of a 3d voxel to be used for training (Note: Ideal selection would have a number which is iteratively divisible by 2, to avoid upsampling problems)
    dir_path : Folder path for  images
    label_dir_path : Folder path for the ground truth images
    """

    def __init__(self,logger, patch_size, dir_path, label_dir_path, stride_depth=64, stride_length=64, stride_width=64,Size=4000, crossvalidation_set = None):
        self.patch_size = patch_size
        self.stride_depth = stride_depth
        self.stride_length = stride_length
        self.stride_width = stride_width
        self.size = Size
        self.image_names = os.listdir(dir_path) if crossvalidation_set == None else crossvalidation_set
        self.logger = logger

        # Constants
        self.IMAGE_FILE_NAME = "imageFilename"
        self.LABEL_FILE_NAME = "labelFilename"
        self.STARTINDEX_DEPTH = "startIndex_depth"
        self.STARTINDEX_LENGTH = "startIndex_length"
        self.STARTINDEX_WIDTH = "startIndex_width"

        self.trans = transforms.ToTensor()  # used to convert tiffimagefile to tensor
        dataDict = { self.IMAGE_FILE_NAME: [], self.LABEL_FILE_NAME: [], self.STARTINDEX_DEPTH: [],self.STARTINDEX_LENGTH: [],self.STARTINDEX_WIDTH: []}

        column_names = [ self.IMAGE_FILE_NAME, self.LABEL_FILE_NAME, self.STARTINDEX_DEPTH, self.STARTINDEX_LENGTH,self.STARTINDEX_WIDTH]
        self.data = pd.DataFrame(columns=column_names)
                        
        
        for imageFileName in self.image_names:  # Parallelly read image file and groundtruth
            if not imageFileName or 'nfs' in imageFileName:
                self.logger.debug("skipping file:"+ str(imageFileName))
                continue
            imageFile = nibabel.load(os.path.join(dir_path, imageFileName)) # shape (Length X Width X Depth X Channels)
            header_shape = imageFile.header.get_data_shape()
            n_depth,n_length,n_width = header_shape[2],header_shape[0],header_shape[1] # gives depth which is no. of slices

            depth_i =0
            for depth_index in range(int((n_depth-patch_size)/stride_depth)+1):  # iterate through the whole image voxel, and extract patch
                length_i = 0
                # print("depth")
                # print(depth_i)
                for length_index in range(int((n_length-patch_size)/stride_length)+1):
                    width_i = 0
                    # print("length")
                    # print(length_i)

                    for width_index in range(int((n_width - patch_size)/stride_width)+1):
                        # print("width")
                        # print(width_i)
                        dataDict[self.IMAGE_FILE_NAME].append(os.path.join(dir_path, imageFileName))
                        dataDict[self.LABEL_FILE_NAME].append(_check_label_filename(label_dir_path, imageFileName))
                        dataDict[self.STARTINDEX_DEPTH].append(depth_i)
                        dataDict[self.STARTINDEX_LENGTH].append(length_i)
                        dataDict[self.STARTINDEX_WIDTH].append(width_i)
                        width_i += stride_width          
                    length_i += stride_length                
                depth_i += stride_depth
            
        self.data = pd.DataFrame.from_dict(dataDict)
        print(len(self.data))
 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
        imageFilename: 0
        labelFilename: 1
        startIndex_depth : 2
        startIndex_length : 3
        startIndex_width : 4

        '''
        
        
        images = nibabel.load(self.data.iloc[idx, 0])
        groundTruthImages = nibabel.load(self.data.iloc[idx, 1])
        
        startIndex_depth = self.data.iloc[idx, 2]
        startIndex_length = self.data.iloc[idx, 3]
        startIndex_width = self.data.iloc[idx, 4]

        # voxel = images.dataobj[startIndex_length:startIndex_length+self.patch_size, startIndex_width:startIndex_width+self.patch_size, startIndex_depth:startIndex_depth+self.patch_size,0]
        voxel = images.dataobj[startIndex_length:startIndex_length+self.patch_size, startIndex_width:startIndex_width+self.patch_size, startIndex_depth:startIndex_depth+self.patch_size].squeeze()
        slices = np.moveaxis(np.array(voxel),-1, 0).astype(np.float32) #get slices in range, convert to array, change axis of depth (because nibabel gives LXWXD, but we need in DXLXW)
        patch =  torch.from_numpy(slices)
        patch = patch/torch.max(patch)# normalisation
        
        # target_voxel = groundTruthImages.dataobj[startIndex_length:startIndex_length+self.patch_size, startIndex_width:startIndex_width+self.patch_size, startIndex_depth:startIndex_depth+self.patch_size,0]
        target_voxel = groundTruthImages.dataobj[startIndex_length:startIndex_length+self.patch_size, startIndex_width:startIndex_width+self.patch_size, startIndex_depth:startIndex_depth+self.patch_size].squeeze()
        target_slices = np.moveaxis(np.array(target_voxel), -1, 0).astype( np.float32)  # get slices in range, convert to array, change axis of depth (because nibabel gives LXWXD, but we need in DXLXW)
        targetPatch = torch.from_numpy(target_slices)
        # targetPatch = targetPatch/255.0
        # targetPatch = torch.where(torch.eq(targetPatch, 2), 0 * torch.ones_like(targetPatch), targetPatch)#convert all 2's to 0's (2 means background, so make it 0)

        return patch, targetPatch

