#!/usr/bin/env python
"""

"""

import os
import random

import nibabel
import apex
from apex import amp

import sys
import torch
import torch.utils.data

from Evaluation.evaluate import Dice, FocalTverskyLoss, IOU, getLosses, getMetric
from Utils.customdataset import VesselDataset, validation_VesselDataset
from Utils.elastic_transform import RandomElasticDeformation, warp_image
from Utils.vessel_utils import write_summary, save_model, load_model_with_amp, load_model, create_mask, \
    convert_and_save_tif, create_diff_mask
import numpy as np
import torchvision.transforms as transforms

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


class Pipeline:
    def __init__(self, model, optimizer, logger, with_apex, num_epochs,
                 checkpoint_path, dir_path, writer_training, writer_validating, deform=False,
                 patch_size=64, stride_depth=64, stride_length=64, stride_width=64,
                 samples_per_epoch=8000, batch_size=16,
                 training_set=None, validation_set=None, test_set=None):

        self.model = model
        self.optimizer = optimizer
        self.logger = logger
        self.num_epochs = num_epochs

        self.writer_training = writer_training
        self.writer_validating = writer_validating
        self.checkpoint_path = checkpoint_path
        self.DATASET_FOLDER = dir_path

        self.with_apex = with_apex
        self.deform = deform

        # image input parameters
        self.patch_size = patch_size
        self.stride_depth = stride_depth
        self.stride_length = stride_length
        self.stride_width = stride_width
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size

        # Losses
        self.dice = Dice()
        self.focalTverskyLoss = FocalTverskyLoss()
        self.iou = IOU()

        self.LOWEST_LOSS = 1
        self.test_set = test_set

        traindataset = VesselDataset(logger, self.patch_size,
                                     self.DATASET_FOLDER + 'train/', self.DATASET_FOLDER + 'train_label/',
                                     stride_depth=self.stride_depth, stride_length=self.stride_length,
                                     stride_width=self.stride_width, Size=self.samples_per_epoch,
                                     crossvalidation_set=training_set)
        validationdataset = validation_VesselDataset(logger, self.patch_size, self.DATASET_FOLDER + 'validation/',
                                                     self.DATASET_FOLDER + 'validation_label/',
                                                     stride_depth=self.stride_depth, stride_length=self.stride_length,
                                                     stride_width=self.stride_width, Size=self.samples_per_epoch,
                                                     crossvalidation_set=validation_set)

        self.train_loader = torch.utils.data.DataLoader(traindataset, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=8)
        self.validate_loader = torch.utils.data.DataLoader(validationdataset, batch_size=self.batch_size, shuffle=True,
                                                           num_workers=8)

        self.elastic = RandomElasticDeformation(
            num_control_points=5,
            max_displacement=0.02,
            locked_borders=2
        )
        self.elastic.cuda()

    def train(self):
        self.logger.debug("Training...")

        training_batch_index = 0
        for epoch in range(self.num_epochs):
            self.model.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
            total_floss, total_dloss, total_jaccard_index, total_dscore, total_binloss = 0, 0, 0, 0, 0
            batch_index = 0
            for batch_index, (local_batch, local_labels) in enumerate(self.train_loader):

                # Transfer to GPU
                self.logger.debug('Epoch: {} Batch Index: {}'.format(epoch, batch_index))
                local_batch, local_labels = local_batch[:, None, :].cuda(), local_labels[:, None, :].cuda()

                # Clear gradients
                self.optimizer.zero_grad()

                try:
                    loss_ratios = [1, 0.66, 0.34]

                    floss = torch.empty()
                    output1 = torch.empty()
                    level = 0

                    # --------------------------------------------------------------------------------------------------
                    # First Branch Supervised error
                    for output in self.model(local_batch):
                        if level == 0:
                            output1 = output
                        if level > 0:  # then the output size is reduced, and hence interpolate to patch_size
                            output = torch.nn.functional.interpolate(input=output, size=(64, 64, 64))

                        output = torch.sigmoid(output)
                        floss += loss_ratios[level] * self.focalTverskyLoss(output, local_labels)
                        level += 1

                    # Elastic Deformations
                    if self.deform:
                        local_batch_xt, displacement, inv_displacement = self.elastic(local_batch.cuda())
                        local_labels_xt = warp_image(local_labels.cuda(), displacement, multi=True)
                        floss2 = torch.empty()
                        output2 = torch.empty()

                        level = 0
                        # ------------------------------------------------------------------------------
                        # Second Branch Supervised error
                        for output in self.model(local_batch):
                            if level == 0:
                                output2 = output
                            if level > 0:  # then the output size is reduced, and hence interpolate to patch_size
                                output = torch.nn.functional.interpolate(input=output, size=(64, 64, 64))

                            output = torch.sigmoid(output)
                            floss2 += loss_ratios[level] * self.focalTverskyLoss(output, local_labels_xt)
                            level += 1

                        # -------------------------------------------------------------------------------------------
                        # Consistency loss
                        output1T = warp_image(output1.squeeze(1), displacement, multi=True)
                        floss_c = self.focalTverskyLoss(output2, output1T)

                        # -------------------------------------------------------------------------------------------
                        # Total loss
                        floss = floss + floss2 + floss_c

                except Exception as error:
                    self.logger.exception(error)
                    sys.exit()

                self.logger.info("Epoch:" + str(epoch) + " Batch_Index:" + str(batch_index) + "Training..." +
                                 "\n focalTverskyLoss:" + str(floss))

                # Calculating gradients
                if self.with_apex:
                    with amp.scale_loss(floss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_value_(amp.master_params(self.optimizer), 1)
                else:
                    floss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                self.optimizer.step()

                if training_batch_index % 50 == 0:  # Save best metric evaluation weights
                    write_summary(self.writer_training, self.logger, training_batch_index, local_labels[0][0][6],
                                  output1[0][0][6],
                                  # 6 because in some cases we have padded with 5 which returns background
                                  floss, 0, 0, 0)
                training_batch_index += 1

                # Initialising the average loss metrics
                total_floss += floss.detach().item()

            # Calculate the average loss per batch in one epoch
            total_floss /= (batch_index + 1.0)

            # Print every epoch
            self.logger.info("Epoch:" + str(epoch) + " Average Training..." +
                             "\n focalTverskyLoss:" + str(total_floss))

            torch.cuda.empty_cache()  # to avoid memory errors
            self.validate(training_batch_index, self.model)
            torch.cuda.empty_cache()  # to avoid memory errors

        return self.model

    def validate(self, tainingIndex, currentModel):
        """
        Method to validate
        :param tainingIndex: Epoch after which validation is performed(can be anything for test)
        :param currentModel: model for which validation/test is being performed
        :return:
        """
        self.logger.debug('Validating...')

        floss, binloss, dloss, dscore, jaccard_index = 0, 0, 0, 0, 0
        no_patches = 0
        currentModel.eval()
        data_loader = self.validate_loader
        writer = self.writer_validating
        with torch.no_grad():
            for index, (batch, labels) in enumerate(data_loader):
                self.logger.info("loading" + str(index))
                no_patches += 1
                # Transfer to GPU
                batch, labels = batch[:, None, :].cuda(), labels[:, None, :].cuda()

                floss_iter = torch.empty()
                output1 = torch.empty()
                try:
                    # Forward propagation
                    loss_ratios = [1, 0.66, 0.34]
                    level = 0

                    # Forward propagation
                    for output in currentModel(batch):
                        if level == 0:
                            output1 = output
                        if level > 0:  # then the output size is reduced, and hence interpolate to patch_size
                            output = torch.nn.functional.interpolate(input=output, size=(64, 64, 64))

                        output = torch.sigmoid(output)
                        floss_iter += loss_ratios[level] * self.focalTverskyLoss(output, labels)
                        level += 1
                except Exception as error:
                    self.logger.exception(error)

                floss += floss_iter
                dl, ds = self.dice(output1, labels)
                dloss += dl.detach().item()

        # Average the losses
        floss = floss / no_patches
        dloss = dloss / no_patches
        process = ' Validating'
        self.logger.info("Epoch:" + str(tainingIndex) + process + "..." +
                         "\n FocalTverskyLoss:" + str(floss) +
                         "\n DiceLoss:" + str(dloss))

        write_summary(writer, self.logger, tainingIndex, labels[0][0][6], output1[0][0][6], floss, dloss, 0, 0)

        save_model(self.checkpoint_path, {
            'epoch': 'last',
            # Let is always overwrite, we need just the last checkpoint and best checkpoint(saved below)
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'amp': amp.state_dict()
        })

        if self.LOWEST_LOSS > floss:  # Save best metric evaluation weights
            self.LOWEST_LOSS = floss
            self.logger.info(
                'Best metric... @ epoch:' + str(tainingIndex) + ' Current Lowest loss:' + str(self.LOWEST_LOSS))

            save_model(self.checkpoint_path, {
                'epoch': 'best',
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'amp': amp.state_dict()}, best_metric=True)

    def test(self, test_logger, batch_index='best'):
        test_logger.debug('Testing...')

        test_folder_path = self.DATASET_FOLDER + 'test/'
        test_label_path = self.DATASET_FOLDER + 'test_label/'

        if self.with_apex:
            model, _, __ = load_model_with_amp(self.model, self.checkpoint_path, batch_index)
        else:
            model, _ = load_model(self.model, self.checkpoint_path, batch_index)

        final_true_pos, final_false_neg, final_false_pos, final_intersection, final_union = 0, 0, 0, 0, 0


        image_names = os.listdir(self.test_folder_path) if self.test_set == None else self.test_set
        for image_file_name in image_names:  # Parallelly read image file and groundtruth
            image_file = nibabel.load(
                os.path.join(test_folder_path, image_file_name))  # shape (Length X Width X Depth X Channels)
            header_shape = image_file.header.get_data_shape()
            n_depth, n_length, n_width = header_shape[2], header_shape[0], header_shape[
                1]  # gives depth which is no. of slices

            test_logger.debug("ImageName:" + str(image_file_name))
            test_logger.debug("Dimensions:" + str(n_depth) + " X " + str(n_length) + " X " + str(n_width))

            total_true_pos, total_false_neg, total_false_pos, total_intersection, total_union = 0, 0, 0, 0, 0
            depth_i = 0
            for depth_index in range(int((n_depth - self.patch_size) / self.stride_depth) + 1):
                test_logger.debug('depth...')
                test_logger.debug(depth_i)

                length_i = 0
                for length_index in range(int((n_length - self.patch_size) / self.stride_length) + 1):
                    width_i = 0
                    for width_index in range(int((n_width - self.patch_size) / self.stride_width) + 1):
                        images = nibabel.load(
                            os.path.join(test_folder_path, image_file_name))  # shape (Length X Width X Depth X Channels)
                        ground_truth_images = nibabel.load(os.path.join(test_label_path, image_file_name))

                        # get patch
                        voxel = images.dataobj[length_i:length_i + self.patch_size,
                                width_i:width_i + self.patch_size,
                                depth_i:depth_i + self.patch_size, 0]
                        slices = np.moveaxis(np.array(voxel), -1, 0).astype(
                            np.float32)  # get slices in range, convert to array, change axis of depth (because nibabel gives LXWXD, but we need in DXLXW)
                        patch = torch.from_numpy(slices)
                        patch = patch / torch.max(patch)  # normalisation

                        # get target patch
                        target_voxel = ground_truth_images.dataobj[length_i:length_i + self.patch_size,
                                       width_i:width_i + self.patch_size,
                                       depth_i:depth_i + self.patch_size, 0]
                        target_slices = np.moveaxis(np.array(target_voxel), -1, 0).astype(
                            np.float32)  # get slices in range, convert to array, change axis of depth (because nibabel gives LXWXD, but we need in DXLXW)
                        target_patch = torch.from_numpy(target_slices)
                        # targetPatch = targetPatch / 255

                        target_patch = torch.where(torch.eq(target_patch, 2), 0 * torch.ones_like(target_patch),
                                                  target_patch)  # convert all 2's to 0's (2 means background, so make it 0)
                        # targetPatch = targetPatch / 255.0

                        batch, label = patch, target_patch
                        batch, label = batch[None, None, :].cuda(), label[None, None, :].cuda()

                        output = torch.empty()
                        for output1 in model(batch):
                            output = torch.sigmoid(output1)
                            break  # We need only the output from last level

                        # image_output[depth_i:depth_i+stride_depth, length_i:length_i+stride_length, width_i:width_i+stride_width] =  output
                        # label_output[depth_i:depth_i+stride_depth, length_i:length_i+stride_length, width_i:width_i+stride_width] =  label

                        true_pos, false_neg, false_pos, intersection, union = getMetric(test_logger, output, label)
                        total_true_pos += true_pos.detach().item()
                        total_false_neg += false_neg.detach().item()
                        total_false_pos += false_pos.detach().item()
                        total_intersection += intersection.detach().item()
                        total_union += union.detach().item()

                        del images, ground_truth_images
                        width_i += self.stride_width
                    length_i += self.stride_length
                depth_i += self.stride_depth

            test_logger.debug("Testing for image: " + str(image_file_name))
            # Average the losses
            # floss = focalTverskyLoss(image_output, label_output).detach().item()
            # dloss, dscore  = dice(image_output, label_output)
            # jaccardIndex = iou(image_output, label_output)
            floss, dloss, jaccard_index = getLosses(test_logger, total_true_pos, total_false_neg, total_false_pos,
                                                   total_intersection, total_union)
            test_logger.info("Testing..." +
                             "\n FocalTverskyLoss:" + str(floss) +
                             "\n DiceLoss:" + str(dloss) +
                             "\n JacardIndex:" + str(jaccard_index))

            final_true_pos += total_true_pos
            final_false_neg += total_false_neg
            final_false_pos += total_false_pos
            final_intersection += total_intersection
            final_union += total_union

        floss, dloss, jaccard_index = getLosses(test_logger, final_true_pos, final_false_neg, final_false_pos,
                                               final_intersection, final_union)
        test_logger.info("Average Testing..." +
                         "\n FocalTverskyLoss:" + str(floss) +
                         "\n DiceLoss:" + str(dloss) +
                         "\n JacardIndex:" + str(jaccard_index))



    def get_patch(self, images, isLabel, patch_size, startindex_depth, startindex_length, startindex_width):
        voxel = images.dataobj[startindex_length:startindex_length + patch_size,
                startindex_width:startindex_width + patch_size,
                startindex_depth:startindex_depth + patch_size, 0]
        slices = np.moveaxis(np.array(voxel), -1, 0).astype(np.float32)  # get slices in range, convert to array, change axis of depth (because nibabel gives LXWXD, but we need in DXLXW)
        patch = torch.from_numpy(slices)

        if not isLabel:
            patch = patch / torch.max(patch)  # normalisation
        else:
            patch = torch.where(torch.eq(patch, 2), 0 * torch.ones_like(patch),patch)  # convert all 2's to 0's (2 means background, so make it 0)

        return patch


    def predict(self,model_name, image_path, label_path, output_path):
        given_label = bool(label_path)

        image_file = nibabel.load(image_path)

        if given_label:
            groudtruth_file = nibabel.load(label_path)

        header_shape = image_file.header.get_data_shape()
        n_depth, n_length, n_width = header_shape[2], header_shape[0], header_shape[1]  # gives depth which is no. of slices

        image_output = torch.zeros([n_depth, n_length, n_width])
        image_output.cuda()

        self.logger.debug('Predicting...')

        with torch.no_grad():
            header_shape = image_file.header.get_data_shape()
            n_depth, n_length, n_width = header_shape[2], header_shape[0], header_shape[1]  # gives depth which is no. of slices

            image_output = torch.zeros([n_depth, n_length, n_width])
            image_output.cuda()

            flag_holder = torch.zeros([n_depth, n_length, n_width])
            flag_holder.cuda()

            all_ones_patch = torch.ones([self.patch_size, self.patch_size, self.patch_size])
            all_ones_patch.cuda()

            depth_i = 0
            for depth_index in range(1 + int(
                    (
                            n_depth - self.patch_size) / self.stride_depth)):  # iterate through the whole image voxel, and extract patch
                length_i = 0
                print("depth")
                print(depth_i)
                for length_index in range(1 + int((n_length - self.patch_size) / self.stride_length)):
                    width_i = 0
                    for width_index in range(1 + int((n_width - self.patch_size) / self.stride_width)):
                        batch, label = self.get_patch(image_file, False, self.patch_size, depth_i, length_i,
                                                      width_i)
                        batch = batch[None, None, :].cuda()

                        output = torch.empty()
                        for output1 in self.model(batch)(batch):
                            output = torch.sigmoid(output1)
                            break  # We need only the output from last level

                        if (torch.sum(flag_holder[depth_i:depth_i + self.patch_size, length_i:length_i + self.patch_size,
                                      width_i:width_i + self.patch_size]) == 0):  # fresh start
                            image_output[depth_i:depth_i + self.patch_size, length_i:length_i + self.patch_size,
                            width_i:width_i + self.patch_size] = output
                            flag_holder[depth_i:depth_i + self.patch_size, length_i:length_i + self.patch_size,
                            width_i:width_i + self.patch_size] += all_ones_patch

                        else:
                            temp_tensor = image_output[depth_i:depth_i + self.patch_size, length_i:length_i + self.patch_size,
                                         width_i:width_i + self.patch_size] * flag_holder[depth_i:depth_i + self.patch_size,
                                                                         length_i:length_i + self.patch_size,
                                                                         width_i:width_i + self.patch_size]
                            flag_holder[depth_i:depth_i + self.patch_size, length_i:length_i + self.patch_size,
                            width_i:width_i + self.patch_size] += all_ones_patch
                            image_output[depth_i:depth_i + self.patch_size, length_i:length_i + self.patch_size,
                            width_i:width_i + self.patch_size] = (output + temp_tensor) / flag_holder[
                                                                                    depth_i:depth_i + self.patch_size,
                                                                                    length_i:length_i + self.patch_size,
                                                                                    width_i:width_i + self.patch_size]

                        width_i += self.stride_width
                    length_i += self.stride_length
                depth_i += self.stride_depth

            del flag_holder  # no need of this file after this point, saves memory

            if given_label:
                label_output =  groudtruth_file.dataobj[::, ::, ::, 0]
                target_slices = np.moveaxis(np.array(label_output), -1, 0).astype(
                np.float32)  # get slices in range, convert to array, change axis of depth (because nibabel gives LXWXD, but we need in DXLXW)
                targetPatch = torch.from_numpy(target_slices)
                targetPatch = torch.where(torch.eq(targetPatch, 2), 0 * torch.ones_like(targetPatch),
                                      targetPatch)  # convert all 2's to 0's (2 means background, so make it 0)
                targetPatch.cuda()

            print("image_output shape:" + str(image_output.shape))

            trans = transforms.ToTensor()
            outputpatch = torch.Tensor();
            for i in range(image_output.shape[0]):
                try:
                    # direct_outputpatch = torch.cat([direct_outputpatch, trans(diff_image)], dim=0)
                    diff_image = create_mask(image_output[i], self.logger)
                    outputpatch = torch.cat([outputpatch, trans(diff_image)], dim=0)
                    print("outputpatch shape:" + str(outputpatch.shape))

                except Exception as error:
                    self.logger.exception(error)
                    # writer.close()
            convert_and_save_tif(outputpatch, output_path, filename=self.MODEL_NAME + '_actual.tif', isColored=False)

            del outputpatch

            if given_label:

                outputpatch = torch.Tensor();
                # Get colored output
                for i in range(image_output.shape[0]):
                    try:
                      # direct_outputpatch = torch.cat([direct_outputpatch, trans(diff_image)], dim=0)
                        diff_image = create_diff_mask(image_output[i],
                                                  targetPatch[i],
                                                  self.logger)  # get the difference between label and predicted
                        outputpatch = torch.cat([outputpatch, trans(diff_image)], dim=0)
                        print("outputpatch shape:" + str(outputpatch.shape))

                    except Exception as error:
                        self.logger.exception(error)
                        # writer.close()
                convert_and_save_tif(outputpatch, output_path, filename=model_name + '_color.tif')