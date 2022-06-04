# !/usr/bin/env python
"""

"""

import os
import random
import sys
from glob import glob

import nibabel
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchio as tio
import torchvision.transforms as transforms
from skimage.filters import threshold_otsu
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from Evaluation.evaluate import (IOU, Dice, FocalTverskyLoss, MIP_Loss, getLosses,
                                 getMetric)
from Utils.elastic_transform import RandomElasticDeformation, warp_image
from Utils.result_analyser import *
from Utils.vessel_utils import (convert_and_save_tif, create_diff_mask,
                                create_mask, load_model, load_model_with_amp,
                                save_model, write_summary, write_Epoch_summary)
from Utils.datasets import SRDataset

__author__ = "Kartik Prabhu, Mahantesh Pattadkal, and Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Kartik Prabhu", "Mahantesh Pattadkal", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"


class Pipeline:

    def __init__(self, cmd_args, model, logger, dir_path, checkpoint_path, writer_training, writer_validating,
                 training_set=None, validation_set=None, test_set=None):

        self.logger = logger
        self.model = model
        self.lr_1 = cmd_args.learning_rate
        self.logger.info("learning rate " + str(self.lr_1))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cmd_args.learning_rate)
        self.num_epochs = cmd_args.num_epochs

        self.writer_training = writer_training
        self.writer_validating = writer_validating
        self.checkpoint_path = checkpoint_path
        self.DATASET_FOLDER = dir_path
        self.output_path = cmd_args.output_path

        self.model_name = cmd_args.model_name

        self.clip_grads = cmd_args.clip_grads
        self.with_apex = cmd_args.apex
        self.deform = cmd_args.deform

        # image input parameters
        self.patch_size = cmd_args.patch_size
        self.stride_depth = cmd_args.stride_depth
        self.stride_length = cmd_args.stride_length
        self.stride_width = cmd_args.stride_width
        self.samples_per_epoch = cmd_args.samples_per_epoch

        # execution configs
        self.batch_size = cmd_args.batch_size
        self.num_worker = cmd_args.num_worker

        # Losses
        self.dice = Dice()
        self.focalTverskyLoss = FocalTverskyLoss()
        self.mip_loss = MIP_Loss()
        self.floss_coeff = cmd_args.floss_coeff
        self.mip_loss_coeff = cmd_args.mip_loss_coeff
        self.iou = IOU()

        self.LOWEST_LOSS = float('inf')
        self.test_set = test_set

        if self.with_apex:
            self.scaler = GradScaler()

        # set probabilistic property
        if "Models.prob" in self.model.__module__:
            self.isProb = True
            from Models.prob_unet.utils import l2_regularisation
            self.l2_regularisation = l2_regularisation
        else:
            self.isProb = False

        if cmd_args.train:  # Only if training is to be performed
                traindataset = self.create_TIOSubDS(vol_path=self.DATASET_FOLDER + '/train/',
                                                    label_path=self.DATASET_FOLDER + '/train_label/',
                                                    crossvalidation_set=training_set)
                validationdataset = self.create_TIOSubDS(vol_path=self.DATASET_FOLDER + '/validate/',
                                                         label_path=self.DATASET_FOLDER + '/validate_label/',
                                                         crossvalidation_set=validation_set, is_train=False, is_validate=True)
                sampler = torch.utils.data.RandomSampler(data_source=traindataset, replacement=True, num_samples=self.samples_per_epoch)
                self.train_loader = torch.utils.data.DataLoader(traindataset, batch_size=self.batch_size, shuffle=False,
                                                                num_workers=self.num_worker, pin_memory=True,
                                                                sampler=sampler)
                self.validate_loader = torch.utils.data.DataLoader(validationdataset, batch_size=self.batch_size,
                                                                   shuffle=False,
                                                                   num_workers=self.num_worker, pin_memory=True)

    def create_TIOSubDS(self, vol_path, label_path, crossvalidation_set=None, is_train=True, is_validate=False, get_subjects_only=False,
                        transforms=None):
        if is_train:
            trainDS = SRDataset(logger=self.logger, patch_size=self.patch_size,
                                dir_path=self.DATASET_FOLDER + '/train/',
                                label_dir_path=self.DATASET_FOLDER + '/train_label/',
                                # TODO: implement non-iso patch-size, now only using the first element
                                stride_depth=self.stride_depth, stride_length=self.stride_length,
                                stride_width=self.stride_width, Size=None, fly_under_percent=None,
                                # TODO: implement fly_under_percent, if needed
                                patch_size_us=self.patch_size, pre_interpolate=None, norm_data=False,
                                pre_load=True,
                                return_coords=True)  # TODO implement patch_size_us if required - patch_size//scaling_factor
            if get_subjects_only:
                return trainDS
            # sampler = tio.data.UniformSampler(self.patch_size)
            # patches_queue = tio.Queue(
            #     trainDS,
            #     max_length=(self.samples_per_epoch // len(trainDS.pre_loaded_data['pre_loaded_img'])) * 2,
            #     samples_per_volume=1,
            #     sampler=sampler,
            #     num_workers=0,
            #     start_background=True
            # )
            # return patches_queue
            return trainDS
        elif is_validate:
            validationDS = SRDataset(logger=self.logger, patch_size=self.patch_size,
                                dir_path=self.DATASET_FOLDER + '/validate/',
                                label_dir_path=self.DATASET_FOLDER + '/validate_label/',
                                # TODO: implement non-iso patch-size, now only using the first element
                                stride_depth=self.stride_depth, stride_length=self.stride_length,
                                stride_width=self.stride_width, Size=None, fly_under_percent=None,
                                # TODO: implement fly_under_percent, if needed
                                patch_size_us=self.patch_size, pre_interpolate=None, norm_data=False,
                                pre_load=True,
                                return_coords=True)  # TODO implement patch_size_us if required - patch_size//scaling_factor
            overlap = np.subtract(self.patch_size, (self.stride_length, self.stride_width, self.stride_depth))
            grid_samplers = []
            for i in range(len(validationDS)):
                grid_sampler = tio.inference.GridSampler(
                    validationDS[i],
                    self.patch_size,
                    overlap,
                )
                grid_samplers.append(grid_sampler)
            return torch.utils.data.ConcatDataset(grid_samplers)
        else:
            vols = glob(vol_path + "*.nii") + glob(vol_path + "*.nii.gz")
            labels = glob(label_path + "*.nii") + glob(label_path + "*.nii.gz")
            subjects = []
            for i in range(len(vols)):
                v = vols[i]
                filename = os.path.basename(v).split('.')[0]
                l = [s for s in labels if filename in s][0]
                subject = tio.Subject(
                    img=tio.ScalarImage(v),
                    label=tio.LabelMap(l),
                    subjectname=filename,
                )
                transforms = tio.ToCanonical(), tio.Resample(tio.ScalarImage(v))
                transform = tio.Compose(transforms)
                subject = transform(subject)
                subjects.append(subject)

            if get_subjects_only:
                return subjects

            overlap = np.subtract(self.patch_size, (self.stride_length, self.stride_width, self.stride_depth))
            grid_samplers = []
            for i in range(len(subjects)):
                grid_sampler = tio.inference.GridSampler(
                    subjects[i],
                    self.patch_size,
                    overlap,
                )
                grid_samplers.append(grid_sampler)
            return torch.utils.data.ConcatDataset(grid_samplers)

    def normaliser(self, batch):
        for i in range(batch.shape[0]):
            if batch[i].max() > 0.0:
                batch[i] = batch[i] / batch[i].max()
        return batch

    def load(self, checkpoint_path=None, load_best=True):
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        if self.with_apex:
            self.model, self.optimizer, self.scaler = load_model_with_amp(self.model, self.optimizer, checkpoint_path,
                                                                          batch_index="best" if load_best else "last")
        else:
            self.model, self.optimizer = load_model(self.model, self.optimizer, checkpoint_path,
                                                    batch_index="best" if load_best else "last")

    def train(self):
        self.logger.debug("Training...")

        training_batch_index = 0
        for epoch in range(self.num_epochs):
            print("Train Epoch: " + str(epoch) + " of " + str(self.num_epochs))
            self.model.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
            total_floss = 0
            total_mipLoss = 0
            total_DiceLoss = 0
            total_IOU = 0
            total_DiceScore = 0
            batch_index = 0
            for batch_index, patches_batch in enumerate(tqdm(self.train_loader)):

                local_batch = self.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                local_labels = patches_batch['label'][tio.DATA].float().cuda()

                local_batch = torch.movedim(local_batch, -1, -3)
                local_labels = torch.movedim(local_labels, -1, -3)

                # Transfer to GPU
                self.logger.debug('Epoch: {} Batch Index: {}'.format(epoch, batch_index))

                # Clear gradients
                self.optimizer.zero_grad()

                # try:
                with autocast(enabled=self.with_apex):
                    loss_ratios = [1, 0.66, 0.34]  # TODO param

                    floss = 0
                    mip_loss = 0
                    output1 = 0
                    level = 0
                    diceLoss_batch = 0
                    diceScore_batch = 0
                    IOU_batch = 0

                    # -------------------------------------------------------------------------------------------------
                    # First Branch Supervised error
                    if not self.isProb:
                        # Compute DiceLoss using batch labels
                        for output in self.model(local_batch):
                            if level == 0:
                                output1 = output
                            if level > 0:  # then the output size is reduced, and hence interpolate to patch_size
                                output = torch.nn.functional.interpolate(input=output, size=(64, 64, 64))
                            output = torch.sigmoid(output)
                            dl_batch, ds_batch = self.dice(output, local_labels)
                            IOU_score = self.iou(output, local_labels)
                            diceLoss_batch += dl_batch.detach().item()
                            diceScore_batch += ds_batch.detach().item()
                            IOU_batch += IOU_score.detach().item()
                            floss += loss_ratios[level] * self.focalTverskyLoss(output, local_labels)
                            # Compute MIP loss from the patch on the MIP of the 3D label and the patch prediction
                            mip_loss_patch = 0
                            num_patches = 0
                            for index, op in enumerate(output):
                                op_mip = torch.amax(op, -1)
                                mip_loss_patch += loss_ratios[level] * self.focalTverskyLoss(op_mip,
                                                  patches_batch['ground_truth_mip_patch'][index].float().cuda())
                            if not torch.any(torch.isnan(mip_loss_patch)):
                                mip_loss += mip_loss_patch / len(output)
                            # mip_loss += loss_ratios[level] * self.mip_loss(output, patches_batch, self.pre_loaded_train_lbl_data, self.focalTverskyLoss, self.patch_size)

                            level += 1
                    else:
                        self.model.forward(local_batch, local_labels, training=True)
                        elbo = self.model.elbo(local_labels, analytic_kl=True)
                        reg_loss = self.l2_regularisation(self.model.posterior) + self.l2_regularisation(
                            self.model.prior) + self.l2_regularisation(self.model.fcomb.layers)
                        if self.with_apex:
                            floss = [
                                self.model.mean_reconstruction_loss if self.model.use_mean_recon_loss else self.model.reconstruction_loss,
                                -(self.model.beta * self.model.kl),
                                self.model.reg_alpha * reg_loss]
                        else:
                            floss = -elbo + self.model.reg_alpha * reg_loss

                    # Elastic Deformations
                    if self.deform:
                        # Each batch must be randomly deformed
                        elastic = RandomElasticDeformation(
                            num_control_points=random.choice([5, 6, 7]),
                            max_displacement=random.choice([0.01, 0.015, 0.02, 0.025, 0.03]),
                            locked_borders=2
                        )
                        elastic.cuda()

                        with autocast(enabled=False):
                            local_batch_xt, displacement, _ = elastic(local_batch)
                            local_labels_xt = warp_image(local_labels, displacement, multi=True)
                        floss2 = 0

                        level = 0
                        # ------------------------------------------------------------------------------
                        # Second Branch Supervised error
                        for output in self.model(local_batch_xt):
                            if level == 0:
                                output2 = output
                            if level > 0:  # then the output size is reduced, and hence interpolate to patch_size
                                output = torch.nn.functional.interpolate(input=output, size=(64, 64, 64))

                            output = torch.sigmoid(output)
                            floss2 += loss_ratios[level] * self.focalTverskyLoss(output, local_labels_xt)
                            level += 1

                        # -------------------------------------------------------------------------------------------
                        # Consistency loss
                        with autocast(enabled=False):
                            output1T = warp_image(output1.float(), displacement, multi=True)
                        floss_c = self.focalTverskyLoss(torch.sigmoid(output2), output1T)

                        # -------------------------------------------------------------------------------------------
                        # Total loss
                        floss = floss + floss2 + floss_c

                    else:
                        floss = floss + (self.mip_loss_coeff * mip_loss)



                # except Exception as error:
                #     self.logger.exception(error)
                #     sys.exit()

                self.logger.info("Epoch:" + str(epoch) + " Batch_Index:" + str(batch_index) + " Training..." +
                                 "\n focalTverskyLoss: " + str(floss) + " diceLoss: " + str(
                    diceLoss_batch) + " diceScore: " + str(diceScore_batch) + " iou: " + str(
                    IOU_batch) + " mipLoss: " + str(mip_loss))

                # Calculating gradients
                if self.with_apex:
                    if type(floss) is list:
                        for i in range(len(floss)):
                            if i + 1 == len(floss):  # final loss
                                self.scaler.scale(floss[i]).backward()
                            else:
                                self.scaler.scale(floss[i]).backward(retain_graph=True)
                        floss = torch.sum(torch.stack(floss))
                    else:
                        self.scaler.scale(floss).backward()

                    if self.clip_grads:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.75)
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    floss.backward()
                    if self.clip_grads:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.75)
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)

                    self.optimizer.step()

                if training_batch_index % 50 == 0:  # Save best metric evaluation weights
                    write_summary(self.writer_training, self.logger, training_batch_index,
                                  focalTverskyLoss=floss.detach().item(), mipLoss=mip_loss.detach().item(),
                                  diceLoss=diceLoss_batch, diceScore=diceScore_batch, iou=IOU_batch)
                training_batch_index += 1

                # Initialising the average loss metrics
                total_floss += floss.detach().item()
                # Compute total DiceLoss, DiceScore and IOU per batch
                if (num_patches > 0):
                    diceLoss_batch = diceLoss_batch / num_patches
                    diceScore_batch = diceScore_batch / num_patches
                    IOU_batch = IOU_batch / num_patches
                    mip_loss = mip_loss / num_patches

                total_DiceLoss += diceLoss_batch
                total_DiceScore += diceScore_batch
                total_IOU += IOU_batch
                total_mipLoss += mip_loss

                if self.deform:
                    del elastic
                torch.cuda.empty_cache()

            # Calculate the average loss per batch in one epoch
            total_floss /= (batch_index + 1.0)
            total_mipLoss /= (batch_index + 1.0)

            # Calculate the average DiceLoss, DiceScore and IOU per epoch
            total_DiceLoss /= (batch_index + 1.0)
            total_DiceScore /= (batch_index + 1.0)
            total_IOU /= (batch_index + 1.0)

            # Print every epoch
            self.logger.info("Epoch:" + str(epoch) + " Average Training..." +
                             "\n focalTverskyLoss:" + str(total_floss) + " diceLoss: " + str(
                total_DiceLoss) + " diceScore: " + str(total_DiceScore) + " iou: " + str(
                total_IOU) + " mipLoss: " + str(total_mipLoss))
            write_Epoch_summary(self.writer_training, epoch, focalTverskyLoss=total_floss, mipLoss=total_mipLoss,
                                diceLoss=total_DiceLoss, diceScore=total_DiceScore, iou=total_IOU)

            save_model(self.checkpoint_path, {
                'epoch_type': 'last',
                'epoch': epoch,
                # Let is always overwrite, we need just the last checkpoint and best checkpoint(saved after validate)
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'amp': self.scaler.state_dict()
            })

            torch.cuda.empty_cache()  # to avoid memory errors
            self.validate(training_batch_index, epoch)
            torch.cuda.empty_cache()  # to avoid memory errors

        return self.model

    def validate(self, tainingIndex, epoch):
        """
        Method to validate
        :param tainingIndex: Epoch after which validation is performed(can be anything for test)
        :return:
        """
        self.logger.debug('Validating...')
        print("Validate Epoch: " + str(epoch) + " of " + str(self.num_epochs))

        floss, mipLoss, binloss, dloss, dscore, jaccard_index = 0, 0, 0, 0, 0, 0
        no_patches = 0
        self.model.eval()
        data_loader = self.validate_loader
        writer = self.writer_validating
        with torch.no_grad():
            for index, patches_batch in enumerate(tqdm(data_loader)):
                self.logger.info("loading" + str(index))
                no_patches += 1

                local_batch = self.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                local_labels = patches_batch['label'][tio.DATA].float().cuda()

                local_batch = torch.movedim(local_batch, -1, -3)
                local_labels = torch.movedim(local_labels, -1, -3)

                floss_iter = 0
                mipLoss_iter = 0
                output1 = 0
                try:
                    with autocast(enabled=self.with_apex):
                        # Forward propagation
                        loss_ratios = [1, 0.66, 0.34]  # TODO param
                        level = 0
                        # Forward propagation
                        if not self.isProb:
                            for output in self.model(local_batch):
                                if level == 0:
                                    output1 = output
                                if level > 0:  # then the output size is reduced, and hence interpolate to patch_size
                                    output = torch.nn.functional.interpolate(input=output, size=(64, 64, 64))
                                output = torch.sigmoid(output)

                                # Compute MIP loss from the patch on the MIP of the 3D label and the patch prediction
                                mip_loss_patch = 0
                                for idx, op in enumerate(output):
                                    op_mip = torch.amax(op, -1)
                                    mip_loss_patch += loss_ratios[level] * self.focalTverskyLoss(op_mip,
                                                      patches_batch['ground_truth_mip_patch'][idx].float().cuda())
                                mipLoss_iter += mip_loss_patch / len(op) if not torch.isnan(mip_loss_patch / len(op)) \
                                    else torch.tensor(0).float().cuda()
                                # mipLoss_iter += loss_ratios[level] * self.mip_loss(output, patches_batch, self.pre_loaded_validate_lbl_data, self.focalTverskyLoss, self.patch_size)
                                floss_iter += loss_ratios[level] * self.focalTverskyLoss(output, local_labels)
                                level += 1
                        else:
                            self.model.forward(local_batch, training=False)
                            output1 = torch.sigmoid(self.model.sample(testing=True))
                            floss_iter = self.focalTverskyLoss(output1, local_labels)
                            # elbo = self.model.elbo(local_labels)
                            # reg_loss = self.l2_regularisation(self.model.posterior) + self.l2_regularisation(self.model.prior) + self.l2_regularisation(self.model.fcomb.layers)
                            # floss_iter = -elbo + 1e-5 * reg_loss
                except Exception as error:
                    self.logger.exception(error)

                floss += floss_iter
                mipLoss += mipLoss_iter
                dl, ds = self.dice(torch.sigmoid(output1), local_labels)
                dloss += dl.detach().item()

                # Log validation losses
                self.logger.info("Batch_Index:" + str(index) + " Validation..." +
                                 "\n focalTverskyLoss:" + str(floss) + "\n DiceLoss: " + str(
                    dloss) + "\n MipLoss: " + str(mipLoss))

        # Average the losses
        floss = floss / no_patches
        mipLoss = mipLoss / no_patches
        dloss = dloss / no_patches
        process = ' Validating'
        self.logger.info("Epoch:" + str(tainingIndex) + process + "..." +
                         "\n FocalTverskyLoss:" + str(floss) +
                         "\n DiceLoss:" + str(dloss) +
                         "\n MipLoss:" + str(mipLoss))

        write_summary(writer, self.logger, tainingIndex, local_labels[0][0][6], output1[0][0][6], floss, mipLoss, dloss,
                      0, 0)

        write_Epoch_summary(writer, epoch, focalTverskyLoss=floss, mipLoss=mipLoss, diceLoss=dloss, diceScore=0, iou=0)

        if self.LOWEST_LOSS > floss:  # Save best metric evaluation weights
            self.LOWEST_LOSS = floss
            self.logger.info(
                'Best metric... @ epoch:' + str(tainingIndex) + ' Current Lowest loss:' + str(self.LOWEST_LOSS))

            save_model(self.checkpoint_path, {
                'epoch_type': 'best',
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'amp': self.scaler.state_dict()})

    def test_with_MIP(self, test_logger, save_results=True, test_subjects=None):
        test_logger.debug('Testing...')

        if test_subjects is None:
            test_folder_path = self.DATASET_FOLDER + '/test/'
            test_label_path = self.DATASET_FOLDER + '/test_label/'

            test_subjects = self.create_TIOSubDS(vol_path=test_folder_path, label_path=test_label_path,
                                                 get_subjects_only=True)

        overlap = np.subtract(self.patch_size, (self.stride_length, self.stride_width, self.stride_depth))

        df = pd.DataFrame(columns=["Subject", "Dice", "IoU"])
        result_root = os.path.join(self.output_path, self.model_name, "results")
        os.makedirs(result_root, exist_ok=True)

        self.model.eval()

        with torch.no_grad():
            for test_subject in test_subjects:
                label_MIP = None
                if 'label' in test_subject:
                    label_MIP = test_subject['label'][tio.DATA].float().cuda()
                    label = test_subject['label'][tio.DATA].float().squeeze().numpy()
                    del test_subject['label']
                else:
                    label_MIP = None
                    label = None
                subjectname = test_subject['subjectname']
                del test_subject['subjectname']

                grid_sampler = tio.inference.GridSampler(
                    test_subject,
                    self.patch_size,
                    overlap,
                )
                aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size, shuffle=False,
                                                           num_workers=self.num_worker)

                for index, patches_batch in enumerate(tqdm(patch_loader)):
                    local_batch = self.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                    locations = patches_batch[tio.LOCATION]

                    local_batch = torch.movedim(local_batch, -1, -3)

                    with autocast(enabled=self.with_apex):
                        if not self.isProb:
                            output = self.model(local_batch)
                            if type(output) is tuple or type(output) is list:
                                output = output[0]
                            output = torch.sigmoid(output).detach().cpu()
                        else:
                            self.model.forward(local_batch, training=False)
                            output = self.model.sample(
                                testing=True).detach().cpu()  # TODO: need to check whether sigmoid is needed for prob

                    output = torch.movedim(output, -3, -1).type(local_batch.type())
                    aggregator.add_batch(output, locations)

                predicted_tensor = aggregator.get_output_tensor().cuda()
                predicted = aggregator.get_output_tensor().squeeze().numpy()

                try:
                    thresh = threshold_otsu(predicted)
                    result = predicted > thresh
                except Exception as error:
                    test_logger.exception(error)
                    result = predicted > 0.5  # exception will be thrown only if input image seems to have just one color 1.0.
                result = result.astype(np.float32)

                if label is not None:
                    mip_loss = self.mip_loss(predicted_tensor, label_MIP)
                    datum = {"Subject": subjectname}
                    dice3D = dice(result, label)
                    iou3D = IoU(result, label)
                    datum = pd.DataFrame.from_dict({**datum, "Dice": [dice3D], "IoU": [iou3D], "mip_loss": [mip_loss]})
                    df = pd.concat([df, datum], ignore_index=True)

                if save_results:
                    save_nifti(result, os.path.join(result_root, subjectname + ".nii.gz"))

                    resultMIP = np.max(result, axis=-1)
                    Image.fromarray((resultMIP * 255).astype('uint8'), 'L').save(
                        os.path.join(result_root, subjectname + "_MIP.tif"))

                    predicted_mip = torch.max(predicted_tensor, -1)[0].detach().cpu().squeeze().numpy()
                    Image.fromarray((predicted_mip * 255).astype('uint8'), 'L').save(
                        os.path.join(result_root, subjectname + "_Predicted_MIP.tif"))

                    if label is not None:

                        true_MIP = torch.max(label_MIP, -1)[0].detach().cpu().squeeze().numpy()
                        Image.fromarray((true_MIP * 255).astype('uint8'), 'L').save(
                        os.path.join(result_root, subjectname + "_True_MIP.tif"))

                        overlay = create_diff_mask_binary(result, label)
                        save_tifRGB(overlay, os.path.join(result_root, subjectname + "_colour.tif"))

                        overlayMIP = create_diff_mask_binary(resultMIP, np.max(label, axis=-1))
                        Image.fromarray(overlayMIP.astype('uint8'), 'RGB').save(
                            os.path.join(result_root, subjectname + "_colourMIP.tif"))

                test_logger.info("Testing " + subjectname + "..." +
                                 "\n Dice:" + str(dice3D) +
                                 "\n JacardIndex:" + str(iou3D) +
                                 "\n Mip Loss: " + str(mip_loss))

        df.to_excel(os.path.join(result_root, "Results_Main.xlsx"))

    def test(self, test_logger, save_results=True, test_subjects=None):
        test_logger.debug('Testing...')

        if test_subjects is None:
            test_folder_path = self.DATASET_FOLDER + '/test/'
            test_label_path = self.DATASET_FOLDER + '/test_label/'

            test_subjects = self.create_TIOSubDS(vol_path=test_folder_path, is_train=False, label_path=test_label_path,
                                                 get_subjects_only=True)

        overlap = np.subtract(self.patch_size, (self.stride_length, self.stride_width, self.stride_depth))

        df = pd.DataFrame(columns=["Subject", "Dice", "IoU"])
        result_root = os.path.join(self.output_path, self.model_name, "results")
        os.makedirs(result_root, exist_ok=True)

        self.model.eval()

        with torch.no_grad():
            for test_subject in test_subjects:
                if 'label' in test_subject:
                    label = test_subject['label'][tio.DATA].float().squeeze().numpy()
                    del test_subject['label']
                else:
                    label = None
                subjectname = test_subject['subjectname']
                del test_subject['subjectname']

                grid_sampler = tio.inference.GridSampler(
                    test_subject,
                    self.patch_size,
                    overlap,
                )
                aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size, shuffle=False,
                                                           num_workers=self.num_worker)

                for index, patches_batch in enumerate(tqdm(patch_loader)):
                    local_batch = self.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                    locations = patches_batch[tio.LOCATION]

                    local_batch = torch.movedim(local_batch, -1, -3)

                    with autocast(enabled=self.with_apex):
                        if not self.isProb:
                            output = self.model(local_batch)
                            if type(output) is tuple or type(output) is list:
                                output = output[0]
                            output = torch.sigmoid(output).detach().cpu()
                        else:
                            self.model.forward(local_batch, training=False)
                            output = self.model.sample(
                                testing=True).detach().cpu()  # TODO: need to check whether sigmoid is needed for prob

                    output = torch.movedim(output, -3, -1).type(local_batch.type())
                    aggregator.add_batch(output, locations)

                predicted = aggregator.get_output_tensor().squeeze().numpy()

                try:
                    thresh = threshold_otsu(predicted)
                    result = predicted > thresh
                except Exception as error:
                    test_logger.exception(error)
                    result = predicted > 0.5  # exception will be thrown only if input image seems to have just one color 1.0.
                result = result.astype(np.float32)

                if label is not None:
                    datum = {"Subject": subjectname}
                    dice3D = dice(result, label)
                    iou3D = IoU(result, label)
                    datum = pd.DataFrame.from_dict({**datum, "Dice": [dice3D], "IoU": [iou3D]})
                    df = pd.concat([df, datum], ignore_index=True)

                if save_results:
                    save_nifti(result, os.path.join(result_root, subjectname + ".nii.gz"))

                    resultMIP = np.max(result, axis=-1)
                    Image.fromarray((resultMIP * 255).astype('uint8'), 'L').save(
                        os.path.join(result_root, subjectname + "_MIP.tif"))

                    if label is not None:
                        overlay = create_diff_mask_binary(result, label)
                        save_tifRGB(overlay, os.path.join(result_root, subjectname + "_colour.tif"))

                        overlayMIP = create_diff_mask_binary(resultMIP, np.max(label, axis=-1))
                        Image.fromarray(overlayMIP.astype('uint8'), 'RGB').save(
                            os.path.join(result_root, subjectname + "_colourMIP.tif"))

                test_logger.info("Testing " + subjectname + "..." +
                                 "\n Dice:" + str(dice3D) +
                                 "\n JacardIndex:" + str(iou3D))

        df.to_excel(os.path.join(result_root, "Results_Main.xlsx"))

    def predict(self, image_path, label_path, predict_logger):
        image_name = os.path.basename(image_path).split('.')[0]

        subdict = {
            "img": tio.ScalarImage(image_path),
            "subjectname": image_name,
        }

        if bool(label_path):
            subdict["label"] = tio.LabelMap(label_path)

        subject = tio.Subject(**subdict)

        self.test(predict_logger, save_results=True, test_subjects=[subject])

