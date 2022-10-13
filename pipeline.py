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
from Utils.model_manager import getModel

__author__ = "Kartik Prabhu, Mahantesh Pattadkal, and Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Kartik Prabhu", "Mahantesh Pattadkal", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"


class Pipeline:

    def __init__(self, cmd_args, model, logger, dir_path, checkpoint_path, writer_training, writer_validating, test_logger,
                 training_set=None, validation_set=None, test_set=None, wandb=None):

        self.logger = logger
        self.wandb = wandb
        self.model = model
        self.MODEL_NAME = cmd_args.model_name
        self.model_type = cmd_args.model
        self.lr_1 = cmd_args.learning_rate
        self.logger.info("learning rate " + str(self.lr_1))
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cmd_args.learning_rate)
        self.num_epochs = cmd_args.num_epochs
        self.k_folds = cmd_args.k_folds
        self.learning_rate = cmd_args.learning_rate

        self.writer_training = writer_training
        self.writer_validating = writer_validating
        self.test_logger = test_logger
        self.checkpoint_path = checkpoint_path
        self.load_path = cmd_args.load_path
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
        self.floss_param_smooth = cmd_args.floss_param_smooth
        self.floss_param_gamma = cmd_args.floss_param_gamma
        self.floss_param_alpha = cmd_args.floss_param_alpha
        self.mip_loss_param_smooth = cmd_args.mip_loss_param_smooth
        self.mip_loss_param_gamma = cmd_args.mip_loss_param_gamma
        self.mip_loss_param_alpha = cmd_args.mip_loss_param_alpha
        self.dice = Dice()
        self.focalTverskyLoss = FocalTverskyLoss(smooth=self.floss_param_smooth, gamma=self.floss_param_gamma,
                                                 alpha=self.floss_param_alpha)
        self.mip_loss = FocalTverskyLoss(smooth=self.mip_loss_param_smooth, gamma=self.mip_loss_param_gamma,
                                         alpha=self.mip_loss_param_alpha)
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

    def create_TIOSubDS(self, vol_path, label_path, crossvalidation_set=None, is_train=True, is_validate=False,
                        get_subjects_only=False,
                        transforms=None):
        if is_train:
            trainDS = SRDataset(logger=self.logger, patch_size=self.patch_size,
                                dir_path=vol_path,
                                label_dir_path=label_path,
                                # TODO: implement non-iso patch-size, now only using the first element
                                stride_depth=self.stride_depth, stride_length=self.stride_length,
                                stride_width=self.stride_width, Size=None, fly_under_percent=None,
                                # TODO: implement fly_under_percent, if needed
                                patch_size_us=self.patch_size, pre_interpolate=None, norm_data=False,
                                pre_load=True,
                                return_coords=True,
                                files_us=crossvalidation_set)  # TODO implement patch_size_us if required - patch_size//scaling_factor
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
                                     dir_path=vol_path,
                                     label_dir_path=label_path,
                                     # TODO: implement non-iso patch-size, now only using the first element
                                     stride_depth=self.stride_depth, stride_length=self.stride_length,
                                     stride_width=self.stride_width, Size=None, fly_under_percent=None,
                                     # TODO: implement fly_under_percent, if needed
                                     patch_size_us=self.patch_size, pre_interpolate=None, norm_data=False,
                                     pre_load=True,
                                     return_coords=True,
                                     files_us=crossvalidation_set)  # TODO implement patch_size_us if required - patch_size//scaling_factor
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

    def load(self, checkpoint_path=None, load_best=True, fold_index=""):
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        if self.with_apex:
            self.model, self.optimizer, self.scaler = load_model_with_amp(self.model, self.optimizer, checkpoint_path,
                                                                          batch_index="best" if load_best else "last", fold_index=fold_index)
        else:
            self.model, self.optimizer = load_model(self.model, self.optimizer, checkpoint_path,
                                                    batch_index="best" if load_best else "last", fold_index=fold_index)

    def reset(self):
        del self.model
        self.model = torch.nn.DataParallel(getModel(self.model_type, self.output_path + "/" + self.MODEL_NAME))
        self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.with_apex:
            self.scaler = GradScaler()
        self.LOWEST_LOSS = float('inf')

    def train(self):
        self.logger.debug("Training...")
        vol_path = self.DATASET_FOLDER + '/train/'
        vols = glob(vol_path + "*.nii") + glob(vol_path + "*.nii.gz")
        random.shuffle(vols)
        # get k folds for cross validation
        folds = [vols[i::self.k_folds] for i in range(self.k_folds)]

        for fold_index in range(self.k_folds):
            train_vols = []
            for idx, fold in enumerate(folds):
                if idx != fold_index:
                    train_vols.extend([*fold])
            validation_vols = [*folds[fold_index]]

            traindataset = self.create_TIOSubDS(vol_path=self.DATASET_FOLDER + '/train/',
                                                label_path=self.DATASET_FOLDER + '/train_label/',
                                                crossvalidation_set=train_vols)
            validationdataset = self.create_TIOSubDS(vol_path=self.DATASET_FOLDER + '/train/',
                                                     label_path=self.DATASET_FOLDER + '/train_label/',
                                                     crossvalidation_set=validation_vols, is_train=False,
                                                     is_validate=True)
            sampler = torch.utils.data.RandomSampler(data_source=traindataset, replacement=True,
                                                     num_samples=self.samples_per_epoch)
            train_loader = torch.utils.data.DataLoader(traindataset, batch_size=self.batch_size, shuffle=False,
                                                       num_workers=self.num_worker, pin_memory=True,
                                                       sampler=sampler)
            validate_loader = torch.utils.data.DataLoader(validationdataset, batch_size=self.batch_size,
                                                          shuffle=False,
                                                          num_workers=self.num_worker, pin_memory=True)
            print("Train Fold: " + str(fold_index) + " of " + str(self.k_folds))
            for epoch in range(self.num_epochs):
                print("Train Epoch: " + str(epoch) + " of " + str(self.num_epochs))
                self.model.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
                total_floss = 0
                total_mipLoss = 0
                total_loss = 0
                total_DiceLoss = 0
                total_IOU = 0
                total_DiceScore = 0
                batch_index = 0
                for batch_index, patches_batch in enumerate(tqdm(train_loader)):

                    local_batch = self.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                    local_labels = patches_batch['label'][tio.DATA].float().cuda()

                    local_batch = torch.movedim(local_batch, -1, -3)
                    local_labels = torch.movedim(local_labels, -1, -3)

                    # Transfer to GPU
                    self.logger.debug('Fold: {} Batch Index: {}'.format(fold_index, batch_index))

                    # Clear gradients
                    self.optimizer.zero_grad()

                    # try:
                    with autocast(enabled=self.with_apex):
                        loss_ratios = [1, 0.66, 0.34]  # TODO param

                        floss = torch.tensor(0.001).float().cuda()
                        mip_loss = torch.tensor(0.001).float().cuda()
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
                                mip_loss_patch = torch.tensor(0.001).float().cuda()
                                num_patches = 0
                                for index, op in enumerate(output):
                                    op_mip = torch.amax(op, 1)
                                    mip_loss_patch += loss_ratios[level] * self.mip_loss(op_mip,
                                                                                         patches_batch[
                                                                                             'ground_truth_mip_patch'][
                                                                                             index].float().cuda())
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
                            loss = (self.floss_coeff * floss) + (self.mip_loss_coeff * mip_loss)

                    # except Exception as error:
                    #     self.logger.exception(error)
                    #     sys.exit()

                    self.logger.info("Fold:" + str(fold_index) + " Epoch:" + str(epoch) + " Batch_Index:" + str(batch_index) + " Training..." +
                                     "\n focalTverskyLoss: " + str(floss) + " diceLoss: " + str(
                        diceLoss_batch) + " diceScore: " + str(diceScore_batch) + " iou: " + str(
                        IOU_batch) + " mipLoss: " + str(mip_loss) + " totalLoss: " + str(loss))

                    # Calculating gradients
                    if self.with_apex:
                        if type(loss) is list:
                            for i in range(len(loss)):
                                if i + 1 == len(loss):  # final loss
                                    self.scaler.scale(loss[i]).backward()
                                else:
                                    self.scaler.scale(loss[i]).backward(retain_graph=True)
                            floss = torch.sum(torch.stack(loss))
                        else:
                            self.scaler.scale(loss).backward()

                        if self.clip_grads:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                            # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)

                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if not torch.any(torch.isnan(floss)):
                            floss.backward()
                        else:
                            self.logger.info("nan found in floss.... no backpropagation!!")
                        if self.clip_grads:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                            # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)

                        self.optimizer.step()

                    # Initialising the average loss metrics
                    total_floss += floss.detach().item()
                    total_loss += loss.detach().item()
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
                total_loss /= (batch_index + 1.0)

                # Calculate the average DiceLoss, DiceScore and IOU per epoch
                total_DiceLoss /= (batch_index + 1.0)
                total_DiceScore /= (batch_index + 1.0)
                total_IOU /= (batch_index + 1.0)
                # Print every epoch
                self.logger.info("Fold:" + str(fold_index) + " Epoch:" + str(epoch) + " Average Training..." +
                                 "\n focalTverskyLoss:" + str(total_floss) + " diceLoss: " + str(
                    total_DiceLoss) + " diceScore: " + str(total_DiceScore) + " iou: " + str(
                    total_IOU) + " mipLoss: " + str(total_mipLoss) + " total_loss" + str(total_loss))
                # TODO: Not Logging to tensorboard currently
                # write_Epoch_summary(self.writer_training, fold_index, focalTverskyLoss=total_floss, mipLoss=total_mipLoss,
                #                     diceLoss=total_DiceLoss, diceScore=total_DiceScore, iou=total_IOU,
                #                     total_loss=total_loss)
                if self.wandb is not None:
                    self.wandb.log({"focalTverskyLoss_train_" + str(fold_index): total_floss, "mipLoss_train_" + str(fold_index): total_mipLoss,
                                    "diceScore_train_" + str(fold_index): total_DiceScore, "IOU_train_" + str(fold_index): total_IOU,
                                    "totalLoss_train_" + str(fold_index): total_loss, "epoch": epoch, "fold_index": fold_index})
                # save_model(self.checkpoint_path, {
                #     'epoch_type': 'last',
                #     'epoch': fold_index,
                #     # Let is always overwrite, we need just the last checkpoint and best checkpoint(saved after validate)
                #     'state_dict': self.model.state_dict(),
                #     'optimizer': self.optimizer.state_dict(),
                #     'amp': self.scaler.state_dict()
                # })
                torch.cuda.empty_cache()  # to avoid memory errors
                self.validate(fold_index, epoch, validate_loader)
                torch.cuda.empty_cache()  # to avoid memory errors

            # Testing for current fold
            torch.cuda.empty_cache()  # to avoid memory errors
            self.load(fold_index=fold_index)
            self.test(self.test_logger, fold_index=fold_index)
            torch.cuda.empty_cache()  # to avoid memory errors

            # Discard the current model and reset training parameters
            self.reset()

        return self.model

    def validate(self, fold_index, epoch, validate_loader=None):
        """
        Method to validate
        :return:
        """
        self.logger.debug('Validating...')
        print("Validate Fold: " + str(fold_index) + " of " + str(self.k_folds) + "Validate Epoch: " + str(epoch) + " of " + str(self.num_epochs))

        floss, mipLoss, total_loss, binloss, dloss, dscore, jaccard_index = 0, 0, 0, 0, 0, 0, 0
        no_patches = 0
        self.model.eval()
        if validate_loader is None:
            validationdataset = self.create_TIOSubDS(vol_path=self.DATASET_FOLDER + '/validate/',
                                                     label_path=self.DATASET_FOLDER + '/validate_label/',
                                                     is_train=False, is_validate=True)
            validate_loader = torch.utils.data.DataLoader(validationdataset, batch_size=self.batch_size,
                                                          shuffle=False,
                                                          num_workers=self.num_worker, pin_memory=True)
        writer = self.writer_validating
        with torch.no_grad():
            for index, patches_batch in enumerate(tqdm(validate_loader)):
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
                                mip_loss_patch = torch.tensor(0.001).float().cuda()
                                for idx, op in enumerate(output):
                                    op_mip = torch.amax(op, 1)
                                    mip_loss_patch += self.mip_loss(op_mip,
                                                                    patches_batch['ground_truth_mip_patch'][
                                                                        idx].float().cuda())
                                if not torch.any(torch.isnan(mip_loss_patch)):
                                    mipLoss_iter += mip_loss_patch / len(output)
                                floss_iter += loss_ratios[level] * self.focalTverskyLoss(output, local_labels)
                                level += 1
                        else:
                            self.model.forward(local_batch, training=False)
                            output1 = torch.sigmoid(self.model.sample(testing=True))
                            floss_iter = self.focalTverskyLoss(output1, local_labels)
                except Exception as error:
                    self.logger.exception(error)

                floss += floss_iter
                mipLoss += mipLoss_iter
                total_loss += (self.floss_coeff * floss_iter) + (self.mip_loss_coeff * mipLoss_iter)
                dl, ds = self.dice(torch.sigmoid(output1), local_labels)
                dloss += dl.detach().item()

                # Log validation losses
                self.logger.info("Batch_Index:" + str(index) + " Validation..." +
                                 "\n focalTverskyLoss:" + str(floss) + "\n DiceLoss: " + str(
                    dloss) + "\n MipLoss: " + str(mipLoss) + "\n totalLoss: " + str(total_loss))

        # Average the losses
        floss = floss / no_patches
        mipLoss = mipLoss / no_patches
        total_loss = total_loss / no_patches
        dloss = dloss / no_patches
        process = ' Validating'
        self.logger.info("Fold:" + str(fold_index) + " Epoch:" + str(epoch) + process + "..." +
                         "\n FocalTverskyLoss:" + str(floss) +
                         "\n DiceLoss:" + str(dloss) +
                         "\n MipLoss:" + str(mipLoss) +
                         "\n TotalLoss:" + str(total_loss))
        # TODO: Not logging to tensorboard currently
        # write_Epoch_summary(writer, fold_index, focalTverskyLoss=floss, mipLoss=mipLoss, diceLoss=dloss, diceScore=0,
        #                     iou=0, total_loss=total_loss)
        if self.wandb is not None:
            self.wandb.log({"focalTverskyLoss_val_" + str(fold_index): floss, "mipLoss_val_" + str(fold_index): mipLoss, "diceLoss_val_" + str(fold_index): dloss,
                            "totalLoss_val_" + str(fold_index): total_loss, "epoch": epoch, "fold_index": fold_index})

        if self.LOWEST_LOSS > total_loss:  # Save best metric evaluation weights
            self.LOWEST_LOSS = total_loss
            self.logger.info(
                'Best metric... @ fold:' + str(fold_index) + ' Current Lowest loss:' + str(self.LOWEST_LOSS))

            save_model(self.checkpoint_path, {
                'epoch_type': 'best',
                'epoch': fold_index,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'amp': self.scaler.state_dict()}, fold_index=fold_index)

    def pseudo_train(self, test_logger):
        test_logger.debug('Testing With MIP...')

        traindataset = self.create_TIOSubDS(vol_path=self.DATASET_FOLDER + '/train/',
                                            label_path=self.DATASET_FOLDER + '/train_label/')
        sampler = torch.utils.data.RandomSampler(data_source=traindataset, replacement=True,
                                                 num_samples=self.samples_per_epoch)
        self.train_loader = torch.utils.data.DataLoader(traindataset, batch_size=self.batch_size, shuffle=False,
                                                        num_workers=self.num_worker, pin_memory=True,
                                                        sampler=sampler)
        result_root = os.path.join(self.output_path, self.model_name, "results")
        result_root = os.path.join(result_root, "mips")
        os.makedirs(result_root, exist_ok=True)
        training_batch_index = 0
        for epoch in range(self.num_epochs):
            print("Train Epoch: " + str(epoch) + " of " + str(self.num_epochs))
            self.model.eval()  # make sure to assign mode:train, because in validation, mode is assigned as eval
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
                # self.optimizer.zero_grad()

                # try:
                with autocast(enabled=self.with_apex):
                    loss_ratios = [1, 0.66, 0.34]  # TODO param

                    floss = torch.tensor(0.001).float().cuda()
                    mip_loss = torch.tensor(0.001).float().cuda()
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

                            floss += loss_ratios[level] * self.focalTverskyLoss(output, local_labels)
                            # Compute MIP loss from the patch on the MIP of the 3D label and the patch prediction
                            mip_loss_patch = torch.tensor(0.001).float().cuda()
                            num_patches = 0
                            for index, op in enumerate(output):
                                op_mip = torch.amax(op, 1)
                                true_mip = patches_batch['ground_truth_mip_patch'][index].float().cuda()
                                mip_loss_patch += self.focalTverskyLoss(op_mip, true_mip)
                                op_mip = op_mip.detach().cpu().squeeze().numpy()
                                true_mip = true_mip.detach().cpu().squeeze().numpy()
                                Image.fromarray((op_mip * 255).astype('uint8'), 'L').save(
                                    os.path.join(result_root,
                                                 "level_" + str(level) + "_patch_" + str(index) + "_op_mip.tif"))
                                Image.fromarray((true_mip * 255).astype('uint8'), 'L').save(
                                    os.path.join(result_root,
                                                 "level_" + str(level) + "_patch_" + str(index) + "_true_mip.tif"))
                                test_logger.info("Testing with mip..." +
                                                 "\n floss:" + str(floss) +
                                                 "\n mip_loss:" + str(mip_loss_patch))
                            if not torch.any(torch.isnan(mip_loss_patch)):
                                mip_loss += mip_loss_patch / len(output)
                            level += 1

                    test_logger.info("Testing with mip..." +
                                     "\n Average mip_loss:" + str(mip_loss))
                break

    def test_with_MIP(self, test_logger, test_subjects=None):
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
                    # del test_subject['label']
                else:
                    label = None
                subjectname = test_subject['subjectname']
                del test_subject['subjectname']

                result_root = os.path.join(result_root, subjectname + "_MIPs")
                os.makedirs(result_root, exist_ok=True)

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
                    local_label = patches_batch['label'][tio.DATA].float()

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

                    output = torch.movedim(output, -3, -1)
                    for idx, op in enumerate(output):
                        op_mip = torch.amax(op.squeeze().numpy(), 1)
                        label_mip = torch.amax(local_label[idx].squeeze().numpy(), 1)
                        Image.fromarray((op_mip * 255).astype('uint8'), 'L').save(
                            os.path.join(result_root, subjectname + "_patch" + str(idx) + "_pred_MIP.tif"))
                        Image.fromarray((label_mip * 255).astype('uint8'), 'L').save(
                            os.path.join(result_root, subjectname + "_patch" + str(idx) + "_true_MIP.tif"))

                    aggregator.add_batch(output, locations)

        #         predicted = aggregator.get_output_tensor().squeeze().numpy()
        #
        #         try:
        #             thresh = threshold_otsu(predicted)
        #             result = predicted > thresh
        #         except Exception as error:
        #             test_logger.exception(error)
        #             result = predicted > 0.5  # exception will be thrown only if input image seems to have just one color 1.0.
        #         result = result.astype(np.float32)
        #
        #         if label is not None:
        #             datum = {"Subject": subjectname}
        #             dice3D = dice(result, label)
        #             iou3D = IoU(result, label)
        #             datum = pd.DataFrame.from_dict({**datum, "Dice": [dice3D], "IoU": [iou3D]})
        #             df = pd.concat([df, datum], ignore_index=True)
        #
        #         if save_results:
        #             save_nifti(result, os.path.join(result_root, subjectname + ".nii.gz"))
        #
        #             resultMIP = np.max(result, axis=-1)
        #             Image.fromarray((resultMIP * 255).astype('uint8'), 'L').save(
        #                 os.path.join(result_root, subjectname + "_MIP.tif"))
        #
        #             if label is not None:
        #                 overlay = create_diff_mask_binary(result, label)
        #                 save_tifRGB(overlay, os.path.join(result_root, subjectname + "_colour.tif"))
        #
        #                 overlayMIP = create_diff_mask_binary(resultMIP, np.max(label, axis=-1))
        #                 Image.fromarray(overlayMIP.astype('uint8'), 'RGB').save(
        #                     os.path.join(result_root, subjectname + "_colourMIP.tif"))
        #
        #         test_logger.info("Testing " + subjectname + "..." +
        #                          "\n Dice:" + str(dice3D) +
        #                          "\n JacardIndex:" + str(iou3D))
        #
        # df.to_excel(os.path.join(result_root, "Results_Main.xlsx"))

    def test(self, test_logger, save_results=True, test_subjects=None, fold_index=""):
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
                    save_nifti(result, os.path.join(result_root, subjectname + "_fld" + str(fold_index) + ".nii.gz"))

                    resultMIP = np.max(result, axis=-1)
                    Image.fromarray((resultMIP * 255).astype('uint8'), 'L').save(
                        os.path.join(result_root, subjectname + str(fold_index) + "_MIP.tif"))

                    if label is not None:
                        overlay = create_diff_mask_binary(result, label)
                        save_tifRGB(overlay, os.path.join(result_root, subjectname + "_fld" + str(fold_index) + "_colour.tif"))

                        overlayMIP = create_diff_mask_binary(resultMIP, np.max(label, axis=-1))
                        color_mip = Image.fromarray(overlayMIP.astype('uint8'), 'RGB')
                        color_mip.save(
                            os.path.join(result_root, subjectname + "_fld" + str(fold_index) + "_colourMIP.tif"))
                        if self.wandb is not None:
                            self.wandb.log({"" + subjectname + "_fld" + str(fold_index): self.wandb.Image(color_mip)})


                test_logger.info("Testing " + subjectname + "..." +
                                 "\n Dice:" + str(dice3D) +
                                 "\n JacardIndex:" + str(iou3D))

        df.to_excel(os.path.join(result_root, "Results_Main_fld" + str(fold_index) + ".xlsx"))

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
