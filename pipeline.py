#!/usr/bin/env python
"""

"""
import sys
import os
import random
import sys
from glob import glob

import wandb
import nibabel
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchio as tio
import torchvision.transforms as transforms
from skimage.filters import threshold_otsu
from torch import nn, optim, distributions
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from scipy.stats import energy_distance
from geomloss import SamplesLoss as GeomDistLoss
from geomloss.kernel_samples import energy_kernel

# from Utils.fid.fastfid import fastfid 

from Evaluation.evaluate import (IOU, Dice, FocalTverskyLoss, getLosses,
                                 getMetric)
from Models.ProbUNetV2.eval import calc_energy_distances, get_energy_distance_components
from Utils.elastic_transform import RandomElasticDeformation, warp_image
from Utils.fid.fidloss import FastFID
from Models.SSN.trainer.losses import StochasticSegmentationNetworkLossMCIntegral
from Models.VIMH.loss import VIMHLoss
from Utils.result_analyser import *
from Utils.vessel_utils import (convert_and_save_tif, create_diff_mask,
                                create_mask, load_model, load_model_with_amp,
                                save_model, write_summary)

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

        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=cmd_args.learning_rate)
        self.logger = logger
        self.starting_epoch = 0
        self.num_epochs = cmd_args.num_epochs

        self.writer_training = writer_training
        self.writer_validating = writer_validating
        self.checkpoint_path = checkpoint_path
        self.DATASET_FOLDER = dir_path
        self.output_path = cmd_args.output_path
        self.plauslabels = cmd_args.plauslabels
        self.plauslabel_mode = cmd_args.plauslabel_mode
        self.model_name = cmd_args.model_name
        self.modelID = cmd_args.model

        self.load_best = cmd_args.load_best
        self.clip_grads = cmd_args.clip_grads
        self.with_apex = cmd_args.apex
        self.deform = cmd_args.deform
        self.distloss = cmd_args.distloss
        self.n_prob_test = cmd_args.n_prob_test

        # image input parameters
        if bool(cmd_args.slice2D_shape):
            shp = cmd_args.slice2D_shape+",1"
            self.patch_size = tuple([int(i) for i in shp.split(",")])
            self.dimMode = 2
        else:
            self.patch_size = (cmd_args.patch_size, cmd_args.patch_size, cmd_args.patch_size)
            self.dimMode = 3
            
        
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
        self.iou = IOU()
        if self.modelID == 6: #SSN
            self.ssnloss = StochasticSegmentationNetworkLossMCIntegral(num_mc_samples=self.n_prob_test)
        elif self.modelID == 7: #VIMH
            self.vimhloss = VIMHLoss(loss_func=nn.NLLLoss(), NUM_MODELS=self.model.num_models)

        self.LOWEST_LOSS = float('inf')
        self.test_set = test_set

        if self.with_apex:
            self.scaler = GradScaler()

        #set probabilistic property
        if "Models.prob_unet" in self.model.__module__:
            self.ProbFlag = 1
            from Models.prob_unet.utils import l2_regularisation
            self.l2_regularisation = l2_regularisation
        elif "Models.ProbUNetV2" in self.model.__module__:
            self.ProbFlag = 2
            self.criterion_segmentation = FocalTverskyLoss()
            self.criterion_latent = distributions.kl_divergence
            self.criterion_latent_weight = 1.0
            self.criterion_segmentation_weight = 1.0
        else:
            self.ProbFlag = 0

        if self.ProbFlag > 0:
            self.fid_calc = FastFID(useInceptionNet=True, batch_size=-1, gradient_checkpointing=True)

        self.distloss_mode = cmd_args.distloss_mode

        if self.plauslabels and self.ProbFlag and self.distloss:
            if cmd_args.distloss_mode in [0,1]:
                self.op_distloss = FastFID(useInceptionNet=True if cmd_args.distloss_mode==0 else False, batch_size=cmd_args.batch_size_fidloss, gradient_checkpointing=True)
                self.op_distloss.cuda()
            elif cmd_args.distloss_mode in [2,3]:
                if cmd_args.distloss_mode == 2:
                    dist_loss_type = "sinkhorn"
                    kernel = None
                elif cmd_args.distloss_mode == 3:
                    dist_loss_type = "hausdorff"
                    kernel = energy_kernel
                op_distloss_func = GeomDistLoss(loss=dist_loss_type, kernel=kernel)
                op_distloss_func.cuda()
                def GeomDistLossWrapper(x, y):
                    return op_distloss_func(x.reshape(x.shape[0], x.shape[1], -1), y.reshape(y.shape[0], y.shape[1], -1)).mean()
                self.op_distloss = GeomDistLossWrapper
            else:
                sys.exit("Error: Invalid distloss_mode")
            

        if cmd_args.train: #Only if training is to be performed
            traindataset = self.create_TIOSubDS(vol_path=self.DATASET_FOLDER + '/train/', label_path=self.DATASET_FOLDER + '/train_label/', crossvalidation_set=training_set, plauslabels_path=self.DATASET_FOLDER + '/train_plausiblelabel/' if cmd_args.plauslabels else "")
            validationdataset = self.create_TIOSubDS(vol_path=self.DATASET_FOLDER + '/validate/', label_path=self.DATASET_FOLDER + '/validate_label/', crossvalidation_set=validation_set, is_train=False, plauslabels_path=self.DATASET_FOLDER + '/validate_plausiblelabel/' if cmd_args.plauslabels else "")

            self.train_loader = torch.utils.data.DataLoader(traindataset, batch_size=self.batch_size, shuffle=True,
                                                            num_workers=0) 
            self.validate_loader = torch.utils.data.DataLoader(validationdataset, batch_size=self.batch_size, shuffle=False,
                                                                num_workers=self.num_worker)
    
    def create_TIOSubDS(self, vol_path, label_path, crossvalidation_set=None, is_train=True, get_subjects_only=False, transforms=None, plauslabels_path=""):
        vols = glob(vol_path + "*.nii") + glob(vol_path + "*.nii.gz")
        labels = glob(label_path + "*.nii") + glob(label_path + "*.nii.gz")
        subjects = []
        for i in range(len(vols)):
            v = vols[i]
            filename = os.path.basename(v).split('.')[0]
            l = [s for s in labels if filename in s][0]
            kwargs_dict = {
                "img":tio.ScalarImage(v),
                "label":tio.LabelMap(l),
                "subjectname":filename,
            }
            if plauslabels_path != "":
                p_labels = sorted(glob(plauslabels_path + filename + "*.nii*"))
                for i, pLbl in enumerate(p_labels):
                    kwargs_dict["p_label_"+str(i)] = tio.LabelMap(pLbl)
            subject = tio.Subject(**kwargs_dict)
            subjects.append(subject)   

        if get_subjects_only:
            return subjects

        if is_train:
            subjects_dataset = tio.SubjectsDataset(subjects)
            sampler = tio.data.UniformSampler(self.patch_size)
            patches_queue = tio.Queue(
                                        subjects_dataset,
                                        max_length=(self.samples_per_epoch//len(subjects))*2,
                                        samples_per_volume=self.samples_per_epoch//len(subjects),
                                        sampler=sampler,
                                        num_workers=self.num_worker,
                                        start_background=True
                                    )
            return patches_queue
        else:
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
            batch[i] = batch[i] / batch[i].max()
        return batch

    def load(self, checkpoint_path=None, load_best=True):
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        if self.with_apex:
            self.model, self.optimizer, self.scaler, self.starting_epoch = load_model_with_amp(self.model, self.optimizer, checkpoint_path, batch_index="best" if load_best else "last")
        else:
            self.model, self.optimizer, self.starting_epoch = load_model(self.model, self.optimizer, checkpoint_path, batch_index="best" if load_best else "last")

        self.starting_epoch += 1

    def train(self):
        self.logger.debug("Training...")

        training_batch_index = 0
        for epoch in range(self.starting_epoch, self.num_epochs):
            print("Train Epoch: "+str(epoch) +" of "+ str(self.num_epochs))
            self.model.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
            total_floss = 0
            batch_index = 0
            for batch_index, patches_batch in enumerate(tqdm(self.train_loader)):

                local_batch = self.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                if self.plauslabels:
                    labels = [k for k in patches_batch.keys() if "p_label" in k]
                    if self.plauslabel_mode in [1,3]:
                        labels += ["label"]
                    lbl = random.choice(labels)
                else:
                    lbl = "label"
                local_labels = patches_batch[lbl][tio.DATA].float().cuda()
                if self.plauslabels and self.distloss:
                    local_plauslabels = []
                    for l in labels:
                        local_plauslabels.append(patches_batch[l][tio.DATA])
                    local_plauslabels = torch.concat(local_plauslabels, dim=0 if self.distloss_mode in [0,1] else 1).float().cuda()
                    if self.dimMode == 3:
                        local_plauslabels = torch.movedim(local_plauslabels, -1, -3) 
                    else:
                        local_plauslabels = local_plauslabels.squeeze(-1)
                
                if self.dimMode == 3:
                    local_batch = torch.movedim(local_batch, -1, -3)
                    local_labels = torch.movedim(local_labels, -1, -3) 
                else:
                    local_batch = local_batch.squeeze(-1)
                    local_labels = local_labels.squeeze(-1)

                # Transfer to GPU
                self.logger.debug('Epoch: {} Batch Index: {}'.format(epoch, batch_index))

                if self.ProbFlag == 2:
                    self.model.reset()
                    self.model.train()
                    
                # Clear gradients
                self.optimizer.zero_grad()

                # try:
                with autocast(enabled=self.with_apex):
                    loss_ratios = [1, 0.66, 0.34]  #TODO param

                    floss = 0
                    output1 = 0
                    level = 0

                    # -------------------------------------------------------------------------------------------------
                    # First Branch Supervised error
                    if self.ProbFlag == 0:
                        if self.modelID == 6: #SSN
                            logits, state = self.model(local_batch)
                            floss = self.ssnloss(logits, state['distribution'], local_labels)
                        elif self.modelID == 7: #VIMH
                            soft_out, _, kl = self.model(local_batch, samples=self.n_prob_test, num_classes=1)
                            floss, output = self.vimhloss(soft_out, kl, local_labels, train=True)
                        else:
                            for output in self.model(local_batch): 
                                if level == 0:
                                    output1 = output
                                if level > 0:  # then the output size is reduced, and hence interpolate to patch_size
                                    output = torch.nn.functional.interpolate(input=output, size=self.patch_size[:2] if self.dimMode == 2 else self.patch_size)
                                output = torch.sigmoid(output)
                                floss += loss_ratios[level] * self.focalTverskyLoss(output, local_labels)
                                level += 1
                    elif self.ProbFlag == 1:
                        self.model.forward(local_batch, local_labels, training=True)
                        elbo = self.model.elbo(local_labels, analytic_kl=True)
                        reg_loss = self.l2_regularisation(self.model.posterior) + self.l2_regularisation(self.model.prior) + self.l2_regularisation(self.model.fcomb.layers)
                        floss = -elbo + self.model.reg_alpha * reg_loss
                        # if self.with_apex:
                        #     floss = [self.model.mean_reconstruction_loss if self.model.use_mean_recon_loss else self.model.reconstruction_loss, 
                        #             -(self.model.beta * self.model.kl), 
                        #             self.model.reg_alpha * reg_loss] 
                        # else:
                        #     floss = -elbo + self.model.reg_alpha * reg_loss
                    else:
                        if not self.plauslabels or (self.plauslabels and not self.distloss):
                            output = self.model(local_batch, local_labels, make_onehot=False)
                            loss_segmentation = self.criterion_segmentation(output, local_labels).sum()
                        else:
                            output = self.model(local_batch, local_labels, make_onehot=False, distlossN=len(labels))
                            # loss_segmentation = self.op_distloss(output, local_plauslabels)
                            loss_segmentation = self.op_distloss(torch.concat(output, dim=0 if self.distloss_mode in [0,1] else 1), local_plauslabels)
                        loss_latent = self.criterion_latent(self.model.posterior, self.model.prior).sum()
                        floss = self.criterion_segmentation_weight * loss_segmentation + self.criterion_latent_weight * loss_latent

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
                                output = torch.nn.functional.interpolate(input=output, size=self.patch_size[:2] if self.dimMode == 2 else self.patch_size)

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

                # except Exception as error:
                #     self.logger.exception(error)
                #     sys.exit()

                self.logger.info("Epoch:" + str(epoch) + " Batch_Index:" + str(batch_index) + "Training..." +
                                 "\n MainLoss:" + str(floss))

                # Calculating gradients
                if self.with_apex:
                    if type(floss) is list:
                        for i in range(len(floss)):
                            if i+1 == len(floss): #final loss
                                self.scaler.scale(floss[i]).backward()
                            else:
                                self.scaler.scale(floss[i]).backward(retain_graph=True)
                        floss = torch.sum(torch.stack(floss))
                    else:
                        self.scaler.scale(floss).backward()

                    if self.clip_grads:
                        self.scaler.unscale_(self.optimizer)
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1) 
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), 1) 
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    floss.backward()
                    if self.clip_grads:
                        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)

                    self.optimizer.step()

                if training_batch_index % 50 == 0:  # Save best metric evaluation weights                        
                    write_summary(self.writer_training, self.logger, training_batch_index, focalTverskyLoss=floss.detach().item())
                    wandb.log({"loss": floss.detach().item()})
                training_batch_index += 1

                # Initialising the average loss metrics
                total_floss += floss.detach().item()

                if self.deform:
                    del elastic
                    torch.cuda.empty_cache()

            # Calculate the average loss per batch in one epoch
            total_floss /= (batch_index + 1.0)

            # Print every epoch
            self.logger.info("Epoch:" + str(epoch) + " Average Training..." +
                             "\n MainLoss:" + str(total_floss))

            chk_dict = {
                'epoch_type': 'last',
                'epoch': epoch,
                # Let is always overwrite, we need just the last checkpoint and best checkpoint(saved after validate)
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }

            if self.with_apex:
                chk_dict["amp"] = self.scaler.state_dict()

            save_model(self.checkpoint_path, chk_dict)

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
        print("Validate Epoch: "+str(epoch) +" of "+ str(self.num_epochs))

        floss, binloss, dScore, dscore, jaccard_index = 0, 0, 0, 0, 0
        no_patches = 0
        self.model.eval()
        data_loader = self.validate_loader
        writer = self.writer_validating
        with torch.no_grad():
            for index, patches_batch in enumerate(tqdm(data_loader)):
                self.logger.info("loading" + str(index))
                no_patches += 1

                local_batch = self.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                if self.plauslabels and self.plauslabel_mode>=3 and not self.distloss:
                    labels = [k for k in patches_batch.keys() if "p_label" in k]
                    if self.plauslabel_mode == 3:
                        labels += ["label"]
                    lbl = random.choice(labels)
                else:
                    lbl = "label"
                local_labels = patches_batch[lbl][tio.DATA].float().cuda()
                if self.plauslabels and self.distloss:
                    labels = [k for k in patches_batch.keys() if "p_label" in k]
                    if self.plauslabel_mode in [1,3]:
                        labels += ["label"]
                    local_plauslabels = []
                    for l in labels:
                        local_plauslabels.append(patches_batch[l][tio.DATA])
                    local_plauslabels = torch.concat(local_plauslabels, dim=0 if self.distloss_mode in [0,1] else 1).float().cuda()
                    if self.dimMode == 3:
                        local_plauslabels = torch.movedim(local_plauslabels, -1, -3) 
                    else:
                        local_plauslabels = local_plauslabels.squeeze(-1)

                if self.dimMode == 3:
                    local_batch = torch.movedim(local_batch, -1, -3)
                    local_labels = torch.movedim(local_labels, -1, -3) 
                else:
                    local_batch = local_batch.squeeze(-1)
                    local_labels = local_labels.squeeze(-1)

                floss_iter = 0
                output1 = 0
                try:
                    with autocast(enabled=self.with_apex):
                        # Forward propagation
                        loss_ratios = [1, 0.66, 0.34] #TODO param
                        level = 0

                        # Forward propagation
                        if self.ProbFlag == 0:
                            if self.modelID == 6: #SSN
                                logits, state = self.model(local_batch)
                                floss_iter = self.ssnloss(logits, state['distribution'], local_labels)
                                # prob = torch.nn.functional.softmax(logits, dim=1)
                                # _, output = torch.max(logits, dim=1)
                                output = logits
                            elif self.modelID == 7: #VIMH
                                soft_out, _, kl = self.model(local_batch, samples=self.n_prob_test, num_classes=1)
                                floss_iter, output = self.vimhloss(soft_out, kl, local_labels, train=False)
                            else:
                                for output in self.model(local_batch):
                                    if level == 0:
                                        output1 = output
                                    if level > 0:  # then the output size is reduced, and hence interpolate to patch_size
                                        output = torch.nn.functional.interpolate(input=output, size=self.patch_size[:2] if self.dimMode == 2 else self.patch_size)

                                    output = torch.sigmoid(output)
                                    floss_iter += loss_ratios[level] * self.focalTverskyLoss(output, local_labels)
                                    level += 1
                        elif self.ProbFlag == 1:
                            self.model.forward(local_batch, training=False)
                            output1 = self.model.sample(testing=True)
                            # output1 = torch.sigmoid(self.model.sample(testing=True))
                            # floss_iter = self.focalTverskyLoss(output1, local_labels)
                            elbo = self.model.elbo(local_labels)
                            reg_loss = self.l2_regularisation(self.model.posterior) + self.l2_regularisation(self.model.prior) + self.l2_regularisation(self.model.fcomb.layers)
                            floss_iter = -elbo + self.model.reg_alpha * reg_loss
                        else:
                            if not self.plauslabels or (self.plauslabels and not self.distloss):
                                output1 = self.model.sample_prior(N=1, out_device=local_batch.device, input_=local_batch)
                                floss_iter = self.criterion_segmentation(output1, local_labels).sum()
                                # loss_latent = self.criterion_latent(self.model.posterior, self.model.prior).sum() #posterior isn't present when there is no gradient (in eval mode)
                                # floss_iter = self.criterion_segmentation_weight * loss_segmentation + self.criterion_latent_weight * loss_latent
                            else:
                                output = self.model.sample_prior(N=len(labels), out_device=local_batch.device, input_=local_batch)
                                try:
                                    floss_iter = self.op_distloss(torch.concat(output, dim=0 if self.distloss_mode in [0,1] else 1), local_plauslabels).detach().item()
                                except:
                                    pass
                                output1 = random.choice(output)

                                #as there is no loss_latent, this is used as the final loss. 
                                # loss_latent = self.criterion_latent(self.model.posterior, self.model.prior).sum() #posterior isn't present when there is no gradient (in eval mode)
                                # floss_iter = self.criterion_segmentation_weight * loss_segmentation + self.criterion_latent_weight * loss_latent
                        
                except Exception as error:
                    self.logger.exception(error)
                
                floss += floss_iter
                if self.ProbFlag == 0:
                    dl, ds = self.dice(torch.sigmoid(output1), local_labels)
                else:
                    dl, ds = self.dice(output1, local_labels)
                dScore += ds.detach().item()

        # Average the losses
        floss = floss / no_patches
        dScore = dScore / no_patches
        process = ' Validating'
        self.logger.info("Epoch:" + str(tainingIndex) + process + "..." +
                         "\n MainLoss:" + str(floss) +
                         "\n DiceScore:" + str(dScore))
        if self.dimMode == 3:
            write_summary(writer, self.logger, tainingIndex, local_labels[0][0][6], output1[0][0][6], floss, dScore, 0, 0)
        else:
            write_summary(writer, self.logger, tainingIndex, local_labels[0][0], output1[0][0], floss, dScore, 0, 0)
        wandb.log({"loss": floss})
        wandb.log({"Dice": dScore})

        if self.LOWEST_LOSS > floss:  # Save best metric evaluation weights
            self.LOWEST_LOSS = floss
            self.logger.info(
                'Best metric... @ epoch:' + str(tainingIndex) + ' Current Lowest loss:' + str(self.LOWEST_LOSS))

            chk_dict = {
                'epoch_type': 'best',
                'epoch': epoch,
                'lowest_loss': self.LOWEST_LOSS,
                # Let is always overwrite, we need just the last checkpoint and best checkpoint(saved after validate)
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }

            if self.with_apex:
                chk_dict["amp"] = self.scaler.state_dict()

            save_model(self.checkpoint_path, chk_dict)

    def test(self, test_logger, save_results=True, test_subjects=None): #for testing models other than Probabilistic UNets
        test_logger.debug('Testing...')

        if test_subjects is None:
            test_folder_path = self.DATASET_FOLDER + '/test/'
            test_label_path = self.DATASET_FOLDER + '/test_label/'

            test_subjects = self.create_TIOSubDS(vol_path=test_folder_path, label_path=test_label_path, get_subjects_only=True)

        overlap = np.subtract(self.patch_size, (self.stride_length, self.stride_width, self.stride_depth))

        df = pd.DataFrame(columns = ["Subject", "Dice", "IoU"])
        result_root = os.path.join(self.output_path, self.model_name, "results_"+("best" if self.load_best else "last"))
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
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size, shuffle=False, num_workers=self.num_worker)

                for index, patches_batch in enumerate(tqdm(patch_loader)):
                    local_batch = self.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                    local_labels = patches_batch['label'][tio.DATA].float().cuda()
                    locations = patches_batch[tio.LOCATION]

                    if self.dimMode == 3:
                        local_batch = torch.movedim(local_batch, -1, -3)
                        local_labels = torch.movedim(local_labels, -1, -3) 
                    else:
                        local_batch = local_batch.squeeze(-1)
                        local_labels = local_labels.squeeze(-1)

                    with autocast(enabled=self.with_apex):
                        if self.ProbFlag == 0:
                            output = self.model(local_batch)
                            if type(output) is tuple or type(output) is list:
                                output = output[0]
                            output = torch.sigmoid(output).detach().cpu()
                        else:                            
                            sys.exit("This test function is not capable of testing Probabilistic UNets. Please use the test_prob functin.")

                    if self.dimMode == 2:
                        output = output.unsqueeze(-3)
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
                    save_nifti(result, os.path.join(result_root, subjectname+".nii.gz"))

                    resultMIP = np.max(result, axis=-1)
                    Image.fromarray((resultMIP*255).astype('uint8'), 'L').save(os.path.join(result_root, subjectname+"_MIP.tif"))

                    if label is not None:
                        overlay = create_diff_mask_binary(result, label)
                        save_tifRGB(overlay, os.path.join(result_root, subjectname+"_colour.tif"))

                        overlayMIP = create_diff_mask_binary(resultMIP, np.max(label, axis=-1))
                        Image.fromarray(overlayMIP.astype('uint8'), 'RGB').save(os.path.join(result_root, subjectname+"_colourMIP.tif"))

                test_logger.info("Testing "+subjectname+"..." +
                                "\n Dice:" + str(dice3D) +
                                "\n JacardIndex:" + str(iou3D))

        df.to_csv(os.path.join(result_root, "Results_Main.csv"))

    def test_prob(self, test_logger, save_results=True, test_subjects=None): #for testing Probabilistic UNets
        test_logger.debug('Probabilistic Testing...')
        if self.ProbFlag == 0:
            test_logger.debug("Warning: The probabilistic flag is set to 0, but currently using the probabilistic testing function. This test function is mainly for testing Probabilistic UNets. test() function is meant for the other models. However, if it is desired to sample N-sampled from those models using MC-Dropout, then using this function is acqurate...")

        if test_subjects is None:
            test_folder_path = self.DATASET_FOLDER + '/test/'
            test_label_path = self.DATASET_FOLDER + '/test_label/'
            test_plauslabel_path = self.DATASET_FOLDER + '/test_plausiblelabel/'

            test_subjects = self.create_TIOSubDS(vol_path=test_folder_path, label_path=test_label_path, get_subjects_only=True, plauslabels_path=test_plauslabel_path if os.path.exists(test_plauslabel_path) else "")

        overlap = np.subtract(self.patch_size, (self.stride_length, self.stride_width, self.stride_depth))

        df = pd.DataFrame(columns = ["Subject", "ProbPredID", "Dice", "IoU"])
        dfProb = pd.DataFrame(columns = ["Subject", "maxDice", "maxIoU", "DistLoss", "FID", "GenEngDist"])
        result_root = os.path.join(self.output_path, self.model_name, "results_"+("best" if self.load_best else "last"))
        os.makedirs(result_root, exist_ok=True)

        self.model.eval()
    
        with torch.no_grad():
            masterResults = []
            masterGTs = []
            for test_subject in test_subjects:
                if 'label' in test_subject:
                    label = test_subject['label'][tio.DATA].float().squeeze().numpy()
                    del test_subject['label']
                else:
                    label = None

                labels = [k for k in test_subject.keys() if "p_label" in k]
                plauslabels = []
                for l in labels:
                    plauslabels.append(test_subject[l][tio.DATA])
                if len(plauslabels) > 0:
                    plauslabels = torch.concat(plauslabels, dim=0).float()

                subjectname = test_subject['subjectname']
                del test_subject['subjectname']

                grid_sampler = tio.inference.GridSampler(
                                                            test_subject,
                                                            self.patch_size,
                                                            overlap,
                                                        )
                                        
                aggregators = []
                for _ in range(self.n_prob_test):
                    aggregators.append(tio.inference.GridAggregator(grid_sampler, overlap_mode="average"))

                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size, shuffle=False, num_workers=self.num_worker)

                distloss = 0
                for index, patches_batch in enumerate(tqdm(patch_loader)):
                    local_batch = self.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                    # local_labels = patches_batch['label'][tio.DATA].float().cuda()
                    locations = patches_batch[tio.LOCATION]

                    labels = [k for k in patches_batch.keys() if "p_label" in k]
                    local_plauslabels = []
                    for l in labels:
                        local_plauslabels.append(patches_batch[l][tio.DATA])
                    if len(local_plauslabels) > 0:
                        local_plauslabels = torch.concat(local_plauslabels, dim=0).float().cuda()
                        if self.dimMode == 3:
                            local_plauslabels = torch.movedim(local_plauslabels, -1, -3) 
                        else:
                            local_plauslabels = local_plauslabels.squeeze(-1)

                    if self.dimMode == 3:
                        local_batch = torch.movedim(local_batch, -1, -3)
                        # local_labels = torch.movedim(local_labels, -1, -3) 
                    else:
                        local_batch = local_batch.squeeze(-1)
                        # local_labels = local_labels.squeeze(-1)

                    with autocast(enabled=self.with_apex):
                        if self.ProbFlag == 0:          
                            outputs = []
                            for nP in range(self.n_prob_test):              
                                if self.modelID == 6: #SSN
                                    logits, _ = self.model(local_batch)
                                    # prob = torch.nn.functional.softmax(logits, dim=1)
                                    # _, output = torch.max(logits, dim=1)
                                    outputs.append(logits.detach().cpu())
                                elif self.modelID == 7: #VIMH
                                    soft_out, _, kl = self.model(local_batch, samples=self.n_prob_test, num_classes=1)
                                    _, output = self.vimhloss(soft_out, kl, None, train=False)
                                    outputs.append(output.detach().cpu())
                                else:
                                    output = self.model(local_batch)
                                    if type(output) is tuple or type(output) is list:
                                        output = output[0]
                                    outputs.append(torch.sigmoid(output).detach().cpu())
                        elif self.ProbFlag == 1:
                            sys.exit("Testing function for ProbFlag 1 (ProbUNetV1) is not ready!")
                            self.model.forward(local_batch, training=False)
                            output = self.model.sample(testing=True).detach().cpu() #TODO: need to check whether sigmoid is needed for prob
                        else:                            
                            outputs = self.model.sample_prior(N=self.n_prob_test, out_device=local_batch.device, input_=local_batch)
                            if len(local_plauslabels) > 0:
                                try:
                                    distloss += self.op_distloss(torch.concat(outputs, dim=0), local_plauslabels).detach().item()
                                except:
                                    pass

                    # wandb.log({"loss": floss.detach().item()})

                    for nP in range(self.n_prob_test):
                        output = outputs[nP].detach().cpu()
                        if self.dimMode == 2:
                            output = output.unsqueeze(-3)
                        output = torch.movedim(output, -3, -1).type(local_batch.type()) 
                        aggregators[nP].add_batch(output, locations)
                        del output
                    del outputs

                results = []
                maxDice = 0
                maxIoU = 0
                for nP in range(self.n_prob_test):

                    predicted = aggregators[nP].get_output_tensor().squeeze().numpy()

                    try:
                        thresh = threshold_otsu(predicted)
                        result = predicted > thresh
                    except Exception as error:
                        test_logger.exception(error)
                        result = predicted > 0.5  # exception will be thrown only if input image seems to have just one color 1.0.
                    result = result.astype(np.float32)
                    results.append(result)
                    del predicted
                    
                    if label is not None:
                        datum = {"Subject": subjectname, "ProbPredID": nP}
                        dice3D = dice(result, label)
                        iou3D = IoU(result, label)
                        datum = pd.DataFrame.from_dict({**datum, "Dice": [dice3D], "IoU": [iou3D]})
                        df = pd.concat([df, datum], ignore_index=True)
                        if maxDice < dice3D:
                            maxDice = dice3D
                        if maxIoU < iou3D:
                            maxIoU = iou3D

                    if save_results:
                        save_nifti(result, os.path.join(result_root, f"{subjectname}_Segmentation_{nP}.nii.gz"))

                        resultMIP = np.max(result, axis=-1)
                        Image.fromarray((resultMIP*255).astype('uint8'), 'L').save(os.path.join(result_root, f"{subjectname}_Segmentation_{nP}_MIP.tif"))

                        if label is not None:
                            overlay = create_diff_mask_binary(result, label)
                            save_tifRGB(overlay, os.path.join(result_root, f"{subjectname}_Segmentation_{nP}_colour.tif"))
                            del overlay

                            overlayMIP = create_diff_mask_binary(resultMIP, np.max(label, axis=-1))
                            Image.fromarray(overlayMIP.astype('uint8'), 'RGB').save(os.path.join(result_root, f"{subjectname}_Segmentation_{nP}_colourMIP.tif"))
                            del overlayMIP, resultMIP

                    test_logger.info("Testing "+subjectname+"..." +
                                    "\n Dice:" + str(dice3D) +
                                    "\n JacardIndex:" + str(iou3D))
                del aggregators

                datumProb = {"Subject": subjectname, "maxDice": maxDice, "maxIoU": maxIoU}
                if len(plauslabels) > 0:
                    gt_segs = plauslabels.numpy()
                    # masterGTs.append(gt_segs)
                    seg_samples = np.stack(results, axis=0)
                    # masterResults.append(seg_samples)
                    fidVal = self.fid_calc(torch.from_numpy(seg_samples), torch.from_numpy(gt_segs)).item()
                    genEDist = calc_energy_distances(get_energy_distance_components(gt_segs, seg_samples, eval_class_ids=[1]))
                    datumProb = pd.DataFrame.from_dict({**datumProb, "DistLoss": [distloss], "FID": [fidVal], "GenEngDist": [genEDist]})
                    del gt_segs, seg_samples, plauslabels
                else:
                    datumProb = pd.DataFrame.from_dict(datumProb)
                dfProb = pd.concat([dfProb, datumProb], ignore_index=True)

        wandb.summary['test_'+("best" if self.load_best else "last")+'_maxDice'] = datumProb.maxDice.median()
        wandb.summary['test_'+("best" if self.load_best else "last")+'_maxIoU'] = datumProb.maxIoU.median()
        wandb.summary['test_'+("best" if self.load_best else "last")+'_DistLoss'] = datumProb.DistLoss.median()
        wandb.summary['test_'+("best" if self.load_best else "last")+'_FID'] = datumProb.FID.median()
        wandb.summary['test_'+("best" if self.load_best else "last")+'_genEDist'] = datumProb.GenEngDist.median()

        # masterGTs = np.concatenate(masterGTs, axis=0)
        # masterResults = np.concatenate(masterResults, axis=0)
        # fidVal = self.fid_calc(torch.from_numpy(masterResults), torch.from_numpy(masterGTs)).item()
        # genEDist = calc_energy_distances(get_energy_distance_components(masterGTs, masterResults, eval_class_ids=[1]))
        # datumProb = pd.DataFrame.from_dict({"Subject": "TotalDistCompare", "maxDice": -1, "maxIoU": -1, "DistLoss": -1, "FID": fidVal, "GenEngDist": -1})
        # dfProb = pd.concat([dfProb, datumProb], ignore_index=True)

        df.to_csv(os.path.join(result_root, "Results_Main.csv"))
        dfProb.to_csv(os.path.join(result_root, "Results_Probabilistic.csv"))

    def predict(self, image_path, label_path, predict_logger):
        image_name = os.path.basename(image_path).split('.')[0]

        subdict = {
                        "img":tio.ScalarImage(image_path),
                        "subjectname":image_name,
                    }
        
        if bool(label_path):
            subdict["label"] = tio.LabelMap(label_path)

        subject = tio.Subject(**subdict)

        self.test(predict_logger, save_results=True, test_subjects=[subject])
