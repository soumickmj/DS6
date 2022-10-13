#!/usr/bin/env python
"""
"""

import argparse
import random
import os
import numpy as np
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from pipeline import Pipeline
from Utils.logger import Logger
from Utils.model_manager import getModel
from Utils.vessel_utils import load_model, load_model_with_amp

__author__ = "Kartik Prabhu, Mahantesh Pattadkal, and Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Kartik Prabhu", "Mahantesh Pattadkal", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(2020)
np.random.seed(2020)
random.seed(2020)

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-model",
                        type=int,
                        default=2,
                        help="1{U-Net}; \n"
                             "2{U-Net_Deepsup}; \n"
                             "3{Attention-U-Net}; \n"
                             "4{Probabilistic-U-Net};")
    parser.add_argument("-model_name",
                        default="Model_v1",
                        help="Name of the model")
    parser.add_argument("-dataset_path",
                        default="/vol3/schatter/DS6/Dataset/OriginalVols/300",
                        help="Path to folder containing dataset."
                             "Further divide folders into train,validate,test, train_label,validate_label and test_label."
                             "Example: /home/dataset/")
    parser.add_argument("-output_path",
                        default="/home/schatter/Soumick/Output/DS6/OriginalVols_FDPv0",
                        help="Folder path to store output "
                             "Example: /home/output/")

    parser.add_argument('-train',
                        default=True,
                        help="To train the model")
    parser.add_argument('-test',
                        default=True,
                        help="To test the model")
    parser.add_argument('-test_with_mip',
                        default=False,
                        help="To test the model with MIP")
    parser.add_argument('-pseudo_train',
                        default=False,
                        help="To test the model with MIP")
    parser.add_argument('-predict',
                        default=False,
                        help="To predict a segmentation output of the model and to get a diff between label and output")
    parser.add_argument('-predictor_path',
                        default="/vol3/schatter/DS6/Dataset/BiasFieldCorrected/300/test/vk04.nii",
                        help="Path to the input image to predict an output, ex:/home/test/ww25.nii ")
    parser.add_argument('-predictor_label_path',
                        default="/vol3/schatter/DS6/Dataset/BiasFieldCorrected/300/test_label/vk04.nii.gz",
                        help="Path to the label image to find the diff between label an output, ex:/home/test/ww25_label.nii ")

    parser.add_argument('-load_path',
                        # default="/home/schatter/Soumick/Output/DS6/OrigVol_MaskedFDIPv0_UNetV2/checkpoint",
                        default="/home/schatter/Soumick/Output/DS6/OriginalVols_FDPv0/UNetMSS_X2_Deform/checkpoint/",
                        help="Path to checkpoint of existing model to load, ex:/home/model/checkpoint")
    parser.add_argument('-load_best',
                        default=True,
                        help="Specifiy whether to load the best checkpoiont or the last. Also to be used if Train and Test both are true.")
    parser.add_argument('-deform',
                        default=False,
                        action="store_true",
                        help="To use deformation for training")
    parser.add_argument('-clip_grads',
                        default=True,
                        action="store_true",
                        help="To use deformation for training")
    parser.add_argument('-apex',
                        default=True,
                        help="To use half precision on model weights.")

    parser.add_argument("-batch_size",
                        type=int,
                        default=15,
                        help="Batch size for training")
    parser.add_argument("-num_epochs",
                        type=int,
                        default=50,
                        help="Number of epochs for training")
    parser.add_argument("-learning_rate",
                        type=float,
                        default=0.001,
                        help="Learning rate")
    parser.add_argument("-patch_size",
                        type=int,
                        default=64,
                        help="Patch size of the input volume")
    parser.add_argument("-stride_depth",
                        type=int,
                        default=16,
                        help="Strides for dividing the input volume into patches in depth dimension (To be used during validation and inference)")
    parser.add_argument("-stride_width",
                        type=int,
                        default=32,
                        help="Strides for dividing the input volume into patches in width dimension (To be used during validation and inference)")
    parser.add_argument("-stride_length",
                        type=int,
                        default=32,
                        help="Strides for dividing the input volume into patches in length dimension (To be used during validation and inference)")
    parser.add_argument("-samples_per_epoch",
                        type=int,
                        default=8000,
                        help="Number of samples per epoch")
    parser.add_argument("-num_worker",
                        type=int,
                        default=8,
                        help="Number of worker threads")
    parser.add_argument("-floss_coeff",
                        type=float,
                        default=0.7,
                        help="Loss coefficient for floss in total loss")
    parser.add_argument("-mip_loss_coeff",
                        type=float,
                        default=0.3,
                        help="Loss coefficient for mip_loss in total loss")
    parser.add_argument("-floss_param_smooth",
                        type=float,
                        default=1,
                        help="Loss coefficient for floss_param_smooth")
    parser.add_argument("-floss_param_gamma",
                        type=float,
                        default=0.75,
                        help="Loss coefficient for floss_param_gamma")
    parser.add_argument("-floss_param_alpha",
                        type=float,
                        default=0.7,
                        help="Loss coefficient for floss_param_alpha")
    parser.add_argument("-mip_loss_param_smooth",
                        type=float,
                        default=1,
                        help="Loss coefficient for mip_loss_param_smooth")
    parser.add_argument("-mip_loss_param_gamma",
                        type=float,
                        default=0.75,
                        help="Loss coefficient for mip_loss_param_gamma")
    parser.add_argument("-mip_loss_param_alpha",
                        type=float,
                        default=0.7,
                        help="Loss coefficient for mip_loss_param_alpha")
    parser.add_argument("-k_folds",
                        type=int,
                        default=5,
                        help="Set the number of folds for cross validation")
    parser.add_argument("-wandb",
                        default=True,
                        help="Set this to true to include wandb logging")

    args = parser.parse_args()

    if args.deform:
        args.model_name += "_Deform"

    MODEL_NAME = args.model_name
    DATASET_FOLDER = args.dataset_path
    OUTPUT_PATH = args.output_path

    LOAD_PATH = args.load_path
    CHECKPOINT_PATH = OUTPUT_PATH + "/" + MODEL_NAME + '/checkpoint/'
    TENSORBOARD_PATH_TRAINING = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_training/'
    TENSORBOARD_PATH_VALIDATION = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_validation/'
    TENSORBOARD_PATH_TESTING = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_testing/'

    LOGGER_PATH = OUTPUT_PATH + "/" + MODEL_NAME + '.log'

    logger = Logger(MODEL_NAME, LOGGER_PATH).get_logger()
    test_logger = Logger(MODEL_NAME + '_test', LOGGER_PATH).get_logger()
    wandb = None
    if str(args.wandb).lower() == "true":
        import wandb

        wandb.init(project="DS6_VesselSeg2", entity="ds6_vessel_seg2", notes=args.model_name)
        wandb.config = {
            "learning_rate": args.learning_rate,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "patch_size": args.patch_size,
            "samples_per_epoch": args.samples_per_epoch,
            "mip_loss_coeff": args.mip_loss_coeff,
            "floss_coeff": args.floss_coeff
        }


    # Model
    model = torch.nn.DataParallel(getModel(args.model, OUTPUT_PATH + "/" + MODEL_NAME))
    model.cuda()

    writer_training = SummaryWriter(TENSORBOARD_PATH_TRAINING)
    writer_validating = SummaryWriter(TENSORBOARD_PATH_VALIDATION)

    pipeline = Pipeline(cmd_args=args, model=model, logger=logger,
                        dir_path=DATASET_FOLDER, checkpoint_path=CHECKPOINT_PATH,
                        writer_training=writer_training, writer_validating=writer_validating, test_logger=test_logger, wandb=wandb)

    # loading existing checkpoint if supplied
    if bool(LOAD_PATH):
        pipeline.load(checkpoint_path=LOAD_PATH, load_best=args.load_best)

    # try:

    if args.train:
        pipeline.train()
        torch.cuda.empty_cache()  # to avoid memory errors

    if args.test:
        if args.load_best:
            if bool(LOAD_PATH):
                pipeline.load(checkpoint_path=LOAD_PATH, load_best=args.load_best)
            else:
                pipeline.load(load_best=args.load_best)
        pipeline.test(test_logger=test_logger)
        torch.cuda.empty_cache()  # to avoid memory errors

    if args.test_with_mip:
        if args.load_best:
            if bool(LOAD_PATH):
                pipeline.load(checkpoint_path=LOAD_PATH, load_best=args.load_best)
            else:
                pipeline.load(load_best=args.load_best)
        pipeline.test_with_MIP(test_logger=test_logger)
        torch.cuda.empty_cache()  # to avoid memory errors

    if args.pseudo_train:
        if args.load_best:
            if bool(LOAD_PATH):
                pipeline.load(checkpoint_path=LOAD_PATH, load_best=args.load_best)
            else:
                pipeline.load(load_best=args.load_best)
        pipeline.pseudo_train(test_logger=test_logger)
        torch.cuda.empty_cache()  # to avoid memory errors

    if args.predict:
        pipeline.predict(args.predictor_path, args.predictor_label_path, predict_logger=test_logger)


    # except Exception as error:
    #     logger.exception(error)
    writer_training.close()
    writer_validating.close()