#!/usr/bin/env python
"""

"""
import json
import argparse
import random
import os
import numpy as np
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import wandb

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
torch.set_num_threads(2)

# torch.autograd.set_detect_anomaly(True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=int,
                        default=5,
                        help="1{U-Net}; \n"
                             "2{U-Net_Deepsup}; \n"
                             "3{Attention-U-Net}; \n"
                             "4{Probabilistic-U-Net};\n"
                             "5{V2-Probabilistic-U-Net};\n"
                             "6{S-S-N};\n"
                             "7{VI-MH};\n"
                             "8{DO-UNet};")
    parser.add_argument("--model_name",
                        default="prova_3DMSSeg",
                        help="Name of the model")
    parser.add_argument("--datajson_path", 
                        default="MSSeg_FCM",
                        help="Path to the json file (without the .json extension, only the name) which contains the path to the data and output. Must be present inside the support folder."
                             "Must contain dataset_path, the folder which must be further divided folders into train,validate,test, train_label,validate_label and test_label."
                             "Must contain output_path")
    parser.add_argument('--plauslabels',
                        default=True, action=argparse.BooleanOptionalAction,
                        help="Whether or not to use the plausable labels (training with multiple labels randomly). This will required three additional folders inside the dataset_path: train_plausiblelabel, test_plausiblelabel, validate_plausiblelabel")
    parser.add_argument("--plauslabel_mode",
                        type=int,
                        default=4,
                        help="1{Use-Plausable-And-Main-For-Training}; \n"
                             "2{Use-Plausable-Only-For-Training}; \n"
                             "3{Use-Plausable-And-Main-For-TrainAndValid}; \n"
                             "4{Use-Plausable-Only-For-TrainAndValid};")
    
    parser.add_argument('--train',
                        default=True, action=argparse.BooleanOptionalAction,
                        help="To train the model")
    parser.add_argument('--test',
                        default=False, action=argparse.BooleanOptionalAction,
                        help="To test the model")
    parser.add_argument("--n_prob_test",
                        type=int,
                        default=7,
                        help="N number of predictions are to be optained during testing for the ProbUNets")
    parser.add_argument('--predict',
                        default=False, action=argparse.BooleanOptionalAction,
                        help="To predict a segmentation output of the model and to get a diff between label and output")
    parser.add_argument('--predictor_path',
                        default="/vol3/schatter/DS6/Dataset/BiasFieldCorrected/300/test/vk04.nii",
                        help="Path to the input image to predict an output, ex:/home/test/ww25.nii ")
    parser.add_argument('--predictor_label_path',
                        default="/vol3/schatter/DS6/Dataset/BiasFieldCorrected/300/test_label/vk04.nii.gz",
                        help="Path to the label image to find the diff between label an output, ex:/home/test/ww25_label.nii ")

    parser.add_argument('--load_path',
                        # default="/home/schatter/Soumick/Output/DS6/OrigVol_MaskedFDIPv0_UNetV2/checkpoint",
                        default="",
                        help="Path to checkpoint of existing model to load, ex:/home/model/checkpoint")
    parser.add_argument('--load_best',
                        default=True, action=argparse.BooleanOptionalAction,
                        help="Specifiy whether to load the best checkpoiont or the last. Also to be used if Train and Test both are true.")
    parser.add_argument('--deform',
                        default=False, action=argparse.BooleanOptionalAction,
                        help="To use deformation for training")
    parser.add_argument('--clip_grads',
                        default=False, action=argparse.BooleanOptionalAction,
                        help="To use deformation for training")
    parser.add_argument('--distloss',
                        default=False, action=argparse.BooleanOptionalAction,
                        help="To compute loss by comparing distributions of output and GT (for ProbUNet)")
    parser.add_argument('--distloss_mode',
                        default=3, type=int,
                        help="0: Pure FID for distloss (repeats the input to make 3 channels as pretrained on RGB imagenet) \n"
                             "1: For Fréchet ResNeXt Distance (trained on single-channel MRIs) \n"
                             "2: GeomLoss Sinkhorn (Default cost function) \n"
                             "3: GeomLoss Hausdorff (Default cost function) using energy kernel (squared distances)")
    parser.add_argument('--apex',
                        default=True, action=argparse.BooleanOptionalAction,
                        help="To use half precision on model weights.")

    parser.add_argument("--batch_size",
                        type=int,
                        default=20,
                        help="Batch size for training")
    parser.add_argument("--batch_size_fidloss",
                        type=int,
                        default=4,
                        help="Batch size for FID loss computation. Set it to -1 if the complete batch is supposed to be processed together")
    parser.add_argument("--num_epochs",
                        type=int,
                        default=500,
                        help="Number of epochs for training")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-4,
                        help="Learning rate")
    parser.add_argument("--patch_size",
                        type=int,
                        default=64,
                        help="Patch size of the input volume")
    parser.add_argument("--slice2D_shape",
                        default="",
                        help="For 2D models, set it to the desired shape. Or blank")
    parser.add_argument("--stride_depth",
                        type=int,
                        default=16,
                        help="Strides for dividing the input volume into patches in depth dimension (To be used during validation and inference)")
    parser.add_argument("--stride_width",
                        type=int,
                        default=32,
                        help="Strides for dividing the input volume into patches in width dimension (To be used during validation and inference)")
    parser.add_argument("--stride_length",
                        type=int,
                        default=32,
                        help="Strides for dividing the input volume into patches in length dimension (To be used during validation and inference)")
    parser.add_argument("--samples_per_epoch",
                        type=int,
                        default=10000,
                        help="Number of samples per epoch")
    parser.add_argument("--num_worker",
                        type=int,
                        default=5,
                        help="Number of worker threads")

    args = parser.parse_args()

    with open(f"./support/{args.datajson_path}.json", 'r') as json_file:
            json_data = json.load(json_file)
    args.__dict__.update(json_data)

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

    # Model
    model = getModel(args.model, is2D=bool(args.slice2D_shape), n_prob_test=args.n_prob_test)
    model.cuda()
    print("It's a 2D model!!" if bool(args.slice2D_shape) else "It's a 3D model!!")

    writer_training = SummaryWriter(TENSORBOARD_PATH_TRAINING)
    writer_validating = SummaryWriter(TENSORBOARD_PATH_VALIDATION)
    
    wandb.init(project="ProbMSSegFranzi", entity="mickchimp", id=MODEL_NAME, name=MODEL_NAME, resume=True, config=args.__dict__)
    wandb.watch(model, log_freq=100)

    pipeline = Pipeline(cmd_args=args, model=model, logger=logger,
                        dir_path=DATASET_FOLDER, checkpoint_path=CHECKPOINT_PATH, 
                        writer_training=writer_training, writer_validating=writer_validating)

    # loading existing checkpoint if supplied
    if bool(LOAD_PATH):
        pipeline.load(checkpoint_path=LOAD_PATH, load_best=args.load_best)

    # try:

    if args.train:
        pipeline.train()
        # pipeline.validate(13,13)
        torch.cuda.empty_cache()  # to avoid memory errors

    if args.test:
        # if args.load_best:
        #     pipeline.load(load_best=True)
        if pipeline.ProbFlag in [1, 2]:
            pipeline.test_prob(test_logger=test_logger)
        else:
            pipeline.test(test_logger=test_logger)
        torch.cuda.empty_cache()  # to avoid memory errors

    if args.predict:
        pipeline.predict(args.predictor_path, args.predictor_label_path, predict_logger=test_logger)


    # except Exception as error:
    #     logger.exception(error)
    writer_training.close()
    writer_validating.close()
