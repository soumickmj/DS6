#!/usr/bin/env python
"""

"""

import argparse
import os
import apex
import torch.utils.data
from apex import amp
from torch.utils.tensorboard import SummaryWriter

from crossvalidation import FoldManager
from pipeline import Pipeline
from Utils.logger import Logger
from Utils.model_manager import getModel
from Utils.vessel_utils import load_model, load_model_with_amp, load_model_huggingface

__author__ = "Kartik Prabhu, Mahantesh Pattadkal, and Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Kartik Prabhu", "Mahantesh Pattadkal", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-model",
                        type=int,
                        default=1,
                        help="1{U-Net}; \n"
                             "2{U-Net_Deepsup}; \n"
                             "3{Attention-U-Net};")
    parser.add_argument("-model_name",
                        default="Model_v1",
                        help="Name of the model")
    parser.add_argument("-dataset_path",
                        help="Path to folder containing dataset."
                             "Further divide folders into train,validate,test, train_label,validate_label and test_label."
                             "Example: /home/dataset/")
    parser.add_argument("-output_path",
                        help="Folder path to store output "
                             "Example: /home/output/")

    parser.add_argument('-train',
                        default=True,
                        help="To train the model")
    parser.add_argument('-test',
                        default=True,
                        help="To test the model")
    parser.add_argument('-predict',
                        default=False,
                        help="To predict a segmentation output of the model and to get a diff between label and output")
    parser.add_argument('-predictor_path',
                        default="",
                        help="Path to the input image to predict an output, ex:/home/test/ww25.nii ")
    parser.add_argument('-predictor_label_path',
                        default="",
                        help="Path to the label image to find the diff between label an output, ex:/home/test/ww25_label.nii ")

    parser.add_argument('-load_huggingface',
                        default="",
                        help="Load model from huggingface model hub ex: 'soumickmj/DS6_UNetMSS3D_wDeform' [model param will be ignored]")
    
    parser.add_argument('-load_path',
                        default="",
                        help="Path to checkpoint of existing model to load, ex:/home/model/checkpoint/ [If this is supplied, load_huggingface will be ignored] ")
    parser.add_argument('-load_best',
                        default=False,
                        help="Specifiy whether to load the best checkpoiont or the last [Only if load_path is supplied]")
    
    parser.add_argument('-deform',
                        default=False,
                        help="To use deformation for training")
    parser.add_argument('-apex',
                        default=True,
                        help="To use half precision on model weights.")

    parser.add_argument("-batch_size",
                        type=int,
                        default=20,
                        help="Batch size for training")
    parser.add_argument("-num_epochs",
                        type=int,
                        default=50,
                        help="Number of epochs for training")
    parser.add_argument("-learning_rate",
                        type=float,
                        default=0.01,
                        help="Learning rate")
    parser.add_argument("-patch_size",
                        type=int,
                        default=64,
                        help="Patch size of the input volume")
    parser.add_argument("-stride_depth",
                        type=int,
                        default=16,
                        help="Strides for dividing the input volume into patches in depth dimension")
    parser.add_argument("-stride_width",
                        type=int,
                        default=32,
                        help="Strides for dividing the input volume into patches in width dimension")
    parser.add_argument("-stride_length",
                        type=int,
                        default=32,
                        help="Strides for dividing the input volume into patches in length dimension")
    parser.add_argument("-samples_per_epoch",
                        type=int,
                        default=8000,
                        help="Number of samples per epoch")
    parser.add_argument("-num_worker",
                        type=int,
                        default=8,
                        help="Number of worker threads")

    parser.add_argument("-set_number",
                        default=6,
                        help="Set number to select the folds")

    args = parser.parse_args()

    MODEL_NAME = args.model_name
    DATASET_FOLDER = args.dataset_path
    OUTPUT_PATH = args.output_path
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    LOAD_PATH = args.load_path
    SET_NUMBER = args.set_number

    if args.deform:
        old_model_names = ["model1", "model2", "model3"]  #TODO: add previously pretrained model names
    else:
        old_model_names = ["set" + str(SET_NUMBER) + "_fold1", "set" + str(SET_NUMBER) + "_fold2",
                           "set" + str(SET_NUMBER) + "_fold3"]

    for training_set, validation_set, test_set, old_model_name in zip(FoldManager.getTrainingFolds(SET_NUMBER),
                                                                      FoldManager.getValidationFolds(SET_NUMBER),
                                                                      FoldManager.getTestingFolds(SET_NUMBER),
                                                                      old_model_names):

        if args.deform:
            NEW_MODEL_NAME = old_model_name + '_deformation'  # TODO: change this depending on best deformation model acheived
        else:
            NEW_MODEL_NAME = MODEL_NAME + old_model_name

        CHECKPOINT_PATH = OUTPUT_PATH + "/" + MODEL_NAME + '/checkpoint/'
        TENSORBOARD_PATH_TRAINING = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_training/'
        TENSORBOARD_PATH_VALIDATION = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_validation/'
        TENSORBOARD_PATH_TESTING = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_testing/'

        LOGGER_PATH = OUTPUT_PATH + "/" + MODEL_NAME + '.log'
        logger = Logger(MODEL_NAME, LOGGER_PATH).get_logger()
        test_logger = Logger(MODEL_NAME + '_test', LOGGER_PATH).get_logger()

        # Model
        if args.load_huggingface:
            model = load_model_huggingface(args.load_huggingface)
        else:
            model = getModel(args.model)
        model.cuda()

        # No loading
        if not bool(LOAD_PATH):
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            if args.apex:
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        else:
            if args.apex:
                model, optimizer, amp = load_model_with_amp(model, LOAD_PATH)
            else:
                model, optimizer = load_model(model, LOAD_PATH)

        writer_training = SummaryWriter(TENSORBOARD_PATH_TRAINING)
        writer_validating = SummaryWriter(TENSORBOARD_PATH_VALIDATION)
    
        pipeline = Pipeline(model=model, optimizer=optimizer, logger=logger, with_apex=args.apex, num_epochs=args.num_epochs,
                        dir_path=DATASET_FOLDER, checkpoint_path=CHECKPOINT_PATH, deform=args.deform,
                        writer_training=writer_training, writer_validating=writer_validating,
                        stride_depth=args.stride_depth, stride_length=args.stride_length, stride_width=args.stride_width,
                        training_set=training_set, validation_set=validation_set, test_set=test_set,
                        predict_only=(not args.train) and (not args.test))    
        try:
            if args.train:
                pipeline.train()
                torch.cuda.empty_cache()  # to avoid memory errors

            if args.test:
                pipeline.test(test_logger=test_logger)
                torch.cuda.empty_cache()  # to avoid memory errors

            if args.predict:
                pipeline.predict(MODEL_NAME, args.predictor_path, args.predictor_label_path, OUTPUT_PATH)
        except Exception as error:
            logger.exception(error)
        writer_training.close()
        writer_validating.close()
