#!/usr/bin/env python
"""

"""

import argparse

from Utils.logger import Logger
from Utils.model_manager import getModel
from Utils.vessel_utils import load_model_with_amp, load_model
from pipeline import Pipeline
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import apex
from apex import amp

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

    parser.add_argument('-load_path',
                        default="",
                        help="Path to checkpoint of existing model to load, ex:/home/model/checkpoint/ ")
    parser.add_argument('-deform',
                        default=False,
                        help="To use deformation for training")
    parser.add_argument('-apex',
                        default=True,
                        help="To use half precision on model weights.")

    parser.add_argument("-batch_size",
                        default=20,
                        help="Batch size for training")
    parser.add_argument("-num_epochs",
                        default=50,
                        help="Number of epochs for training")
    parser.add_argument("-learning_rate",
                        default=0.01,
                        help="Learning rate")
    parser.add_argument("-patch_size",
                        default=64,
                        help="Patch size of the input volume")
    parser.add_argument("-stride_depth",
                        default=16,
                        help="Strides for dividing the input volume into patches in depth dimension")
    parser.add_argument("-stride_width",
                        default=32,
                        help="Strides for dividing the input volume into patches in width dimension")
    parser.add_argument("-stride_length",
                        default=32,
                        help="Strides for dividing the input volume into patches in length dimension")
    parser.add_argument("-stride_length",
                        default=32,
                        help="Strides for dividing the input volume into patches in length dimension")
    parser.add_argument("-samples_per_epoch",
                        default=8000,
                        help="Number of samples per epoch")
    parser.add_argument("-num_worker",
                        default=8,
                        help="Number of worker threads")

    args = parser.parse_args()

    MODEL_NAME = args.model_name
    DATASET_FOLDER = args.dataset_path
    OUTPUT_PATH = args.output_path

    LOAD_PATH = args.load_path
    CHECKPOINT_PATH = OUTPUT_PATH + MODEL_NAME + '/checkpoint/'
    TENSORBOARD_PATH_TRAINING = OUTPUT_PATH + MODEL_NAME + '/tensorboard/tensorboard_training/'
    TENSORBOARD_PATH_VALIDATION = OUTPUT_PATH + MODEL_NAME + '/tensorboard/tensorboard_validation/'
    TENSORBOARD_PATH_TESTING = OUTPUT_PATH + MODEL_NAME + '/tensorboard/tensorboard_testing/'

    LOGGER_PATH = OUTPUT_PATH + MODEL_NAME + '.log'

    logger = Logger(MODEL_NAME, OUTPUT_PATH).get_logger()
    test_logger = Logger(MODEL_NAME + '_test', OUTPUT_PATH).get_logger()

    # Model
    model = getModel(args.model)
    model.cuda()

    # No loading
    if bool(LOAD_PATH):
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
    try:

        if args.train:
            pipeline = Pipeline(model=model, optimizer=optimizer, logger=logger, with_apex=args.apex, num_epochs=args.num_epochs,
                            dir_path=DATASET_FOLDER, checkpoint_path=CHECKPOINT_PATH, deform=args.deform,
                            writer_training=writer_training, writer_validating=writer_validating,
                            stride_depth=args.stride_depth, stride_length=args.stride_length, stride_width=args.stride_width)
            pipeline.train()

            del model, pipeline
            torch.cuda.empty_cache()  # to avoid memory errors

        if args.test:
            # Note: The below initialisation is just an example of how PIPELINE can be used for testing, pipeline.test can be even used directly after train.
            pipeline = Pipeline(model=getModel(args.model), optimizer=optimizer, logger=logger, with_apex=args.apex,
                            num_epochs=args.num_epochs, dir_path=DATASET_FOLDER, checkpoint_path=CHECKPOINT_PATH,
                            writer_training=writer_training, writer_validating=writer_validating,
                            stride_depth=args.stride_depth, stride_length=args.stride_length, stride_width=args.stride_width)

            pipeline.test(test_logger=test_logger)

        if args.predict:
            pipeline = Pipeline(model=getModel(args.model), optimizer=optimizer, logger=logger, with_apex=args.apex,
                                num_epochs=args.num_epochs, dir_path=DATASET_FOLDER, checkpoint_path=CHECKPOINT_PATH,
                                writer_training=writer_training, writer_validating=writer_validating,
                                stride_depth=args.stride_depth, stride_length=args.stride_length,
                                stride_width=args.stride_width)
            pipeline.predict(MODEL_NAME, args.predictor_path, args.predictor_label_path, OUTPUT_PATH)


    except Exception as error:
        logger.exception(error)
    writer_training.close()
    writer_validating.close()
