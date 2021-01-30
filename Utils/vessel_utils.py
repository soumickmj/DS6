#!/usr/bin/env python
# from __future__ import print_function, division
'''

Purpose : 

'''


import os
from apex import amp

import torch
import torch.utils.data
import torchvision.transforms as transforms
from PIL import TiffImagePlugin
from skimage.filters import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt

# import GPUtil as GPU
# import psutil
# import humanize

__author__ = "Kartik Prabhu, Mahantesh Pattadkal, and Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Kartik Prabhu", "Mahantesh Pattadkal", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

def write_summary(writer, logger, index, original, reconstructed, focalTverskyLoss, diceLoss, diceScore, iou):
    """
    Method to write summary to the tensorboard.
    index: global_index for the visualisation
    original,reconstructer: image input of dimenstion [channel, Height, Width]
    Losses: all losses used as metric
    """
    print('Writing Summary...')
    writer.add_scalar('FocalTverskyLoss', focalTverskyLoss, index)
    writer.add_scalar('DiceLoss', diceLoss, index)
    writer.add_scalar('DiceScore', diceScore, index)
    writer.add_scalar('IOU', iou, index)

    writer.add_image('original', original.cpu().data.numpy()[None,:],index)
    writer.add_image('reconstructed', reconstructed.cpu().data.numpy()[None,:], index)
    writer.add_image('diff', np.moveaxis(create_diff_mask(reconstructed,original,logger), -1, 0), index) #create_diff_mask is of the format HXWXC, but CXHXW is needed

def save_model(CHECKPOINT_PATH, state, best_metric = False,filename='checkpoint'):
    """
    Method to save model
    """
    print('Saving model...')
    if not os.path.exists(CHECKPOINT_PATH):
        os.mkdir(CHECKPOINT_PATH)
        if best_metric:
            if not os.path.exists(CHECKPOINT_PATH + 'best_metric/'):
                CHECKPOINT_PATH = CHECKPOINT_PATH + 'best_metric/'
                os.mkdir(CHECKPOINT_PATH)
    torch.save(state, CHECKPOINT_PATH + filename + str(state['epoch']) + '.pth')


def load_model(model, CHECKPOINT_PATH, batch_index='best', learning_rate = 0.01, filename='checkpoint'):
    """
    Method to load model, make sure to set the model to eval, use optimiser if want to continue training
    """
    print('Loading model...')
    checkpoint = torch.load(CHECKPOINT_PATH + filename + str(batch_index) + '.pth')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    model.eval()
    return model, optimizer


def load_model_with_amp(model, CHECKPOINT_PATH, batch_index='best', learning_rate = 0.01, filename='checkpoint'):
    """
    Method to load model, make sure to set the model to eval, use optimiser if want to continue training
    opt_level="O1"
    """
    print('Loading model...')
    model.cuda()
    try: #TODO dirty fix for now
        checkpoint = torch.load(CHECKPOINT_PATH + filename + str(batch_index) + '.pth')
    except:
        checkpoint = torch.load(CHECKPOINT_PATH + filename + str(batch_index) + '50.pth')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])
    model.eval()
    return model, optimizer, amp


def load_model_with_amp_v2(model, CHECKPOINT_PATH, batch_index='best', learning_rate = 0.01, filename='checkpoint'):
    """
    Method to load model, make sure to set the model to eval, use optimiser if want to continue training
    opt_level="O2"
    """
    print('Loading model...')
    model.cuda()
    checkpoint = torch.load(CHECKPOINT_PATH + filename + str(batch_index) + '.pth')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    amp.load_state_dict(checkpoint['amp'])
    model.eval()
    return model, optimizer, amp


def convert_and_save_tif(image3D, output_path, filename='output.tif', isColored=True):
    """
    Method to convert 3D tensor to tiff image
    """
    image_list = []
    num = 3# if isColored else 1
    for i in range(0, int(image3D.shape[0] / num)):
        index = i * num
        tensor_image = image3D[index:(index + num), :, :]
        image = transforms.ToPILImage(mode='RGB')(tensor_image)
        image_list.append(image)

    print('convert_and_save_tif:size of image:'+ str(len(image_list)))
    with TiffImagePlugin.AppendingTiffWriter(output_path + filename, True) as tifWriter:
        for im in image_list:
            # with open(DATASET_FOLDER+tiff_in) as tiff_in:
            im.save(tifWriter)
            tifWriter.newFrame()
    print("Conversion to tiff completed, image saved as {}".format(filename))

def convert_and_save_tif_greyscale(image3D, output_path, filename='output.tif'):
    """
    Method to convert 3D tensor to tiff image
    """
    image_list = []

    for i in range(0, int(image3D.shape[0])):
        tensor_image = image3D[i]
        image = transforms.ToPILImage(mode='F')(tensor_image)
        image_list.append(image)

    print('convert_and_save_tif:size of image:' + str(len(image_list)))
    with TiffImagePlugin.AppendingTiffWriter(output_path + filename, True) as tifWriter:
        for im in image_list:
            # with open(DATASET_FOLDER+tiff_in) as tiff_in:
            im.save(tifWriter)
            tifWriter.newFrame()
    print("Conversion to tiff completed, image saved as {}".format(filename))


def create_mask(predicted, logger):
    """
    Method find the difference between the 2 images and overlay colors
    predicted, label : slices , 2D tensor
    """
    predicted = predicted.cpu().data.numpy()

    try:
        thresh = threshold_otsu(predicted)
        predicted_binary = predicted > thresh
    except Exception as error:
        logger.exception(error)
        predicted_binary = predicted > 0.5  # exception will be thrown only if input image seems to have just one color 1.0.

    # Define colors
    black = np.array([0, 0, 0], dtype=np.uint8)  # background
    white = np.array([255, 255, 255], dtype=np.uint8)  # prediction_output

    # Make RGB array, pre-filled with black(background)
    rgb_image = np.zeros((predicted_binary.shape[0], predicted_binary.shape[1], 3), dtype=np.uint8) + black

    # Overwrite with red where threshold exceeded, i.e. where mask is True
    rgb_image[predicted_binary] = white

    return rgb_image

def create_diff_mask(predicted, label, logger):
    """
    Method find the difference between the 2 images(predicted being grescale, label being binary) and overlay colors
    predicted, label : slices , 2D tensor
    """
    label = label.cpu().data.numpy()
    predicted = predicted.cpu().data.numpy()

    try:
        thresh = threshold_otsu(predicted)
        predicted_binary = predicted > thresh
    except Exception as error:
        logger.exception(error)
        predicted_binary = predicted > 0.5  # exception will be thrown only if input image seems to have just one color 1.0.

    # fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    # ax = axes.ravel()
    # ax[0] = plt.subplot(1, 3, 1)
    # ax[1] = plt.subplot(1, 3, 2)
    # ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])
    #
    # ax[0].imshow(predicted, cmap=plt.cm.gray)
    # ax[0].set_title('Original')
    # ax[0].axis('off')
    #
    # ax[1].hist(predicted.ravel(), bins=256)
    # ax[1].set_title('Histogram')
    # ax[1].axvline(thresh, color='r')
    #
    # ax[2].imshow(predicted_binary, cmap=plt.cm.gray)
    # ax[2].set_title('Thresholded')
    # ax[2].axis('off')
    #
    # plt.show()

    diff1 = np.subtract(label, predicted_binary) > 0
    diff2 = np.subtract(predicted_binary, label) > 0

    # Define colors
    red = np.array([255, 0, 0], dtype=np.uint8)  # under_detected
    green = np.array([0, 255, 0], dtype=np.uint8)  # over_detected
    black = np.array([0, 0, 0], dtype=np.uint8)  # background
    white = np.array([255, 255, 255], dtype=np.uint8)  # prediction_output
    blue = np.array([0, 0, 255], dtype=np.uint8) # over_detected
    yellow = np.array([255, 255, 0], dtype=np.uint8)  # under_detected

    # Make RGB array, pre-filled with black(background)
    rgb_image = np.zeros((predicted_binary.shape[0], predicted_binary.shape[1], 3), dtype=np.uint8) + black

    # Overwrite with red where threshold exceeded, i.e. where mask is True
    rgb_image[predicted_binary] = white
    rgb_image[diff1] = red
    rgb_image[diff2] = blue

    return rgb_image

def create_diff_mask_binary(predicted, label, logger):
    """
    Method find the difference between the 2 binary images and overlay colors
    predicted, label : slices , 2D tensor
    """
    predicted_label = label.cpu().data.numpy()
    predicted_binary = predicted.cpu().data.numpy()

    diff1 = np.subtract(predicted_label, predicted_binary) > 0
    diff2 = np.subtract(predicted_binary, predicted_label) > 0

    predicted_binary = predicted_binary > 0

    # Define colors
    red = np.array([255, 0, 0], dtype=np.uint8)  # under_detected
    green = np.array([0, 255, 0], dtype=np.uint8)  # over_detected
    black = np.array([0, 0, 0], dtype=np.uint8)  # background
    white = np.array([255, 255, 255], dtype=np.uint8)  # prediction_output
    blue = np.array([0, 0, 255], dtype=np.uint8) # over_detected
    yellow = np.array([255, 255, 0], dtype=np.uint8)  # under_detected

    # Make RGB array, pre-filled with black(background)
    rgb_image = np.zeros((predicted_binary.shape[0], predicted_binary.shape[1], 3), dtype=np.uint8) + black

    # Overwrite with red where threshold exceeded, i.e. where mask is True
    rgb_image[predicted_binary] = white
    rgb_image[diff1] = red
    rgb_image[diff2] = blue
    return rgb_image



def show_diff(label, predicted, diff_image):
    '''
   Method to display the differences between label, predicted and diff_image
   '''
    fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
    ax = axes
    ax[0] = plt.subplot(1, 3, 1)
    ax[1] = plt.subplot(1, 3, 2)
    ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

    ax[0].imshow(label, cmap=plt.cm.gray)
    ax[0].set_title('GroundTruth')
    ax[0].axis('off')

    ax[1].imshow(predicted, cmap=plt.cm.gray)
    ax[1].set_title('Predicted')
    ax[0].axis('off')

    ax[2].imshow(diff_image, cmap=plt.cm.gray)
    ax[2].set_title('Difference image')
    ax[2].axis('off')

    plt.show()
