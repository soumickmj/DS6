#!/usr/bin/env python
'''

Purpose : model selector

'''
from Models.attentionunet3d import AttU_Net
from Models.unet3d import U_Net, U_Net_DeepSup

__author__ = "Kartik Prabhu, Mahantesh Pattadkal, and Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Kartik Prabhu", "Mahantesh Pattadkal", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

MODEL_UNET = 1
MODEL_UNET_DEEPSUP = 2
MODEL_ATTENTION_UNET = 3

def getModel(model_no):
    defaultModel = U_Net() #Default
    model_list = {
        1: U_Net(),
        2: U_Net_DeepSup(),
        3: AttU_Net()
    }
    return model_list.get(model_no, defaultModel)
