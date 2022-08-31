#!/usr/bin/env python
'''

Purpose : model selector

'''
from ssl import SSLSocket
from Models.attentionunet3d import AttU_Net
from Models.prob_unet.probabilistic_unet import ProbabilisticUnet
from Models.prob_unet2D.probabilistic_unet import ProbabilisticUnet as ProbabilisticUnet2D
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
MODEL_PROBABILISTIC_UNET = 4

def getModel(model_no, is2D=False): #Send model params from outside
    defaultModel = U_Net() #Default
    if is2D:
        model_list = {
            # 1: U_Net(),
            # 2: U_Net_DeepSup(), 
            # 3: AttU_Net(),
            4: ProbabilisticUnet2D(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0)
        }
    else:
        model_list = {
            1: U_Net(),
            2: U_Net_DeepSup(), 
            3: AttU_Net(),
            4: ProbabilisticUnet(num_filters=[32,64,128,192])
            # 4: ProbabilisticUnet(num_filters=[64,128,256,512,1024])
        }
    return model_list.get(model_no, defaultModel)
