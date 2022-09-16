#!/usr/bin/env python
'''

Purpose : model selector

'''
from ssl import SSLSocket
import torch.nn as nn
from Models.ProbUNetV2.model import InjectionConvEncoder2D, InjectionUNet2D, ProbabilisticSegmentationNet
from Models.attentionunet3d import AttU_Net
from Models.attentionunet2d import AttU_Net as AttU_Net2D
from Models.prob_unet.probabilistic_unet import ProbabilisticUnet
from Models.prob_unet2D.probabilistic_unet import ProbabilisticUnet as ProbabilisticUnet2D
from Models.unet3d import U_Net, U_Net_DeepSup
from Models.unet2d import U_Net as U_Net2D, U_Net_DeepSup as U_Net_DeepSup2D


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
            1: U_Net2D(),
            2: U_Net_DeepSup2D(), 
            3: AttU_Net2D(),
            4: ProbabilisticUnet2D(input_channels=1, num_classes=1, num_filters=[32,64,128,192], latent_dim=2, no_convs_fcomb=4, beta=10.0),
            5: ProbabilisticSegmentationNet(in_channels=1, out_channels=1, 
                                            task_op=InjectionUNet2D,
                                            task_kwargs={"output_activation_op": nn.Sigmoid, "activation_kwargs": {"inplace": True}}, 
                                            prior_op=InjectionConvEncoder2D,
                                            prior_kwargs={"activation_kwargs": {"inplace": True}, "norm_depth": 2}, 
                                            posterior_op=InjectionConvEncoder2D,
                                            posterior_kwargs={"activation_kwargs": {"inplace": True}, "norm_depth": 2},
                                            ) 
        }
    else:
        model_list = {
            1: U_Net(),
            2: U_Net_DeepSup(), 
            3: AttU_Net(),
            4: ProbabilisticUnet(num_filters=[32,64,128,192]),
            # 4: ProbabilisticUnet(num_filters=[64,128,256,512,1024]),
            5: ProbabilisticSegmentationNet(in_channels=1, out_channels=1, 
                                            task_kwargs={"output_activation_op": nn.Sigmoid, "activation_kwargs": {"inplace": True}}, 
                                            prior_kwargs={"activation_kwargs": {"inplace": True}, "norm_depth": 2}, 
                                            posterior_kwargs={"activation_kwargs": {"inplace": True}, "norm_depth": 2},
                                            ) 
        }
    model = model_list.get(model_no, defaultModel)
    
    if model_no == 5:
        model.init_weights(*[nn.init.kaiming_uniform_, 0])
        model.init_bias(*[nn.init.constant_, 0])
        
    return model
