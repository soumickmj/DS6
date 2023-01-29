
import torch
import torch.nn as nn
import torch.nn.functional as F

class VIMHLoss(nn.Module):
    def __init__(self, loss_func, NUM_MODELS=4, LAMBDA=0.5, klfactor=1e-8):
        super(VIMHLoss, self).__init__()
        self.loss_func = loss_func
        self.LAMBDA = LAMBDA
        self.klfactor = klfactor
        self.NUM_MODELS = NUM_MODELS

    def forward(self, soft_out, kl, mask, train=True):
        if train:
            # calculate headwise loss
            l_dis = 0.0
            for ens in range(soft_out.size(0)):
                l_dis += (1 - self.LAMBDA) * (self.loss_func(torch.log(soft_out[ens, :, :] + 1e-18), mask) + self.klfactor * kl[
                    ens]) / self.NUM_MODELS

            # make ensemble prediction
            soft = F.softmax(soft_out.sum(0), dim=1)
            l = self.loss_func(torch.log(soft + 1e-18), mask)

            # Sum whole loss and return it, along with the final predictions
            return self.NUM_MODELS * self.LAMBDA * (l + self.klfactor * torch.sum(kl)) + l_dis, torch.argmax(soft, 1)
        else:
            soft_out = soft_out.sum(0) # make ensemble prediction during eval (test or predict)
            if mask is not None:
                l = self.loss_func(torch.log(soft_out / self.NUM_MODELS + 1e-18), mask)
            else:
                l = 0.0           
            return l, torch.argmax(soft_out, 1)