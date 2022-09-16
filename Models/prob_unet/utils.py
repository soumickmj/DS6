import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def save_mask_prediction_example(mask, pred, iter):
	plt.imshow(pred[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_prediction.png")
	plt.imshow(mask[0,:,:],cmap='Greys')
	plt.savefig('images/'+str(iter)+"_mask.png")

def ce_loss(labels, logits, n_classes, loss_mask=None, one_hot_labels=True):
    """
    Cross-entropy loss.
    :param labels: 4D tensor
    :param logits: 4D tensor
    :param n_classes: integer for number of classes
    :param loss_mask: binary 4D tensor, pixels to mask should be marked by 1s
    :param data_format: string
    :param one_hot_labels: bool, indicator for whether labels are to be expected in one-hot representation
    :param name: string
    :return: dict of (pixel-wise) mean and sum of cross-entropy loss
    """
    # permute class channels into last axis
    labels = labels.permute([0,2,3,4,1]).to(torch.int64)
    logits = logits.permute([0,2,3,4,1])

    batch_size = float(labels.shape[0])

    if n_classes==1 or not one_hot_labels:
        flat_labels = labels.reshape([-1])
    else:
        _, flat_labels = labels.max(dim=1)
    flat_logits = logits.reshape([-1, n_classes]) + 1e-11

    # do not compute gradients wrt the labels
    flat_labels.requires_grad = False

    ce_per_pixel = torch.nn.functional.cross_entropy(flat_logits, flat_labels)

    # optional element-wise masking with binary loss mask
    if loss_mask is None:
        ce_sum = torch.sum(ce_per_pixel) / batch_size
        ce_mean = torch.mean(ce_per_pixel)
    else:
        loss_mask_flat = loss_mask.reshape([-1,]).float()
        loss_mask_flat = (1. - loss_mask_flat)
        ce_sum = torch.sum(loss_mask_flat * ce_per_pixel) / batch_size
        n_valid_pixels = torch.sum(loss_mask_flat)
        ce_mean = torch.sum(loss_mask_flat * ce_per_pixel) / n_valid_pixels

    return {'sum': ce_sum, 'mean': ce_mean}
