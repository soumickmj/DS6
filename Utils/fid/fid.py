import torch
import torchvision 
from torch import nn
from torchvision.models import inception_v3
# import cv2 #without this preprocessing won't work
import numpy as np
import glob
import os
from scipy import linalg
import time 

class PartialInceptionNetwork(nn.Module):

    def __init__(self, transform_input=True):
        super().__init__()
        self.inception_network = inception_v3(pretrained=True, progress=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
                                             ", but got {}".format(x.shape)
        x = x * 2 -1 # Normalize to [-1, 1]

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1 
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations


# Only load model once. 
t0 = time.time()
print("Loading inception... ", end="", flush=True)
inception_network = PartialInceptionNetwork()
inception_network = inception_network.cuda()
inception_network.eval() 
print("DONE! %.4fs"%(time.time()-t0))


def get_activations(images, batch_size, inception_network):
    num_images = images.shape[0]
    n_batches = int(np.ceil(num_images  / batch_size))
    inception_activations = np.zeros((num_images, 2048), dtype=np.float32)
    for batch_idx in range(n_batches):
        start_idx = batch_size * batch_idx
        end_idx = batch_size * (batch_idx + 1)

        ims = images[start_idx:end_idx]
        ims = ims.cuda()
        activations = inception_network(ims)
        activations = activations.detach().cpu().numpy()
        assert activations.shape == (ims.shape[0], 2048), "Expexted output shape to be: {}, but was: {}".format((ims.shape[0], 2048), activations.shape)
        inception_activations[start_idx:end_idx, :] = activations

    return inception_activations

def calculate_activation_statistics(images1, images2, batch_size):
    act1 = get_activations(images1, batch_size, inception_network)
    act2 = get_activations(images2, batch_size, inception_network)

    mu1  = np.mean(act1, axis=0)
    sigma1 = np.cov(act1, rowvar=False)

    mu2  = np.mean(act2, axis=0)
    sigma2 = np.cov(act2, rowvar=False)

    return mu1, sigma1, mu2, sigma2


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def preprocess_image(im):
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255
    im = cv2.resize(im, (299, 299))
    im = np.rollaxis(im, axis=2)
    im = torch.from_numpy(im)
    return im

def preprocess_images(images):
    final_images = torch.stack([preprocess_image(im) for im in images], dim=0)
    return final_images

def fid(images1, images2, batch_size=32, preprocess=True):
    if preprocess: 
        images1 = preprocess_images(images1)
        images2 = preprocess_images(images2)

    mu1, sigma1, mu2, sigma2 = calculate_activation_statistics(images1, images2, batch_size) 
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid
