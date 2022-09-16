import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.cuda.amp import autocast

from Utils.fid.fast_sqrt import trace_of_matrix_sqrt

class PartialInceptionNetwork(nn.Module):

    def __init__(self, transform_input=True):
        super().__init__()
        self.inception_network = torchvision.models.inception_v3(pretrained=True, progress=True)
        self.inception_network.Mixed_7c.register_forward_hook(self.output_hook)
        self.transform_input = transform_input

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.mixed_7c_output = output

    def forward(self, x):
        # assert x.shape[1:] == (3, 299, 299), "Expected input shape to be: (N,3,299,299)" +\
        #                                      ", but got {}".format(x.shape)
        x = x * 2 -1 # Normalize to [-1, 1]

        if x.shape[1] == 1: #it's single channel input, need to repeat the dim
            x = torch.concat([x,x,x], dim=1)

        # Trigger output hook
        self.inception_network(x)

        # Output: N x 2048 x 1 x 1 
        activations = self.mixed_7c_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations

class PartialResNeXt(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = self.get_model()
        self.net.layer4.register_forward_hook(self.output_hook)

    def get_model(self):
        model = torchvision.models.resnext101_32x8d()
        model.conv1 = nn.Conv2d(1, model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                                stride=model.conv1.stride, padding=model.conv1.padding, bias=False if model.conv1.bias is None else True)
        model.fc = nn.Linear(in_features=model.fc.in_features,
                                out_features=33, bias=False if model.fc.bias is None else True)
        chk = torch.load(r"./Utils/fid/ResNeXt-3-class-best-latest.pth", map_location="cpu")
        model.load_state_dict(chk)
        return model

    def output_hook(self, module, input, output):
        # N x 2048 x 8 x 8
        self.layer4_output = output

    def forward(self, x):
        x = x * 2 -1 # Normalize to [-1, 1]

        # Trigger output hook
        self.net(x)

        # Output: N x 2048 x 1 x 1 
        activations = self.layer4_output
        activations = torch.nn.functional.adaptive_avg_pool2d(activations, (1,1))
        activations = activations.view(x.shape[0], 2048)
        return activations


class FastFID(nn.Module):
    def __init__(self, useInceptionNet=True, batch_size=-1, gradient_checkpointing=False) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.gradient_checkpointing = gradient_checkpointing
        if useInceptionNet:
            self.net = PartialInceptionNetwork()
        else:
            self.net = PartialResNeXt()
        self.net.eval() 
        self.act_size = 2048

    def calculate_activation_statistics(self, input, target, batch_size):
        if batch_size != input.shape[0]:
            # this has gradient require_true due to gradient checkpointing hack.
            # batch_size = int(batch_size.detach().cpu().numpy()[0])
            num_images = target.shape[0]
            assert num_images % batch_size == 0, "Please choose batch_size to divide number of images."
            n_batches = int(num_images / batch_size)
            act1 = torch.zeros((num_images, self.act_size), device=input.device)

            for batch_idx in range(n_batches):
                start_idx = batch_size * batch_idx
                end_idx = batch_size * (batch_idx + 1)

                if self.gradient_checkpointing:
                    act1[start_idx:end_idx, :] = torch.utils.checkpoint.checkpoint(
                        self.net, input[start_idx:end_idx])
                else:
                    act1[start_idx:end_idx, :] = self.net(
                        input[start_idx:end_idx])

            act1 = act1.t()

        else:
            if self.gradient_checkpointing:
                act1 = torch.utils.checkpoint.checkpoint(self.net, input).t()
            else:
                act1 = self.net(input).t()

        with torch.no_grad():
            act2 = self.net(target).t()  # This is real data, no grad.

        d, bs = act1.shape

        # compute the covariance matrices represented implicitly by S1 and S2.
        # when training we can even use pre-computed image statistics!
        # these are way smaller than real images!

        all_ones = torch.ones((1, bs), device=input.device)

        mu1 = torch.mean(act1, axis=1).view(d, 1)
        S1 = np.sqrt(1/(bs-1)) * (act1 - mu1 @ all_ones)

        mu2 = torch.mean(act2, axis=1).view(d, 1)
        S2 = np.sqrt(1/(bs-1)) * (act2 - mu2 @ all_ones)

        return mu1, S1, mu2, S2

    def compute_trace(self, S):  
        tr = torch.sum(torch.norm(S, dim=0)**2)
        return tr

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2):
        diff = mu1 - mu2
        diff = diff[:, 0] # the shape was (d, 1) before. 

        tr_covmean = trace_of_matrix_sqrt(sigma1, sigma2) 

        mu  = diff.dot(diff) 
        tr1 = self.compute_trace(sigma1) 
        tr2 = self.compute_trace(sigma2) 

        return mu + tr1 + tr2 - 2 * tr_covmean
    
    def forward(self, input, target):
        batch_size = input.shape[0] if self.batch_size == - \
            1 else self.batch_size
        mu1, sigma1, mu2, sigma2 = self.calculate_activation_statistics(
            input, target, batch_size)
        fid = self.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid
