import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# to stablize the gradient of the sqrt
class StableSqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        result = input.sqrt()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result = ctx.saved_tensors[0]
        grad = grad_output / (2.0 * result)
        grad[result == 0] = 0
        return grad

def sample_weights(W_mu, b_mu, W_p, b_p):
    """Quick method for sampling weights and exporting weights"""
    eps_W = W_mu.data.new(W_mu.size()).normal_()
    # sample parameters
    std_w = 1e-6 + F.softplus(W_p, beta=1, threshold=20)
    W = W_mu + 1 * std_w * eps_W
    if b_mu is not None:
        std_b = 1e-6 + F.softplus(b_p, beta=1, threshold=20)
        eps_b = b_mu.data.new(b_mu.size()).normal_()
        b = b_mu + 1 * std_b * eps_b
    else:
        b = None
    return W, b

def KLD_cost(mu_p, sig_p, mu_q, sig_q):
    #Compute the KLD for all parameters in the layer. Use the multinomal version
    #https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    return  0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q/sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()

def reverse_softplus(p):
    return math.log(math.exp(p)- 1)

class BayesLinear_local_reparam(nn.Module):
    """Linear Layer where activations are sampled from a fully factorised normal which is given by aggregating
     the moments of each weight's normal distribution. The KL divergence is obtained in closed form. Only works
      with gaussian priors.
    """

    def __init__(self, n_in, n_out, prior_sig):
        super(BayesLinear_local_reparam, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior_sig = prior_sig

        # Learnable parameters
        self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).fill_(0))
        self.W_p = nn.Parameter(torch.Tensor(self.n_in, self.n_out).fill_(reverse_softplus(prior_sig)))

        self.b_mu = nn.Parameter(torch.Tensor(self.n_out).normal_(0,1))
        self.b_p = nn.Parameter(torch.Tensor(self.n_out).fill_(reverse_softplus(prior_sig)))

    def forward(self, X, sample, return_KL=True):
        if not sample:  # This is just a placeholder function
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_out)
            return output, KLD_cost(mu_p=0, sig_p=self.prior_sig,
                                    mu_q=self.W_mu, sig_q=torch.tensor([self.prior_sig], device=self.W_mu.device))
        else:
            # calculate std
            std_w = 1e-10 + F.softplus(self.W_p, beta=1, threshold=20)
            std_b = 1e-10 + F.softplus(self.b_p, beta=1, threshold=20)

            act_W_mu = torch.mm(X, self.W_mu)  # self.W_mu + std_w * eps_W

            act_W_std = torch.mm(X.pow(2), std_w.pow(2))

            act_W_std = StableSqrt.apply(act_W_std)

            # Tensor.new()  Constructs a new tensor of the same data type as self tensor.
            # the same random sample is used for every element in the minibatch output
            eps_W = self.W_mu.data.new(act_W_std.size()).normal_(mean=0, std=1)
            eps_b = self.b_mu.data.new(std_b.size()).normal_(mean=0, std=1)

            act_W_out = act_W_mu + act_W_std * eps_W  # (batch_size, n_output)

            act_b_out = self.b_mu + std_b * eps_b

            output = act_W_out + act_b_out.unsqueeze(0).expand(X.shape[0], -1)

            if return_KL:
                kld = KLD_cost(mu_p=torch.zeros(1), sig_p=self.prior_sig, mu_q=self.W_mu, sig_q=std_w) \
                      + KLD_cost(mu_p=0, sig_p=0.1, mu_q=self.b_mu, sig_q=std_b)
                return output, kld
            return output, 0

class BayesConv_local_reparam(nn.Module):
    """Conv Layer where activations are sampled from a fully factorised normal which is given by aggregating
     the moments of each weight's normal distribution. The KL divergence is obtained in closed form. Only works
      with gaussian priors.
    """
    def __init__(self, in_channels, out_channels, kernel_size, prior_sig, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__()
        self.n_in = in_channels
        self.n_out = out_channels
        self.prior_sig = prior_sig
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        self.groups = groups
        self.padding_mode = padding_mode

        # Learnable parameters
        self.W_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size, kernel_size).fill_(0.))
        self.W_p = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size, kernel_size).fill_(reverse_softplus(prior_sig)))

        self.W_mu_prior = torch.Tensor(1).fill_(0.0)#.cuda()
        self.W_p_prior = torch.Tensor(1).fill_(reverse_softplus(prior_sig))#.cuda()


    def forward(self, X, sample, returnKL=True):
        if not sample:  # This is just a placeholder function
            output = F.conv3d(X, self.W_mu, stride=(self.stride,), padding=(self.padding,), dilation=(self.dilation,),
                              groups=self.groups)
            return output, KLD_cost(mu_p=0, sig_p=self.prior_sig.to(self.W_mu.device),
                                    mu_q=self.W_mu, sig_q=torch.tensor([self.prior_sig], device=self.W_mu.device))
        else:

            # calculate std may not need eps stablization
            std_w = 1e-10 + F.softplus(self.W_p, beta=1, threshold=20)
            std_w_prior = 1e-10 + F.softplus(self.W_p_prior, beta=1, threshold=20)

            act_W_mu = F.conv3d(X, self.W_mu, stride=(self.stride,), padding=(self.padding,), dilation=(self.dilation,), groups=self.groups)# self.W_mu + std_w * eps_W

            act_W_std = F.conv3d(X.pow(2), std_w.pow(2), stride=(self.stride,), padding=(self.padding,), dilation=(self.dilation,), groups=self.groups)

            act_W_std = StableSqrt.apply(act_W_std)

            # Tensor.new()  Constructs a new tensor of the same data type as self tensor.
            # the same random sample is used for every element in the minibatch output
            eps_W = torch.empty(act_W_std.size(),device=X.device).normal_(mean=0, std=1)

            act_W_out = act_W_mu + act_W_std * eps_W  # (batch_size, n_output)
            if torch.isnan(act_W_out).any():
                o = 2
            output = act_W_out
            if torch.isnan(output).any():
                o = 2
            if returnKL:
                kld = KLD_cost(mu_p=self.W_mu_prior.to(self.W_mu.device), sig_p=std_w_prior.to(self.W_mu.device),
                               mu_q=self.W_mu, sig_q=std_w)
                return output, kld
            return output, 0

    def update_priors(self, uniformly=False):
        if uniformly:
            self.W_mu_prior = self.W_mu.detach().clone().mean()
            self.W_p_prior = self.W_p.detach().clone().mean()
        else:
            self.W_mu_prior = self.W_mu.detach().clone()
            self.W_p_prior = self.W_p.detach().clone()