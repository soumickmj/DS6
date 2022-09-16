"""
    Modification of fid.py that computes trace of matrix square root fast. 
"""

import torch 
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

from Utils.fid.fast_sqrt import trace_of_matrix_sqrt

from Utils.fid.fid import * 

from warnings import filterwarnings # nn.Upscale throws a legacy warning 

def calculate_activation_statistics(images1, images2, batch_size, gradient_checkpointing):
    if batch_size != images1.shape[0]: 
        batch_size = int(batch_size.detach().cpu().numpy()[0])  # this has gradient require_true due to gradient checkpointing hack.
        num_images = images1.shape[0]
        assert num_images % batch_size == 0, "Please choose batch_size to divide number of images."
        n_batches = int(num_images  / batch_size)
        act1 = torch.zeros((num_images, 2048))

        for batch_idx in range(n_batches):
            start_idx   = batch_size * batch_idx
            end_idx     = batch_size * (batch_idx + 1)

            if gradient_checkpointing: 
                act1[start_idx:end_idx, :] = torch.utils.checkpoint.checkpoint(inception_network, images1[start_idx:end_idx])
            else: 
                act1[start_idx:end_idx, :] = inception_network(images1[start_idx:end_idx])

        act1 = act1.t()

    else: 
        if gradient_checkpointing: 
            act1 = torch.utils.checkpoint.checkpoint(inception_network, images1).t()
        else: 
            act1 = inception_network(images1).t()

    with torch.no_grad(): 
        act2 = inception_network(images2).t() # This is real data, no grad.

    d, bs = act1.shape

    # compute the covariance matrices represented implicitly by S1 and S2.  
    # when training we can even use pre-computed image statistics! 
    # these are way smaller than real images! 

    all_ones = torch.ones((1, bs), device="cuda")

    mu1      = torch.mean(act1, axis=1).view(d, 1)
    S1       = np.sqrt(1/(bs-1)) * (act1 - mu1 @ all_ones)


    mu2      = torch.mean(act2, axis=1).view(d, 1)
    S2       = np.sqrt(1/(bs-1)) * (act2 - mu2 @ all_ones)

    return mu1, S1, mu2, S2


def fastfid(images1, images2, batch_size=-1, preprocess=True, gradient_checkpointing=False):
    if batch_size == -1: batch_size = images1.shape[0]

    if preprocess: 
        # Documentation of cv2.resize(..) says it uses bilinear. 
        # Using other mode cause computed fid to be apart. 
        # Supress the following warning: 
        #   ... /lib/python3.7/site-packages/torch/nn/functional.py:2494: 
        #   UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False 
        #   since 0.4.0. Please specify align_corners=True if the old behavior is desired. 
        #   See the documentation of nn.Upsample for details.
        filterwarnings("ignore") 
        up = nn.Upsample( (299, 299),  mode='bilinear') 
        images1 = up( images1 ) 
        images2 = up( images2 ) 
        filterwarnings("default")

        assert images1[0].shape == (3, 299, 299), images1[0].shape

    mu1, sigma1, mu2, sigma2 = calculate_activation_statistics(images1, images2, batch_size, gradient_checkpointing) 
    fid = calculate_frechet_distance_fast(mu1, sigma1, mu2, sigma2)
    return fid



###  ------------------------------------------------------------------------ ## 

def compute_trace(S):  
    tr = torch.sum(torch.norm(S, dim=0)**2)
    return tr

def calculate_frechet_distance_fast(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    diff = diff[:, 0] # the shape was (d, 1) before. 

    tr_covmean = trace_of_matrix_sqrt(sigma1, sigma2) 

    mu  = diff.dot(diff) 
    tr1 = compute_trace(sigma1) 
    tr2 = compute_trace(sigma2) 


    return mu + tr1 + tr2 - 2 * tr_covmean



#### SLOW METHOD 

def calculate_activation_statistics_normal(images1, images2, batch_size):
    act1 = inception_network(images1)
    act2 = inception_network(images2)

    act1 = act1.detach().cpu().numpy()
    act2 = act2.detach().cpu().numpy()

    mu1     = np.mean(act1, axis=0)
    sigma1  = np.cov(act1, rowvar=False)

    mu2     = np.mean(act2, axis=0)
    sigma2  = np.cov(act2, rowvar=False)

    return mu1, sigma1, mu2, sigma2


def calculate_frechet_distance_normal(mu1, sigma1, mu2, sigma2):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    tr_covmean = np.trace(covmean)


    mu = diff.dot(diff) 
    tr1 = np.trace(sigma1) 
    tr2 = np.trace(sigma2) 

    return mu + tr1 + tr2 - 2 * tr_covmean


