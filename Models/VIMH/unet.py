# source: https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/models/u_net.py


import torch
import torch.nn.functional as F
from torch import nn
import torch.distributions as td
from typing import Tuple

# source: https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/utils/misc.py
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class ReshapedDistribution(td.Distribution):
    def __init__(self, base_distribution: td.Distribution, new_event_shape: Tuple[int, ...]):
        super().__init__(batch_shape=base_distribution.batch_shape, event_shape=new_event_shape)
        self.base_distribution = base_distribution
        self.new_shape = base_distribution.batch_shape + new_event_shape

    @property
    def support(self):
        return self.base_distribution.support

    # @property
    # def arg_constraints(self):
    #     return self.base_distribution.arg_constraints

    @property
    def mean(self):
        return self.base_distribution.mean.view(self.new_shape)

    @property
    def variance(self):
        return self.base_distribution.variance.view(self.new_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_distribution.rsample(sample_shape).view(sample_shape + self.new_shape)

    def log_prob(self, value):
        return self.base_distribution.log_prob(value.view(self.batch_shape + (-1,)))

    def entropy(self):
        return self.base_distribution.entropy()


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers1 = [
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
        layers2 = [
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers2.append(nn.Dropout())
        layers2.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode1 = nn.Sequential(*layers1)
        self.encode2 = nn.Sequential(*layers2)

    def forward(self, x):
        x = self.c1(x)
        x = self.encode1(x)
        x = self.c2(x)
        return self.encode2(x)

class _EncoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers1 = [
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)]

        self.c1 = nn.Conv3d(in_channels, out_channels, kernel_size=3)
        self.c2 = nn.Conv3d(out_channels, out_channels, kernel_size=3)
        layers2 = [
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers2.append(nn.Dropout())
        layers2.append(nn.MaxPool3d(kernel_size=2, stride=2))
        self.encode1 = nn.Sequential(*layers1)
        self.encode2 = nn.Sequential(*layers2)

    def forward(self, x):
        x = self.c1(x)
        x = self.encode1(x)
        x = self.c2(x)
        return self.encode2(x)

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels,dropout=0.0):
        super(_DecoderBlock, self).__init__()
        self.c1 = nn.Sequential(nn.Conv2d(in_channels, middle_channels, kernel_size=3))
        self.decode = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv2d(middle_channels, middle_channels, kernel_size=3))
        self.decode2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.c1(x)
        x = self.decode(x)
        x = self.c2(x)
        return self.decode2(x)

class _DecoderBlock3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels,dropout=0.0):
        super(_DecoderBlock, self).__init__()
        self.c1 = nn.Sequential(nn.Conv3d(in_channels, middle_channels, kernel_size=3))
        self.decode = nn.Sequential(
            nn.Dropout3d(p=dropout),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True))
        self.c2 = nn.Sequential(nn.Conv3d(middle_channels, middle_channels, kernel_size=3))
        self.decode2 = nn.Sequential(
            nn.Dropout3d(p=dropout),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.c1(x)
        x = self.decode(x)
        x = self.c2(x)
        return self.decode2(x)

class UNet(nn.Module):
    def __init__(self, num_classes, num_in=3):
        super(UNet, self).__init__()
        n = 2
        self.enc1 = _EncoderBlock(num_in, 64//n)
        self.enc2 = _EncoderBlock(64//n, 128//n)
        self.enc3 = _EncoderBlock(128//n, 256//n)
        self.center = _DecoderBlock(256//n, 512//n, 256//n, dropout=0.5)
        self.dec3 = _DecoderBlock(512//n, 256//n, 128//n, dropout=0.5)
        self.dec2 = _DecoderBlock(256//n, 128//n, 64//n, dropout=0.5)
        self.d1 = nn.Conv2d(128//n, 64//n, kernel_size=3)
        self.dec1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.BatchNorm2d(64//n),
            nn.ReLU(inplace=True))
        self.d2 = nn.Conv2d(64//n, 64//n, kernel_size=3)
        self.dec12 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.BatchNorm2d(64//n),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64//n, num_classes, kernel_size=1)
        self.finalB = nn.BatchNorm2d(num_classes)
        self.softm = nn.Softmax2d()
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        center = self.center(enc3)
        dec3 = self.dec3(torch.cat([center, F.upsample(enc3, center.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))

        dec1 = self.d1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(dec1)
        dec1 = self.d2(dec1)
        dec1 = self.dec12(dec1)
        final = self.final(dec1)

        final = self.finalB(final)
        final = F.upsample(final, x.size()[2:], mode='bilinear')
        return F.softmax(final, dim=1)

    def sample_forward(self, x, num_samples, num_classes):
        softmax_result = torch.zeros([ num_samples,x.size(0), num_classes, x.size(2), x.size(3)], device=x.device,dtype=torch.float32)
        for i in range(num_samples):
            softmax_result[i] = self.forward(x)
        return softmax_result.mean(0), softmax_result.std(0)

class UNet3D(nn.Module):
    def __init__(self, num_classes, num_in=3):
        super(UNet, self).__init__()
        n = 2
        self.enc1 = _EncoderBlock3D(num_in, 64//n)
        self.enc2 = _EncoderBlock3D(64//n, 128//n)
        self.enc3 = _EncoderBlock3D(128//n, 256//n)
        self.center = _DecoderBlock3D(256//n, 512//n, 256//n, dropout=0.5)
        self.dec3 = _DecoderBlock3D(512//n, 256//n, 128//n, dropout=0.5)
        self.dec2 = _DecoderBlock3D(256//n, 128//n, 64//n, dropout=0.5)
        self.d1 = nn.Conv3d(128//n, 64//n, kernel_size=3)
        self.dec1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.BatchNorm3d(64//n),
            nn.ReLU(inplace=True))
        self.d2 = nn.Conv3d(64//n, 64//n, kernel_size=3)
        self.dec12 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.BatchNorm3d(64//n),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv3d(64//n, num_classes, kernel_size=1)
        self.finalB = nn.BatchNorm3d(num_classes)
        self.softm = nn.Softmax3d()
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        center = self.center(enc3)
        dec3 = self.dec3(torch.cat([center, F.upsample(enc3, center.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))

        dec1 = self.d1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(dec1)
        dec1 = self.d2(dec1)
        dec1 = self.dec12(dec1)
        final = self.final(dec1)

        final = self.finalB(final)
        final = F.upsample(final, x.size()[2:], mode='bilinear')
        return F.softmax(final, dim=1)

    def sample_forward(self, x, mask, num_samples, num_classes):
        softmax_result = torch.zeros([ num_samples,x.size(0), num_classes, x.size(2), x.size(3)], device=x.device,dtype=torch.float32)
        for i in range(num_samples):
            softmax_result[i] = self.forward(x)
        return softmax_result.mean(0), softmax_result.std(0)

class UNet2(nn.Module):
    def __init__(self, num_classes, num_in=3):
        super(UNet2, self).__init__()
        n = 2
        self.enc1 = _EncoderBlock(num_in, 64//n)
        self.enc2 = _EncoderBlock(64//n, 128//n)
        self.enc3 = _EncoderBlock(128//n, 256//n)
        self.center = _DecoderBlock(256//n, 512//n, 256//n, dropout=0.0)
        self.dec3 = _DecoderBlock(512//n, 256//n, 128//n, dropout=0.0)
        self.dec2 = _DecoderBlock(256//n, 128//n, 64//n, dropout=0.0)
        self.d1 = nn.Conv2d(128//n, 64//n, kernel_size=3)
        self.dec1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.BatchNorm2d(64//n),
            nn.ReLU(inplace=True))
        self.d2 = nn.Conv2d(64//n, 64//n, kernel_size=3)
        self.dec12 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.BatchNorm2d(64//n),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64//n, num_classes, kernel_size=1)
        self.finalB = nn.BatchNorm2d(num_classes)
        self.softm = nn.Softmax2d()
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        center = self.center(enc3)
        dec3 = self.dec3(torch.cat([center, F.upsample(enc3, center.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.d1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(dec1)
        dec1 = self.d2(dec1)
        dec1 = self.dec12(dec1)
        return dec1

    def sample_forward(self, x, mask, num_samples, num_classes):
        softmax_result = torch.zeros([num_samples, x.size(0), num_classes, x.size(2), x.size(3)], device=x.device,dtype=torch.float32)
        for i in range(num_samples):
            softmax_result[i] = self.forward(x)
        return softmax_result, softmax_result.std(0)

class UNetClassic(nn.Module):
    def __init__(self, num_classes, num_in=3, p=0):
        super(UNetClassic, self).__init__()
        n = 1
        self.enc1 = _EncoderBlock(num_in, 64//n)
        self.enc2 = _EncoderBlock(64//n, 128//n)
        self.enc3 = _EncoderBlock(128//n, 256//n)
        self.enc4 = _EncoderBlock(256//n, 512//n, dropout=True)
        self.center = _DecoderBlock(512//n, 1024//n, 512//n, dropout=p)
        self.dec4 = _DecoderBlock(1024//n, 512//n, 256//n, dropout=p)
        self.dec3 = _DecoderBlock(512//n, 256//n, 128//n, dropout=p)
        self.dec2 = _DecoderBlock(256//n, 128//n, 64//n, dropout=p)
        self.d1 = nn.Conv2d(128//n, 64//n, kernel_size=3)
        self.dec1 = nn.Sequential(
            nn.Dropout(p=p),
            nn.BatchNorm2d(64//n),
            nn.ReLU(inplace=True))
        self.d2 = nn.Conv2d(64//n, 64//n, kernel_size=3)
        self.dec12 = nn.Sequential(
            nn.Dropout(p=p),
            nn.BatchNorm2d(64//n),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64//n, num_classes, kernel_size=1)
        self.finalB = nn.BatchNorm2d(num_classes)
        self.softm = nn.Softmax2d()
        initialize_weights(self)

    def forward(self, x, num_samples=1, num_classes=4):
        softmax_result = torch.zeros([x.size(0), num_classes, x.size(2), x.size(3)], device=x.device,
                                     dtype=torch.float32)
        for i in range(num_samples):
            enc1 = self.enc1(x)
            enc2 = self.enc2(enc1)
            enc3 = self.enc3(enc2)
            enc4 = self.enc4(enc3)
            center = self.center(enc4)
            dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
            dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
            dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
            dec1 = self.d1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
            dec1 = self.dec1(dec1)
            dec1 = self.d2(dec1)
            dec1 = self.dec12(dec1)
            final = self.finalB(dec1)
            final = F.upsample(final, x.size()[2:], mode='bilinear')
            softmax_result += F.softmax(final, dim=1)
        return softmax_result / num_samples

class StochasticUNet2(UNet2):
    def __init__(self,
                 num_classes,
                 input_channels,
                 dimension=2,
                 rank: int = 10,
                 epsilon=1e-5,
                 diagonal=False):
        super().__init__(num_classes=input_channels, num_in=input_channels)
        self.dim = dimension
        conv_fn = nn.Conv3d if self.dim == 3 else nn.Conv2d
        self.rank = rank
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.diagonal = diagonal  # whether to use only the diagonal (independent normals)
        features =32
        self.mean_l = conv_fn(features, num_classes, kernel_size=(1, ) * self.dim)
        self.log_cov_diag_l = conv_fn(features, num_classes, kernel_size=(1, ) * self.dim)
        self.cov_factor_l = conv_fn(features, num_classes * rank, kernel_size=(1, ) * self.dim)

    @staticmethod
    def fixed_re_parametrization_trick(dist, num_samples):
        assert num_samples % 2 == 0
        samples = dist.rsample((num_samples // 2,))
        mean = dist.mean.unsqueeze(0)
        samples = samples - mean
        return torch.cat([samples, -samples]) + mean

    def forward(self, image, mask):
        logits = F.upsample(F.relu(super().forward(image)),image.size()[2:], mode='bilinear')
        batch_size = image.shape[0]
        event_shape = (self.num_classes,) + logits.shape[2:]
        mean = self.mean_l(logits)
        cov_diag = self.log_cov_diag_l(logits).exp() + self.epsilon
        mean = mean.view((batch_size, -1))
        cov_diag = cov_diag.view((batch_size, -1))

        cov_factor = self.cov_factor_l(logits)
        cov_factor = cov_factor.view((batch_size, self.rank, self.num_classes, -1))
        cov_factor = cov_factor.flatten(2, 3)
        cov_factor = cov_factor.transpose(1, 2)

        # covariance in the background tens to blow up to infinity, hence set to 0 outside the ROI
        mask = image.sum(1) > 0
        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1).float()
        cov_factor = cov_factor * mask.unsqueeze(-1)
        cov_diag = cov_diag * mask + self.epsilon

        if self.diagonal:
            base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        else:
            try:
                base_distribution = td.LowRankMultivariateNormal(loc=mean, cov_factor=cov_factor, cov_diag=cov_diag)
            except:
                print('Covariance became not invertible using independent normals for this batch!')
                base_distribution = td.Independent(td.Normal(loc=mean, scale=torch.sqrt(cov_diag)), 1)
        distribution = ReshapedDistribution(base_distribution, event_shape)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        cov_diag_view = cov_diag.view(shape).detach()
        cov_factor_view = cov_factor.transpose(2, 1).view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return F.softmax(logit_mean, dim=1), output_dict

    def sample_forward(self, x, mask, num_samples=None, num_classes=None):
        return self.forward(x, mask)



