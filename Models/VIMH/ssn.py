import torch.nn as nn
import torch
import torch.distributions as td
import torch.nn.functional as F
# from trainer.distributions import ReshapedDistribution
# source: https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/utils/misc.py
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


# Normal Encoder
class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, dim=2):
        super(_EncoderBlock, self).__init__()
        if dim == 2:
            layers1 = [
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)]

            self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1)
            self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,padding=1)
            layers2 = [
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                layers2.append(nn.Dropout())
            # layers2.append(nn.MaxPool2d(kernel_size=2, stride=2))
        elif dim == 3:
            layers1 = [
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)]

            self.c1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,padding=1)
            self.c2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,padding=1)
            layers2 = [
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                layers2.append(nn.Dropout())
            # layers2.append(nn.MaxPool3d(kernel_size=2, stride=2))
        self.encode1 = nn.Sequential(*layers1)
        self.encode2 = nn.Sequential(*layers2)

    def forward(self, x):
        x = self.c1(x)
        x = self.encode1(x)
        x = self.c2(x)
        return self.encode2(x)

# Normal Decoder
class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dim=2):
        super(_DecoderBlock, self).__init__()
        if dim == 2:
            self.c1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3,padding=1)
            self.decode = nn.Sequential(
                nn.BatchNorm2d(middle_channels),
                nn.ReLU(inplace=True))
            self.c2 = nn.Conv2d(middle_channels, middle_channels, kernel_size=3,padding=1)
            self.decode2 = nn.Sequential(
                nn.BatchNorm2d(middle_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2)
            )
        elif dim == 3:
            self.c1 = nn.Conv3d(in_channels, middle_channels, kernel_size=3,padding=1)
            self.decode = nn.Sequential(
                nn.BatchNorm3d(middle_channels),
                nn.ReLU(inplace=True))
            self.c2 = nn.Conv3d(middle_channels, middle_channels, kernel_size=3,padding=1)
            self.decode2 = nn.Sequential(
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
    def __init__(self, num_classes, num_in=4, dim=2):
        super(UNet, self).__init__()
        n = 4
        self.up_mode = 'trilinear' if dim == 3 else 'bilinear'
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2) if dim == 3 else nn.MaxPool3d(kernel_size=2, stride=2)
        self.enc1 = _EncoderBlock(num_in, 64//n, dim=dim)
        self.enc2 = _EncoderBlock(64//n, 128//n, dim=dim)
        self.enc3 = _EncoderBlock(128//n, 256//n, dim=dim)
        self.enc4 = _EncoderBlock(256//n, 512//n, dim=dim)
        self.center = _DecoderBlock(512//n, 1024//n, 512//n, dim=dim)
        self.dec4 = _DecoderBlock(1024//n, 512//n, 256//n, dim=dim)
        self.dec3 = _DecoderBlock(512//n, 256//n, 128//n, dim=dim)
        self.dec2 = _DecoderBlock(256//n, 128//n, 64//n, dim=dim)
        if dim ==2:

            self.d1 = nn.Conv2d(128 // n, 64 // n, kernel_size=3)
            self.dec1 = nn.Sequential(
                nn.BatchNorm2d(64//n),
                nn.ReLU(inplace=True))
            self.d2 = nn.Conv2d(64//n, 64//n, kernel_size=3)
            self.dec12 = nn.Sequential(
                nn.BatchNorm2d(64//n),
                nn.ReLU(inplace=True),
            )
            self.final = nn.Conv2d(64 // n, num_classes, kernel_size=1)
            self.finalB = nn.BatchNorm2d(num_classes)
            self.softm = nn.Softmax2d()
        elif dim == 3:
            self.d1 = nn.Conv3d(128 // n, 64 // n, kernel_size=3)
            self.dec1 = nn.Sequential(
                nn.BatchNorm3d(64//n),
                nn.ReLU(inplace=True))
            self.d2 = nn.Conv3d(64//n, 64//n, kernel_size=3)
            self.dec12 = nn.Sequential(
                nn.BatchNorm3d(64//n),
                nn.ReLU(inplace=True),
            )
            self.final = nn.Conv3d(64//n, num_classes, kernel_size=1)
            self.finalB = nn.BatchNorm3d(num_classes)
            self.softm = nn.Softmax()

        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        center = self.center(self.pool(enc4))
        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode=self.up_mode)], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode=self.up_mode)], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode=self.up_mode)], 1))
        dec1 = self.d1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode=self.up_mode)], 1))
        dec1 = self.dec1(dec1)
        dec1 = self.d2(dec1)
        dec1 = self.dec12(dec1)
        final = self.final(dec1)
        final = self.finalB(final)
        final = F.interpolate(final, x.size()[2:], mode=self.up_mode)
        return F.softmax(final, dim=1),0

class StochasticUNet(UNet):
    def __init__(self,
                 num_classes,
                 input_channels,
                 dimension=2,
                 rank: int = 10,
                 epsilon=1e-5,
                 diagonal=False):
        super().__init__(num_classes=input_channels, num_in=input_channels,dim=dimension)
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

    def forward(self, image, **kwargs):
        logits = F.relu(super().forward(image, **kwargs)[0])
        batch_size = logits.shape[0]
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
        mask = kwargs['sampling_mask']
        mask = mask.unsqueeze(1).expand((batch_size, self.num_classes) + mask.shape[1:]).reshape(batch_size, -1)
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
        distribution = base_distribution#ReshapedDistribution(base_distribution, event_shape)

        shape = (batch_size,) + event_shape
        logit_mean = mean.view(shape)
        cov_diag_view = cov_diag.view(shape).detach()
        cov_factor_view = cov_factor.transpose(2, 1).view((batch_size, self.num_classes * self.rank) + event_shape[1:]).detach()

        output_dict = {'logit_mean': logit_mean.detach(),
                       'cov_diag': cov_diag_view,
                       'cov_factor': cov_factor_view,
                       'distribution': distribution}

        return logit_mean, output_dict