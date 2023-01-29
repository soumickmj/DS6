import torch
import torch.nn.functional as F
from torch import nn

try:
    from .layers.bayes_Layers import BayesConv_local_reparam
except:
    from Models.VIMH.layers.bayes_Layers import BayesConv_local_reparam


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


# Bayes Encoder
class _EncoderBlockB(nn.Module):
    def __init__(self, in_channels, out_channels, prior, dropout=False):
        super(_EncoderBlockB, self).__init__()
        layers1 = [
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]

        self.c1 = BayesConv_local_reparam(in_channels, out_channels, kernel_size=3, prior_sig=prior)
        self.c2 = BayesConv_local_reparam(out_channels, out_channels, kernel_size=3, prior_sig=prior)
        layers2 = [
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers2.append(nn.Dropout())
        layers2.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode1 = nn.Sequential(*layers1)
        self.encode2 = nn.Sequential(*layers2)

    def forward(self, x, sample, returnKL=True):
        sum_kl = 0
        x, kl = self.c1(x, sample, returnKL)
        sum_kl += kl
        x = self.encode1(x)
        x, kl = self.c2(x, sample, returnKL)
        sum_kl += kl
        return self.encode2(x), sum_kl


# Normal Encoder
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


# Bayes Decoder
class _DecoderBlockB(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, prior):
        super(_DecoderBlockB, self).__init__()
        self.c1 = BayesConv_local_reparam(in_channels, middle_channels, kernel_size=3, prior_sig=prior)
        self.decode = nn.Sequential(
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True))
        self.c2 = BayesConv_local_reparam(middle_channels, middle_channels, kernel_size=3, prior_sig=prior)
        self.decode2 = nn.Sequential(
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x, sample, returnKL=True):
        sum_kl = 0
        x, kl = self.c1(x, sample, returnKL)
        sum_kl += kl

        x = self.decode(x)
        x, kl = self.c2(x, sample, returnKL)
        sum_kl += kl
        return self.decode2(x), sum_kl


# Normal Decoder
class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.c1 = nn.Conv2d(in_channels, middle_channels, kernel_size=3)
        self.decode = nn.Sequential(
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True))
        self.c2 = nn.Conv2d(middle_channels, middle_channels, kernel_size=3)
        self.decode2 = nn.Sequential(
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.c1(x)
        x = self.decode(x)
        x = self.c2(x)
        return self.decode2(x)


class _DecoderBlockD(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=0.0):
        super(_DecoderBlockD, self).__init__()
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


class BUNet(nn.Module):
    def __init__(self, num_classes, num_in=3, prior=1.0):
        super().__init__()
        n = 2
        self.enc1 = _EncoderBlockB(num_in, 64 // n, prior=prior)
        self.enc2 = _EncoderBlockB(64 // n, 128 // n, prior=prior)
        self.enc3 = _EncoderBlockB(128 // n, 256 // n, prior=prior)
        self.center = _DecoderBlockB(256 // n, 512 // n, 256 // n, prior=prior)
        self.dec3 = _DecoderBlockB(512 // n, 256 // n, 128 // n, prior=prior)
        self.dec2 = _DecoderBlockB(256 // n, 128 // n, 64 // n, prior=prior)

        self.d1 = BayesConv_local_reparam(128 // n, 64 // n, kernel_size=3, prior_sig=prior)
        self.dec1 = nn.Sequential(
            nn.BatchNorm2d(64 // n),
            nn.ReLU(inplace=True))
        self.d2 = BayesConv_local_reparam(64 // n, 64 // n, kernel_size=3, prior_sig=prior)
        self.dec12 = nn.Sequential(
            nn.BatchNorm2d(64 // n),
            nn.ReLU(inplace=True),
        )
        self.final = BayesConv_local_reparam(64 // n, num_classes, kernel_size=1, prior_sig=prior)
        self.finalB = nn.BatchNorm2d(num_classes)

        initialize_weights(self)

    def forward(self, x, sample, returnKL=True):
        sum_kl = 0
        enc1, kl = self.enc1(x, sample, returnKL)
        sum_kl += kl
        enc2, kl = self.enc2(enc1, sample, returnKL)
        sum_kl += kl
        enc3, kl = self.enc3(enc2, sample, returnKL)
        sum_kl += kl
        center, kl = self.center(enc3, sample, returnKL)
        sum_kl += kl
        dec3, kl = self.dec3(torch.cat([center, F.upsample(enc3, center.size()[2:], mode='bilinear')], 1), sample,
                             returnKL)
        sum_kl += kl
        dec2, kl = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1), sample, returnKL)
        sum_kl += kl
        dec1, kl = self.d1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1), sample, returnKL)
        sum_kl += kl
        dec1 = self.dec1(dec1)
        dec1, kl = self.d2(dec1, sample, returnKL)
        sum_kl += kl
        dec1 = self.dec12(dec1)
        final, kl = self.final(dec1, sample, returnKL)
        sum_kl += kl
        final = self.finalB(final)
        return F.upsample(final, x.size()[2:], mode='bilinear'), sum_kl

    def sample_forward(self, x, num_samples, num_classes):
        softmax_result = torch.zeros([x.size(0), num_classes, x.size(2), x.size(3)], device=x.device,
                                     dtype=torch.float32)
        cum_kl = 0.
        for i in range(num_samples):
            tmp_result, tmp_kl = self.forward(x, True)

            softmax_result += F.softmax(tmp_result, 1, )
            # print(softmax_result[0, :, 0, 0])
            # print(softmax_result.sum(0).sum(1).sum(1),i)
            cum_kl += tmp_kl
        return softmax_result / num_samples, cum_kl / num_samples


class UNet(nn.Module):
    def __init__(self, num_classes, num_in=3):
        super(UNet, self).__init__()
        n = 2
        self.enc1 = _EncoderBlock(num_in, 64 // n)
        self.enc2 = _EncoderBlock(64 // n, 128 // n)
        self.enc3 = _EncoderBlock(128 // n, 256 // n)
        self.center = _DecoderBlock(256 // n, 512 // n, 256 // n)
        self.dec3 = _DecoderBlock(512 // n, 256 // n, 128 // n)
        self.dec2 = _DecoderBlock(256 // n, 128 // n, 64 // n)
        self.d1 = nn.Conv2d(128 // n, 64 // n, kernel_size=3)
        self.dec1 = nn.Sequential(
            nn.BatchNorm2d(64 // n),
            nn.ReLU(inplace=True))
        self.d2 = nn.Conv2d(64 // n, 64 // n, kernel_size=3)
        self.dec12 = nn.Sequential(
            nn.BatchNorm2d(64 // n),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64 // n, num_classes, kernel_size=1)
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


class UNet_Enc_base(nn.Module):
    def __init__(self, num_in=3):
        super(UNet_Enc_base, self).__init__()
        n = 2
        self.enc1 = _EncoderBlock(num_in, 64 // n)
        self.enc2 = _EncoderBlock(64 // n, 128 // n)
        self.enc3 = _EncoderBlock(128 // n, 256 // n)
        self.center = _DecoderBlock(256 // n, 512 // n, 256 // n)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        center = self.center(enc3)
        return center, enc3, enc2, enc1


class UNet_Dec3_base(nn.Module):
    def __init__(self, num_in=3):
        super(UNet_Dec3_base, self).__init__()
        n = 2
        self.enc1 = _EncoderBlock(num_in, 64 // n)
        self.enc2 = _EncoderBlock(64 // n, 128 // n)
        self.enc3 = _EncoderBlock(128 // n, 256 // n)
        self.center = _DecoderBlock(256 // n, 512 // n, 256 // n)
        self.dec3 = _DecoderBlock(512 // n, 256 // n, 128 // n)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        center = self.center(enc3)
        dec3 = self.dec3(torch.cat([center, F.upsample(enc3, center.size()[2:], mode='bilinear')], 1))
        return dec3, enc2, enc1


class UNet_Dec3_base_d(nn.Module):
    def __init__(self, num_in=3):
        super(UNet_Dec3_base_d, self).__init__()
        n = 2
        self.enc1 = _EncoderBlock(num_in, 64 // n)
        self.enc2 = _EncoderBlock(64 // n, 128 // n)
        self.enc3 = _EncoderBlock(128 // n, 256 // n)
        self.center = _DecoderBlockD(256 // n, 512 // n, 256 // n, dropout=0.5)
        self.dec3 = _DecoderBlockD(512 // n, 256 // n, 128 // n, dropout=0.5)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        center = self.center(enc3)
        dec3 = self.dec3(torch.cat([center, F.upsample(enc3, center.size()[2:], mode='bilinear')], 1))
        return dec3, enc2, enc1


class UNet_Dec2_base(nn.Module):
    def __init__(self, num_in=3):
        super(UNet_Dec2_base, self).__init__()
        n = 2
        self.enc1 = _EncoderBlock(num_in, 64 // n)
        self.enc2 = _EncoderBlock(64 // n, 128 // n)
        self.enc3 = _EncoderBlock(128 // n, 256 // n)
        self.center = _DecoderBlock(256 // n, 512 // n, 256 // n)
        self.dec3 = _DecoderBlock(512 // n, 256 // n, 128 // n)
        self.dec2 = _DecoderBlock(256 // n, 128 // n, 64 // n)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        center = self.center(enc3)
        dec3 = self.dec3(torch.cat([center, F.upsample(enc3, center.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        return dec2, enc1


class UNet_Dec2_base_D(nn.Module):
    def __init__(self, num_in=3):
        super(UNet_Dec2_base_D, self).__init__()
        n = 2
        self.enc1 = _EncoderBlock(num_in, 64 // n)
        self.enc2 = _EncoderBlock(64 // n, 128 // n)
        self.enc3 = _EncoderBlock(128 // n, 256 // n)
        self.center = _DecoderBlockD(256 // n, 512 // n, 256 // n, dropout=0.5)
        self.dec3 = _DecoderBlockD(512 // n, 256 // n, 128 // n, dropout=0.5)
        self.dec2 = _DecoderBlockD(256 // n, 128 // n, 64 // n, dropout=0.5)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        center = self.center(enc3)
        dec3 = self.dec3(torch.cat([center, F.upsample(enc3, center.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        return dec2, enc1


class UNetBIG_Dec2_base(nn.Module):
    def __init__(self, num_in=3):
        super(UNetBIG_Dec2_base, self).__init__()
        n = 1
        self.enc1 = _EncoderBlock(num_in, 64 // n)
        self.enc2 = _EncoderBlock(64 // n, 128 // n)
        self.enc3 = _EncoderBlock(128 // n, 256 // n)
        self.enc4 = _EncoderBlock(256 // n, 512 // n)
        self.center = _DecoderBlock(512 // n, 1024 // n, 512 // n)
        self.dec4 = _DecoderBlock(1024 // n, 512 // n, 256 // n)
        self.dec3 = _DecoderBlock(512 // n, 256 // n, 128 // n)
        self.dec2 = _DecoderBlock(256 // n, 128 // n, 64 // n)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        return dec2, enc1


class UNet_Dec1_base(nn.Module):
    def __init__(self, num_in=3):
        super(UNet_Dec1_base, self).__init__()
        n = 2
        self.enc1 = _EncoderBlock(num_in, 64 // n)
        self.enc2 = _EncoderBlock(64 // n, 128 // n)
        self.enc3 = _EncoderBlock(128 // n, 256 // n)
        self.center = _DecoderBlock(256 // n, 512 // n, 256 // n)
        self.dec3 = _DecoderBlock(512 // n, 256 // n, 128 // n)
        self.dec2 = _DecoderBlock(256 // n, 128 // n, 64 // n)
        self.d1 = nn.Conv2d(128 // n, 64 // n, kernel_size=3)
        self.dec1 = nn.Sequential(
            nn.BatchNorm2d(64 // n),
            nn.ReLU(inplace=True))
        self.d2 = nn.Conv2d(64 // n, 64 // n, kernel_size=3)
        self.dec12 = nn.Sequential(
            nn.BatchNorm2d(64 // n),
            nn.ReLU(inplace=True),
        )
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


class BUNet_Enc_head(nn.Module):
    def __init__(self, num_classes, num_in=3, prior=1.0):
        super().__init__()
        n = 2
        self.dec3 = _DecoderBlockB(512 // n, 256 // n, 128 // n, prior=prior)
        self.dec2 = _DecoderBlockB(256 // n, 128 // n, 64 // n, prior=prior)
        self.d1 = BayesConv_local_reparam(128 // n, 64 // n, kernel_size=3, prior_sig=prior)
        self.dec1 = nn.Sequential(
            nn.BatchNorm2d(64 // n),
            nn.ReLU(inplace=True))
        self.d2 = BayesConv_local_reparam(64 // n, 64 // n, kernel_size=3, prior_sig=prior)
        self.dec12 = nn.Sequential(
            nn.BatchNorm2d(64 // n),
            nn.ReLU(inplace=True),
        )
        self.final = BayesConv_local_reparam(64 // n, num_classes, kernel_size=1, prior_sig=prior)
        self.finalB = nn.BatchNorm2d(num_classes)
        initialize_weights(self)

    def forward(self, x_size, center, enc3, enc2, enc1, sample=True, returnKL=True):
        sum_kl = 0
        dec3, kl = self.dec3(torch.cat([center, F.upsample(enc3, center.size()[2:], mode='bilinear')], 1), sample,
                             returnKL)
        sum_kl += kl
        dec2, kl = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1), sample, returnKL)
        sum_kl += kl
        dec1, kl = self.d1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1), sample, returnKL)
        sum_kl += kl
        dec1 = self.dec1(dec1)
        dec1, kl = self.d2(dec1, sample, returnKL)
        sum_kl += kl
        dec1 = self.dec12(dec1)
        final, kl = self.final(dec1, sample, returnKL)
        sum_kl += kl
        final = self.finalB(final)
        return F.upsample(final, x_size[2:], mode='bilinear'), sum_kl

    def sample_forward(self, x_size, center, enc3, enc2, enc1, num_samples, num_classes):
        softmax_result = torch.zeros([x_size[0], num_classes, x_size[2], x_size[3]], device=center.device,
                                     dtype=torch.float32)
        cum_kl = 0.
        for i in range(num_samples):
            tmp_result, tmp_kl = self.forward(x_size, center, enc3, enc2, enc1, True)
            softmax_result += F.softmax(tmp_result, 1, )
            cum_kl += tmp_kl
        return softmax_result / num_samples, cum_kl / num_samples


class BUNet_Dec3_head(nn.Module):
    def __init__(self, num_classes, num_in=3, prior=1.0):
        super().__init__()
        n = 2
        self.dec2 = _DecoderBlockB(256 // n, 128 // n, 64 // n, prior=prior)
        self.d1 = BayesConv_local_reparam(128 // n, 64 // n, kernel_size=3, prior_sig=prior)
        self.dec1 = nn.Sequential(
            nn.BatchNorm2d(64 // n),
            nn.ReLU(inplace=True))
        self.d2 = BayesConv_local_reparam(64 // n, 64 // n, kernel_size=3, prior_sig=prior)
        self.dec12 = nn.Sequential(
            nn.BatchNorm2d(64 // n),
            nn.ReLU(inplace=True),
        )
        self.final = BayesConv_local_reparam(64 // n, num_classes, kernel_size=1, prior_sig=prior)
        self.finalB = nn.BatchNorm2d(num_classes)
        initialize_weights(self)

    def forward(self, x_size, dec3, enc2, enc1, sample=True, returnKL=True):
        sum_kl = 0
        dec2, kl = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1), sample, returnKL)
        sum_kl += kl
        dec1, kl = self.d1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1), sample, returnKL)
        sum_kl += kl
        dec1 = self.dec1(dec1)
        dec1, kl = self.d2(dec1, sample, returnKL)
        sum_kl += kl
        dec1 = self.dec12(dec1)
        final, kl = self.final(dec1, sample, returnKL)
        sum_kl += kl
        final = self.finalB(final)
        return F.upsample(final, x_size[2:], mode='bilinear'), sum_kl

    def sample_forward(self, x_size, dec3, enc2, enc1, num_samples, num_classes):
        softmax_result = torch.zeros([x_size[0], num_classes, x_size[2], x_size[3]], device=dec3.device,
                                     dtype=torch.float32)
        cum_kl = 0.
        for i in range(num_samples):
            tmp_result, tmp_kl = self.forward(x_size, dec3, enc2, enc1, True)
            softmax_result += F.softmax(tmp_result, 1, )
            cum_kl += tmp_kl
        return softmax_result / num_samples, cum_kl / num_samples


class BUNet_Dec2_head(nn.Module):
    def __init__(self, num_classes, prior=1.0):
        super().__init__()
        n = 2
        self.d1 = BayesConv_local_reparam(128 // n, 64 // n, kernel_size=3, prior_sig=prior)
        self.dec1 = nn.Sequential(
            nn.BatchNorm2d(64 // n),
            nn.ReLU(inplace=True))
        self.d2 = BayesConv_local_reparam(64 // n, 64 // n, kernel_size=3, prior_sig=prior)
        self.dec12 = nn.Sequential(
            nn.BatchNorm2d(64 // n),
            nn.ReLU(inplace=True),
        )
        self.final = BayesConv_local_reparam(64 // n, num_classes, kernel_size=1, prior_sig=prior)
        self.finalB = nn.BatchNorm2d(num_classes)
        initialize_weights(self)

    def forward(self, x_size, dec2, enc1, sample=True, returnKL=True):
        sum_kl = 0
        dec1, kl = self.d1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1), sample, returnKL)
        sum_kl += kl
        dec1 = self.dec1(dec1)
        dec1, kl = self.d2(dec1, sample, returnKL)
        sum_kl += kl
        dec1 = self.dec12(dec1)
        final, kl = self.final(dec1, sample, returnKL)
        sum_kl += kl
        final = self.finalB(final)
        return F.upsample(final, x_size[2:], mode='bilinear'), sum_kl

    def sample_forward(self, x_size, dec2, enc1, num_samples, num_classes):
        softmax_result = torch.zeros([num_samples, x_size[0], num_classes, x_size[2], x_size[3]], device=dec2.device,
                                     dtype=torch.float32)
        cum_kl = 0.
        for i in range(num_samples):
            tmp_result, tmp_kl = self.forward(x_size, dec2, enc1, True)
            softmax_result[i] = F.softmax(tmp_result, 1, )
            cum_kl += tmp_kl
        return softmax_result.mean(0), softmax_result.std(0), cum_kl / num_samples


class BUNetBIG_Dec2_head(nn.Module):
    def __init__(self, num_classes, prior=1.0):
        super().__init__()
        n = 1
        self.d1 = BayesConv_local_reparam(128 // n, 64 // n, kernel_size=3, prior_sig=prior)
        self.dec1 = nn.Sequential(
            nn.BatchNorm2d(64 // n),
            nn.ReLU(inplace=True))
        self.d2 = BayesConv_local_reparam(64 // n, 64 // n, kernel_size=3, prior_sig=prior)
        self.dec12 = nn.Sequential(
            nn.BatchNorm2d(64 // n),
            nn.ReLU(inplace=True),
        )
        self.final = BayesConv_local_reparam(64 // n, num_classes, kernel_size=1, prior_sig=prior)
        self.finalB = nn.BatchNorm2d(num_classes)
        initialize_weights(self)

    def forward(self, x_size, dec2, enc1, sample=True, returnKL=True):
        sum_kl = 0
        dec1, kl = self.d1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1), sample, returnKL)
        sum_kl += kl
        dec1 = self.dec1(dec1)
        dec1, kl = self.d2(dec1, sample, returnKL)
        sum_kl += kl
        dec1 = self.dec12(dec1)
        final, kl = self.final(dec1, sample, returnKL)
        sum_kl += kl
        final = self.finalB(final)
        return F.upsample(final, x_size[2:], mode='bilinear'), sum_kl

    def sample_forward(self, x_size, dec2, enc1, num_samples, num_classes):
        softmax_result = torch.zeros([x_size[0], num_classes, x_size[2], x_size[3]], device=dec2.device,
                                     dtype=torch.float32)
        cum_kl = 0.
        for i in range(num_samples):
            tmp_result, tmp_kl = self.forward(x_size, dec2, enc1, True)
            softmax_result += F.softmax(tmp_result, 1, )
            cum_kl += tmp_kl
        return softmax_result / num_samples, cum_kl / num_samples


class BUNet_Dec1_head(nn.Module):
    def __init__(self, num_classes, prior=1.0):
        super().__init__()
        n = 2
        self.final = BayesConv_local_reparam(64 // n, num_classes, kernel_size=1, prior_sig=prior)
        self.finalB = nn.BatchNorm2d(num_classes)
        initialize_weights(self)

    def forward(self, x_size, dec1, sample=True, returnKL=True):
        sum_kl = 0
        final, kl = self.final(dec1, sample, returnKL)
        sum_kl += kl
        final = self.finalB(final)
        return F.upsample(final, x_size[2:], mode='bilinear'), sum_kl

    def sample_forward(self, x_size, dec1, num_samples, num_classes):
        softmax_result = torch.zeros([x_size[0], num_classes, x_size[2], x_size[3]], device=dec1.device,
                                     dtype=torch.float32)
        cum_kl = 0.
        for i in range(num_samples):
            tmp_result, tmp_kl = self.forward(x_size, dec1, True)
            softmax_result += F.softmax(tmp_result, 1, )
            cum_kl += tmp_kl
        return softmax_result / num_samples, cum_kl / num_samples


class UNet_Dec1_head(nn.Module):
    def __init__(self, num_classes, num_in=3):
        super(UNet_Dec1_head, self).__init__()
        n = 2
        self.final = nn.Conv2d(64 // n, num_classes, kernel_size=1)
        self.finalB = nn.BatchNorm2d(num_classes)
        self.softm = nn.Softmax2d()
        initialize_weights(self)

    def forward(self, x_size, dec1):
        final = self.final(dec1)
        final = self.finalB(final)
        final = F.upsample(final, x_size[2:], mode='bilinear')
        return F.softmax(final, dim=1)


class UNet_Dec2_head(nn.Module):
    def __init__(self, num_classes, num_in=3):
        super(UNet_Dec2_head, self).__init__()
        n = 2
        self.d1 = nn.Conv2d(128 // n, 64 // n, kernel_size=3)
        self.dec1 = nn.Sequential(
            nn.BatchNorm2d(64 // n),
            nn.ReLU(inplace=True))
        self.d2 = nn.Conv2d(64 // n, 64 // n, kernel_size=3)
        self.dec12 = nn.Sequential(
            nn.BatchNorm2d(64 // n),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64 // n, num_classes, kernel_size=1)
        self.finalB = nn.BatchNorm2d(num_classes)
        self.softm = nn.Softmax2d()
        initialize_weights(self)

    def forward(self, x_size, dec2, enc1):
        dec1 = self.d1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(dec1)
        dec1 = self.d2(dec1)
        dec1 = self.dec12(dec1)
        final = self.final(dec1)
        final = self.finalB(final)
        final = F.upsample(final, x_size[2:], mode='bilinear')
        return F.softmax(final, dim=1)


class UNet_Ensemble(nn.Module):
    def __init__(self, num_models=4, mutliHead_layer="Dec1", prior=0.1, num_in=4, num_classes=4):
        super(UNet_Ensemble, self).__init__()
        self.num_models = num_models
        self.multiHead = mutliHead_layer
        if self.multiHead is "BDec1":
            self.base = UNet_Dec1_base(num_in=num_in)
            self.nets = nn.ModuleList(
                [BUNet_Dec1_head(num_classes=num_classes, prior=prior) for _ in range(num_models)])
        elif self.multiHead is "BDec2":
            self.base = UNet_Dec2_base(num_in=num_in)
            self.nets = nn.ModuleList(
                [BUNet_Dec2_head(num_classes=num_classes, prior=prior) for _ in range(num_models)])
        elif self.multiHead is "DBDec2":
            self.base = UNet_Dec2_base_D(num_in=num_in)
            self.nets = nn.ModuleList(
                [BUNet_Dec2_head(num_classes=num_classes, prior=prior) for _ in range(num_models)])
        elif self.multiHead is "BDec3":
            self.base = UNet_Dec3_base(num_in=num_in)
            self.nets = nn.ModuleList(
                [BUNet_Dec3_head(num_classes=num_classes, prior=prior) for _ in range(num_models)])
        elif self.multiHead is "DBDec3":
            self.base = UNet_Dec3_base_d(num_in=num_in)
            self.nets = nn.ModuleList(
                [BUNet_Dec3_head(num_classes=num_classes, prior=prior) for _ in range(num_models)])
        elif self.multiHead is "BEnc":
            self.base = UNet_Enc_base(num_in=num_in)
            self.nets = nn.ModuleList([BUNet_Enc_head(num_classes=num_classes, prior=prior) for _ in range(num_models)])
        elif self.multiHead is "Dec1":
            self.base = UNet_Dec1_base(num_in=num_in)
            self.nets = nn.ModuleList([UNet_Dec1_head(num_classes=num_classes) for _ in range(num_models)])
        elif self.multiHead is "Dec2":
            self.base = UNet_Dec2_base(num_in=num_in)
            self.nets = nn.ModuleList([UNet_Dec2_head(num_classes=num_classes) for _ in range(num_models)])
        elif self.multiHead is "DDec2":
            self.base = UNet_Dec2_base_D(num_in=num_in)
            self.nets = nn.ModuleList([UNet_Dec2_head(num_classes=num_classes) for _ in range(num_models)])
        elif self.multiHead is "BIGBDec2":
            self.base = UNetBIG_Dec2_base(num_in=num_in)
            self.nets = nn.ModuleList(
                [BUNetBIG_Dec2_head(num_classes=num_classes, prior=prior) for _ in range(num_models)])
        else:
            print("Only Multihead Layers for BDec1, BDec2, BDec3, BEnc, Dec1, Dec2, DDec2 and BIGBDec2 are implemented!!")

    def forward(self, x, samples=3, num_classes=10, num_models=4, num_gpu=4):
        outputs = []
        stds = []
        sum_kl = []
        x_size = x.size()
        if self.multiHead is "BDec1":
            dec1 = self.base(x)
            for i in range(self.num_models):
                o, kl = self.nets[i].sample_forward(x_size, dec1, samples, num_classes)
                sum_kl.append(kl)
                outputs.append(o)
            return torch.stack(outputs), 0, torch.stack(sum_kl)
        elif self.multiHead is "BDec2":
            dec2, enc1 = self.base(x)
            for i in range(self.num_models):
                o, s, kl = self.nets[i].sample_forward(x_size, dec2, enc1, samples, num_classes)
                sum_kl.append(kl)
                outputs.append(o)
                stds.append(s)
            return torch.stack(outputs), torch.stack(stds), torch.stack(sum_kl)
        elif self.multiHead is "DBDec2":
            dec2, enc1 = self.base(x)
            for i in range(self.num_models):
                o, s, kl = self.nets[i].sample_forward(x_size, dec2, enc1, samples, num_classes)
                sum_kl.append(kl)
                outputs.append(o)
                stds.append(s)
            return torch.stack(outputs), torch.stack(stds), torch.stack(sum_kl)
        elif self.multiHead is "BDec3":
            dec3, enc2, enc1 = self.base(x)
            for i in range(self.num_models):
                o, kl = self.nets[i].sample_forward(x_size, dec3, enc2, enc1, samples, num_classes)
                sum_kl.append(kl)
                outputs.append(o)
            return torch.stack(outputs), 0, torch.stack(sum_kl)
        elif self.multiHead is "DBDec3":
            dec3, enc2, enc1 = self.base(x)
            for i in range(self.num_models):
                o, kl = self.nets[i].sample_forward(x_size, dec3, enc2, enc1, samples, num_classes)
                sum_kl.append(kl)
                outputs.append(o)
            return torch.stack(outputs), 0, torch.stack(sum_kl)
        elif self.multiHead is "BEnc":
            center, enc3, enc2, enc1 = self.base(x)
            for i in range(self.num_models):
                o, kl = self.nets[i].sample_forward(x_size, center, enc3, enc2, enc1, samples, num_classes)
                sum_kl.append(kl)
                outputs.append(o)
            return torch.stack(outputs), 0, torch.stack(sum_kl)
        elif self.multiHead is "Dec1":
            dec1 = self.base(x)
            for i in range(self.num_models):
                o = self.nets[i](x_size, dec1)
                outputs.append(o)
            return torch.stack(outputs), torch.stack(outputs).std(0), 0
        elif self.multiHead is "Dec2":
            dec2, enc1 = self.base(x)
            for i in range(self.num_models):
                o = self.nets[i](x_size, dec2, enc1)
                outputs.append(o)
            return torch.stack(outputs), torch.stack(outputs).std(0), 0
        elif self.multiHead is "DDec2":
            dec2, enc1 = self.base(x)
            for i in range(self.num_models):
                o = self.nets[i](x_size, dec2, enc1)
                outputs.append(o)
            return torch.stack(outputs), torch.stack(outputs).std(0), 0
        elif self.multiHead is "BIGBDec2":
            dec2, enc1 = self.base(x)
            sum_kl = torch.zeros(num_models, device="cuda:0")
            outputs = torch.zeros(num_models, x_size[0], x_size[1], x_size[2], x_size[3], device="cuda:0")
            for i in range(self.num_models):
                o, kl = self.nets[i].sample_forward(x_size, dec2, enc1, samples, num_classes)
                sum_kl[i] += kl.to(sum_kl.device)
                outputs[i] = o
            return outputs, 0, sum_kl
        else:
            print("Only Multihead Layers for BDec1, BDec2, BDec3, BEnc, DBDec2, DBDec3, Dec1, Dec2, DDec2, BIGBDec2 are implemented!!")
