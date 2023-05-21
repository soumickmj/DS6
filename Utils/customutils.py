import numpy as np
import scipy.io as sio

__author__ = "Soumick Chatterjee, Chompunuch Sarasaen"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Soumick Chatterjee", "Chompunuch Sarasaen"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"


def fft2c(x, shape=None, axes=(0, 1), shiftAxes=(0, 1),
          normalize=None):  # originally was axes=(-2,-1), shiftAxes = None
    f = np.empty(x.shape, dtype=np.complex128)
    if (len(x.shape) == 4):
        for i in range(x.shape[-1]):
            for j in range(x.shape[-2]):
                f[:, :, j, i] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x[:, :, j, i]), s=shape, norm=normalize))
    elif (len(x.shape) == 3):
        for i in range(x.shape[-1]):
            f[:, :, i] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x[:, :, i]), s=shape, norm=normalize))
    else:
        f = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), s=shape, norm=normalize))
    return f


def ifft2c(x, shape=None, axes=(0, 1), shiftAxes=(0, 1),
           normalize=None):  # originally was axes=(-2,-1), shiftAxes = None
    f = np.empty(x.shape, dtype=np.complex128)
    if (len(x.shape) == 4):
        for i in range(x.shape[-1]):
            for j in range(x.shape[-2]):
                f[:, :, j, i] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x[:, :, j, i]), s=shape, norm=normalize))
    elif (len(x.shape) == 3):
        for i in range(x.shape[-1]):
            f[:, :, i] = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x[:, :, i]), s=shape, norm=normalize))
    else:
        f = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x), s=shape, norm=normalize))
    return f


def createCenterRatioMask(slice, percent, returnNumLinesRemoved=False):
    dim1 = slice.shape[0]
    dim2 = slice.shape[1]
    ratio = dim2 / dim1

    mask = np.ones(slice.shape)
    dim1_now = dim1
    dim2_should = dim2
    i = 0
    currentPercent = 1

    while currentPercent > percent:
        i += 1
        mask[0:i, :] = 0
        mask[slice.shape[0] - i:, :] = 0

        dim1_now = dim1 - (i * 2)
        dim2_should = round(dim1_now * ratio)
        dim2_removal = int((dim2 - dim2_should) / 2)

        mask[:, 0:dim2_removal] = 0
        mask[:, slice.shape[1] - dim2_removal:] = 0

        currentPercent = np.count_nonzero(mask) / mask.size

    if returnNumLinesRemoved:
        linesRemoved_dim1 = dim1 - dim1_now
        linesRemoved_dim2 = dim2 - dim2_should
        return mask, (linesRemoved_dim1, linesRemoved_dim2)
    else:
        return mask


def performUndersampling(fullImgVol, mask=None, maskmatpath=None, zeropad=True):
    # Either send mask, or maskmatpath.
    # path will only be used in mask not supplied
    fullKSPVol = fft2c(fullImgVol)
    underKSPVol = performUndersamplingKSP(fullKSPVol, mask, maskmatpath, zeropad)
    underImgVol = ifft2c(underKSPVol)
    return underImgVol


def performUndersamplingKSP(fullKSPVol, mask=None, maskmatpath=None, zeropad=True):
    # Either send mask, or maskmatpath.
    # path will only be used in mask not supplied
    if mask is None:
        mask = sio.loadmat(maskmatpath)['mask']
    if zeropad:
        underKSPVol = np.multiply(fullKSPVol.transpose((2, 0, 1)), mask).transpose((1, 2, 0))
    else:
        temp = []
        for i in range(mask.shape[0]):
            maskline = mask[i, :]
            if maskline.any():
                temp.append(fullKSPVol[i, ...])
        temp = np.array(temp)
        underKSPVol = []
        for i in range(mask.shape[1]):
            maskline = mask[:, i]
            if maskline.any():
                underKSPVol.append(temp[:, i, ...])
        underKSPVol = np.array(underKSPVol).swapaxes(0, 1)
    return underKSPVol
