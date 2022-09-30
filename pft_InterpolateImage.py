from skimage.transform import resize
from normalize01 import *

def pft_InterpolateImage(Wzor, Mask, Segmentation, Perimeter, OriginalResolution, InterpolationType):
    if InterpolationType == 'Imresize - 0.25 mm pixels - cubic':
        newPixelSize = 0.25
        magnification = OriginalResolution/newPixelSize
    else:
        magnification = 4.

    NR = int(round(magnification * Wzor.shape[0]))
    NC = int(round(magnification * Wzor.shape[1]))

    W = resize(normalizer01(Wzor), (NR, NC), anti_aliasing=True) * (np.max(Wzor) - np.min(Wzor)) + np.min(Wzor)
    W = np.round(W).astype(Wzor.dtype)
    M = resize(Mask, (NR, NC), anti_aliasing=True) > 0.5
    S = resize(normalizer01(Segmentation), (NR, NC), anti_aliasing=True) * np.max(Segmentation)
    S = np.round(S).astype(Segmentation.dtype)
    P = resize(Perimeter, (NR, NC), anti_aliasing=True) > 0.5
    return W, M, S, P
