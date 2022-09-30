from normalize01 import *
from scipy.signal import fftconvolve
from skimage.morphology import convex_hull_image
from skimage.morphology import dilation
from pft_JC_bxct import *
from disk import *
from PyQt5.QtWidgets import QApplication

def updateB(Img, C, M, Ksigma):
    PC1 = np.zeros(Img.shape, Img.dtype)
    PC2 = np.copy(PC1)
    N_class = M.shape[2]

    for kk in range(N_class):
        PC1 += C[kk] * M[:, :, kk]
        PC2 += C[kk]**2 * M[:, :, kk]

    KNm1 = fftconvolve(PC1 * Img, Ksigma, 'same')
    KDn1 = fftconvolve(PC2, Ksigma, 'same')

    b = KNm1 / KDn1

    return b

def updateC(Img, u, Kb1, Kb2, epsilon):
    Hu = Heaviside(u, epsilon)
    M = np.zeros((Hu.shape[0], Hu.shape[1], 2), Hu.dtype)
    M[:, :, 0] = Hu
    M[:, :, 1] = 1 - Hu
    N_class = 2
    C_new = np.zeros(N_class)
    for i in range(N_class):
        Nm2 = Kb1 * Img * M[:, :, i]
        Dn2 = Kb2 * M[:, :, i]
        C_new[i] = np.sum(Nm2) / np.sum(Dn2)

    return C_new

def del2(u):
    core = u[1:-1, 1:-1]
    up = u[0:-2, 1:-1]
    down = u[2:, 1:-1]
    left = u[1:-1, 0:-2]
    right = u[1:-1, 2:]

    laplacian = np.zeros(u.shape, dtype=u.dtype)
    laplacian[1:-1, 1:-1] = (up + down + left + right) * 0.25 - core

    laplacian[0, 1:-1] = 2 * laplacian[1, 1:-1] - laplacian[2, 1:-1]
    laplacian[-1, 1:-1] = 2 * laplacian[-2, 1:-1] - laplacian[-3, 1:-1]

    laplacian[:, 0] = 2 * laplacian[:, 1] - laplacian[:, 2]
    laplacian[:, -1] = 2 * laplacian[:, -2] - laplacian[:, -3]

    return laplacian


def curvature_central(u):
    uy, ux = np.gradient(u)
    normDu = np.sqrt(ux**2 + uy**2 + 1e-10)
    Nx = ux / normDu
    Ny = uy / normDu
    _, nxx = np.gradient(Nx)
    nyy, _ = np.gradient(Ny)
    return nxx + nyy

def Heaviside(x, epsilon):
    h = 0.5 * (1. + (2. / np.pi) * np.arctan(x / epsilon))
    return h

def Dirac(x, epsilon):
    return (epsilon / np.pi) / (epsilon**2 + x**2)

def NeumannBoundCond(f):
    nrow, ncol = f.shape
    g = f
    g[[0, nrow - 1, 0, nrow - 1], [0, 0, ncol - 1, ncol - 1]] = g[[2, nrow - 3, 2, nrow - 3], [2, 2, ncol -3, ncol -3]]
    g[[0, nrow - 1], 1:ncol - 1] = g[[2, nrow - 3], 1:ncol - 1]
    g[1:nrow - 1, [0, ncol - 1]] = g[1:nrow - 1, [2, ncol - 3]]
    return g

def updateLSF(Img, u0, C, KONE_Img, KB1, KB2, mu, nu, timestep, epsilon, iter_lse):
    u = u0
    Hu = Heaviside(u, epsilon)

    M = np.zeros((Hu.shape[0], Hu.shape[1], 2), dtype = Hu.dtype)
    M[:, :, 0] = Hu
    M[:, :, 1] = 1 - Hu
    N_class = 2
    e = np.zeros(M.shape)

    for i in range(N_class):
        e[:, :, i] = KONE_Img - 2 * Img * C[i] * KB1 + C[i]**2 * KB2

    for kk in range(iter_lse):
        u = NeumannBoundCond(u)
        K = curvature_central(u)
        DiracU = Dirac(u, epsilon)
        ImageTerm = -DiracU * (e[:, :, 0] - e[:, :, 1])
        penalizeTerm = mu * (4 * del2(u) - K)
        lengthTerm = nu * DiracU * K
        u = u + timestep * (lengthTerm + penalizeTerm + ImageTerm)

    return u

def lse_bfe(u0, Img, b, Ksigma, KONE, nu, timestep, mu, epsilon, iter_lse):
    u = u0

    KB1 = fftconvolve(b, Ksigma, 'same')
    KB2 = fftconvolve(b * b, Ksigma, 'same')

    C = updateC(Img, u, KB1, KB2, epsilon)

    KONE_Img = Img * Img * KONE
    u = updateLSF(Img, u, C, KONE_Img, KB1, KB2, mu, nu, timestep, epsilon, iter_lse)

    Hu = Heaviside(u, epsilon)

    M = np.zeros((Hu.shape[0], Hu.shape[1], 2), Hu.dtype)
    M[:, :, 0] = Hu
    M[:, :, 1] = 1 - Hu
    b = updateB(Img, C, M, Ksigma)

    return u, b, C

def fspecial_gauss(sigma):
    x, y = np.mgrid[(-2 * sigma):(2 * sigma + 1), (-2 * sigma):(2 * sigma + 1)]
    g = np.exp(-((x**2 + y**2) / (2. * sigma**2)))
    g = g / np.sum(g)
    return g

def imadjust(img):
    values = img.flatten()
    values = np.sort(values)

    low = values[int(len(values) * 0.01)]
    high = values[int(len(values) * 0.99)]
    trim = np.clip(img, a_min=low, a_max=high)

    if img.dtype == 'uint16':
        after = normalizer01(trim) * (2**16 - 1) - 2**15
        after = after.astype(np.int16)
    elif img.dtype == 'uint8':
        after = normalizer01(trim) * (2**8 - 1) - 2**7
        after = after.astype(np.int8)
    else:
        print('Wrong--------------',img.dtype)
        sys.exit("Wrong format")

    return after

def pft_JC_LevelSetDetection(inputImg, pBar):
    sigma = 4
    epsilon = 3
    Img = inputImg.astype(np.float)
    A = 255
    Img = A * normalizer01(Img)
    nu = 0.001 * A**2

    iter_outer = 100
    iter_inner = 10

    timestep = 0.1
    mu = 1
    c0 = 1

    initialLSF = c0 * np.ones(Img.shape)
    xSize, ySize = Img.shape

    initialLSF[int(round(0.2 * xSize)):int(round(0.8 * ySize)), int(round(0.2 * xSize)):int(round(0.8*ySize))] = - c0
    u = initialLSF
    b = np.ones(Img.shape)

    K = fspecial_gauss(sigma)
    KI = fftconvolve(Img, K, 'same')
    KONE = fftconvolve(np.ones(Img.shape), K, 'same')

    row, col = Img.shape
    N = row * col
    _v = 0
    for i in range(iter_outer):
        u, b, _ = lse_bfe(u, Img, b, K, KONE, nu, timestep, mu, epsilon, iter_inner)
        pBar.setValue(_v)
        _v += 1
        QApplication.processEvents()
        
    contourI = u > 0
    return contourI

def findEllipse(LargestArea):
    Stats = regionprops(LargestArea)[0]
    y0, x0 = Stats.centroid
    aa = 1.05 * Stats.major_axis_length * 0.5
    bb = 1.05 * Stats.minor_axis_length * 0.5
    Theta = -Stats.orientation
    cc = np.cos(Theta)
    ss = np.sin(Theta)
    rows, cols = LargestArea.shape

    yy, xx = np.meshgrid(np.arange(cols), np.arange(rows))

    XX = cc * (xx - y0) - ss * (yy - x0)
    YY = ss * (xx - y0) + cc * (yy - x0)

    Delta = int(round(0.025 * (aa + bb)))

    EquivalentEllipse = (XX ** 2 / aa ** 2 + YY ** 2 / bb ** 2 <= 1.0)
    '''
    plt.subplot(121)
    plt.imshow(LargestArea, cmap='gray')
    plt.subplot(122)
    plt.scatter(x0, y0, marker='+', edgecolors='r')
    plt.scatter(x0 + bb * np.cos(Theta), y0 + bb * np.sin(Theta), marker='o', c = 'r')
    plt.scatter(x0 + aa * np.cos(Theta + np.pi * 0.5), y0 + aa * np.sin(Theta + np.pi * 0.5), marker='+', c = 'r')
    plt.imshow(EquivalentEllipse + LargestArea, cmap='gray')
    plt.show()
    '''
    return EquivalentEllipse, Delta


def pft_JC_FractalDimensionCalculation(CroppedImage, BinaryMask, Slice, pBar):

    AdjustedCroppedImage = imadjust(CroppedImage)
    # print(AdjustedCroppedImage.dtype, np.max(AdjustedCroppedImage), np.min(AdjustedCroppedImage))

    PreThresholdImage = np.zeros(AdjustedCroppedImage.shape, AdjustedCroppedImage.dtype)
    PreThresholdImage[BinaryMask] = AdjustedCroppedImage[BinaryMask]

    ThresholdImage = ~pft_JC_LevelSetDetection(PreThresholdImage, pBar)
    ThresholdImage = ThresholdImage * BinaryMask

    LargestArea00 = label(ThresholdImage)

    _areas = [np.sum(LargestArea00 == i) for i in range(1, int(np.max(LargestArea00)) + 1)]

    LargestArea01 = (LargestArea00==(np.argmax(_areas) + 1))
    LargestArea = convex_hull_image(LargestArea01)
    LargestArea = LargestArea.astype(np.uint8)

    EquivalentEllipse, Delta = findEllipse(LargestArea)
    LargestArea = dilation(LargestArea, selem=disk(Delta))
    Surround = LargestArea | EquivalentEllipse
    ThresholdImage = ThresholdImage * Surround
    EdgeImage = bwedge(ThresholdImage)
    '''
    plt.subplot(131)
    plt.imshow(ThresholdImage, cmap='gray')
    plt.subplot(132)
    plt.imshow(EdgeImage, cmap='gray')
    plt.subplot(133)
    plt.imshow(CroppedImage, cmap='gray')
    plt.show()
    '''
    FD = pft_JC_bxct(EdgeImage)
    return FD, EdgeImage, Surround

