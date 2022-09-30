import numpy as np
from skimage.measure import label, regionprops
from numba import njit

def bwedge(bImg, num = 5):
    # print('Original Size', bImg.shape)
    core = bImg[1:-1, 1:-1]

    neighbors = np.zeros(core.shape, np.uint8)
    for i in range(3):
        for j in range(3):
            _cur = bImg[i:i+core.shape[0], j:j+core.shape[1]]
            neighbors += _cur

    _edge = core & (neighbors > num) & (neighbors < 9)
    _blank = np.zeros((_edge.shape[0] + 2, _edge.shape[1] + 2), dtype = _edge.dtype)
    _blank[1:-1, 1:-1] = _edge
    return _blank

@njit
def initGrid(Image):
    dimen = Image.shape[0]

    startBoxSize = int(np.floor(0.45 * dimen))
    curBoxSize = int(startBoxSize)

    nBox = np.zeros((startBoxSize - 2 + 1))
    boxSize = np.arange(2, startBoxSize + 1)[::-1]

    for sizeCount in range(startBoxSize - 1):
        curBoxSize = boxSize[sizeCount]

        for macroY in range(1, int(np.ceil(dimen / curBoxSize)) + 1):
            for macroX in range(1, int(np.ceil(dimen/curBoxSize)) + 1):
                boxYinit = (macroY - 1) * curBoxSize
                boxXinit = (macroX - 1) * curBoxSize
                boxYend = np.fmin(macroY*curBoxSize, dimen)
                boxXend = np.fmin(macroX * curBoxSize, dimen)

                boxFound = False
                for curY in range(boxYinit, boxYend):
                    for curX in range(boxXinit, boxXend):
                        if Image[curY, curX]:
                            boxFound = True
                            nBox[sizeCount] += 1
                            break

                    if boxFound == True:
                        break
    return nBox, boxSize


def pft_JC_bxct(EdgeImage):
    EdgeImage = EdgeImage.astype(np.uint8)
    s = regionprops(EdgeImage)
    box = s[0].bbox
    nWidth = box[3] - box[1]
    nHeight = box[2] - box[0]

    if (nHeight * nWidth) > 0.25 * EdgeImage.shape[0] * EdgeImage.shape[1]:
        EdgeImage = EdgeImage[box[0]:box[2], box[1]:box[3]]

    padI = np.zeros([np.max(EdgeImage.shape), np.max(EdgeImage.shape)], EdgeImage.dtype)

    if EdgeImage.shape[0] > EdgeImage.shape[1]:
        _s = int((EdgeImage.shape[0] - EdgeImage.shape[1]) * 0.5)
        padI[:, _s:_s + EdgeImage.shape[1]] = EdgeImage
    else:
        _s = int((EdgeImage.shape[1] - EdgeImage.shape[0]) * 0.5)
        padI[_s:_s + EdgeImage.shape[0], :] = EdgeImage

    nBoxA, boxSizeA = initGrid(padI)
    nBoxB, _ = initGrid(padI[::-1])
    nBoxC, _ = initGrid(padI[:, ::-1])
    nBoxD, _ = initGrid(padI[::-1, ::-1])

    nBox = np.min([nBoxA, nBoxB, nBoxC, nBoxD], axis = 0)
    boxSize = boxSizeA

    assert(len(nBox) == len(boxSize))
    totalBoxSizes = len(boxSize)
    p = np.polyfit(np.log(boxSize), np.log(nBox), 1)

    return p[0] * -1.

