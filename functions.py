import numpy as np
import nibabel as nib
import os
from skimage.morphology import dilation
from skimage.measure import label, regionprops
import sys
import pydicom as dicom
import pandas as pd
from PyQt5.QtWidgets import QApplication
import operator
from skimage.transform import resize
from skimage.segmentation import flood, flood_fill
# from matplotlib import pyplot as plt

# conNames = ['saendocardialContour', 'saepicardialContour', 'sarvendocardialContour', 'sarvepicardialContour', 'bloodpoolContour']
conNames = ['saendocardialContour', 'saepicardialContour',]
def readCVI(filename):
    
    head = '<Hash:item'
    end = '<!--end of ImageStates-->'
    imgNum = 0
    pixels = []
    user = ''
    with open(filename) as f:
        lines = f.readlines()
        
        for _line in lines:
            if _line.find(head) == 12:
                if _line.find('StudyUid') > 0:
                    studyUID = _line.split('>')[1].split('<')[0]
            '''   
            if _line.find(end) == 12:
                break
            '''
                
            if _line.find(head) == 4 and _line.find('OwnerUserName') > 0:
                user = _line.split('>')[1].split('<')[0]
                
            if _line.find(head) == 16:
                _uid = _line.split('"')[1].split('"')[0]
                _imageBool = True
                points = []
            
            if _line.find('<!--end') == 16 and _line.find(_uid) > 0:
                if len(points) > 0:
                    print(_uid, len(points), '=')
                _imageBool = False
                
            if _line.find(head) == 24 and _line.find('Hash:count="9"') > 0:
                _kind = _line.split('"')[1]
                # print(_kind)
                
            if _line.find('<!--end of Points-->') == 28:
                if len(points) > 0 and _kind in conNames:
                    # print(_uid, len(points), '=', _kind)
                    pixels.append([_uid, _kind, points])
                    imgNum += 1
                points = []
                
            if _line.find(head) == 28 and _line.find('QPolygon') > 0:
                _num = _line.split('=')[-1].split('>')[0].replace('"', '')
                # print(_uid, _num)
                
            if _line.find('<Point:x>') == 36:
                _x = _line.split('>')[1].split('<')[0]
            if _line.find('<Point:y>') == 36:
                _y = _line.split('>')[1].split('<')[0]
                points.append([_x, _y])
                
        return studyUID, user, pixels

def cvi2nii(cviname, dicomlist, volume):

    imglist = dicomlist[2]
    '''
    plt.subplot(121)
    plt.imshow(volume[:, :, 0, 0], cmap = 'gray')
    plt.subplot(122)
    plt.imshow(np.rot90(imglist[0], 3), cmap = 'gray')
    plt.show()
    '''
    studyUID, _, pixels = readCVI(cviname)
    cviUID = [row[0] for row in pixels]
    cviKind = [row[1] for row in pixels]
    cviPts = [row[2] for row in pixels]

    filelist = dicomlist[0]
    uidlist = dicomlist[1]
    imglist = dicomlist[2]
    T = volume.shape[3]
    black = np.zeros(volume.shape)
    
    for i in range(len(filelist)):
        _name, _uid = filelist[i], uidlist[i]
        
        keySeg = []
        while(True):
            try:
                _id = cviUID.index(_uid)
            except ValueError:
                break

            _cvi_con = cviPts[_id]
            _cvi_con = np.asarray(_cvi_con).astype(np.int) // 4
            _cvi_kind = cviKind[_id]

            tmp = np.zeros((black.shape[1], black.shape[0]))
            tmp[_cvi_con[:, 0], _cvi_con[:, 1]] = 1
            _inv = flood_fill(tmp, (0, 0), 255, connectivity=1)

            _back = 255 - _inv
            _seg =  np.nonzero(_back > 122)
            keySeg.append([_cvi_kind, _seg])
            # tmp[_seg[1], _seg[0]] = conNames.index(_cvi_kind)
            # black[:, :, i // T, i % T] = np.rot90(tmp, 3)
            cviUID.pop(_id)
            cviKind.pop(_id)
            cviPts.pop(_id)
        
        if keySeg == []:
            continue

        tmp = np.rot90(black[:, :, i // T, i % T])
        for _ks in keySeg:
            tmp[_ks[1][1], _ks[1][0]] = conNames.index(_ks[0]) + 1
        black[:, :, i // T, i % T] = np.rot90(tmp, 3)

    return black



def readDCM(filename):
    ds = dicom.read_file(filename)
    origin = np.asarray([float(i) for i in ds[0x20, 0x32].value])
    xydir = np.asarray([float(i) for i in ds[0x20, 0x37].value])
    xdir = xydir[:3]
    ydir = xydir[3:]

    origin[:2] = -origin[:2]
    xdir[:2] = -xdir[:2]
    ydir[:2] = -ydir[:2]

    uid = ds[0x8, 0x18].value

    dx = float(ds.PixelSpacing[1])
    dy = float(ds.PixelSpacing[0])

    r, c = np.shape(ds.pixel_array)
    if r >= 512 or c >= 512:
        return resize(ds.pixel_array, (ds.pixel_array.shape[0] // 2, ds.pixel_array.shape[1] // 2),
                      anti_aliasing=True), float(ds[0x20, 0x1041].value), float(ds[0x8, 0x13].value), \
               [origin, xdir, ydir, dx * 2, dy * 2]

    return ds.pixel_array, float(ds[0x20, 0x1041].value), float(ds[0x8, 0x13].value), [origin, xdir, ydir, dx, dy], uid

def cropImage(_img, _mask, _seg, _peri):
    # Crop Image
    beforeShape = _img.shape
    _labels = label(_mask)
    assert (np.max(_labels) == 1)
    yMin = np.min(np.nonzero(_labels)[0]) - 0
    yMax = np.max(np.nonzero(_labels)[0]) + 0
    xMin = np.min(np.nonzero(_labels)[1]) - 0
    xMax = np.max(np.nonzero(_labels)[1]) + 0

    rec = np.array([[xMin, yMax], [xMin, yMin], [xMax, yMin], [xMax, yMax], [xMin, yMax]])

    _img = _img[yMin:yMax + 1, xMin:xMax + 1]
    _mask = _mask[yMin:yMax + 1, xMin:xMax + 1]
    _seg = _seg[yMin:yMax + 1, xMin:xMax + 1]
    _peri = _peri[yMin:yMax + 1, xMin:xMax + 1]

    return _img, _mask, _seg, _peri, [yMin, xMin, _img.shape[0], _img.shape[1]]

def bwperim(bImg):
    # print('Original Size', bImg.shape)
    core = bImg[1:-1, 1:-1]

    neighbors = np.zeros(core.shape, np.uint8)
    for i in range(3):
        for j in range(3):
            _cur = bImg[i:i+core.shape[0], j:j+core.shape[1]]
            if i == 1 and j == 1:
                continue
            neighbors += _cur

    _edge = core & (neighbors < 8)
    _blank = np.zeros((_edge.shape[0] + 2, _edge.shape[1] + 2), dtype = _edge.dtype)
    _blank[1:-1, 1:-1] = _edge
    return _blank

def makeMask(R, Corner):
    background = np.ones((R, R), np.uint8)
    for i in range(R):
        _r = np.min([i, R - 1 - i])
        if _r > Corner:
            continue
        for j in range(R):
            _c = np.min([j, R - 1 - j])
            if _c > Corner:
                continue
            if _r + _c < Corner:
                background[i, j] = 0
    return background

def disk(r):
    if r == 8:
        return makeMask(15, 4)
    elif r == 7:
        return makeMask(13, 2)
    elif r == 6:
        return makeMask(11, 2)
    elif r == 5:
        return makeMask(9, 2)
    elif r == 4:
        return makeMask(7, 2)
    elif r == 3:
        return makeMask(5, 0)
    elif r == 2:
        return makeMask(5, 2)
    elif r == 1:
        return makeMask(3, 1)

    else:
        sys.exit("Too large disk")

def readNii(fileName):
    nfile = nib.load(fileName)
    ndata = nfile.get_fdata()
    pixdim = nfile.header["pixdim"]

    _x = nfile.header["qoffset_x"]
    _y = nfile.header["qoffset_y"]
    _z = nfile.header["qoffset_z"]
    _dim = nfile.header["dim"]
    _affine = nfile.affine
    return ndata.astype(np.uint16), pixdim[1:4], np.asarray([_x, _y, _z]), _affine

def saveNii(imgName, volume, affine):
    nim = nib.Nifti1Image(volume, affine)
    nib.save(nim, imgName)
    print("Save____", imgName)

def pft_RealignImages(img, seg, shift):
    if seg.shape[2] > img.shape[2] and shift < 0:
        seg = seg[:, :, -shift:-shift + img.shape[2], :]
    return img, seg

def pft_ExtractMatchAndShiftedImages(_img, _seg, sliceNum, miniPixelCounts=64):
    bloodPool = 1
    myocardium = 2

    # Conditions
    Conditions = ['OK' for i in range(_seg.shape[2])]
    BinaryMask = np.zeros(_seg.shape, np.bool)
    PerimeterStack = np.zeros(_seg.shape, np.bool)
    PerimeterFound = np.zeros(_seg.shape[2], np.bool)
    MaxRad = 8

    for i in range(_img.shape[2]):
        if i != sliceNum:
            continue

        _pool = _seg[:, :, i] == bloodPool
        _wall = _seg[:, :, i] == myocardium

        if np.sum(_pool) < miniPixelCounts:
            Conditions[i] = 'Meagre blood pool'
            continue
        else:
            for Radius in range(MaxRad, 0, -1):
                _mask = disk(Radius)
                _area = dilation(_pool, selem=_mask)
                _guess = _area ^ _pool
                _edge = bwperim(_area)
                _out = ~_wall & _guess

                if np.sum(_out) == 0:
                    PerimeterFound[i] = True
                    BinaryMask[:, :, i] = _area
                    PerimeterStack[:, :, i] = _edge
                    break

            if PerimeterFound[i] == False:
                Conditions[i] = 'Meagre blood pool'

    return BinaryMask, PerimeterStack, Conditions

def edgeImage2Pts(edge, cropSize, scale):
    rc = np.nonzero(edge > 0)
    r = rc[0].tolist()
    c = rc[1].tolist()
    nrc = []
    for i in range(len(r)):
        _r = r[i] / scale
        _c = c[i] / scale
        _rc = [cropSize[0] + _r, cropSize[1] + _c]
        nrc.append(_rc)
    return nrc

def sortFiles(targetFolder, pBar):
    iter = 0
    metaMatrix = []
    for root, subdirs, files in os.walk(targetFolder):
        # print( files)
        for filename in files:
            curFile = os.path.join(root, filename)
            iter += 1
            if iter % 10 == 0:
                pBar.setValue(iter // 10 % 100)
                QApplication.processEvents()
            try:
                ds = dicom.read_file(curFile)
                try:
                    imageTime = ds[0x8, 0x33].value
                except KeyError:
                    print(filename, 'does not have [0008, 0033]')
                    imageTime = 0
                # imageNum = ds[0x18, 0x1090].value
                # seriesNum = ds[0x20, 0x0011].value
                try:
                    protocol = ds[0x18, 0x1030].value
                except KeyError:
                    print(filename, 'does not have protocol')
                    continue
                studyUID = ds[0x20, 0xD].value
                seriesUID = ds[0x20, 0xE].value
                metaMatrix.append([curFile, protocol, imageTime, seriesUID, studyUID])
            except:
                print(filename, ' is not DICOM file')

    pdMatrix = pd.DataFrame(metaMatrix, columns=["Name", "Protocol", "Time", "Series", "UID"])

    del pdMatrix["UID"]
    del pdMatrix["Series"]

    pdMatrix.sort_values(by=["Time", "Protocol"], inplace=True, ascending=True)
    pdMatrix.index = np.arange(pdMatrix.shape[0])
    return pdMatrix

def dcm2nii(selection):
    table = []
    for index, row in selection.iterrows():
        _name = row["Name"]
        _pixels, _slice, _time, _geo, _uid = readDCM(_name)
        table.append([_pixels, _slice, _time, _uid, _name, _geo])

    table = sorted(table, key=operator.itemgetter(1, 2), reverse=False)

    xdir = [row[-1][1] for row in table[1:] if np.dot(row[-1][1], table[0][-1][1]) < 0.99]
    ydir = [row[-1][2] for row in table[1:] if np.dot(row[-1][2], table[0][-1][2]) < 0.99]

    if len(xdir) > 0 or len(ydir) > 0:
        return []

    xdir = table[0][-1][1]
    ydir = table[0][-1][2]
    zdir = np.cross(xdir, ydir)
    first_loc = table[0][-1][0]
    last_loc = table[-1][-1][0]
    del_z = np.dot(last_loc - first_loc, zdir)

    # Frame rate
    slices = [row[1] for row in table]
    single = []
    for _s in slices:
        if not _s in single:
            single.append(_s)

    framenum = len(slices) / len(single)
    if len(slices) % len(single) != 0:
        print("What's wrong?")

    frTable = []
    for _s in single:
        # print(np.sum(np.asarray(slices) == _s))
        frTable.append([_s, np.sum(np.asarray(slices) == _s)])
    # print(frTable)

    dx = table[0][-1][3]
    dy = table[0][-1][4]
    dz = np.abs(del_z) / (len(frTable) - 1)

    if del_z > 0:
        corner = first_loc
        zdir = (last_loc - first_loc) / np.linalg.norm(last_loc - first_loc)
        order = 1
    else:
        corner = last_loc
        zdir = (first_loc - last_loc) / np.linalg.norm(first_loc - last_loc)
        order = -1

    affine = np.eye(4)
    affine[:3, 0] = xdir * dx
    affine[:3, 1] = ydir * dy
    affine[:3, 2] = zdir * dz
    affine[:3, 3] = corner

    X, Y = table[0][0].shape

    # calculate frameNum
    originlist = [round(k[-1][0][-1], 2) for k in table]
    slicenum = len(set(originlist))
    framenum = len(originlist) // slicenum

    Z = len(table) // framenum
    T = framenum
    volume = np.zeros((Y, X, Z, T), dtype=np.uint16)
    for k, _pixels in enumerate([row[0] for row in table][::order]):
        volume[:, :, k // T, k % T] = np.rot90(_pixels, 3)  # .transpose()
    
    return volume, [dy, dx, dz], corner, affine, [[row[-2] for row in table][::order], [row[-3] for row in table][::order], [row[0] for row in table][::order]]
