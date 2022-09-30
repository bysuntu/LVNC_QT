import sys
sys.path.append('.')
from PyQt5 import QtWidgets
from simple import Ui_MainWindow
import sys
from scipy.interpolate import interp1d, CubicSpline
# from matplotlib import pyplot as plt

from PyQt5.QtGui import QPen, qRgb, QPolygon, QBrush, QColor, QPainter, QIcon, QPixmap, QImage, QFont, QKeySequence
from PyQt5.QtCore import Qt, QPoint, QSize
from skimage import draw
# from matplotlib import image as matImage

from functions import *
from pft_InterpolateImage import *
from pft_JC_FractalDimensionCalculation import *
from matplotlib import pyplot as plt
import time
from skimage.draw import polygon2mask

bloodPool = 1
myocardium = 2

square = 700

class myQLabel(QtWidgets.QLabel):
    def __init__(self, parent, segData):
        super().__init__(parent)
        self.track = []
        self.manualDraw = 0
        self.over = False

        self.setMouseTracking(True)

        # Linkage
        self.circleSize = parent.ui.sizeSlider.value()
        self.opacity = parent.ui.opacity.value()
        self.paintOver = parent.ui.paintOver.currentText().lower()
        self.activeLayer = 'null'

        self.edge = []
        self.drawPts = []
        self.contours = []
        
        self.zoomScale = parent.zoomScale
        self.zoomGeo = parent.zoomGeo
        
        self.clearpts = QtWidgets.QShortcut(QKeySequence('Delete'), self)
        self.clearpts.activated.connect(
            lambda: self.deletePts(self.drawPts))
        
        
    def deletePts(self, path):
        self.drawPts=[]
        self.contours=[]
        
        
    def pts2Contours(self):
        if len(self.drawPts) >= 3: # Process
            localLoop = [k for k in self.drawPts]
            front2 = localLoop[-2]
            front = localLoop[-1]
            back = localLoop[0]
            back2 = localLoop[1]
            localLoop.insert(0, front)
            localLoop.insert(0, front2)
            localLoop.append(back)
            localLoop.append(back2)

            ptNum = len(localLoop) - 4
        
            localLoop = np.array(localLoop)
            original = np.linspace(-2, ptNum + 1, ptNum + 4)
            dense = np.linspace(0, ptNum, (ptNum) * 10 + 1)
        
            csX = interp1d(original, localLoop[:, 0], 'quadratic')
            csY = interp1d(original, localLoop[:, 1], 'quadratic')
            newx, newy = csX(dense), csY(dense)
            self.contours = []
            for i in range(len(newx)):
                self.contours.append(QPoint(newy[i], newx[i]))


    def mousePressEvent(self, event):
        super(myQLabel, self).mousePressEvent(event)
        self.lastpoint = event.pos()

        if event.buttons() == Qt.LeftButton and self.manualDraw == 2:
            self.drawPts.append([event.y(), event.x()])
            self.pts2Contours()
            
        if event.buttons() == Qt.MiddleButton and self.manualDraw == 2 and len(self.contours) >= 3:

            wS = self.pixmap().width() / self.segmentation.shape[1]
            hS = self.pixmap().height() / self.segmentation.shape[0]
            
            localLoop = []
            for pt_ in self.contours:
                localLoop.append([pt_.y() / hS, pt_.x() / wS])
            drawMask = polygon2mask(self.segmentation.shape[:-1], np.array(localLoop))

            if self.paintOver == 'visible labels':
                _mask = self.segmentation[:, :, 3] == self.opacity
            elif self.paintOver == 'all labels':
                _mask = self.segmentation[:, :, 3] >= 0
            elif self.paintOver == 'clear labels':
                _mask = self.segmentation[:, :, 3] == 0
            elif self.paintOver == 'blood pool':
                _mask = (self.segmentation[:, :, 0] > 0) * (self.segmentation[:, :, 1] == 0)
            elif self.paintOver == 'myocardium':
                _mask = (self.segmentation[:, :, 1] > 0) * (self.segmentation[:, :, 0] == 0)
            elif self.paintOver == 'right ventricle':
                _mask = (self.segmentation[:, :, 0] > 0) * (self.segmentation[:, :, 1] > 0)
            else:
                sys.exit("Wrong Paint Over")
                
            rr, cc = np.nonzero(drawMask * _mask)
                
            if self.activeLayer == 'blood pool':
                self.segmentation[rr, cc, 0] = 255
                self.segmentation[rr, cc, 1] = 0
                self.segmentation[rr, cc, 3] = self.opacity
            elif self.activeLayer == 'myocardium':
                self.segmentation[rr, cc, 0] = 0
                self.segmentation[rr, cc, 1] = 255
                self.segmentation[rr, cc, 3] = self.opacity
            elif self.activeLayer == 'right ventricle':
                self.segmentation[rr, cc, 0] = 255
                self.segmentation[rr, cc, 1] = 255
                self.segmentation[rr, cc, 3] = self.opacity
            elif self.activeLayer == 'clear labels':
                self.segmentation[rr, cc] = 0
            else:
                pass
                
            self.drawPts = []
            self.contours = []
        
    '''
    def wheelEvent(self, QWheelEvent):
        self.zoomScale += QWheelEvent.angleDelta().y() / 120. * 0.1
        self.zoomScale = min(max(self.zoomScale, 1.), 5)
        print(self.zoomScale, self.oriData.shape[:2], self.oriData.shape[0] / self.zoomScale)
        
        if QWheelEvent.modifiers() & Qt.ControlModifier:
            print(self.lastpoint.x())
            if QWheelEvent.angleDelta().y() > 0:
                print('Up')
            else:
                print('Down')
    '''
    
    def mouseDoubleClickEvent(self, event):
        self.drawPts = []
        self.contours = []

    def mouseMoveEvent(self, event):
        super(myQLabel, self).mouseMoveEvent(event)
        
        if event.buttons() == Qt.RightButton and self.manualDraw == 2:
            if len(self.drawPts) >= 3:
                x_ = np.array([row[0] for row in self.drawPts]) - event.y() 
                y_ = np.array([row[1] for row in self.drawPts]) - event.x()
                dis_ = np.sqrt(x_ * x_ + y_ * y_)
                if np.min(dis_) < 10.:
                    min_ = np.argmin(dis_)
                    self.drawPts[min_] = [event.y(), event.x()]
                    self.pts2Contours()
        

        self.over = True
        self.lastpoint = event.pos()
        self.track.append(QPoint(event.pos()))
        if event.buttons() == Qt.LeftButton and self.manualDraw == 1:
            wS = self.pixmap().width() / self.segmentation.shape[1]
            hS = self.pixmap().height() / self.segmentation.shape[0]
            rr, cc = draw.circle(event.y(), event.x(), self.circleSize)

            if self.paintOver == 'visible labels':
                _mask = self.segmentation[:, :, 3] == self.opacity
            elif self.paintOver == 'all labels':
                _mask = self.segmentation[:, :, 3] >= 0
            elif self.paintOver == 'clear labels':
                _mask = self.segmentation[:, :, 3] == 0
            elif self.paintOver == 'blood pool':
                _mask = (self.segmentation[:, :, 0] > 0) * (self.segmentation[:, :, 1] == 0)
            elif self.paintOver == 'myocardium':
                _mask = (self.segmentation[:, :, 1] > 0) * (self.segmentation[:, :, 0] == 0)
            elif self.paintOver == 'right ventricle':
                _mask = (self.segmentation[:, :, 0] > 0) * (self.segmentation[:, :, 1] > 0)
            else:
                sys.exit("Wrong Paint Over")

            rc = []
            for k in range(len(rr)):
                _rc = [np.max([0, np.min([int(rr[k] / hS), self.segmentation.shape[0] - 1])]),
                       np.max([0, np.min([int(cc[k] / wS), self.segmentation.shape[1] - 1])])]
                if not _rc in rc and _mask[_rc[0], _rc[1]]:
                    rc.append(_rc)
            rr = [row[0] for row in rc]
            cc = [row[1] for row in rc]

            if self.activeLayer == 'blood pool':
                self.segmentation[rr, cc, 0] = 255
                self.segmentation[rr, cc, 1] = 0
                self.segmentation[rr, cc, 3] = self.opacity
            elif self.activeLayer == 'myocardium':
                self.segmentation[rr, cc, 0] = 0
                self.segmentation[rr, cc, 1] = 255
                self.segmentation[rr, cc, 3] = self.opacity
            elif self.activeLayer == 'right ventricle':
                self.segmentation[rr, cc, 0] = 255
                self.segmentation[rr, cc, 1] = 255
                self.segmentation[rr, cc, 3] = self.opacity
            elif self.activeLayer == 'clear labels':
                self.segmentation[rr, cc] = 0
            else:
                pass

    def leaveEvent(self, event):
        self.over = False

    def paintEvent(self, paintEvent):
        super(myQLabel, self).paintEvent(paintEvent)
        painter = QPainter(self)
        if self.activeLayer == 'blood pool':
            painterColor = QColor(255, 0, 0, 125)
        elif self.activeLayer == 'myocardium':
            painterColor = QColor(0, 255, 0, 125)
        elif self.activeLayer == 'right ventricle':
            painterColor = QColor(255, 255, 0, 125)
        elif self.activeLayer == 'clear labels':
            painterColor = QColor(0, 0, 0, 125)
        else:
            painterColor = QColor(0, 0, 0, 0)
        painter.setBrush(QBrush(painterColor, Qt.SolidPattern))

        if self.over == True and self.manualDraw == 1:
            painter.drawEllipse(self.lastpoint, self.circleSize, self.circleSize)

        # painter.setCompositionMode(QPainter.CompositionMode_Source)
        _mask = QImage(self.segmentation.tobytes(), self.segmentation.shape[1], self.segmentation.shape[0],
                             self.segmentation.shape[1] * 4, QImage.Format_RGBA8888)
        painter.drawPixmap(0, 0, self.pixmap().width(), self.pixmap().height(), QPixmap.fromImage(_mask))

        pen = QPen()
        pen.setWidth(1)
        pen.setColor(QColor(255, 255, 0, 250))
        painter.setPen(pen)

        for p_ in self.drawPts:
            painter.drawEllipse(QPoint(p_[1], p_[0]), 4, 4)
        
        # DRAW
        if len(self.drawPts) >= 3:
            qtPoly_ = QPolygon(self.contours)
            painter.drawPolygon(qtPoly_)

        pen.setColor(QColor(255, 255, 255, 250))

        try:
            for _rc in self.edge:
                painter.drawPoint(_rc[1] * self.pixmap().width() / self.segmentation.shape[1],
                                  _rc[0] * self.pixmap().height() / self.segmentation.shape[0])
        except:
            pass
        
        try:
            painter.setPen(Qt.green)
            painter.setFont(QFont("time", 15))
            painter.drawText(20, 40, "Slice: {}".format(self.curSlice + 1))
            painter.drawText(20, 20, self.IC)
            painter.drawText(20, 60, "Value: {0:.4f}".format(self.text))
        except:
            pass
        self.update()


class nhcWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(nhcWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        curdir = os.path.join(os.path.abspath(os.path.curdir), 'simpleGUI_simple')
        
        print(os.path.join(curdir, 'icons', 'win.png'))

        # Main Window
        self.setWindowIcon(QIcon(os.path.join(curdir, 'icons', 'win.png')))
        self.setWindowTitle("LNVC")

        # Reset
        # self.ui.resetBtn.setIcon(QIcon('icons/reset.png'))

        # Load Data
        self.imgData = np.zeros((100, 100, 1, 1))
        self.segData = np.zeros((100, 100, 1, 1))
        
        self.oriData = np.copy(self.imgData)
        self.oriSegm = np.copy(self.segData)
        self.oriEdge = []

        # Save
        self.ui.saveBtn.setIcon(QIcon(os.path.join(curdir, 'icons', 'save.png')))
        self.ui.saveasBtn.setIcon(QIcon(os.path.join(curdir, 'icons', 'save-as.png')))
        self.ui.imgOrSeg.addItem(QIcon(os.path.join(curdir, 'icons', 'img.png')), "Image")
        self.ui.imgOrSeg.addItem(QIcon(os.path.join(curdir, 'icons', 'seg.png')), "Segmentation")
        self.ui.imgOrSeg.setCurrentIndex(1)

        # FIFO Frame
        self.ui.b1.setIcon(QIcon(os.path.join(curdir, 'icons', 'hand.png')))
        self.ui.b1.setIconSize(QSize(40,40))
        self.ui.b2.setIcon(QIcon(os.path.join(curdir, 'icons', 'circle.png')))
        self.ui.b2.setIconSize(QSize(40,40))
        self.ui.b3.setIcon(QIcon(os.path.join(curdir, 'icons', 'poly.png')))
        self.ui.b3.setIconSize(QSize(40, 40))
        self.ui.b4.setIcon(QIcon(os.path.join(curdir, 'icons', 'preview.png')))
        self.ui.b4.setIconSize(QSize(40, 40))

        self.source = "NIFTI"
        self.ui.niftiButton.toggled.connect(lambda:self.btnstate(self.ui.niftiButton))
        self.ui.dicomButton.toggled.connect(lambda:self.btnstate(self.ui.dicomButton))
        self.ui.batchButton.toggled.connect(lambda:self.btnstate(self.ui.batchButton))

        self.imgFile = ""
        self.segFile = ""
        self.batchDir = ""
        self.saveDir = ""

        self.ui.niiImage.clicked.connect(lambda:self.fileOpen(self.ui.niiImage, "Image nii (*.nii)"))
        self.ui.niiSeg.clicked.connect(lambda: self.fileOpen(self.ui.niiSeg, "Seg nii (*.nii)"))
        self.ui.dicomImage.clicked.connect(lambda :self.folderOpen(self.ui.dicomImage))
        self.ui.workspace.clicked.connect(lambda :self.fileOpen(self.ui.workspace, "Ascii workspace file (*.cvi42wsx)"))
        self.ui.batchButton.clicked.connect(lambda: self.folderOpen(self.ui.batchButton))
        self.ui.output.clicked.connect(lambda :self.folderOpen(self.ui.output))

        # Activate label
        self.ui.activateLabel.addItem(QIcon(os.path.join(curdir, 'icons', 'clear.png')), "Clear labels")
        self.ui.activateLabel.addItem(QIcon(os.path.join(curdir, 'icons', 'red.png')), "Blood pool")
        self.ui.activateLabel.addItem(QIcon(os.path.join(curdir, 'icons', 'green.png')), "Myocardium")
        self.ui.activateLabel.addItem(QIcon(os.path.join(curdir, 'icons', 'yellow.png')), "Right ventricle")
        self.ui.activateLabel.setCurrentIndex(1)

        self.ui.paintOver.addItem(QIcon(os.path.join(curdir, 'icons', 'all.png')), "All labels")
        self.ui.paintOver.addItem(QIcon(os.path.join(curdir, 'icons', 'visible.png')), "Visible labels")
        self.ui.paintOver.addItem(QIcon(os.path.join(curdir, 'icons', 'clear.png')), "Clear labels")
        self.ui.paintOver.addItem(QIcon(os.path.join(curdir, 'icons', 'red.png')), "Blood pool")
        self.ui.paintOver.addItem(QIcon(os.path.join(curdir, 'icons', 'green.png')), "Myocardium")
        self.ui.paintOver.addItem(QIcon(os.path.join(curdir, 'icons', 'yellow.png')), "Right ventricle")

        self.ui.sizeSlider.setMinimum(1)
        self.ui.sizeSlider.setMaximum(30)
        self.ui.sizeSlider.setValue(15)
        self.ui.sizeSlider.valueChanged.connect(lambda:self.sizeFun(self.ui.sizeSlider, self.ui.sizeBox))
        self.ui.sizeBox.setMinimum(1)
        self.ui.sizeBox.setMaximum(30)
        self.ui.sizeBox.setValue(15)
        self.ui.sizeBox.valueChanged.connect(lambda :self.sizeFun(self.ui.sizeBox, self.ui.sizeSlider))

        self.ui.sliceSlider.setMinimum(0)
        self.ui.sliceSlider.setMaximum(self.imgData.shape[2] - 1)
        self.ui.sliceSlider.setValue(0)
        self.ui.frameSlider.setMinimum(0)
        self.ui.frameSlider.setMaximum(0)
        self.ui.opacity.setMinimum(1)
        self.ui.opacity.setMaximum(255)
        self.ui.opacity.setValue(100)

        # MYQLABEL
        imgHeight, imgWidth, _, _ = self.imgData.shape
        if imgHeight > imgWidth:
            labelHeight, labelWidth = [700, round(700./(imgHeight) * imgWidth)]
        else:
            labelHeight, labelWidth = [round(700. / imgWidth * imgHeight), 700]

        _w_scale = labelWidth / (1. * imgWidth)
        _h_scale = labelHeight / (1. * imgHeight)
        _scale = round(np.min([_w_scale, _h_scale]) * imgWidth)
        _scale = labelWidth
        
        self.ui.scaleScroll.valueChanged.connect(lambda:self.zoomImage(self.ui.scaleScroll))
        self.zoomScale = 1.
        self.zoomGeo = [0, 0, self.imgData.shape[0], self.imgData.shape[1]]

        self.suQLabel = myQLabel(self, self.segData[:, :, self.ui.sliceSlider.value()])
        self.suQLabel.resize(labelWidth, labelHeight)
        self.suQLabel.move(240, 56)
        self.suQLabel.scale = _scale
        self.suQLabel.zoomScale = self.zoomScale
        self.suQLabel.zoomGeo = self.zoomGeo
        
        self.arrayToImage(self.suQLabel)
        
        self.ui.sliceSlider.valueChanged.connect(
            lambda :self.arrayToImage(self.suQLabel))
        self.ui.opacity.valueChanged.connect(
            lambda: self.arrayToImage(self.suQLabel))
        self.ui.frameSlider.valueChanged.connect(
            lambda:self.arrayToImage(self.suQLabel)
        )
        # ShortCuts
        self.sliceUp = QtWidgets.QShortcut(QKeySequence('Up'), self)
        self.sliceDown = QtWidgets.QShortcut(QKeySequence('Down'), self)
        self.frameLeft = QtWidgets.QShortcut(QKeySequence('Left'), self)
        self.frameRight = QtWidgets.QShortcut(QKeySequence('Right'), self)
        

        self.sliceUp.activated.connect(
            lambda :self.changeValues(self.ui.sliceSlider, self.sliceUp.key().toString()))
        self.sliceDown.activated.connect(
            lambda: self.changeValues(self.ui.sliceSlider, self.sliceDown.key().toString()))
        self.frameLeft.activated.connect(
            lambda: self.changeValues(self.ui.frameSlider, self.frameLeft.key().toString()))
        self.frameRight.activated.connect(
            lambda: self.changeValues(self.ui.frameSlider, self.frameRight.key().toString()))

        # PAINT
        self.ui.paintOver.currentIndexChanged.connect(
            lambda :self.pixelOperation(self.segData, self.suQLabel))
        self.ui.activateLabel.currentIndexChanged.connect(
            lambda :self.pixelOperation(self.segData, self.suQLabel))

        # button group
        self.ui.b1.setCheckable(True)
        self.ui.b2.setCheckable(True)
        self.ui.b3.setCheckable(True)
        self.ui.b4.setCheckable(True)

        self.ui.b1.setChecked(True)
        self.ui.b1.setFocus()
        self.ui.b1.clicked.connect(lambda: self.btngroup(self.ui.b1, self.suQLabel))
        self.ui.b2.clicked.connect(lambda: self.btngroup(self.ui.b2, self.suQLabel))
        self.ui.b3.clicked.connect(lambda: self.btngroup(self.ui.b3, self.suQLabel))
        self.ui.b4.clicked.connect(lambda: self.btngroup(self.ui.b4, self.suQLabel))
        self.ui.b4.clicked.connect(lambda :self.lvnc(self.suQLabel, self.ui.b4, self.ui.lvncBar))

        self.ui.loadFunBtn.clicked.connect(lambda: self.loadData(self.suQLabel))
        self.ui.Start.clicked.connect(lambda: self.lvnc(self.suQLabel, self.ui.Start, self.ui.progressBar))

        self.saveFileFilter = "seg_sa_ED.nii"
        self.ui.saveBtn.clicked.connect(lambda: self.fileOpen(self.ui.saveBtn, self.saveFileFilter))
        self.ui.saveasBtn.clicked.connect(lambda: self.fileOpen(self.ui.saveasBtn, self.saveFileFilter))
        
        self.outputDir = os.path.abspath(os.path.curdir)

        # Table
        for i in range(25):
            item = QtWidgets.QTableWidgetItem("NULL")
            self.ui.tableWidget.setItem(0, i, item)

        self.ui.exportBtn.clicked.connect(lambda: self.exportValue(self.ui.exportBtn))

        self.ui.resetBtn.clicked.connect(lambda: self.resetValues(self.suQLabel))
        
        self.show()
        
    def zoomImage(self, f):
        corR, corC, newR, newC = self.zoomGeo
        
        self.resetValues(self.suQLabel)
        '''
        print('Slice: ', self.suQLabel.curSlice)
        num = self.suQLabel.curSlice
        scale_ = self.oriData.shape[0] * 1. / newR
        try:
            temp_ = np.array(self.suQLabel.edge) / scale_ + np.array([corR, corC])
            self.edges[num] = temp_.tolist()
        except ValueError:
            pass
        '''
        self.arrayToImage(self.suQLabel)
        # Information exchanges between zoomed and original segmentation
        
        self.oriSegm[corR:corR + newR, corC:corC + newC] = self.segData
        

        # 
        self.zoomScale = (f.value() - 1) * 0.1 + 1

        oriR, oriC = self.oriData.shape[:2]
        corR = int((oriR - oriR / self.zoomScale) * 0.5)
        corC = int((oriC - oriC / self.zoomScale) * 0.5)
        newR = int(oriR / self.zoomScale)
        newC = int(oriC / self.zoomScale)
        self.imgData = self.oriData[corR:corR + newR, corC:corC + newC]
        
        self.zoomGeo = [corR, corC, newR, newC]
        
        # Information exchanges between zoomed and original segmentation
        self.segData = self.oriSegm[corR:corR + newR, corC:corC + newC]
        
        '''
        scale_ = self.oriData.shape[0] * 1. / newR
        temp_ = np.array(self.suQLabel.edge
        '''
        
        self.suQLabel.zoomScale = self.zoomScale
        self.suQLabel.zoomGeo = self.zoomGeo
        
        
        
        self.arrayToImage(self.suQLabel)
        

    def changeValues(self, f, text):
        v = f.value()
        max_ = f.maximum()
        min_ = f.minimum()
        if text == 'Up' or text == 'Right':
            v += 1
        if text == 'Down' or text == 'Left':
            v -= 1
        nV = np.min([np.max([v, min_]), max_])
        f.setValue(nV)
        self.arrayToImage(self.suQLabel)

    def resetValues(self, imgLabel):
        # Table
        for i in range(25):
            if i < self.imgData.shape[2] or i >= 20:
                item = QtWidgets.QTableWidgetItem("Nan")
            else:
                item = QtWidgets.QTableWidgetItem("NULL")
            self.ui.tableWidget.setItem(0, i, item)

        self.edges = [[] for k in range(self.imgData.shape[2])]
        self.texts = [[] for k in range(self.imgData.shape[2])]
        imgLabel.update()
        self.arrayToImage(imgLabel)       
        
    def exportValue(self, f, filterName = '*.csv'):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
        "{};;All Files (*)".format(filterName), options=options)
        
        if fileName.find('.csv') < 0:
            fileName += '.csv'

        with open(fileName, 'w') as f:
            for i in range(20):
                f.write('Slice {},'.format(str(i + 1)))
            f.write('Mean global FD, Mean basal FD, Mean apical FD, Max. basal FD, Max. apical FD\n')
            for i in range(25):
                _text = self.ui.tableWidget.item(0, i).text()
                f.write('{},'.format(_text))
            f.write('\n')

    def singleCase(self, imgLabel, btn, pBar):
        MeagreBloodPool = -111
        SparseMyocardium = -222
        NoROICreated = -333
        FDMeasureFailed = 0.0  # % Signal that an attempt was made, but failed - this will be excluded from the FD statistics

        pBar.setValue(1)
        QApplication.processEvents()

        try:
            if not os.path.isdir(self.outputDir):
                self.outputDir = os.path.abspath(os.path.curdir)
        except AttributeError:
            self.outputDir = os.path.abspath(os.path.curdir)

        outDir = self.outputDir
        pixelCount = int(self.ui.pixelCount.text())
        InterpolationType = self.ui.interpolation.currentText()

        try:
            OriginalResolution = self.imgDim[1]
        except AttributeError:
            return

        if InterpolationType == 'x4 cubic':
            OutputResolution = OriginalResolution / 4.
        elif InterpolationType == '0.25 mm cubic':
            OutputResolution = 0.25
        else:
            print("Wrong interpolation")
            sys.exit("___Wrong Interpolation_____")
        FD = np.zeros(self.imgData.shape[2])
        FractalDimensions = [[] for i in range(self.imgData.shape[2])]

        # Progress
        self.completed = 0
        pBar.setValue(self.completed)
        time.sleep(0.1)

        if btn.objectName() == "b4":
            slices = [self.ui.sliceSlider.value()]
        elif btn.objectName() == "Start":
            slices = np.arange(self.imgData.shape[2]).tolist()
        else:
            sys.exit("___Wrong__Button___")

        for _sliceNum in slices:
            QtWidgets.QApplication.processEvents()
            self.ui.sliceSlider.setValue(_sliceNum)
            self.arrayToImage(imgLabel)

            self.completed += 1
            pBar.setValue(round(self.completed / len(slices)))

            BinaryMask, PerimeterStack, Conditions = pft_ExtractMatchAndShiftedImages(
                self.imgData[:, :, :, self.ui.frameSlider.value()],
                self.segData[:, :, :, self.ui.frameSlider.value()], _sliceNum, pixelCount)

            for k, _c in enumerate(Conditions):
                if k != _sliceNum:
                    continue

                _img = self.imgData[:, :, k, self.ui.frameSlider.value()]
                _seg = self.segData[:, :, k, self.ui.frameSlider.value()]
                if _c == 'Meagre blood pool':
                    FD[k] = MeagreBloodPool
                    FractalDimensions[k].append('Meagre blood pool')
                elif _c == 'OK':
                    _mask = BinaryMask[:, :, k]
                    _peri = PerimeterStack[:, :, k]
                    _img, _mask, _seg, _peri, cropSize = cropImage(_img, _mask, _seg, _peri)
                    _before = _img.shape[0]
                    _img, _mask, _seg, _peri = pft_InterpolateImage(_img, _mask, _seg, _peri, OriginalResolution,
                                                                    InterpolationType)
                    # Progress
                    self.completed += 1
                    self.ui.lvncBar.setValue(self.completed)

                    _after = _img.shape[0]
                    FD[k], _edge, _thres = pft_JC_FractalDimensionCalculation(_img, _mask, k, pBar)
                    _rc = edgeImage2Pts(_edge, cropSize, _after / _before)
                    self.edges[_sliceNum] = _rc
                    imgLabel.edge = _rc

                    self.texts[_sliceNum] = FD[k]
                    imgLabel.text = FD[k]

                    '''
                    # Names
                    _imgName = 'Cropped-Image-Slice-{}-ED.png'.format(k + 1)
                    _maskName = 'Binary-Mask-Slice-{}-ED.png'.format(k + 1)
                    _periName = 'Perimeter-On-Segmentation-Slice-{}-ED.png'.format(k + 1)
                    _edgeName = 'Edge-Image-Slice-{}-ED.png'.format(k + 1)
                    _addName = 'Edge+Thres-Image-Slice-{}-ED.png'.format(k + 1)
                    matImage.imsave(os.path.join(outDir, _imgName), _img, cmap='gray')
                    matImage.imsave(os.path.join(outDir, _maskName), _mask, cmap='gray')
                    matImage.imsave(os.path.join(outDir, _periName), _seg + _peri * 3, cmap='gray')
                    matImage.imsave(os.path.join(outDir, _edgeName), _edge, cmap='gray')
                    _img[_thres == 0] = 0
                    _img[_edge] = 511
                    matImage.imsave(os.path.join(outDir, _addName), _img + 150 * _edge, cmap='gray')
                    '''

        self.ui.b4.setChecked(False)
        self.ui.b1.setChecked(True)
        self.ui.lvncBar.setValue(0)
        self.ui.progressBar.setValue(0)
        return Conditions, FD

    def lvnc(self, imgLabel, btn, pBar):
        if self.imgData.shape[2] < 2:
            return
        if len(self.batchDir) == 0:
            Conditions, FD = self.singleCase(imgLabel, btn, pBar)
            for i, _v in enumerate(FD):
                if _v > 0.1:
                    _item = QtWidgets.QTableWidgetItem("{:2.4f}".format(_v))
                    self.ui.tableWidget.setItem(0, i, _item)
                else:
                    FD[i] = np.nan
            self.ui.tableWidget.setItem(0, 20, QtWidgets.QTableWidgetItem("{:2.4f}".format(np.nanmean(FD))))
            self.ui.tableWidget.setItem(0, 22, QtWidgets.QTableWidgetItem("{:2.4f}".format(np.nanmean(FD[:len(FD)//2]))))
            self.ui.tableWidget.setItem(0, 21, QtWidgets.QTableWidgetItem("{:2.4f}".format(np.nanmean(FD[len(FD)//2:]))))
            self.ui.tableWidget.setItem(0, 23, QtWidgets.QTableWidgetItem("{:2.4f}".format(np.nanmax(FD[len(FD)//2:]))))
            self.ui.tableWidget.setItem(0, 24, QtWidgets.QTableWidgetItem("{:2.4f}".format(np.nanmax(FD[:len(FD)//2]))))
            
            # np.savetxt(os.path.join(self.outputDir, 'values.txt'), FD, delimiter=',')
        elif len(self.batchDir) > 0 and btn.objectName() == 'Start':
            rootdir = self.outputDir
            for _case in os.listdir(self.batchDir):

                _imgname = os.path.join(self.batchDir, _case, 'sa_ED.nii')
                _segname = os.path.join(self.batchDir, _case, 'seg_sa_ED.nii')

                if os.path.isfile(_imgname) and os.path.isfile(_segname):
                    self.imgFile = _imgname
                    self.segFile = _segname
                    self.outputDir = os.path.join(rootdir, _case)
                    if not os.path.isdir(self.outputDir):
                        os.mkdir(self.outputDir)
                    self.loadData(imgLabel)
                    Conditions, FD = self.singleCase(imgLabel, btn, pBar)
                    # np.savetxt(os.path.join(self.outputDir, 'values.txt'), FD, delimiter=',')
        else:
            pass

    def pixelOperation(self, segData, imgLabel):
        _type = self.ui.activateLabel.currentText().lower()
        if self.ui.b1.isChecked() or self.ui.b4.isChecked():
            _type = 'null'
        imgLabel.activeLayer = _type
        imgLabel.paintOver = self.ui.paintOver.currentText().lower()

    def fileOpen(self, f, filterName = "*.nii"):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        if f.objectName().lower().find('save') < 0:
            fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","{};;All Files (*)".format(filterName), options=options)
            if f.objectName().lower().find('seg') >= 0:
                self.segFile = fileName
            elif f.objectName().lower().find('image') >= 0:
                self.imgFile = fileName
            elif filterName.lower().find('cvi42wsx') >= 0:
                self.segFile = fileName
            self.batchDir = ""
        elif f.objectName().lower().find('saveas') >= 0:
            fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                                "{};;All Files (*)".format(filterName), options=options)
            if self.source == "DICOM":
                _img_name = os.path.join(fileName.replace('.nii', '') + '_img.nii')
                saveNii(_img_name, self.imgData, self.imgAffine)
                _seg_name = os.path.join(fileName.replace('.nii', '') + '_seg.nii')
                saveNii(_seg_name, self.segData, self.imgAffine)
            else:
                try:
                    saveNii(fileName, self.segData, self.imgAffine)
                except :
                    return
        else:
            if self.source == "DICOM":
                try:
                    _img_name = os.path.join(self.outputDir, '_img.nii')
                    saveNii(_img_name, self.imgData, self.imgAffine)
                    _seg_name = os.path.join(self.outputDir, '_seg.nii')
                    saveNii(_seg_name, self.segData, self.imgAffine)
                except :
                    return
            elif self.source == 'NIFTI':
                try:
                    saveNii(self.segFile, self.segData, self.imgAffine)
                except :
                    return
            else:
                return
            print('Image File: ', self.imgFile)

    def folderOpen(self, f):
        options = QtWidgets.QFileDialog.DontResolveSymlinks | QtWidgets.QFileDialog.ShowDirsOnly
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Open Folder", options=options)

        if f.text() == "Batch input folder ->" and f.isChecked():
            self.batchDir = directory
        if f.text() == "Image" and self.ui.dicomButton.isChecked():
            self.imgFile = directory
            _matrix = sortFiles(directory, self.ui.loadPBar)
            _protocol, _nums = [], []
            [_protocol.append(k) for k in _matrix["Protocol"].tolist() if not k in _protocol]
            _num = [np.sum(_matrix["Protocol"] == k) for k in _protocol]
            self.ui.protocols.clear()
            [self.ui.protocols.addItem("{} :: {}".format(_protocol[k], _num[k])) for k in range(len(_protocol)) if _num[k] > 100]
            self.matrix = _matrix
            self.ui.loadPBar.setValue(0)
            self.batchDir = ""
        if f.text() == 'Intermediate result folder ->':
            self.outputDir = directory

    def btnstate(self, b):
        if b.text() == 'NIFTI' and b.isChecked() == True:
            self.source = "NIFTI"
            b.setEnabled(True)
            self.ui.niiImage.setEnabled(True)
            self.ui.niiSeg.setEnabled(True)

            self.ui.dicomImage.setDisabled(True)
            self.ui.workspace.setDisabled(True)
            self.ui.protocols.setDisabled(True)
        if b.text() == 'Dicom' and b.isChecked() == True:
            self.source = 'DICOM'
            b.setEnabled(True)
            self.ui.dicomImage.setEnabled(True)
            self.ui.workspace.setEnabled(True)
            self.ui.protocols.setEnabled(True)

            self.ui.niiImage.setDisabled(True)
            self.ui.niiSeg.setDisabled(True)
        if b.text() == 'Batch input folder ->' and b.isChecked() == True:
            self.source = "BATCH"
            self.ui.dicomImage.setDisabled(True)
            self.ui.workspace.setDisabled(True)
            self.ui.protocols.setDisabled(True)
            self.ui.niiImage.setDisabled(True)
            self.ui.niiSeg.setDisabled(True)

    def btngroup(self, b, imgLabel):
        
        if b.objectName() == 'b3':
            imgLabel.activeLayer = self.ui.activateLabel.currentText().lower()
            imgLabel.manualDraw = 2
        elif b.objectName() == 'b2':
            imgLabel.activeLayer = self.ui.activateLabel.currentText().lower()
            imgLabel.manualDraw = 1
        else:
            imgLabel.activeLayer = 'null'
            imgLabel.manualDraw = 0
            
    def sizeFun(self, a, b):
        b.setValue(a.value())
        self.suQLabel.circleSize = a.value()


    def arrayToImage(self, imgLabel):
        num = self.ui.sliceSlider.value()
        fnum = self.ui.frameSlider.value()
        # ======Check======= #
        try:
            red = (imgLabel.segmentation[:, :, 0] > 0) * (imgLabel.segmentation[:, :, 1] == 0)
            green = (imgLabel.segmentation[:, :, 0] == 0) * (imgLabel.segmentation[:, :, 1] > 0)
            yellow = (imgLabel.segmentation[:, :, 0] > 0) * (imgLabel.segmentation[:, :, 1] > 0)
            background = imgLabel.segmentation[:, :, 3] == 0

            modifiedSeg = np.zeros((self.segData.shape[0], self.segData.shape[1]), self.segData.dtype)
            modifiedSeg[red] = 1
            modifiedSeg[green] = 2
            modifiedSeg[yellow] = 4
            modifiedSeg[background] = 0

            _dif = np.abs(modifiedSeg - self.segData[:, :, self.preSliceIndex,
                                        self.preFrameIndex])
            if np.sum(_dif) > 0:
                self.segData[:, :, self.preSliceIndex, self.preFrameIndex] = modifiedSeg
        except AttributeError:
            self.preSliceIndex = num
            self.preFrameIndex = fnum
        except IndexError:
            pass
        # ================== #

        _pixel = self.imgData[:, :, num, fnum].astype(np.float)
        if np.max(_pixel) > 5:
            _max = np.max(_pixel)
            _min = np.min(_pixel)
            _pixel = (_pixel - _min) / (_max - _min) * 255
        _pixel = _pixel.astype(np.uint8)

        # Array to Image
        _qimage = QImage(_pixel.tobytes(), _pixel.shape[1], _pixel.shape[0], _pixel.shape[1], QImage.Format_Grayscale8)
        _scale = imgLabel.scale
        imgLabel.setPixmap(QPixmap.fromImage(_qimage).scaledToWidth(_scale))
        segData = self.segData[:, :, num, fnum]

        segmentation = np.zeros((segData.shape[0], segData.shape[1], 4), dtype = np.uint8)
        for k in range(2):
            segmentation[:, :, k] = (segData == k + 1).astype(np.uint8) * 255

        segmentation[segData != 0, 3] = self.ui.opacity.value()
        segmentation[segData == 0, 3] = 0
        if np.max(segData) == 4:
            segmentation[segData == 4, 0] = 255
            segmentation[segData == 4, 1] = 255

        imgLabel.segmentation = segmentation
        self.preSliceIndex = num
        self.preFrameIndex = fnum
        imgLabel.curSlice = num
        try:
            imgLabel.edge = self.edges[num]
            imgLabel.text = self.texts[num]
        except (AttributeError, IndexError) as e:
            self.edges = [[] for k in range(self.imgData.shape[2])]
            imgLabel.edge = self.edges[num]
            self.texts = [[] for k in range(self.imgData.shape[2])]
            imgLabel.text = self.texts[num]
        imgLabel.show()

    def loadData(self, imgLabel):
        self.ui.frameSlider.setValue(0)
        self.ui.sliceSlider.setValue(0)
        self.ui.scaleScroll.setValue(1)

        filelist = []

        imgLabel.IC = ''
        self.IC = ''
        if os.path.isfile(self.imgFile):
            self.imgData, self.imgDim, self.imgOrigin, self.imgAffine = readNii(self.imgFile)
        elif os.path.isdir(self.imgFile):
            _selection = self.matrix[self.matrix["Protocol"] == self.ui.protocols.currentText().split(' :: ')[0]]
            if len(_selection) == 0:
                return
            ds = dicom.read_file(_selection["Name"].iloc[0])
            imgLabel.IC = ds[0x10, 0x20].value
            self.IC = imgLabel.IC
            info = dcm2nii(_selection)
            if info == []:
                return
            else:
                self.imgData, self.imgDim, self.imgOrigin, self.imgAffine, filelist = info

        else:
            return
        self.segData, self.segDim, self.segOrigin, self.segAffine = np.zeros(self.imgData.shape,
                                                                             dtype=self.imgData.dtype), self.imgDim, self.imgOrigin, self.imgAffine
                
        if os.path.isfile(self.segFile) and self.segFile.find('cvi42wsx') < 0:
            self.segData, self.segDim, self.segOrigin, self.segAffine = readNii(self.segFile)

        if os.path.isfile(self.segFile) and self.segFile.find('cvi42wsx') > 0 and filelist != []:
            self.segData = cvi2nii(self.segFile, filelist, self.imgData)

        if len(self.imgData.shape) == 3:
            self.imgData = np.expand_dims(self.imgData, axis=-1)
        if len(self.segData.shape) == 3:
            self.segData = np.expand_dims(self.segData, axis=-1)

        imgHeight, imgWidth, _, _ = self.imgData.shape
        if imgHeight > imgWidth:
            labelHeight, labelWidth = [700, round(700./(imgHeight) * imgWidth)]
        else:
            labelHeight, labelWidth = [round(700. / imgWidth * imgHeight), 700]

        _w_scale = labelWidth / (1. * imgWidth)
        _h_scale = labelHeight / (1. * imgHeight)
        _scale = round(np.min([_w_scale, _h_scale]) * imgWidth)
        _scale = labelWidth

        imgLabel.scale = _scale
        imgLabel.resize(labelWidth, labelHeight)

        # SHIFT
        _img = self.imgData
        _imgVoxelSize = self.imgDim
        _imgZ = self.imgOrigin[2]
        _seg = self.segData
        _segVoxelSize = self.segDim
        _segZ = self.segOrigin[2]
        _shift = np.round((_segZ - _imgZ) / _imgVoxelSize[2]).astype(np.int)

        self.imgData, self.segData = pft_RealignImages(_img, _seg, _shift)

        self.ui.sliceSlider.setMaximum(self.imgData.shape[2] - 1)
        self.ui.frameSlider.setMaximum(self.imgData.shape[3] - 1)

        self.edges = [[] for k in range(self.imgData.shape[2])]
        self.texts = [[] for k in range(self.imgData.shape[2])]

        imgLabel.edge = []
        imgLabel.text = []

        self.arrayToImage(imgLabel)

        # Table
        for i in range(25):
            if i < self.imgData.shape[2] or i >= 20:
                item = QtWidgets.QTableWidgetItem("Nan")
            else:
                item = QtWidgets.QTableWidgetItem("NULL")
            self.ui.tableWidget.setItem(0, i, item)
        
        self.oriData = self.imgData
        self.oriSegm = self.segData
        
        self.zoomScale = 1.
        self.zoomGeo = [0, 0, self.imgData.shape[0], self.imgData.shape[1]]
        
        imgLabel.zoomScale = self.zoomScale
        imgLabel.zoomGeo = self.zoomGeo
        
        imgLabel.update()


app = QtWidgets.QApplication([])
window = nhcWindow()
window.show()
sys.exit(app.exec_())