# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'demoUiMin.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!


from matplotlib import pyplot as plt
from skimage import data,color, exposure,feature,io
from math import sqrt
from skimage.color import rgb2gray
import skimage.exposure as imexp
from skimage.morphology import binary_opening,disk
from skimage.filters import gabor_kernel
from PIL import Image
from scipy import ndimage as ndi
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from PyQt5 import QtCore, QtGui, QtWidgets










patches = [];
patchesHsv = [];


whitePatches = [];
whitePatchesHsv = [];

blackPatches = [];
blackPatchesHsv = [];


lessionPatches = [];
lessionPatchesHsv = [];

    
for i in range(0,5):
    patches.append(np.array(Image.open('patches/'+str(i+1)+'.jpg')))
                      
for i in range(0,5):
    patchesHsv.append(color.rgb2hsv(patches[i]))
    
    
    
for i in range(0,3):
    whitePatches.append(np.array(Image.open('patches/white'+str(i+1)+'.jpg')))
                      
for i in range(0,3):
    whitePatchesHsv.append(color.rgb2hsv(whitePatches[i]))
    
    
for i in range(0,4):
    blackPatches.append(np.array(Image.open('patches/black'+str(i+1)+'.jpg')))
for i in range(0,4):
    blackPatchesHsv.append(color.rgb2hsv(blackPatches[i]))

    
    
for i in range(0,9):
    lessionPatches.append(np.array(Image.open('patches/lession'+str(i+1)+'.jpg')))
for i in range(0,9):
    lessionPatchesHsv.append(color.rgb2hsv(lessionPatches[i]))




def satThreshold(v,s):    ##return hue or intensity as dominant feature
    th = 1.0 - 0.8*v;
    if(s>th):
        return "h"
    else: 
        return "v"
        
        
        
        

def approximate(img):
    tmpImg = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            dominantVal = satThreshold(img[i,j,2],img[i,j,1])
#             print("dominantVal----"+dominantVal)
            tmpImg[i,j,:] = img[i,j,:]
            if dominantVal == "h":
                tmpImg[i,j,0] = img[i,j,0]
                tmpImg[i,j,1] = 1.0
                tmpImg[i,j,2] = 1.0
                
            else:
                tmpImg[i,j,0] = 1.0
                tmpImg[i,j,1] = 1.0
                tmpImg[i,j,2] = img[i,j,2]
        
    return tmpImg
    
    


###Gabor filter

def filterImage(image, kernels):
    filtered = []
    for k, kernel in enumerate(kernels):
        filtered.append(ndi.convolve(image, kernel, mode='wrap'))
    return filtered
    
    
    
    
    
def mass_compute_and_combine_feats(filteredImg,hsvImg,aprocHsvImg):
    
    coreFeatsLen = hsvImg.shape[2]+aprocHsvImg.shape[2]
    textTureFeatsLen = len(filteredImg)*3
    dim = textTureFeatsLen+coreFeatsLen
    feats = np.zeros((filteredImg[0].shape[0],filteredImg[0].shape[1],dim))
    step = 0
    
    for m_index,image in enumerate(filteredImg):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = image[max(i-1,0):min(i+1,image.shape[0]-1),max(j-1,0):min(j+1,image.shape[1]-1)]
                feats[i,j, step] = window.mean()
                feats[i,j, step+1] = window.var()
                feats[i,j, step+2] = image[i,j]
                if step+3 == textTureFeatsLen:
                    
                    feats[i,j, step+3] = hsvImg[i,j,0]
                    feats[i,j, step+4] = hsvImg[i,j,1]
                    feats[i,j, step+5] = hsvImg[i,j,2]
                    feats[i,j, step+6] = aprocHsvImg[i,j,0]
                    feats[i,j, step+7] = aprocHsvImg[i,j,1]
                    feats[i,j, step+8] = aprocHsvImg[i,j,2]
                
        step += 3
 
    return feats
    
    
def ref_img_feats():
    
#     print("ref images features")
    
    ref_feats = {}
   
    for i in range(len(patchesHsv)):
        imgToFilter = rgb2gray(patches[i])
        filtredImg =  filterImage(imgToFilter,kernels)
        ref_feats[i] = mass_compute_and_combine_feats(filtredImg,patchesHsv[i],approximate(patchesHsv[i]))
        ref_feats[i] = np.mean(ref_feats[i], axis=0) 
    return ref_feats
    
    
    
def ref_white_img_feats():
    
    ref_feats = {}
    
    for i in range(len(whitePatchesHsv)):
        imgToFilter = rgb2gray(whitePatches[i])
        filtredImg = filterImage(imgToFilter,kernels)
        ref_feats[i] = mass_compute_and_combine_feats(filtredImg,whitePatchesHsv[i],approximate(whitePatchesHsv[i]))
        ref_feats[i] = np.mean(ref_feats[i], axis=0)

    return ref_feats
    
    
    
    
    
    
def ref_black_img_feats():
    
    ref_feats = {}
    
    for i in range(len(blackPatchesHsv)):
        imgToFilter = rgb2gray(blackPatches[i])
        filtredImg = filterImage(imgToFilter,kernels)
        ref_feats[i] = mass_compute_and_combine_feats(filtredImg,blackPatchesHsv[i],approximate(blackPatchesHsv[i]))
        ref_feats[i] = np.mean(ref_feats[i], axis=0)

        
    return ref_feats
    
def ref_lession_img_feats():
    
    ref_feats = {}
    
    for i in range(len(lessionPatchesHsv)):
        imgToFilter = rgb2gray(lessionPatches[i])
        filtredImg = filterImage(imgToFilter,kernels)
        ref_feats[i] = mass_compute_and_combine_feats(filtredImg,lessionPatchesHsv[i],approximate(lessionPatchesHsv[i]))
        ref_feats[i] = np.mean(ref_feats[i], axis=0)
        
    return ref_feats
    





class K_Means_Feats:
    def __init__(self, k=10, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.rgbColors = np.array([[0,255,0],[255,255,255],[0,0,0],[255,0,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[51,51,255],[102,102,0],[255,0,127],[160,32,240],[238,130,238]])
        self.rgbColors.astype(float)
#        white, green, ,red, yellow, purple,violet



    
    
    def fit(self,img):

        self.centroids = {}
        

        p_index = len(patches)
        w_p_index = len(patches)+len(whitePatches)
        b_p_index = len(patches)+len(whitePatches)+len(blackPatches)

        ref_feats_n = ref_feats_normal
        ref_feats_w = ref_feats_white
        ref_feats_b = ref_feats_black
        ref_feats_l = ref_feats_lession
            
            
        
        print('indeces----')
        print p_index
        print w_p_index
#         print b_p_index
        
        #centroids for normal parts of the esophagus
        for i in range(len(patches)):
            self.centroids[i] = ref_feats_n[i][0]
        #centroids for white and bright parts
        for i in range(len(whitePatches)):
            self.centroids[p_index+i] = ref_feats_w[i][0]
        #centroids for black parts
        for i in range(len(blackPatches)):
            self.centroids[w_p_index+i] = ref_feats_b[i][0] 
            
        for i in range(len(lessionPatches)):
            self.centroids[b_p_index+i] = ref_feats_l[i][0]     
        #ramdomly select centroid for the lessions   
#         self.centroids[self.k-1] = img[0,0,:]
      
        print("k----")
        print(self.k)
        print("centroids----")
#         print(self.centroids)

        for i in range(self.max_iter):
            self.clusters = {}
            self.clustersIndeces = {}
                ##classes holder
            for i in range(self.k):
                self.clusters[i] = []
                self.clustersIndeces[i] = []
        
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    distances = [np.linalg.norm(img[i,j,:]-self.centroids[centroid]) for centroid in self.centroids]
                    clust_index = distances.index(min(distances))
                    self.clustersIndeces[clust_index].append([i,j])
                    self.clusters[clust_index].append(img[i,j,:])
                    
            prev_centroids = dict(self.centroids)         
            
            ##  re-assign centroids 
            for item in self.clusters:
                self.centroids[item] = np.average(self.clusters[item],axis=0)

            optimized = True
            
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/((original_centroid)*100.0)) > self.tol:
                    optimized = False

            if optimized:
                break
        
        output = np.zeros((img.shape[0],img.shape[1],3),np.uint8); 
        
##assigning colors  

        for cent in self.centroids:
            for pair in self.clustersIndeces[cent]:
                
                if  (cent < p_index): #normal pixels---assign green color
                    output[pair[0],pair[1],0] = self.rgbColors[0][0]
                    output[pair[0],pair[1],1] = self.rgbColors[0][1]
                    output[pair[0],pair[1],2] = self.rgbColors[0][2]
                    
                elif  (cent >= p_index) & (cent < w_p_index): #white pixels----assign white color
                    output[pair[0],pair[1],0] = self.rgbColors[1][0]
                    output[pair[0],pair[1],1] = self.rgbColors[1][1]
                    output[pair[0],pair[1],2] = self.rgbColors[1][2]
                    
                elif  (cent >= w_p_index) & (cent < b_p_index): # black pixels---assign black color
                    output[pair[0],pair[1],0] = self.rgbColors[2][0]
                    output[pair[0],pair[1],1] = self.rgbColors[2][1]
                    output[pair[0],pair[1],2] = self.rgbColors[2][2] 
                    
                elif  (cent >= b_p_index): #lessions------assign red color
                    output[pair[0],pair[1],0] = self.rgbColors[3][0]
                    output[pair[0],pair[1],1] = self.rgbColors[3][1]
                    output[pair[0],pair[1],2] = self.rgbColors[3][2] 
                    
        print "kmean done"
        return output
    
    
    def fit_and_filter(self,img):

        self.centroids = {}
        

        p_index = len(patches)
        w_p_index = len(patches)+len(whitePatches)
        b_p_index = len(patches)+len(whitePatches)+len(blackPatches)

        ref_feats_n = ref_feats_normal
        ref_feats_w = ref_feats_white
        ref_feats_b = ref_feats_black
        ref_feats_l = ref_feats_lession
            
            
        
        print('indeces----')
        print p_index
        print w_p_index
        
        for i in range(len(patches)):
            self.centroids[i] = ref_feats_n[i][0]
        #centroids for white and bright parts
        for i in range(len(whitePatches)):
            self.centroids[p_index+i] = ref_feats_w[i][0]
        #centroids for black parts
        for i in range(len(blackPatches)):
            self.centroids[w_p_index+i] = ref_feats_b[i][0] 
            
        for i in range(len(lessionPatches)):
            self.centroids[b_p_index+i] = ref_feats_l[i][0]     


        print("k----")
        print(self.k)
        print("centroids----")
            

        for i in range(self.max_iter):
            self.clusters = {}
            self.clustersIndeces = {}
                ##classes holder
            for i in range(self.k):
                self.clusters[i] = []
                self.clustersIndeces[i] = []
        
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    distances = [np.linalg.norm(img[i,j,:]-self.centroids[centroid]) for centroid in self.centroids]
                    clust_index = distances.index(min(distances))
                    self.clustersIndeces[clust_index].append([i,j])
                    self.clusters[clust_index].append(img[i,j,:])
                    
            prev_centroids = dict(self.centroids)         
            
            ##  re-assign centroids 
            for item in self.clusters:
                self.centroids[item] = np.average(self.clusters[item],axis=0)

            optimized = True
            
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/((original_centroid)*100.0)) > self.tol:
                    optimized = False

            if optimized:
                break
        
        output = np.zeros((img.shape[0],img.shape[1]),np.uint8); 
        
##assigning colors  

        for cent in self.centroids:
            for pair in self.clustersIndeces[cent]: 
                if  (cent >= b_p_index): #lessions------assign red color
                    output[pair[0],pair[1]] = 255

        
        print "kmean done"
                    
        return binary_opening(output, disk(3))    
        

def filterLession(img):
     lessions = img == [255,0,0]   
     return binary_opening(lessions, disk(3))    
     
     
     

class Ui_MainWindow(QtWidgets.QWidget):
    #def setupUi(self, MainWindow):
     #   MainWindow.setObjectName("MainWindow")
      #  MainWindow.resize(738, 596)
        
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(742, 645)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.OriginalImageBtn = QtWidgets.QPushButton(self.centralwidget)
        self.OriginalImageBtn.setObjectName("OriginalImageBtn")
        self.gridLayout.addWidget(self.OriginalImageBtn, 0, 0, 1, 1)
        self.OriginalImageLabel = QtWidgets.QLabel(self.centralwidget)
        self.OriginalImageLabel.setText("")
        self.OriginalImageLabel.setObjectName("OriginalImageLabel")
        self.gridLayout.addWidget(self.OriginalImageLabel, 1, 0, 1, 1)
        self.FinalOutputImageBtn = QtWidgets.QPushButton(self.centralwidget)
        self.FinalOutputImageBtn.setObjectName("FinalOutputImageBtn")
        self.gridLayout.addWidget(self.FinalOutputImageBtn, 0, 2, 1, 1)
        self.SegmentationImageLabel = QtWidgets.QLabel(self.centralwidget)
        self.SegmentationImageLabel.setText("")
        self.SegmentationImageLabel.setObjectName("SegmentationImageLabel")
        self.gridLayout.addWidget(self.SegmentationImageLabel, 1, 1, 1, 1)
        self.FinalOuputImageLabel = QtWidgets.QLabel(self.centralwidget)
        self.FinalOuputImageLabel.setText("")
        self.FinalOuputImageLabel.setObjectName("FinalOuputImageLabel")
        self.gridLayout.addWidget(self.FinalOuputImageLabel, 1, 2, 1, 1)
        self.SegmentationImageBtn = QtWidgets.QPushButton(self.centralwidget)
        self.SegmentationImageBtn.setObjectName("SegmentationImageBtn")
        self.gridLayout.addWidget(self.SegmentationImageBtn, 0, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 3, 2, 1, 1)
        self.PrepareRefImagesBtn = QtWidgets.QPushButton(self.centralwidget)
        self.PrepareRefImagesBtn.setObjectName("PrepareRefImagesBtn")
        self.gridLayout.addWidget(self.PrepareRefImagesBtn, 2, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 738, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        
        
        self.PrepareRefImagesBtn.clicked.connect(self.preperRefImages)
        self.OriginalImageBtn.clicked.connect(self.pickAndShowImage)
        self.SegmentationImageBtn.clicked.connect(self.showSegImage)
        self.FinalOutputImageBtn.clicked.connect(self.showFinalSegImage)
        
        
          
        self.originalImg = None
        self.segImg = None 
        self.finalOutPutImg = None 
        self.hsvImg = None 
        self.approxHsvImg = None 
        
        
        
        
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Esophageal Image Segmentation"))
        self.OriginalImageBtn.setText(_translate("MainWindow", "Choose Image"))
        self.FinalOutputImageBtn.setText(_translate("MainWindow", "Final Ouput"))
        self.SegmentationImageBtn.setText(_translate("MainWindow", "Segmentation"))
        self.PrepareRefImagesBtn.setText(_translate("MainWindow", "Prepare Reference Images"))

        
         
        
    def pickAndShowImage(self):
        
        name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File')
        print("filePath")
        if name[0] != '':
            img = QtGui.QPixmap(name[0])
            
            self.OriginalImageLabel.setPixmap(img.scaled(self.OriginalImageLabel.width(), self.OriginalImageLabel.height(),
                QtCore.Qt.KeepAspectRatio))
            self.OriginalImageLabel.setAlignment(QtCore.Qt.AlignCenter)
   
            #keep reference to original image
            self.originalImg = np.array(Image.open(name[0]))
            self.hsvImg = color.rgb2hsv(self.originalImg)
            self.approxHsvImg = approximate(self.hsvImg)
  
    def showSegImage(self):
        print "showSeg1Image--"
        imgToFilter = rgb2gray(self.originalImg)
        filtredImg = filterImage(imgToFilter,kernels)
        self.segImg = kmeanf.fit(mass_compute_and_combine_feats(filtredImg, self.hsvImg ,self.approxHsvImg))
        
        
        height, width, bytesPerComponent = self.segImg.shape
        bytesPerLine = bytesPerComponent * width
    
        img=QtGui.QImage(self.segImg, width, height,bytesPerLine,QtGui.QImage.Format_RGB888)
        print "Qimage img"
        print img
        

        self.SegmentationImageLabel.setPixmap(QtGui.QPixmap.fromImage(img).scaled(self.SegmentationImageLabel.width(), self.SegmentationImageLabel.height(),
                QtCore.Qt.KeepAspectRatio))
        self.SegmentationImageLabel.setAlignment(QtCore.Qt.AlignCenter)
     
    
    def showFinalSegImage(self):
        print "Final output--"
        imgToFilter = rgb2gray(self.originalImg)
        filtredImg = filterImage(imgToFilter,kernels)
        self.finalOutPutImg = kmeanf.fit_and_filter(mass_compute_and_combine_feats(filtredImg, self.hsvImg ,self.approxHsvImg))
        #self.finalOutPutImg = filterLession(self.segImg)
        
        height, width = self.finalOutPutImg.shape
        bytesPerLine = width
        
        
        print "self.finalOutPutImg"
        print self.finalOutPutImg
    
        img=QtGui.QImage(self.finalOutPutImg, width, height,bytesPerLine,QtGui.QImage.Format_Indexed8)
        print "Qimage img"
        print img
        

        self.FinalOuputImageLabel.setPixmap(QtGui.QPixmap.fromImage(img).scaled(self.FinalOuputImageLabel.width(), self.FinalOuputImageLabel.height(),
                QtCore.Qt.KeepAspectRatio))
        self.FinalOuputImageLabel.setAlignment(QtCore.Qt.AlignCenter)
        
        
    def preperRefImages(self):
            
        ##########
        ##the code below takes two min to run
        ##########
              
        # preparing Gabor filter bank kernels
        print "preparging"
        
            

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    
    
    
    
           # preparing Gabor filter bank kernels
    print "preparging ref images"
    kernels = []
    for theta in np.arange(0, np.pi, np.pi / 6):
        for sigma in (1.0,1.5):  #gausian kernel window size
            for frequency in (0.1, 0.2):
                kernel = np.real(gabor_kernel(frequency, theta=theta,sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
                
                
    ##prepare reference features
    ref_feats_normal = ref_img_feats()
    ref_feats_white = ref_white_img_feats()
    ref_feats_black = ref_black_img_feats()
    ref_feats_lession = ref_lession_img_feats() 
    
    
    # divide into white, black, normal and abonomal parts of the esophagus
    kmeanf = K_Means_Feats(len(patches)+len(whitePatches)+len(blackPatches)+len(lessionPatches))
    
    print "Done !!!"
    
    
    
    sys.exit(app.exec_())

