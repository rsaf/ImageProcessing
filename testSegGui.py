# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'testSegGui.ui'
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
from matplotlib import style


from PyQt5 import QtCore, QtGui, QtWidgets






onlineSamples = []
onlineHsvSamples = []

imgSamples = [];
imgHsvSamples = [];


patches = [];
patchesHsv = [];


whitePatches = [];
whitePatchesHsv = [];

blackPatches = [];
blackPatchesHsv = [];


lessionPatches = [];
lessionPatchesHsv = [];



for i in range(0,3):
    onlineSamples.append(np.array(Image.open('onlineSamples/textures/'+str(i+1)+'.jpg')))
                      
for i in range(0,3):
    onlineHsvSamples.append(color.rgb2hsv(onlineSamples[i]))

for i in range(0,5):
    imgSamples.append(np.array(Image.open('samples_croped/'+str(i+1)+'.jpg')))
                      
for i in range(0,5):
    imgHsvSamples.append(color.rgb2hsv(imgSamples[i]))
    
    
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


print(len(imgHsvSamples))  
print(len(onlineHsvSamples)) 
print(len(patchesHsv)) 
print(len(whitePatchesHsv)) 
print(len(blackPatchesHsv)) 
print(len(lessionPatches)) 
print(imgHsvSamples[0].shape)
print(onlineHsvSamples[0].shape)
print(patchesHsv[0].shape)
print(whitePatchesHsv[0].shape)
print(blackPatchesHsv[0].shape)
print(lessionPatchesHsv[0].shape)




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
                    
        return binary_opening(output, disk(3))    
        

        
        
# preparing Gabor filter bank kernels
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

colnum = 4
# rownum = int(math.floor((float(len(imgSamples))/ float(colnum))+1))
rownum = len(onlineSamples)
# print(rownum)


fig, axes = plt.subplots(rownum, colnum, figsize=(65, rownum*10), sharex=True, sharey=True)
ax = axes.ravel()
index = 0

# divide into white, black, normal and abonomal parts of the esophagus
kmeanf = K_Means_Feats(len(patches)+len(whitePatches)+len(blackPatches)+len(lessionPatches))

for i in range(0, len(onlineHsvSamples)):
    
    imgToFilter = rgb2gray(onlineSamples[i])
    
    filtredImg = filterImage(imgToFilter,kernels)
   
    
    print("len(filtredImg)")
    print(len(filtredImg))
    print(filtredImg[0].shape)
    
    ax[index].imshow(onlineSamples[i],cmap='gray')
    ax[index].set_title("original rgb "+str(i))
    ax[index].axis('off') 
   
    ax[index+1].imshow(approximate(onlineHsvSamples[i]),cmap='gray')
    ax[index+1].set_title("segmented "+str(i))
    ax[index+1].axis('off')

         
    ax[index+2].imshow(kmeanf.fit(mass_compute_and_combine_feats(filtredImg,onlineHsvSamples[i],approximate(onlineHsvSamples[i]))) ,cmap='gray')
    ax[index+2].set_title("both "+str(i))
    ax[index+2].axis('off')
    
    ax[index+3].imshow(kmeanf.fit_and_filter(mass_compute_and_combine_feats(filtredImg,onlineHsvSamples[i],approximate(onlineHsvSamples[i]))) ,cmap='gray')
    ax[index+3].set_title("both "+str(i))
    ax[index+3].axis('off')
    
    index += colnum
    
plt.show()




class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(794, 632)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.OriginalImgLabel = QtWidgets.QLabel(self.centralwidget)
        self.OriginalImgLabel.setObjectName("OriginalImgLabel")
        self.horizontalLayout_2.addWidget(self.OriginalImgLabel)
        self.HSVImgLabel = QtWidgets.QLabel(self.centralwidget)
        self.HSVImgLabel.setObjectName("HSVImgLabel")
        self.horizontalLayout_2.addWidget(self.HSVImgLabel)
        self.HSVApproxLabel = QtWidgets.QLabel(self.centralwidget)
        self.HSVApproxLabel.setObjectName("HSVApproxLabel")
        self.horizontalLayout_2.addWidget(self.HSVApproxLabel)
        self.Segmentation1Label = QtWidgets.QLabel(self.centralwidget)
        self.Segmentation1Label.setObjectName("Segmentation1Label")
        self.horizontalLayout_2.addWidget(self.Segmentation1Label)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 0, 0, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 1, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.OriginalImg = QtWidgets.QPushButton(self.centralwidget)
        self.OriginalImg.setObjectName("OriginalImg")
        self.horizontalLayout.addWidget(self.OriginalImg)
        self.HSVImg = QtWidgets.QPushButton(self.centralwidget)
        self.HSVImg.setObjectName("HSVImg")
        self.horizontalLayout.addWidget(self.HSVImg)
        self.HSVApprox = QtWidgets.QPushButton(self.centralwidget)
        self.HSVApprox.setObjectName("HSVApprox")
        self.horizontalLayout.addWidget(self.HSVApprox)
        self.Segmentation1 = QtWidgets.QPushButton(self.centralwidget)
        self.Segmentation1.setObjectName("Segmentation1")
        self.horizontalLayout.addWidget(self.Segmentation1)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 794, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.OriginalImg.clicked.connect(self.OriginalImgLabel.clear)
        self.HSVImg.clicked.connect(self.HSVImgLabel.clear)
        self.HSVApprox.clicked.connect(self.HSVApproxLabel.clear)
        self.Segmentation1.clicked.connect(self.Segmentation1Label.clear)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "EsophagusSeg"))
        self.OriginalImgLabel.setText(_translate("MainWindow", "TextLabel"))
        self.HSVImgLabel.setText(_translate("MainWindow", "TextLabel"))
        self.HSVApproxLabel.setText(_translate("MainWindow", "TextLabel"))
        self.Segmentation1Label.setText(_translate("MainWindow", "TextLabel"))
        self.OriginalImg.setText(_translate("MainWindow", "Original Image"))
        self.HSVImg.setText(_translate("MainWindow", "HSV"))
        self.HSVApprox.setText(_translate("MainWindow", "HSV-Approx"))
        self.Segmentation1.setText(_translate("MainWindow", "Segmentation1"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

