# -*- coding: utf-8 -*-
import numpy as np
import cv2
#from matplotlib import pyplot as plt
from os.path import dirname, join, basename
import sys
from glob import glob


bin_n = 16*16 # Number of bins

def hog(img):
    x_pixel,y_pixel=194,259
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:x_pixel/2,:y_pixel/2], bins[x_pixel/2:,:y_pixel/2], bins[:x_pixel/2,y_pixel/2:], bins[x_pixel/2:,y_pixel/2:]
    mag_cells = mag[:x_pixel/2,:y_pixel/2], mag[x_pixel/2:,:y_pixel/2], mag[:x_pixel/2,y_pixel/2:], mag[x_pixel/2:,y_pixel/2:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
#    print hist.shape
#    print type(hist)
    return hist

#print glob(join(dirname(__file__)+'/cat','*.jpg'))
img={}
num=0
for fn in glob(join(dirname(__file__)+'/cat', '*.jpg')):
    img[num] = cv2.imread(fn,0)#参数加0，只读取黑白数据，去掉0，就是彩色读取。
#    print img[num].shape
    num=num+1
print num,' num'
print 'the file path is ', dirname(__file__)
positive=num
for fn in glob(join(dirname(__file__)+'/other', '*.jpg')):
    img[num] = cv2.imread(fn,0)#参数加0，只读取黑白数据，去掉0，就是彩色读取。
#    print img[num].shape
    num=num+1
print num,' num'
print positive,' positive'

trainpic=[]
for i in img:
#    print type(i)
    trainpic.append(img[i])

svm_params = dict( kernel_type = cv2.SVM_LINEAR,
                    svm_type = cv2.SVM_C_SVC,
                    C=2.67, gamma=5.383 )
#img = cv2.imread('02.jpg',0)
#hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
#print hist_full
#plt.plot(hist_full)
#plt.show()

#img1 = cv2.imread('02.jpg',0)
#temp=img[0].ravel()
#print temp
#print len(temp)
temp=hog(img[0])
print temp.shape

#hogdata = [map(hog,img[i]) for i in img]
hogdata = map(hog,trainpic)
print np.float32(hogdata).shape,' hogdata'
trainData = np.float32(hogdata).reshape(-1,bin_n*4)
print trainData.shape,' trainData'
responses = np.float32(np.repeat(1.0,trainData.shape[0])[:,np.newaxis])
responses[positive:trainData.shape[0]]=-1.0
print responses.shape,' responses'
print len(trainData)
print len(responses)
print type(trainData)

svm = cv2.SVM()
svm.train(trainData,responses, params=svm_params)
svm.save('svm_cat_data.dat')

