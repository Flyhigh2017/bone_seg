import numpy as np
import os
from os.path import isfile, join
from os import listdir
import scipy.misc as misc
import cv2
from PIL import Image
imagepath = "/test_images/1_1_9.jpg"
maskpath = "/output/Label/1_1_9.png"
#3-2-8  35 is the reference image
img = cv2.imread(maskpath)
img = img[:,:,2]
img = (img * 100)
section1 = img.copy()
section2 = img.copy()
#Spine
section1[section1 == 200] = 0
mask1 = cv2.medianBlur(section1,3)
mask1[mask1 != 0] = 1.5
#cord
section2[section2 == 100] = 0
section2 = cv2.GaussianBlur(section2,(3,3),0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
section2 = section2 * 255
section2 = cv2.erode(section2, kernel)
mask2 = cv2.medianBlur(section2,9)
mask2[mask2 != 0] = 1
#read raw image
orig = cv2.imread(imagepath,0) * 2
rgbimage = cv2.imread(imagepath)
#normalize all images
norm = 32.99
print "average:", np.average(orig)
ave = np.average(orig)
rate = norm / ave
print "rate:", rate
orig = (rate * orig.astype('float64')).astype('uint8')
#####process spine
spine = orig * mask1
cord = orig * mask2

ret,th2 = cv2.threshold(spine,35,255,cv2.THRESH_BINARY)#####

k2 = cv2.getStructuringElement(cv2.MORPH_RECT,(8, 8))
ks = cv2.getStructuringElement(cv2.MORPH_RECT,(29, 29))
th2 = cv2.erode(th2, k2)
spine1 = spine.copy()
spine1[spine1 >= 3] = 255
spine1 = cv2.erode(spine1, ks)
inbetween = spine1 - th2
inbetween[inbetween <= 3] = 0
cv2.imshow('bones',inbetween)
cv2.waitKey(0)
'''
spine1 = spine.copy()
spine2 = spine1.copy()
spine2[spine2 <= 2] = 255
spine1[spine1 <= 40] = 255

k3 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
inbetween = spine1 - spine2
inbetween = cv2.erode(inbetween,k3)
cv2.imshow('bones',inbetween)
cv2.waitKey(0)
'''
contours, hierarchy = cv2.findContours(th2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
contours_inbetween, hierarchy_inbetween = cv2.findContours(inbetween,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(rgbimage,contours,-1,(0,0,255),3)

cv2.drawContours(rgbimage,contours_inbetween,-1,(255,0,0),3)
####process cord
retcord,thcord = cv2.threshold(cord,10,255,cv2.THRESH_BINARY)
contours_cord, hierarchy_cord = cv2.findContours(thcord,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(rgbimage,contours_cord,-1,(0,255,0),3)
cv2.imshow('Image2',rgbimage)
cv2.waitKey(0)




