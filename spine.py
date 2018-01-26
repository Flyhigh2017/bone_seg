import numpy as np
import os
from os.path import isfile, join
from os import listdir
import scipy.misc as misc
import cv2
from PIL import Image
import pdb
import copy
from matplotlib import pyplot as plt
imagepath = "./test_images/1_1_9.jpg"
maskpath = "./Label/1_1_9.png"


def showimg(img):
	cv2.imshow('Window', img)
	cv2.waitKey(0)


def curvefit(x, vec, height, width, degree, heuristic):
	z = np.polyfit(x[heuristic:-heuristic], vec[heuristic:-heuristic], degree)
	p = np.poly1d(z)
	xx = np.arange(height)
	extrapolation = np.uint16(p(xx))
	new_matrix = np.zeros([height, width])
	new_matrix[xx, extrapolation] = 1
	return new_matrix


def compute_shape(shape_mask):
	y = shape_mask.copy()
	[height, width] = np.shape(y)
	shape_matrix = np.zeros([height, width])
	vec = []
	x = []
	for i in range(height):
		print i
		white_pixel = np.where(y[i, :]>0)[0]
		id = np.where(white_pixel<350)[0]
		if len(id) == 0:
			continue
		white_pixel = white_pixel[id]
		mean_pos = int(np.mean(white_pixel))
		shape_matrix[i, mean_pos] = 1.0
		vec.append(mean_pos)
		x.append(i)


	shift_matrix = np.float32([[1,0,1], [0,1,0]])
	shifted = cv2.warpAffine(shape_matrix.copy(), shift_matrix, (height, width))
	shape_matrix += shifted
	#shift_matrix = np.float32([[1,0,2], [0,1,0]])
	#shifted = cv2.warpAffine(shape_matrix, shift_matrix, (height, width))
	#shape_matrix += shifted
	return shape_matrix, x, vec

def inbetween_contour(imagepath, maskpath, rgbimage):
    mask = cv2.imread(maskpath)[:, :, 2]
    img = cv2.imread(imagepath)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    spine_mask = 1.0*(mask == 1)
    #rgbimage = cv2.imread(imagepath)

    # compute the shape
    kernel = np.ones((3,3), np.uint8)
    th1 = cv2.morphologyEx(spine_mask, cv2.MORPH_CLOSE, kernel)
    th1 = cv2.erode(th1, kernel, iterations=1)
    spine_mask = (th1>0)
    shape_mask = 1.0*spine_mask

    spine = gray_img * spine_mask
    outer = np.where(spine==0)
    mean = np.mean(spine[np.where(spine > 0)])*0.8
    ret, th1 = cv2.threshold(spine, mean, 255, cv2.THRESH_BINARY)
    th1 = cv2.erode(th1, kernel, iterations=1)

    inv_th1 = np.zeros_like(spine)
    inv_th1[np.where(th1==255)] = 0
    inv_th1[np.where(th1==0)] = 255
    inv_th1[outer] = 0
    #showimg(inv_th1)


    # then filter by shape
    # compute shape by using mean point
    [height, width] = np.shape(shape_mask)
    shape_matrix, x, vec = compute_shape(shape_mask)



    #showimg(shape_matrix)
    curve = curvefit(x, vec, height, width, 4, 20)
    #showimg(curve)


    ## slide the img using shape

    #showimg(inv_th1)


    start = min(vec[10:-10])
    end = max(vec[10:-10])

    new_mask = inv_th1.copy()
    #showimg(new_mask)


    for i in range(-100,100):
        shift_matrix = np.float32([[1,0,i], [0,1,0]])
        shifted = cv2.warpAffine(shape_matrix.copy(), shift_matrix, (height, width))
        prod = np.sum(shifted * inv_th1)
        if prod > 120000:
            new_mask[np.where(shifted>0)] = 0
	#print prod


    #showimg(new_mask)
    kernel = np.ones([3,3])
    new_mask = cv2.erode(new_mask, kernel, iterations=1)
    #showimg(new_mask)
    contours, hierarchy = cv2.findContours(new_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours, key = cv2.contourArea, reverse=True)[:7]

    cv2.drawContours(rgbimage,contours,-1,(255, 0,0),3)
    showimg(rgbimage)
    return rgbimage

## contour refinement

#3-2-8  35 is the reference image
def spine_cord_contour(imagepath,maskpath):
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
    contours, hierarchy = cv2.findContours(th2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key = cv2.contourArea, reverse=True)[:8]
    cv2.drawContours(rgbimage,contours,-1,(0,0,255),3)
    
    ####process cord
    retcord,thcord = cv2.threshold(cord,10,255,cv2.THRESH_BINARY)
    contours_cord, hierarchy_cord = cv2.findContours(thcord,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgbimage,contours_cord,-1,(0,255,0),3)
    cv2.imshow('Image1',rgbimage)
    cv2.waitKey(0)
    return rgbimage

data_path = "./test_images/"
label_path = "./Label/"
image_list = listdir(data_path)
label_list = listdir(label_path)
for i in range(1,len(image_list)):
    imagepath = data_path+image_list[i]
    maskpath = label_path+label_list[i-1]
    img = spine_cord_contour(imagepath,maskpath)
    inbetween_contour(imagepath,maskpath,img)
