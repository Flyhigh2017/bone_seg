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
import Crop
def compare_img(img1, img2):
    figure = plt.figure(1)
    plt.subplot(211)
    plt.imshow(img1, cmap='gray')
    plt.subplot(212)
    plt.imshow(img2, cmap='gray')
    plt.show()


def showimg(img):
    cv2.imshow('Window', img)
    cv2.waitKey(0)


def curvefit(x, vec, height, width, degree, heuristic):
    z = np.polyfit(x[heuristic:-heuristic], vec[heuristic:-heuristic], degree)
    p = np.poly1d(z)
    xx = np.arange(height)
    extrapolation = np.uint16(p(xx))
    new_matrix = np.zeros([height, width])
    extrapolation[extrapolation >= height] = 0
    extrapolation[extrapolation >= width] = 0
    new_matrix[xx, extrapolation] = 1
    return new_matrix


def compute_right_shape(shape_mask):
    y = shape_mask.copy()
    [height, width] = np.shape(y)
    shape_matrix = np.zeros([height, width])
    vec = []
    x = []
    for i in range(height):
        white_pixel = np.where(y[i, :] > 0)[0]
        id = np.where(white_pixel < height * 0.7)[0]
        if len(id) == 0:
            continue
        white_pixel = white_pixel[id]
        mean_pos = int(np.max(white_pixel))
        shape_matrix[i, mean_pos] = 1.0
        vec.append(mean_pos)
        x.append(i)
    
    #shift_matrix = np.float32([[1, 0, 1], [0, 1, 0]])
    #shifted = cv2.warpAffine(shape_matrix.copy(), shift_matrix, (height, width))
    #shape_matrix += shifted
    return shape_matrix, x, vec



def compute_shape(shape_mask):
    y = shape_mask.copy()
    [height, width] = np.shape(y)
    shape_matrix = np.zeros([height, width])
    vec = []
    x = []
    for i in range(height):
        white_pixel = np.where(y[i, :]>0)[0]
        id = np.where(white_pixel<height*0.7)[0]
        if len(id) == 0:
            continue
        white_pixel = white_pixel[id]
        mean_pos = int(np.min(white_pixel))
        shape_matrix[i, mean_pos] = 1.0
        vec.append(mean_pos)
        x.append(i)
    
    
    #shift_matrix = np.float32([[1,0,1], [0,1,0]])
    #shifted = cv2.warpAffine(shape_matrix.copy(), shift_matrix, (height, width))
    #shape_matrix += shifted
    return shape_matrix, x, vec


def left_bound_filter(shape, img, th):
    height, width = np.shape(img)
    new_img = img.copy()
    for i in range(0, 15):
        shift_matrix = np.float32([[1,0,i], [0,1,0]])
        shifted = cv2.warpAffine(shape, shift_matrix, (height, width))
        prod = np.sum(shifted * img)
        if prod > th:
            new_img[np.where(shifted > 0)] = 0
    return new_img


def right_bound_filter(shape, img, th):
    height, width = np.shape(img)
    new_img = img.copy()
    for i in range(-20, 0):
        shift_matrix = np.float32([[1,0,i], [0,1,0]])
        shifted = cv2.warpAffine(shape, shift_matrix, (height, width))
        prod = np.sum(shifted * img)
        if prod > th:
            new_img[np.where(shifted > 0)] = 0
    return new_img

def inbetween_contour(imagepath, maskpath, rgbimage):
    mask = cv2.imread(maskpath)[:, :, 2]
    img = cv2.imread(imagepath)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
    cl1 = clahe.apply(gray_img)
    #compare_img(gray_img, cl1)
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
    right_bound, r, rvec = compute_right_shape(shape_mask)


    #showimg(shape_matrix)
    #curve = curvefit(x, vec, height, width, 4, 20)
    #showimg(curve)


    ## slide the img using shape

    #showimg(inv_th1)
    ## slide the img using shape
    new_mask = inv_th1.copy()
    #showimg(new_mask)

    up_shape = np.zeros_like(shape_matrix)
    down_shape = np.zeros_like(shape_matrix)
    up_right = np.zeros_like(shape_matrix)
    down_right = np.zeros_like(shape_matrix)

    bbd = 210
    up_shape[0:bbd, :] = shape_matrix[0:bbd, :]
    down_shape[bbd:-1, :] = shape_matrix[bbd:-1, :]
    up_right[0:bbd, :] = right_bound[0:bbd, :]
    down_right[bbd:-1, :] = right_bound[bbd:-1, :]

    new_mask = left_bound_filter(up_shape,  new_mask, 30000)
    new_mask = left_bound_filter(down_shape,  new_mask, 20000)
    new_mask = right_bound_filter(up_right, new_mask, 30000)
    new_mask = right_bound_filter(down_right,  new_mask, 20000)

    #showimg(new_mask)


    kernel = np.ones([3,3])
    new_mask = cv2.erode(new_mask, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(new_mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key = cv2.contourArea, reverse=True)[:7]
    cv2.drawContours(rgbimage, contours,-1,(255, 0,0),3)
    #compare_img(new_mask, gray_img)
    showimg(rgbimage)
    '''
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
    '''
    return rgbimage

## contour refinement

#3-2-8 is the reference image
def spine_cord_contour(imagepath,maskpath,prefix):
    img = cv2.imread(maskpath)
    img = img[:,:,2]
    img = (img * 100)
    section1 = img.copy()#mask used for spine
    section2 = img.copy()#mask used for cord
    #Spine
    section1[section1 == 200] = 0
    mask1 = cv2.medianBlur(section1,5)
    mask1[mask1 != 0] = 1.5
    #cord
    section2[section2 == 100] = 0
    section2 = cv2.GaussianBlur(section2,(3,3),0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
    section2 = section2 * 255
    section2 = cv2.erode(section2, kernel)
    mask2 = cv2.medianBlur(section2,9)
    mask2[mask2 != 0] = 1
    #read raw image
    orig = cv2.imread(imagepath,0) * 2
    rgbimage = cv2.imread(imagepath)
    feature_img = rgbimage.copy()
    #normalize all images, reference image is 3-2-8
    norm = 32.99
    print "average:", np.average(orig)
    ave = np.average(orig)
    rate = norm / ave
    print "rate:", rate
    orig = (rate * orig.astype('float64')).astype('uint8')
    #####process spine
    spine = orig * mask1
    cord = orig * mask2 * 100
    
    ret,th2 = cv2.threshold(spine,40,255,cv2.THRESH_BINARY)#####threshhold
    
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT,(21, 7))
    th2 = cv2.erode(th2, k2)#erode
    contours, hierarchy = cv2.findContours(th2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#find contour
    contours = sorted(contours, key = cv2.contourArea, reverse=True)[:8]
    cv2.drawContours(rgbimage,contours,-1,(0,0,255),3)
    #cv2.imshow(prefix,rgbimage)
    #cv2.waitKey(0)
    #put text and point line
    '''
    names = ["bone"+str(i) for i in range(8)]
    count = 0
    store_mp = {}
    lst = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            store_mp[cY] = cX
            lst.append(cY)

    lst = sorted(lst)
    for cY in lst:
        cX = store_mp[cY]
        cv2.circle(rgbimage, (cX, cY), 3, (255, 255, 255), -1)
        cv2.line(rgbimage, (cX, cY), (cX+50+20*count, cY), (255,255,255), 1)
        cv2.putText(rgbimage, names[count], (cX+50+20*count+ 5, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.CV_AA)
        count += 1
    index = 1
    for i in range(len(lst)-1):
        y_cur = lst[i]
        x_cur = store_mp[y_cur]
        y_next = lst[i+1]
        x_next = store_mp[y_next]
        point1 = (x_cur,y_cur)
        point2 = (x_next,y_next)
        gray_img = cv2.cvtColor(feature_img, cv2.COLOR_BGR2GRAY)
        out, point3, angle = Crop.find_square(gray_img, point1, point2)
        if point2[0] > point1[0]:
            counterclockwise = -1
        else:
            counterclockwise = 1
#M = cv2.getRotationMatrix2D(((point1[0]+point3[0])/2,(point1[1]+point3[1])/2),counterclockwise * angle,1)
#res = cv2.warpAffine(out,M,(gray_img.shape[1],gray_img.shape[0]))
        cropped = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR) #res
        contour_rect, hierarchy_rect = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)#res
        cv2.putText(feature_img, str(index), (point1[0]+20, point1[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.CV_AA)
        index += 1
        cv2.drawContours(feature_img,contour_rect,-1,(0,255,0),2)
        (x, y, w, h) = cv2.boundingRect(contour_rect[0])
        cropped = cropped[y:y+h,x:x+w]
#cv2.imshow('image',cropped)
#cv2.waitKey(0)
#cv2.imwrite("./test_result/" + prefix + str(i)+".png",feature_img)
    ####process cord
    retcord,thcord = cv2.threshold(cord,10,255,cv2.THRESH_BINARY)
    contours_cord, hierarchy_cord = cv2.findContours(thcord,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgbimage,contours_cord,-1,(0,255,0),3)
    #cv2.imshow('Image1',rgbimage)
    #cv2.waitKey(0)
    '''
    return rgbimage, feature_img

data_path = "./test_images/"
data_again = "./again/"
save_path = "./save/"
label_path = "./Label/"
image_list = listdir(data_again)
label_list = listdir(label_path)
for i in range(1,len(image_list)):
    print image_list[i]
    imagepath = data_path+image_list[i]
    maskpath = label_path+image_list[i][:-4]+".png"
    img, feature_img = spine_cord_contour(imagepath,maskpath,image_list[i])
    #cv2.imshow('res',feature_img)
    #cv2.waitKey(0)
    img = inbetween_contour(imagepath,maskpath,img)
#cv2.imwrite(save_path + image_list[i],feature_img)
