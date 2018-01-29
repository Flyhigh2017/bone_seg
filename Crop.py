import numpy as np
import os
from os.path import isfile, join
from os import listdir
import scipy.misc as misc
import cv2
from PIL import Image
imagepath = "./test_images/1_1_9.jpg"
maskpath = "./Label/1_1_9.png"
img = cv2.imread(imagepath,0)
def find_next(point1,point2):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    d = np.sqrt((y2 - y1)*(y2 - y1) + (x2 - x1)*(x2 - x1))
    if x2 == x1:
        y3 = y2
        x3 = x2 + d
        return (int(x3),int(y3))
    k = (y2 - y1 + 0.0) / (x2 - x1 + 0.0)
    y31 = y2 - np.sqrt(d*d / (k*k+1))
    y32 = y2 + np.sqrt(d*d / (k*k+1))
    x31 = np.sqrt(d*d-(y31-y2)*(y31-y2)) + x2
    x32 = np.sqrt(d*d-(y32-y2)*(y32-y2))+x2
    point31 = (int(x31),int(y31))
    point32 = (int(x32),int(y32))
    if x2 < x1:
        return point32
    if x2 > x1:
        return point31


def find_square(img, point1, point2):
    #img is grayscale
    mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
    point3 = find_next(point1,point2)
    cv2.line(mask, point1, point2, 255, thickness=2)
    cv2.line(mask, point2, point3, 255, thickness=2)
    dx = point2[0] - point3[0]
    dy = point2[1] - point3[1]
    point4 = (point1[0] - dx, point1[1] - dy)
    cv2.line(mask, point3, point4, 255, thickness=2)
    cv2.line(mask, point1, point4, 255, thickness=2)
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mask, contours, 0, 255, -1) # Draw filled contour in mask
    out = np.zeros_like(img) # Extract out the object and place into output image
    out[mask == 255] = img[mask == 255]
    angle = np.arctan2((point2[0] - point1[0] + 0.0), (point2[1] - point1[1] + 0.0)) * 180 / np.pi
    angle = np.absolute(angle)

    return out, point3, angle
'''
point1 = (100,100)
point2 = (80,300)
find_square(img,point1,point2)
'''



