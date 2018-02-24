import numpy as np
import os
from os.path import isfile, join
from os import listdir
import scipy.misc as misc
import cv2
from PIL import Image
data_path = "/Users/anekisei/Documents/UAV_dataset/2.Sagital"
file_name = [i for i in range(31,61)]
'''
for dirname in listdir(data_path):
    
    image_path = join(data_path, dirname)
    

    #delete = join(image_path,".DS_Store")
    #os.remove(delete)
    image_list = listdir(image_path)
    for image_name in image_list:
        print image_name

    if dirname == '.DS_Store':
        continue
    print dirname
    count = 1
    index = 1
    for i, filename in enumerate(os.listdir(image_path)):
        print filename
        os.rename(image_path + "/" + filename, image_path + "/" + dirname + "_" + str(count) + "_" + str(index) + ".jpg")
        index += 1
        if index == 13:
            index = 1
            count += 1
'''
'''
for dirname in listdir(data_path):
    if dirname == '.DS_Store':
        continue
    image_path = join(data_path, dirname)
    image_list = listdir(image_path)
    for i in range(len(image_list)):
        if image_list[i] == '.DS_Store':
            continue
        image_list[i] = '"' + 'data/annotations/' + image_list[i][:-3] + 'png' + '"'
    print image_list
'''
Label = cv2.imread("31_1_6.png")
#LabelPred = Label * 1000
#print LabelPred.shape
#LabelPred = cv2.cvtColor( LabelPred, cv2.COLOR_RGB2GRAY )
#LabelPred = LabelPred.reshape((LabelPred.shape[1],LabelPred.shape[2])).astype(np.uint8)
LabelPred = Label[:,:,2]
for i in range(LabelPred.shape[0]):
    print LabelPred[i,:]
im = Image.fromarray(LabelPred)
im.show()


