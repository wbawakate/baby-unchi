#!/usr/bin/env python                                                                                                 
#-*- coding: utf-8 -*-                                                                                                

import os
import re
import cv2

import numpy as np

direct = "image3/"
files = os.listdir(direct)

files.remove(".DS_Store")

width = 128
height = 128
"""
for i, file in enumerate(files):
    print file
    src = cv2.imread("image/"+file, 1)
    print src.shape
    y = src.shape[0] / 2
    x = src.shape[1] / 2
    dst = src[y:y+height, x:x+width]
    cv2.imwrite("trem/t/"+str(i)+".bmp", dst)
""" 

# save image function
def save_image(file_number, h_start, h_end, w_start, w_end):
    # clip an image
    dst = src[h_start:h_end, w_start:w_end]
    
    # define file path
    folder_path = "trem/f2/" + str(file_number) + ".bmp"

    # save image
    cv2.imwrite(folder_path, dst)
    
    return


for i, file in enumerate(files):
    print file

    # load images
    src = cv2.imread("image3/"+file, 1)

    # show image parameters
    print src.shape
    
    y = 0
    x = src.shape[1] - width

    # extract shape data
    img_y = src.shape[0]
    img_x = src.shape[1]
    
    # original code
    #dst = src[y:y+height, x:x+width]
    #cv2.imwrite("trem/f/"+str(i)+".bmp", dst)

    # define an offset number
    offset_number = 745 + i

    # save image    
    # (0,0) - (128, 128)
    save_image(offset_number, y, y+height, x, x+width)

    

