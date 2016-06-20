import os
import re
import cv2

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

def intaractive_patch_creator(src_dir, dis_t_dir, dis_f_dir,
                              show_size=256, patch_size=128, 
                              n_patch=float("inf")):

    """
    This function make batches by human (especially doctors).
    repeat:
    1. show a picture and range randomly.
    2. push 1 if the patch is effective for diagnosis else 0.
    3. save the patch to dis_(t or f)_dir.
    """

    """ dis dir should be empty """
    if len([file for file in os.listdir(dis_t_dir) if (".bmp" in file)]) != 0:
        print "error:", dis_t_dir, "does not emply"
        return
    if len([file for file in os.listdir(dis_f_dir) if (".bmp" in file)]) != 0:
        print "error:", dis_f_dir, "does not emply"
        return

    files = [file for file in os.listdir(src_dir) if (".jpg" in file)]

    id = 0
    while True:

        """ select image randomly """
        file = random.choice(files)
        print file
        img = cv2.imread(src_dir+file)
        
        """ resize image """
        resized_y = show_size
        resized_x = int(float(show_size) / img.shape[0] * img.shape[1])
        resized_img = cv2.resize(img, (resized_x, resized_y))
        resized_img_show = cv2.resize(img, (resized_x, resized_y))
       
        """ select patch's position randomly"""
        x = random.randint(0, resized_x-patch_size)
        y = random.randint(0, resized_y-patch_size)
        cv2.rectangle(resized_img_show,(x,y),(x+patch_size,y+patch_size),(0,0,255),3)
        patch = resized_img[y:y+patch_size, x:x+patch_size]

        """ show and save """
        print "effective: 1, ineffective: 0, quit: q, skip: other key"
        cv2.imshow("result", resized_img_show)
        keycode = cv2.waitKey(0)
        if keycode == ord("1"):
            cv2.imwrite(dis_t_dir+str(id)+".bmp", patch)
        elif keycode == ord("0"):
            cv2.imwrite(dis_f_dir+str(id)+".bmp", patch)
        elif keycode == ord("q"):
            print "exit program"
            break
        else:
            print "skip"
            continue
        

        id += 1
        if id >= n_patch:
            break
                              

if __name__ == "__main__":
    intaractive_patch_creator(src_dir="image/", 
                              dis_t_dir="trem/t/", 
                              dis_f_dir="trem/f/",
                              n_patch=3)
