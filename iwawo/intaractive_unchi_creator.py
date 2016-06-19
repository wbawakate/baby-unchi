import os
import re
import cv2

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

src_dir = "image/"
t_dir = "tremed_data/t/"
f_dir = "tremed_data/f/"

files = os.listdir(src_dir)
files.remove(".DS_Store")
dataset = np.empty((0, 100*100*3), np.float64)

id = 0
while True:
    file = random.choice(files)
    print file
    img = cv2.imread(src_dir+file)
    #print img
    #img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

    print img.shape

    resized_y = 256
    resized_x = int(256. / img.shape[0] * img.shape[1])
    
    resized_img = cv2.resize(img, (resized_x, resized_y))
    resized_img_show = cv2.resize(img, (resized_x, resized_y))
    patch_size = 128
    x = random.randint(0, resized_x-patch_size)
    y = random.randint(0, resized_y-patch_size)
    cv2.rectangle(resized_img_show,(x,y),(x+patch_size,y+patch_size),(0,0,255),3)
    #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", resized_img_show)

    keycode = cv2.waitKey(0)
    if keycode == ord("1"):
        data = resized_img[y:y+patch_size, x:x+patch_size]
        cv2.imwrite("trem/t/"+str(id)+".bmp", data)
    elif keycode == ord("0"):
        data = resized_img[y:y+patch_size, x:x+patch_size]
        cv2.imwrite("trem/f/"+str(id)+".bmp", data)
    else:
        break

    id += 1
    
