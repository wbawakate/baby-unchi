#!/usr/bin/env python
#-*- coding: utf-8 -*-

import os
import re
import cv2

import numpy as np

t_dir = "trem/t/"
f_dir = "trem/f/"


files = os.listdir(t_dir)
files.remove(".DS_Store")
dataset = np.empty((0, 128*128*3), np.float64)
for i, file in enumerate(files):
    print file
    src = cv2.imread(t_dir+file, 1).flatten()
    src = np.array([src.astype(np.float64) / 255.])
    print src.shape
    dataset = np.append(dataset, src, axis=0)
np.save("t_data.npy", dataset)

files = os.listdir(f_dir)
files.remove(".DS_Store")
dataset = np.empty((0, 128*128*3), np.float64)
for i, file in enumerate(files):
    print file
    src = cv2.imread(f_dir+file, 1).flatten()
    src = np.array([src.astype(np.float64) / 255.])
    print src.shape
    dataset = np.append(dataset, src, axis=0)
np.save("f_data.npy", dataset)


