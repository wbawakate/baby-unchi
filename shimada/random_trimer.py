#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
from PIL import Image
import random

patch_size = 128
patch_n = 100
resized = 256
categories = ['positive', 'negative']
src_dir = 'original'
dst_dir = 'trim'

for c in categories:
    files = os.listdir(src_dir + '/' + c)
    for idx, f in enumerate(files):
        img = Image.open(src_dir+'/'+c+'/'+f)
        resized_x = int(float(resized) * img.size[0] /  img.size[1])
        resized_y = resized
        img = img.resize((resized_x, resized_y))
        img_ary = np.asarray(img)

        for i in range(patch_n):
            x = random.randint(0, resized_x-patch_size)
            y = random.randint(0, resized_y-patch_size)
            trimed = img_ary[y:y+patch_size, x:x+patch_size, :]
            out = Image.fromarray(trimed)
            out.save(dst_dir + '/' + c + '/' + str(idx) + '_' + str(i) + '.bmp')
