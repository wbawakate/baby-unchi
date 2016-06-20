#!/usr/bin/env python
#-*- coding: utf-8 -*-

from PIL import Image
import glob
import argparse

import numpy as np

parser = argparse.ArgumentParser(description='bmp images to npz')
parser.add_argument('src_dir', type=str, help='src images directory which must have positive and negative directory')
args = parser.parse_args()

categories = ['positive', 'negative']
for c in categories:
    files = glob.glob(args.src_dir+'/'+ c + '/*.bmp')
    dataset_list = []
    for f in files:
        src = np.asarray(Image.open(f)).flatten()
        src = np.array([src.astype(np.float32) / 255.])
        dataset_list.append(src)
    dataset = np.asarray(dataset_list)
    np.save(c + '_data.npy', dataset)
