#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import pickle

import random
from PIL import Image

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-gpu', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

# GPUが使えるか確認
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

# 学習のパラメータ
n_units = 512
n_input = 128 * 128 * 3

# データセットをロード                                                                                     
print('load BABY dataset')
t_data = np.load("./positive_data.npy").astype(np.float32)
f_data = np.load("./negative_data.npy").astype(np.float32)

all_data = [t_data, f_data]
N_all_data = t_data.shape[0] + f_data.shape[0]

# 多層パーセプトロンのモデル（パラメータ集合）
model = pickle.load(open("../trained-model/model", "rb"), encoding='latin1')

# GPU使用のときはGPUにモデルを転送
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

def forward(x_data, y_data, train=False):
    # 順伝播の処理を定義
    x = chainer.Variable(x_data)
    # 隠れ層1の出力
    h1 = F.dropout(F.relu(model.l1(x)), train=train)
    # 隠れ層2の出力
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    # 出力層の出力
    y = model.l3(h2)
    return y

idx = 0
dst = './detected'
for cat, data in enumerate(all_data):
    c = 'positive' if cat==0 else 'negative'
    for i in data:
        idx += 1
        label = np.argmax(forward(np.array([i], dtype=np.float32), None, train=False).data) 
        d = 'unchi' if label==1 else 'other'
        im = Image.fromarray(np.uint8(i.reshape((128, 128, 3)) * 255))
        im.save(dst+'/'+d+'/'+c+'/'+str(idx)+'.bmp')
