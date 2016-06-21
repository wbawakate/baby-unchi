#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import glob
import shutil
import numpy as np
import chainer
import chainer.functions as F
from chainer import cuda, serializers
from PIL import Image

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('src_dir', type=str, help='src images directory which must have positive and negative directory')
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

# 多層パーセプトロンのモデル（パラメータ集合）
model = chainer.FunctionSet(l1=F.Linear(n_input, n_units), # 入力層-隠れ層1
                            l2=F.Linear(n_units, n_units), # 隠れ層1-隠れ層2
                            l3=F.Linear(n_units, 2))       # 隠れ層2-出力層

serializers.load_npz('../model_by_chainer_serializers.npz', model)

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

dst = './detected'
categories = ['positive', 'negative']
for c in categories:
    files = glob.glob(args.src_dir+'/'+ c + '/*.bmp')
    for f in files:
        data = np.asarray(Image.open(f)).astype(np.float32)

        # preprocess
        data /= 255.
        data = np.expand_dims(data[:,:,::-1].flatten(), 0)

        label = np.argmax(forward(xp.asarray(data, dtype=np.float32), None, train=False).data) 
        d = 'unchi' if label==1 else 'other'
        shutil.copy(f,  dst+'/'+d+'/'+c+'/')
