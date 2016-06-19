#coding: utf-8
import argparse
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import time
import pickle

import random
import os
import cv2

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-gpu', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')

# GPUが使えるか確認
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

# 学習のパラメータ
batchsize = 1
n_epoch = 1
n_units = 512
n_input = 100 * 100 * 3

# テスト精度を出力するファイル名
accfile = "accuracy.txt"

# データセットをロード                                                                                     
print 'load BABY dataset'
t_data = np.load("t_data.npy").astype(np.float32)
f_data = np.load("f_data.npy").astype(np.float32)
t_label = np.ones((t_data.shape[0])).astype(np.int32)
f_label = np.zeros((f_data.shape[0])).astype(np.int32)

N_all_data = t_data.shape[0] + f_data.shape[0]

x_train_a, x_test_a = np.split(t_data, [int(t_data.shape[0] * 0.8)])
x_train_b, x_test_b = np.split(f_data, [int(f_data.shape[0] * 0.8)])

x_train = np.r_[x_train_a, x_train_b]
x_test = np.r_[x_test_a, x_test_b]

y_train_a, y_test_a = np.split(t_label, [int(t_data.shape[0] * 0.8)])
y_train_b, y_test_b = np.split(f_label, [int(f_data.shape[0] * 0.8)])
y_train = np.r_[y_train_a, y_train_b]
y_test = np.r_[y_test_a, y_test_b]

N = y_train.size
N_test = y_test.size

print N, N_test, N_all_data

# 多層パーセプトロンのモデル（パラメータ集合）
model = pickle.load(open("model", "rb"))

# GPU使用のときはGPUにモデルを転送
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

def forward(x_data, y_data, train=True):
    # 順伝播の処理を定義
    x = chainer.Variable(x_data)
    # 隠れ層1の出力
    h1 = F.dropout(F.relu(model.l1(x)), train=train)
    # 隠れ層2の出力
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    # 出力層の出力
    y = model.l3(h2)

    return y

# Optimizerをセット
# 最適化対象であるパラメータ集合のmodelを渡しておく
optimizer = optimizers.Adam()
optimizer.setup(model)

src_dir = "image/"
files = os.listdir(src_dir)
files.remove(".DS_Store")

i = 0
for i in xrange(1000):

    file = random.choice(files)
    print file
    img = cv2.imread(src_dir+file)
    resized_x = 512
    resized_y = int(512 / img.shape[0] * img.shape[1])

    resized_img = cv2.resize(img, (resized_x, resized_y))
    patch_size = 128
    x = random.randint(0, resized_x-patch_size)
    y = random.randint(0, resized_y-patch_size)
    data = resized_img[y:y+patch_size, x:x+patch_size]

    label = np.argmax(forward(np.array([data.flatten()], dtype=np.float32), 
                              None, train=False).data)

    if label == 1:
        #img = np.array(x_train[i].reshape(100, 100, 3))*255
        #img = img.astype(np.int32)
        cv2.imwrite("detected_unchi/t/"+str(i)+".bmp", data)
    elif label == 0:
        #img = np.array(x_train[i].reshape(100, 100, 3))*255
        #img = img.astype(np.int32)
        cv2.imwrite("detected_unchi/f/"+str(i)+".bmp", data)
