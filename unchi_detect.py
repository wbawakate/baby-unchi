#coding: utf-8
import argparse
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import time
import pickle

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-gpu', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')

# GPUが使えるか確認
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

# 学習のパラメータ
batchsize = 10
n_epoch = 20
n_units = 512
n_input = 128 * 128 * 3

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
model = chainer.FunctionSet(l1=F.Linear(n_input, n_units),      # 入力層-隠れ層1
                            l2=F.Linear(n_units, n_units),  # 隠れ層1-隠れ層2
                            l3=F.Linear(n_units, 2))       # 隠れ層2-出力層

# GPU使用のときはGPUにモデルを転送
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

def forward(x_data, y_data, train=True):
    # 順伝播の処理を定義
    # 入力と教師データ
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    # 隠れ層1の出力
    h1 = F.dropout(F.relu(model.l1(x)), train=train)
    # 隠れ層2の出力
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    # 出力層の出力
    y = model.l3(h2)

    # 訓練時とテスト時で返す値を変える
    if train:
        # 訓練時は損失を返す
        # 多値分類なのでクロスエントロピーを使う
        loss = F.softmax_cross_entropy(y, t)
        return loss
    else:
        # テスト時は精度を返す
        acc = F.accuracy(y, t)
        return acc

# Optimizerをセット
# 最適化対象であるパラメータ集合のmodelを渡しておく
optimizer = optimizers.Adam()
optimizer.setup(model)

fp = open(accfile, "w")
fp.write("epoch\ttest_accuracy\n")

# 訓練ループ
# 各エポックでテスト精度を求める
start_time = time.clock()
for epoch in range(1, n_epoch + 1):
    print "epoch: %d" % epoch

    # 訓練データを用いてパラメータを更新する
    perm = np.random.permutation(N)
    sum_loss = 0
    for i in range(0, N, batchsize):
        x_batch = xp.asarray(x_train[perm[i:i + batchsize]])
        y_batch = xp.asarray(y_train[perm[i:i + batchsize]])

        optimizer.zero_grads()
        loss = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(y_batch)

    print "train mean loss: %f" % (sum_loss / N)

    # テストデータを用いて精度を評価する
    sum_accuracy = 0
    for i in range(0, N_test, batchsize):
        x_batch = xp.asarray(x_test[i:i + batchsize])
        y_batch = xp.asarray(y_test[i:i + batchsize])

        acc = forward(x_batch, y_batch, train=False)
        sum_accuracy += float(acc.data) * len(y_batch)

    print "test accuracy: %f" % (sum_accuracy / N_test)
    fp.write("%d\t%f\n" % (epoch, sum_accuracy / N_test))
    fp.flush()

end_time = time.clock()
print end_time - start_time

fp.close()
pickle.dump(model, open("model", "wb"), -1)
