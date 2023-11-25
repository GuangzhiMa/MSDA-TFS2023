# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 19:11:26 2022

@author: 14025959_admin
"""
import torch
from train import train_FDIM
import random
from utils import Gau_data, DF_interval, mmd
from test2 import test_multi

import torch.backends.cudnn as cudnn

import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("..") 

from sklearn.datasets import make_blobs

SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

cuda = True
cudnn.benchmark = True


num_inputs = 2
class_number = 3
sys_examples = 500
s1_feature, s1_labels = make_blobs(n_features=num_inputs, n_samples=sys_examples, centers=3, random_state=120,
                                      cluster_std=[0.5, 1.8, 3.5], center_box=(-5, 10))

s2_feature, s2_labels = make_blobs(n_features=num_inputs, n_samples=sys_examples, centers=3, random_state=120,
                                      cluster_std=[1.2, 1.5, 2], center_box=(5, 15))

s3_feature, s3_labels = make_blobs(n_features=num_inputs, n_samples=sys_examples, centers=3, random_state=120,
                                      cluster_std=[2, 1, 1.5], center_box=(-15, 5))

batch_size = 100

source_feature = np.vstack((s1_feature, s2_feature, s3_feature))
source_labels = np.hstack((s1_labels, s2_labels, s3_labels))

source = torch.utils.data.TensorDataset(torch.tensor(source_feature, dtype=torch.float32), torch.tensor(source_labels).long())
Source_loader = torch.utils.data.DataLoader(source, batch_size)

source_1 = torch.utils.data.TensorDataset(torch.tensor(s1_feature, dtype=torch.float32), torch.tensor(s1_labels).long())
source1_loader = torch.utils.data.DataLoader(source_1, batch_size)
source_2 = torch.utils.data.TensorDataset(torch.tensor(s2_feature, dtype=torch.float32), torch.tensor(s2_labels).long())
source2_loader = torch.utils.data.DataLoader(source_2, batch_size)
source_3 = torch.utils.data.TensorDataset(torch.tensor(s3_feature, dtype=torch.float32), torch.tensor(s3_labels).long())
source3_loader = torch.utils.data.DataLoader(source_3, batch_size)

source_loader = [source1_loader, source2_loader, source3_loader]


t_interval, t_labels = Gau_data(sys_examples = sys_examples, random_state = 120)

e = 0.0001
s1_e_f = np.zeros((s1_feature.shape[0], num_inputs * 3))
s2_e_f = np.zeros((s2_feature.shape[0], num_inputs * 3))
s3_e_f = np.zeros((s3_feature.shape[0], num_inputs * 3))
for i in range(num_inputs):
    for j in range(s1_feature.shape[0]):
        s1_e_f[j, 3 * i] = s1_feature[j, i] - e
        s1_e_f[j, 3 * i + 1] = s1_feature[j, i]
        s1_e_f[j, 3 * i + 2] = s1_feature[j, i] + e
        s2_e_f[j, 3 * i] = s2_feature[j, i] - e
        s2_e_f[j, 3 * i + 1] = s2_feature[j, i]
        s2_e_f[j, 3 * i + 2] = s2_feature[j, i] + e
        s3_e_f[j, 3 * i] = s3_feature[j, i] - e
        s3_e_f[j, 3 * i + 1] = s3_feature[j, i]
        s3_e_f[j, 3 * i + 2] = s3_feature[j, i] + e
source = [s1_e_f, s2_e_f, s3_e_f]


# select beta
type = 1
deta = 0.5
B = np.arange(0, 1.02, 0.1)

for q in range(len(B)):
    t_fuzzy = np.zeros((t_interval.shape[0], num_inputs * 3))

    for i in range(num_inputs):
        for j in range(t_interval.shape[0]):
            t_fuzzy[j, 3 * i] = t_interval[j, 2 * i]
            t_fuzzy[j, 3 * i + 1] = B[q] * t_interval[j, 2 * i] + (1 - B[q]) * t_interval[j, 2 * i + 1]
            t_fuzzy[j, 3 * i + 2] = t_interval[j, 2 * i + 1]
    # correlation weight vector
    MMD = np.zeros(len(source))
    for i in range(len(source)):
        MMD[i] = mmd(source[i], t_fuzzy, deta, type)

    if q == 0:
        best_mmd = np.sum(MMD)
        best_beta = B[q]

    if np.sum(MMD) < best_mmd:
         best_mmd= np.sum(MMD)
         best_beta = B[q]

print(best_beta)

t_fuzzy = np.zeros((t_interval.shape[0], num_inputs * 3))

for i in range(num_inputs):
    for j in range(t_interval.shape[0]):
        t_fuzzy[j, 3 * i] = t_interval[j, 2 * i]
        t_fuzzy[j, 3 * i + 1] = best_beta * t_interval[j, 2 * i] + (1 - best_beta) * t_interval[j, 2 * i + 1]
        t_fuzzy[j, 3 * i + 2] = t_interval[j, 2 * i + 1]

MMD = np.zeros(len(source))
for i in range(len(source)):
    MMD[i] = mmd(source[i], t_fuzzy, deta, type)

print(MMD)

MMD = 1 - MMD/(np.max(np.abs(MMD))*1.5)
w_corr = MMD/np.sum(MMD)

print(w_corr)

# fuzzy technique to extract crisp-valued
t_df = DF_interval(t_interval, num_inputs, best_beta)
target_data = torch.utils.data.TensorDataset(torch.tensor(t_df, dtype=torch.float32), torch.tensor(t_labels).long())
target_loader = torch.utils.data.DataLoader(target_data, batch_size)


model_root = 'models'
T = 5
score_FDIM1 = np.zeros(T)
n_epoch = 500
lr = 0.01

# FRAT
train_hist_FDIM = {}
train_hist_FDIM['Total_loss'] = []
train_hist_FDIM['disc_loss'] = []

test_hist_FDIM1 = []

for i in range(T):
    train_FDIM(source_loader, target_loader, lr, n_epoch, num_inputs, class_number, w_corr, train_hist_FDIM, test_hist_FDIM1, cuda)
    score_FDIM1[i] = test_multi(target_loader, cuda, w_corr, 'corr')
    print('DEMAT-Net with corr : %.4f' % (score_FDIM1[i]))

test_Acc1=np.array(test_hist_FDIM1)
test_Acc1.resize((T,n_epoch))
test_Accmean1 = np.mean(test_Acc1,0)
test_Accstd1 = np.std(test_Acc1,0)

#
type = 2
B = np.arange(0, 1.02, 0.1)

for q in range(len(B)):
    t_fuzzy = np.zeros((t_interval.shape[0], num_inputs * 3))

    for i in range(num_inputs):
        for j in range(t_interval.shape[0]):
            t_fuzzy[j, 3 * i] = t_interval[j, 2 * i]
            t_fuzzy[j, 3 * i + 1] = B[q] * t_interval[j, 2 * i] + (1 - B[q]) * t_interval[j, 2 * i + 1]
            t_fuzzy[j, 3 * i + 2] = t_interval[j, 2 * i + 1]
    # correlation weight vector
    MMD = np.zeros(len(source))
    for i in range(len(source)):
        MMD[i] = mmd(source[i], t_fuzzy, deta, type)

    if q == 0:
        best_mmd = np.sum(MMD)
        best_beta = B[q]

    if np.sum(MMD) < best_mmd:
         best_mmd= np.sum(MMD)
         best_beta = B[q]

print(best_beta)
# best_beta = 0.4
t_fuzzy = np.zeros((t_interval.shape[0], num_inputs * 3))

for i in range(num_inputs):
    for j in range(t_interval.shape[0]):
        t_fuzzy[j, 3 * i] = t_interval[j, 2 * i]
        t_fuzzy[j, 3 * i + 1] = best_beta * t_interval[j, 2 * i] + (1 - best_beta) * t_interval[j, 2 * i + 1]
        t_fuzzy[j, 3 * i + 2] = t_interval[j, 2 * i + 1]

MMD = np.zeros(len(source))
for i in range(len(source)):
    MMD[i] = mmd(source[i], t_fuzzy, deta, type)

print(MMD)

MMD = 1 - MMD/(np.max(np.abs(MMD))*1.5)
w_corr = MMD/np.sum(MMD)

print(w_corr)

# fuzzy technique to extract crisp-valued
t_df = DF_interval(t_interval, num_inputs, best_beta)
target_data = torch.utils.data.TensorDataset(torch.tensor(t_df, dtype=torch.float32), torch.tensor(t_labels).long())
target_loader = torch.utils.data.DataLoader(target_data, batch_size)


model_root = 'models'
score_FDIM2 = np.zeros(T)

# FRAT
train_hist_FDIM = {}
train_hist_FDIM['Total_loss'] = []
train_hist_FDIM['disc_loss'] = []

test_hist_FDIM2 = []

for i in range(T):
    train_FDIM(source_loader, target_loader, lr, n_epoch, num_inputs, class_number, w_corr, train_hist_FDIM, test_hist_FDIM2, cuda)
    score_FDIM2[i] = test_multi(target_loader, cuda, w_corr, 'corr')
    print('DEMAT-Net with corr : %.4f' % (score_FDIM2[i]))

test_Acc2=np.array(test_hist_FDIM2)
test_Acc2.resize((T,n_epoch))
test_Accmean2 = np.mean(test_Acc2,0)
test_Accstd2 = np.std(test_Acc2,0)

#
type = 3
B = np.arange(0, 1.02, 0.1)

for q in range(len(B)):
    t_fuzzy = np.zeros((t_interval.shape[0], num_inputs * 3))

    for i in range(num_inputs):
        for j in range(t_interval.shape[0]):
            t_fuzzy[j, 3 * i] = t_interval[j, 2 * i]
            t_fuzzy[j, 3 * i + 1] = B[q] * t_interval[j, 2 * i] + (1 - B[q]) * t_interval[j, 2 * i + 1]
            t_fuzzy[j, 3 * i + 2] = t_interval[j, 2 * i + 1]
    # correlation weight vector
    MMD = np.zeros(len(source))
    for i in range(len(source)):
        MMD[i] = mmd(source[i], t_fuzzy, deta, type)

    if q == 0:
        best_mmd = np.sum(MMD)
        best_beta = B[q]

    if np.sum(MMD) < best_mmd:
         best_mmd= np.sum(MMD)
         best_beta = B[q]

print(best_beta)

t_fuzzy = np.zeros((t_interval.shape[0], num_inputs * 3))

for i in range(num_inputs):
    for j in range(t_interval.shape[0]):
        t_fuzzy[j, 3 * i] = t_interval[j, 2 * i]
        t_fuzzy[j, 3 * i + 1] = best_beta * t_interval[j, 2 * i] + (1 - best_beta) * t_interval[j, 2 * i + 1]
        t_fuzzy[j, 3 * i + 2] = t_interval[j, 2 * i + 1]

MMD = np.zeros(len(source))
for i in range(len(source)):
    MMD[i] = mmd(source[i], t_fuzzy, deta, type)

print(MMD)

MMD = 1 - MMD/(np.max(np.abs(MMD))*1.5)
w_corr = MMD/np.sum(MMD)

print(w_corr)

# fuzzy technique to extract crisp-valued
t_df = DF_interval(t_interval, num_inputs, best_beta)
target_data = torch.utils.data.TensorDataset(torch.tensor(t_df, dtype=torch.float32), torch.tensor(t_labels).long())
target_loader = torch.utils.data.DataLoader(target_data, batch_size)


model_root = 'models'
score_FDIM3 = np.zeros(T)


# FRAT
train_hist_FDIM = {}
train_hist_FDIM['Total_loss'] = []
train_hist_FDIM['disc_loss'] = []

test_hist_FDIM3 = []

for i in range(T):
    train_FDIM(source_loader, target_loader, lr, n_epoch, num_inputs, class_number, w_corr, train_hist_FDIM, test_hist_FDIM3, cuda)
    score_FDIM3[i] = test_multi(target_loader, cuda, w_corr, 'corr')
    print('DEMAT-Net with corr : %.4f' % (score_FDIM3[i]))

test_Acc3=np.array(test_hist_FDIM3)
test_Acc3.resize((T,n_epoch))
test_Accmean3 = np.mean(test_Acc3,0)
test_Accstd3 = np.std(test_Acc3,0)

#
type = 4
B = np.arange(0, 1.02, 0.1)

for q in range(len(B)):
    t_fuzzy = np.zeros((t_interval.shape[0], num_inputs * 3))

    for i in range(num_inputs):
        for j in range(t_interval.shape[0]):
            t_fuzzy[j, 3 * i] = t_interval[j, 2 * i]
            t_fuzzy[j, 3 * i + 1] = B[q] * t_interval[j, 2 * i] + (1 - B[q]) * t_interval[j, 2 * i + 1]
            t_fuzzy[j, 3 * i + 2] = t_interval[j, 2 * i + 1]
    # correlation weight vector
    MMD = np.zeros(len(source))
    for i in range(len(source)):
        MMD[i] = mmd(source[i], t_fuzzy, deta, type)

    if q == 0:
        best_mmd = np.sum(MMD)
        best_beta = B[q]

    if np.sum(MMD) < best_mmd:
         best_mmd= np.sum(MMD)
         best_beta = B[q]

print(best_beta)

t_fuzzy = np.zeros((t_interval.shape[0], num_inputs * 3))

for i in range(num_inputs):
    for j in range(t_interval.shape[0]):
        t_fuzzy[j, 3 * i] = t_interval[j, 2 * i]
        t_fuzzy[j, 3 * i + 1] = best_beta * t_interval[j, 2 * i] + (1 - best_beta) * t_interval[j, 2 * i + 1]
        t_fuzzy[j, 3 * i + 2] = t_interval[j, 2 * i + 1]

MMD = np.zeros(len(source))
for i in range(len(source)):
    MMD[i] = mmd(source[i], t_fuzzy, deta, type)

print(MMD)

MMD = 1 - MMD/(np.max(np.abs(MMD))*1.5)
w_corr = MMD/np.sum(MMD)

print(w_corr)

# fuzzy technique to extract crisp-valued
t_df = DF_interval(t_interval, num_inputs, best_beta)
target_data = torch.utils.data.TensorDataset(torch.tensor(t_df, dtype=torch.float32), torch.tensor(t_labels).long())
target_loader = torch.utils.data.DataLoader(target_data, batch_size)


model_root = 'models'
score_FDIM4 = np.zeros(T)

# FRAT
train_hist_FDIM = {}
train_hist_FDIM['Total_loss'] = []
train_hist_FDIM['disc_loss'] = []

test_hist_FDIM4 = []

for i in range(T):
    train_FDIM(source_loader, target_loader, lr, n_epoch, num_inputs, class_number, w_corr, train_hist_FDIM, test_hist_FDIM4, cuda)
    score_FDIM4[i] = test_multi(target_loader, cuda, w_corr, 'corr')
    print('DEMAT-Net with corr : %.4f' % (score_FDIM4[i]))

test_Acc4=np.array(test_hist_FDIM4)
test_Acc4.resize((T,n_epoch))
test_Accmean4 = np.mean(test_Acc4,0)
test_Accstd4 = np.std(test_Acc4,0)

plt.plot(range(1, n_epoch + 1),test_Accmean1,color='tomato',linestyle='-',linewidth=2)
plt.plot(range(1, n_epoch + 1),test_Accmean2,color='limegreen',linestyle='-',linewidth=2)
plt.plot(range(1, n_epoch + 1),test_Accmean3,color='deepskyblue',linestyle='-',linewidth=2)
plt.plot(range(1, n_epoch + 1),test_Accmean4,color='fuchsia',linestyle='-',linewidth=2)
plt.fill_between(range(1, n_epoch + 1), test_Accmean1 - test_Accstd1, test_Accmean1 + test_Accstd1, facecolor='tomato', alpha=0.1)
plt.fill_between(range(1, n_epoch + 1), test_Accmean2 - test_Accstd2, test_Accmean2 + test_Accstd2, facecolor='blue', alpha=0.1)
plt.fill_between(range(1, n_epoch + 1), test_Accmean3 - test_Accstd3, test_Accmean3 + test_Accstd3, facecolor='limegreen', alpha=0.1)
plt.fill_between(range(1, n_epoch + 1), test_Accmean4 - test_Accstd4, test_Accmean4 + test_Accstd4, facecolor='fuchsia', alpha=0.1)
plt.xlabel("Epochs")
plt.ylabel("Accuracy(%)")
plt.legend(['type 1', 'type 2', 'type 3', 'type 4'], loc='lower right')
plt.ylim((0.5, 0.91))
my_y_ticks = np.arange(0.5, 0.91, 0.05)
plt.yticks(my_y_ticks)
plt.savefig('/data/guanma/ml_p/FRAT/compare_fuzzy.pdf', bbox_inches='tight')
plt.show()


      

