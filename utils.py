import torch
import torch.nn as nn
import sys
from torch.autograd import Variable
import torch.nn.functional as F
sys.path.append("..")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
from sympy import *

import sys
sys.path.append("..")

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pylab

def Gau_data(sys_examples, random_state):

    num_inputs = 2

    t_features, t_labels = make_blobs(n_features=num_inputs, n_samples=sys_examples, centers=3, random_state=random_state,
                                      cluster_std=[0.6, 1, 3.5], center_box=(-10, 10))

    bias1_t = np.random.uniform(0.1, 0.5, (sys_examples, num_inputs))
    bias2_t = np.random.uniform(2, 4, (sys_examples, num_inputs))

    t_interval = np.zeros((t_features.shape[0], t_features.shape[1] * 2))

    for i in range(t_features.shape[1]):
        for j in range(t_features.shape[0]):
            t_interval[j, 2 * i] = t_features[j, i] - bias1_t[j, i]
            t_interval[j, 2 * i + 1] = t_features[j, i] + bias2_t[j, i]

    return t_interval, t_labels

def DF_fuzzy(feature, n_inputs):

    df_feature = np.zeros((feature.shape[0], n_inputs))
    for i in range(n_inputs):
        for j in range(feature.shape[0]):
            df_feature[j, i] = 2 * feature[j, 3 * i + 1] / 3 + (feature[j, 3 * i] + feature[j, 3 * i + 2]) / 6

    return df_feature

def DF_interval(feature, n_inputs, beta):

    df_feature = np.zeros((feature.shape[0], n_inputs))
    for i in range(n_inputs):
        for j in range(feature.shape[0]):
            df_feature[j, i] = (2 * beta / 3 + 1 / 6) * feature[j, 2 * i] + (5 / 6 - 2 * beta / 3) * feature[j, 2 * i + 1]

    return df_feature

def DF_mean(feature, n_inputs):

    m_feature = np.zeros((feature.shape[0], n_inputs))
    for i in range(n_inputs):
        for j in range(feature.shape[0]):
            m_feature[j, i] = feature[j, 3 * i + 1] / 2 + (feature[j, 3 * i] + feature[j, 3 * i + 2]) / 4

    return m_feature

def DF_M(feature, n_inputs):

    m_feature = np.zeros((feature.shape[0], n_inputs))
    for i in range(n_inputs):
        for j in range(feature.shape[0]):
            m_feature[j, i] = (feature[j, 2 * i] + feature[j, 2 * i + 1]) / 2

    return m_feature

def Split_dataset(data, label, vali_size, test_size, random_state):
    letter_train, letter_vali, y_train, y_vali = train_test_split(data, label, test_size=vali_size, random_state=random_state)
    letter_train, letter_test, y_train, y_test = train_test_split(letter_train, y_train, test_size=test_size, random_state=random_state)
    return letter_train, letter_vali, letter_test, y_train, y_vali, y_test

def M_fuzzy(feature, num_inputs):

    M_fuzzy = np.zeros((feature.shape[0], num_inputs))

    for i in range(M_fuzzy.shape[0]):
        for j in range(M_fuzzy.shape[1]):
            M_fuzzy[i, j] = (feature[i, 3 * j + 2] - feature[i, 3 * j])*(feature[i, 3 * j] + feature[i, 3 * j + 1] + feature[i, 3 * j + 2])/6

    return M_fuzzy

def dis(s1, s2, type):

    if type == 1:
        dist = ((s1[2]-s1[0]-s2[2]+s2[0])**2)/4
    elif type == 2:
        dist = ((-s1[0]*s1[1]-s1[0]**2+s1[2]*s1[1]+s1[2]**2)/6-(-s2[0]*s2[1]-s2[0]**2+s2[2]*s2[1]+s1[2]**2)/6)**2
    elif type == 3:
        dist = 2 * (s1[1] - s2[1]) ** 2 / 3 + (s1[0] - s2[0]) / 3 + (s1[2] - s2[2]) / 3 + (s1[0] - s2[0]) * (
                    s1[1] - s2[1]) / 3 + (s1[2] - s2[2]) * (s1[1] - s2[1]) / 3
    elif type == 4:
        dist = ((s1[0] - s1[2]) ** 2 + (s2[0] - s2[2]) ** 2) / 72 - ((s1[0] + s1[2] - s2[0] - s2[2]) ** 2) / 6 + 2 * (
                    s1[1] - s2[1]) ** 2 / 3 + 2 * s1[1] * (s1[0] + s1[2] - s2[0] - s2[2]) - 2 * s2[1] * (s1[0] + s1[2] - s2[0] - s2[2])

    return dist

def dis_vec(s_f, t_f, type):

    num = int(len(s_f)/3)
    dis_vec = np.zeros(num)
    for i in range(num):
        dis_vec[i] = dis(s_f[3*i:(3*i+3)], t_f[3*i:(3*i+3)], type)

    return np.sum(dis_vec)

def DIS(source_feature, tg, type):

    R = np.zeros(source_feature.shape[0])
    for i in range(len(R)):
        R[i] = dis_vec(source_feature[i], tg, type)

    return R

def kern(a, b, deta, type):

    k = np.exp(-(dis_vec(a, b, type)/(2*(deta**2))))

    return k

def mmd(S, T, deta, type):

    n = S.shape[0]
    m = T.shape[0]

    mmd1 = 0
    for i in range(n):
        for j in range(n - (i + 1)):
            mmd1 = mmd1 + kern(S[i], S[j + 1], deta, type)
    mmd1 = mmd1/(n*(n-1))

    mmd2 = 0
    for i in range(m):
        for j in range(m - (i + 1)):
            mmd2 = mmd2 + kern(T[i], T[j + 1], deta, type)
    mmd2 = mmd2 / (m*(m-1))

    mmd3 = 0
    for i in range(n):
        for j in range(m):
            mmd3 = mmd3 + kern(S[i], T[j], deta, type)
    mmd3 = 2*mmd3/(m*n)

    mmd = mmd1 + mmd2 - mmd3

    return mmd

def corr(source, target, R_max, type):

    corr = np.zeros(len(source))

    for i in range(len(source)):
        corr[i] = 0
        for j in range(target.shape[0]):
            R_ts = DIS(source[i], target[j], type)
            for p in range(len(R_ts)):
                if R_ts[p] > R_max[i]:
                   source_f = source[i]
                   R_M = R_max[i]
                   for q in range(target.shape[0]):
                       R_t = dis_vec(source_f[i], target[q], type)
                       if R_t > R_M:
                           R_M = R_t
                   R_ts[p] = 1 - R_ts[p] / R_M
                else:
                    R_ts[p] = 1 - R_ts[p]/R_max[i]
            corr[i] = corr[i] + np.sum(R_ts)

    return corr

def visualize_accuracy(test_hist):
    x = range(len(test_hist))

    target_accuracy = test_hist

    plt.plot(x, target_accuracy, color = 'cornflowerblue', label = 'target accuracy')

    plt.ylim(ymin=0.2, ymax=0.90)
    plt.xlabel('Epoch', fontsize = 16)
    plt.ylabel('Accuracy', fontsize = 16)

    plt.grid(True)
    plt.legend(['target accuracy'], loc='lower right')
    plt.tick_params(labelsize=16)
    plt.savefig('/data/guanma/ml_p/MSDAIMO/acc_line.pdf', bbox_inches='tight')
    pylab.show()

