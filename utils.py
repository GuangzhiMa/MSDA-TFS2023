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


def doublemoon_data(sys_examples, random_state):

    num_inputs = 2

    s_features, s_labels = make_moons(n_samples=(sys_examples, sys_examples), noise=0.05, random_state=random_state)
    t_features, t_labels = make_moons(n_samples=(sys_examples, sys_examples), noise=0.2, random_state=random_state)

    transformation_s = np.ones((2 * sys_examples, num_inputs)) * 10
    transformation_t = np.ones((2 * sys_examples, num_inputs)) * 8
    s_features = s_features + transformation_s
    t_features = t_features + transformation_t

    bias1_s = np.random.uniform(0.5, 1, (2 * sys_examples, num_inputs))
    bias2_s = np.random.uniform(1, 2, (2 * sys_examples, num_inputs))
    bias1_t = np.random.uniform(0.5, 1, (2 * sys_examples, num_inputs))
    bias2_t = np.random.uniform(1, 2, (2 * sys_examples, num_inputs))

    s_interval = np.zeros((s_features.shape[0], s_features.shape[1] * 2))

    for i in range(s_features.shape[1]):
        for j in range(s_features.shape[0]):
            s_interval[j, 2 * i] = s_features[j, i] - bias1_s[j, i]
            s_interval[j, 2 * i + 1] = s_features[j, i] + bias2_s[j, i]

    t_interval = np.zeros((t_features.shape[0], t_features.shape[1] * 2))

    for i in range(t_features.shape[1]):
        for j in range(t_features.shape[0]):
            t_interval[j, 2 * i] = t_features[j, i] - bias1_t[j, i]
            t_interval[j, 2 * i + 1] = t_features[j, i] + bias2_t[j, i]

    return s_features, s_interval, s_labels, t_features, t_interval, t_labels

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

def Gau_fuzzy(sys_examples, random_state):

    num_inputs = 2

    t_features, t_labels = make_blobs(n_features=num_inputs, n_samples=sys_examples, centers=3, random_state=random_state,
                                      cluster_std=[1.5, 1, 2], center_box=(20, 40))

    bias1_t = np.random.uniform(0.5, 1, (sys_examples, num_inputs))
    bias2_t = np.random.uniform(2, 5, (sys_examples, num_inputs))

    t_fuzzy = np.zeros((t_features.shape[0], t_features.shape[1] * 3))

    for i in range(t_features.shape[1]):
        for j in range(t_features.shape[0]):
            t_fuzzy[j, 3 * i] = t_features[j, i] - bias1_t[j, i]
            t_fuzzy[j, 3 * i + 1] = t_features[j, i]
            t_fuzzy[j, 3 * i + 2] = t_features[j, i] + bias2_t[j, i]

    return t_fuzzy, t_labels

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

def DFdataset(feature, labels, beta, test_size, feature_number, batch_size, random_state):
    encry_number = feature.shape[0]
    # VAL
    feature_VAL = np.array([])
    for i in range(feature_number):
        M = (2 * beta / 3 + 1 / 6) * feature[:, 2 * i] + (5 / 6 - 2 * beta / 3) * feature[:, 2 * i + 1]
        feature_VAL = np.concatenate((feature_VAL, M))
    feature_VAL = (feature_VAL.reshape(feature_number, encry_number)).T

    VAL_train, VAL_test, y_train, y_test = train_test_split(feature_VAL, labels, test_size=test_size,
                                                            random_state=random_state)
    train_data = torch.utils.data.TensorDataset(torch.tensor(VAL_train, dtype=torch.float32), torch.tensor(y_train).long())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, drop_last=True)

    test_data = torch.utils.data.TensorDataset(torch.tensor(VAL_test, dtype=torch.float32), torch.tensor(y_test).long())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, drop_last=True)

    return train_loader, test_loader


def DFdataset_s(feature, labels, beta, feature_number, batch_size):
    encry_number = feature.shape[0]
    # VAL
    feature_VAL = np.array([])
    for i in range(feature_number):
        M = (2 * beta / 3 + 1 / 6) * feature[:, 2 * i] + (5 / 6 - 2 * beta / 3) * feature[:, 2 * i + 1]
        feature_VAL = np.concatenate((feature_VAL, M))
    feature_VAL = (feature_VAL.reshape(feature_number, encry_number)).T
    feature_l = np.zeros((encry_number, feature_number))
    for j in range(feature_number):
        feature_l[:, j] = feature[:, 2 * j]
    feature_r = np.zeros((encry_number, feature_number))
    for j in range(feature_number):
        feature_r[:, j] = feature[:, 2 * j + 1]

    source = torch.utils.data.TensorDataset(torch.tensor(feature_VAL, dtype=torch.float32), torch.tensor(labels).long())
    source_loader = torch.utils.data.DataLoader(source, batch_size, drop_last=True)
    source_l = torch.utils.data.TensorDataset(torch.tensor(feature_l, dtype=torch.float32), torch.tensor(labels).long())
    source_l_loader = torch.utils.data.DataLoader(source_l, batch_size, drop_last=True)
    source_r = torch.utils.data.TensorDataset(torch.tensor(feature_r, dtype=torch.float32), torch.tensor(labels).long())
    source_r_loader = torch.utils.data.DataLoader(source_r, batch_size, drop_last=True)

    return source_loader, source_l_loader, source_r_loader


def DFdataset_t(feature, labels, beta, test_size, feature_number, batch_size, random_state):
    encry_number = feature.shape[0]
    # VAL
    feature_VAL = np.array([])
    for i in range(feature_number):
        M = (2 * beta / 3 + 1 / 6) * feature[:, 2 * i] + (5 / 6 - 2 * beta / 3) * feature[:, 2 * i + 1]
        feature_VAL = np.concatenate((feature_VAL, M))
    feature_VAL = (feature_VAL.reshape(feature_number, encry_number)).T
    feature_all = np.zeros((encry_number, feature_number * 3))
    for j in range(feature_number):
        feature_all[:, 3 * j] = feature[:, 2 * j]
        feature_all[:, 3 * j + 1] = feature[:, 2 * j + 1]
        feature_all[:, 3 * j + 2] = feature_VAL[:, j]

    VAL_train, VAL_test, y_train, y_test = train_test_split(feature_all, labels, test_size=test_size, random_state=random_state)
    l_train = np.zeros((VAL_train.shape[0], feature_number))
    for j in range(feature_number):
        l_train[:, j] = VAL_train[:, 3 * j]
    r_train = np.zeros((VAL_train.shape[0], feature_number))
    for j in range(feature_number):
        r_train[:, j] = VAL_train[:, 3 * j + 1]
    d_train = np.zeros((VAL_train.shape[0], feature_number))
    for j in range(feature_number):
        d_train[:, j] = VAL_train[:, 3 * j + 2]
    l_test = np.zeros((VAL_test.shape[0], feature_number))
    for j in range(feature_number):
        l_test[:, j] = VAL_test[:, 3 * j]
    r_test = np.zeros((VAL_test.shape[0], feature_number))
    for j in range(feature_number):
        r_test[:, j] = VAL_test[:, 3 * j + 1]
    d_test = np.zeros((VAL_test.shape[0], feature_number))
    for j in range(feature_number):
        d_test[:, j] = VAL_test[:, 3 * j + 2]
    train_l = torch.utils.data.TensorDataset(torch.tensor(l_train, dtype=torch.float32), torch.tensor(y_train).long())
    train_l_loader = torch.utils.data.DataLoader(train_l, batch_size, drop_last=True)

    train_r = torch.utils.data.TensorDataset(torch.tensor(r_train, dtype=torch.float32), torch.tensor(y_train).long())
    train_r_loader = torch.utils.data.DataLoader(train_r, batch_size, drop_last=True)

    train_d = torch.utils.data.TensorDataset(torch.tensor(d_train, dtype=torch.float32), torch.tensor(y_train).long())
    train_loader = torch.utils.data.DataLoader(train_d, batch_size, drop_last=True)
    test_d = torch.utils.data.TensorDataset(torch.tensor(d_test, dtype=torch.float32), torch.tensor(y_test).long())
    test_loader = torch.utils.data.DataLoader(test_d, batch_size, drop_last=True)

    return train_l_loader, train_r_loader, train_loader, test_loader

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

# def R(s_feature, t):
#
#     R = np.zeros(s_feature.shape[0])
#
#     R_max = 0
#     for i in range(s_feature.shape[0]):
#         for j in range(s_feature.shape[0] - (i + 1)):
#             r = np.linalg.norm(s_feature[i] - s_feature[j + 1])
#             if r > R_max:
#                 R_max = r
#
#     for i in range(s_feature.shape[0]):
#         R[i] = np.linalg.norm(s_feature[i] - t)
#         if R[i] > R_max:
#             R_max = R[i]
#
#     for i in range(len(R)):
#         R[i] = 1 - R[i]/R_max
#
#     return R

# def dis(s1, s2, type):
#
#     if type == 1:
#         if s1[0] > s2[0]:
#             s3 = s1
#             s1 = s2
#             s2 = s3
#
#         if s1[2] < s2[0]:
#             x = symbols('x')
#             f1 = (x / (s1[1] - s1[0]) - s1[0] / (s1[1] - s1[0])) ** 2
#             dist = integrate(f1, (x, s1[0], s1[1]))
#             f2 = (x / (s1[1] - s1[2]) - s1[2] / (s1[1] - s1[2])) ** 2
#             dist += integrate(f2, (x, s1[1], s1[2]))
#             f3 = (x / (s2[1] - s2[0]) - s2[0] / (s2[1] - s2[0])) ** 2
#             dist += integrate(f3, (x, s2[0], s2[1]))
#             f4 = (x / (s2[1] - s2[2]) - s2[2] / (s2[1] - s2[2])) ** 2
#             dist += integrate(f4, (x, s2[1], s2[2]))
#         elif s2[1] < s1[1]:
#             x = symbols('x')
#             f1 = (x / (s1[1] - s1[0]) - s1[0] / (s1[1] - s1[0])) ** 2
#             dist = integrate(f1, (x, s1[0], s2[0]))
#             f2 = (x / (s1[1] - s1[0]) - s1[0] / (s1[1] - s1[0]) - x / (s2[1] - s2[0]) + s2[0] / (s2[1] - s2[0])) ** 2
#             dist += integrate(f2, (x, s2[0], s2[1]))
#             f3 = (x / (s1[1] - s1[0]) - s1[0] / (s1[1] - s1[0]) - x / (s2[1] - s2[2]) + s2[2] / (s2[1] - s2[2])) ** 2
#             dist += integrate(f3, (x, s2[1], s2[2]))
#             f4 = (x / (s1[1] - s1[0]) - s1[0] / (s1[1] - s1[0])) ** 2
#             dist += integrate(f4, (x, s2[2], s1[1]))
#             f5 = (x / (s1[1] - s1[2]) - s1[2] / (s1[1] - s1[2])) ** 2
#             dist += integrate(f5, (x, s1[1], s1[2]))
#         else:
#             x = symbols('x')
#             f1 = (x / (s1[1] - s1[0]) - s1[0] / (s1[1] - s1[0])) ** 2
#             dist = integrate(f1, (x, s1[0], s1[1]))
#             f2 = (x / (s1[1] - s1[2]) - s1[2] / (s1[1] - s1[2])) ** 2
#             dist += integrate(f2, (x, s1[1], s2[0]))
#             f3 = (x / (s1[1] - s1[2]) - s1[2] / (s1[1] - s1[2]) - x / (s2[1] - s2[0]) + s2[0] / (s2[1] - s2[0])) ** 2
#             dist += integrate(f3, (x, s2[0], s2[1]))
#             f4 = (x / (s1[1] - s1[2]) - s1[2] / (s1[1] - s1[2]) - x / (s2[1] - s2[2]) + s2[2] / (s2[1] - s2[2])) ** 2
#             dist += integrate(f4, (x, s2[1], s2[2]))
#             f5 = (x / (s1[1] - s1[2]) - s1[2] / (s1[1] - s1[2])) ** 2
#             dist += integrate(f5, (x, s2[2], s1[2]))
#     elif type == 2:
#         if s1[0] > s2[0]:
#             s3 = s1
#             s1 = s2
#             s2 = s3
#
#         if s1[2] < s2[0]:
#             x = symbols('x')
#             f1 = ((x / (s1[1] - s1[0]) - s1[0] / (s1[1] - s1[0]))*x) ** 2
#             dist = integrate(f1, (x, s1[0], s1[1]))
#             f2 = ((x / (s1[1] - s1[2]) - s1[2] / (s1[1] - s1[2]))*x) ** 2
#             dist += integrate(f2, (x, s1[1], s1[2]))
#             f3 = ((x / (s2[1] - s2[0]) - s2[0] / (s2[1] - s2[0]))*x) ** 2
#             dist += integrate(f3, (x, s2[0], s2[1]))
#             f4 = ((x / (s2[1] - s2[2]) - s2[2] / (s2[1] - s2[2]))*x) ** 2
#             dist += integrate(f4, (x, s2[1], s2[2]))
#         elif s2[1] < s1[1]:
#             x = symbols('x')
#             f1 = ((x / (s1[1] - s1[0]) - s1[0] / (s1[1] - s1[0]))*x) ** 2
#             dist = integrate(f1, (x, s1[0], s2[0]))
#             f2 = ((x / (s1[1] - s1[0]) - s1[0] / (s1[1] - s1[0]) - x / (s2[1] - s2[0]) + s2[0] / (s2[1] - s2[0]))*x) ** 2
#             dist += integrate(f2, (x, s2[0], s2[1]))
#             f3 = ((x / (s1[1] - s1[0]) - s1[0] / (s1[1] - s1[0]) - x / (s2[1] - s2[2]) + s2[2] / (s2[1] - s2[2]))*x) ** 2
#             dist += integrate(f3, (x, s2[1], s2[2]))
#             f4 = ((x / (s1[1] - s1[0]) - s1[0] / (s1[1] - s1[0]))*x) ** 2
#             dist += integrate(f4, (x, s2[2], s1[1]))
#             f5 = ((x / (s1[1] - s1[2]) - s1[2] / (s1[1] - s1[2]))*x) ** 2
#             dist += integrate(f5, (x, s1[1], s1[2]))
#         else:
#             x = symbols('x')
#             f1 = ((x / (s1[1] - s1[0]) - s1[0] / (s1[1] - s1[0]))*x) ** 2
#             dist = integrate(f1, (x, s1[0], s1[1]))
#             f2 = ((x / (s1[1] - s1[2]) - s1[2] / (s1[1] - s1[2]))*x) ** 2
#             dist += integrate(f2, (x, s1[1], s2[0]))
#             f3 = ((x / (s1[1] - s1[2]) - s1[2] / (s1[1] - s1[2]) - x / (s2[1] - s2[0]) + s2[0] / (s2[1] - s2[0]))*x) ** 2
#             dist += integrate(f3, (x, s2[0], s2[1]))
#             f4 = ((x / (s1[1] - s1[2]) - s1[2] / (s1[1] - s1[2]) - x / (s2[1] - s2[2]) + s2[2] / (s2[1] - s2[2]))*x) ** 2
#             dist += integrate(f4, (x, s2[1], s2[2]))
#             f5 = ((x / (s1[1] - s1[2]) - s1[2] / (s1[1] - s1[2]))*x) ** 2
#             dist += integrate(f5, (x, s2[2], s1[2]))
#     elif type == 3:
#         dist = 2 * (s1[1] - s2[1]) ** 2 / 3 + (s1[0] - s2[0]) / 3 + (s1[2] - s2[2]) / 3 + (s1[0] - s2[0]) * (
#                     s1[1] - s2[1]) / 3 + (s1[2] - s2[2]) * (s1[1] - s2[1]) / 3
#     elif type == 4:
#         dist = ((s1[0] - s1[2]) ** 2 + (s2[0] - s2[2]) ** 2) / 72 - ((s1[0] + s1[2] - s2[0] - s2[2]) ** 2) / 6 + 2 * (
#                     s1[1] - s2[1]) ** 2 / 3 + 2 * s1[1] * (s1[0] + s1[2] - s2[0] - s2[2]) - 2 * s2[1] * (s1[0] + s1[2] - s2[0] - s2[2])
#
#     return dist

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

# def mmd(S, T, deta, type):
#
#     n = S.shape[0]
#     m = T.shape[0]
#
#     if n > m:
#         n = m
#
#     mmd = 0
#     for i in range(n):
#         for j in range(n - (i + 1)):
#             mmd = mmd + kern(S[i], S[j + 1], deta, type) + kern(T[i], T[j + 1], deta, type) - kern(S[i], T[j + 1], deta, type) - kern(T[i], S[j + 1], deta, type)
#     mmd = mmd/(n**2-n)
#
#     return mmd

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

# def R(s_feature, t, type):
#
#     R_max, R = DIS(s_feature, t, type)
#
#     for i in range(len(R)):
#         R[i] = 1 - R[i]/R_max
#
#     return R

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

# def loss_cmss(features, s_num, t_num, ad_net, weight_ad, cuda):
#     ad_out = ad_net(features)
#     dc_target = Variable(torch.from_numpy(np.array([[1]] * s_num + [[0]] * t_num)).float())
#     if cuda:
#         dc_target = dc_target.cuda()
#         weight_ad = weight_ad.cuda()
#     return nn.BCELoss(weight=weight_ad.view(-1))(ad_out.view(-1), dc_target.view(-1))

def loss_cmss(features, features_t, weight_ad, cuda):

    out_s = F.softmax(features, 1)
    out_t = F.softmax(features_t, 1)
    # print(torch.mean(torch.mul(weight_ad, -torch.log(out_s[:, 0]+ 1e-8))))
    if cuda:
        out_s = out_s.cuda()
        out_t = out_t.cuda()
        weight_ad = weight_ad.cuda()
    loss_cmss = torch.mean(torch.mul(weight_ad, -torch.log(out_s[:, 0]+ 1e-8))) + torch.mean(-torch.log(out_t[:, 1]+ 1e-8))

    return loss_cmss
