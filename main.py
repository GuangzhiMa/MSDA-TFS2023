import argparse
import os
import torch
from train import train_FDIM, train_FDIM1
import random
from utils import Gau_data, DF_interval, mmd
from test2 import test_multi, test_multi0

import torch.backends.cudnn as cudnn

import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import sys
sys.path.append("..")

from sklearn.datasets import make_blobs

def data_create(num_inputs, class_number, sys_examples, batch_size, epsilon, sigma):
    s1_feature, s1_labels = make_blobs(n_features=num_inputs, n_samples=sys_examples, centers=class_number, random_state=120,
                                       cluster_std=[0.5, 1.8, 3.5], center_box=(-5, 10))
    s2_feature, s2_labels = make_blobs(n_features=num_inputs, n_samples=sys_examples, centers=class_number, random_state=120,
                                       cluster_std=[1.2, 1.5, 2], center_box=(5, 15))
    s3_feature, s3_labels = make_blobs(n_features=num_inputs, n_samples=sys_examples, centers=class_number, random_state=120,
                                       cluster_std=[2, 1, 1.5], center_box=(-15, 5))
    source_1 = torch.utils.data.TensorDataset(torch.tensor(s1_feature, dtype=torch.float32), torch.tensor(s1_labels).long())
    source1_loader = torch.utils.data.DataLoader(source_1, batch_size, shuffle=True)
    source_2 = torch.utils.data.TensorDataset(torch.tensor(s2_feature, dtype=torch.float32), torch.tensor(s2_labels).long())
    source2_loader = torch.utils.data.DataLoader(source_2, batch_size, shuffle=True)
    source_3 = torch.utils.data.TensorDataset(torch.tensor(s3_feature, dtype=torch.float32), torch.tensor(s3_labels).long())
    source3_loader = torch.utils.data.DataLoader(source_3, batch_size, shuffle=True)

    source_loader = [source1_loader, source2_loader, source3_loader]

    t_interval, t_labels = Gau_data(sys_examples=sys_examples, random_state=120)

    # select beta
    e = epsilon
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
            MMD[i] = mmd(source[i], t_fuzzy, sigma, 3)
        if q == 0:
            best_mmd = np.sum(MMD)
            best_beta = B[q]

        if np.sum(MMD) < best_mmd:
            best_mmd = np.sum(MMD)
            best_beta = B[q]
    print('best_beta = ', best_beta)
    t_fuzzy = np.zeros((t_interval.shape[0], num_inputs * 3))

    for i in range(num_inputs):
        for j in range(t_interval.shape[0]):
            t_fuzzy[j, 3 * i] = t_interval[j, 2 * i]
            t_fuzzy[j, 3 * i + 1] = best_beta * t_interval[j, 2 * i] + (1 - best_beta) * t_interval[j, 2 * i + 1]
            t_fuzzy[j, 3 * i + 2] = t_interval[j, 2 * i + 1]

    MMD = np.zeros(len(source))
    for i in range(len(source)):
        MMD[i] = mmd(source[i], t_fuzzy, sigma, 3)
    MMD = 1 - MMD / (np.max(np.abs(MMD)) * 1.5)
    w_corr = MMD / np.sum(MMD)

    # fuzzy technique to extract crisp-valued
    t_df = DF_interval(t_interval, num_inputs, best_beta)
    target_data = torch.utils.data.TensorDataset(torch.tensor(t_df, dtype=torch.float32), torch.tensor(t_labels).long())
    target_loader = torch.utils.data.DataLoader(target_data, batch_size, shuffle=True)

    return source_loader, target_loader, w_corr

def weather_target():
    weatherlabel0 = pd.read_excel('Seattle Tacoma.xlsx', usecols=[6])
    weatherfeature0 = pd.read_excel('Seattle Tacoma.xlsx', usecols=[1, 2, 3, 4, 5])
    weatherfeature0 = np.array(weatherfeature0)
    weatherlabels0 = np.array(weatherlabel0)
    weatherlabels0 = weatherlabels0.reshape(weatherlabels0.shape[0], )
    weatherlabel = np.zeros([2192, ])
    weatherfeature = np.zeros([2192, 10])

    for i in range(2192):
        we = weatherlabels0[4 * i:4 * i + 4].sum()
        if we > 0:
            weatherlabel[i] = 1
        else:
            weatherlabel[i] = 0

    for i in range(2192):
        we_max = np.max(weatherfeature0[4 * i:4 * i + 4, :], 0)
        we_min = np.min(weatherfeature0[4 * i:4 * i + 4, :], 0)
        for j in range(5):
            weatherfeature[i, 2 * j] = we_min[j]
            weatherfeature[i, 2 * j + 1] = we_max[j]

    t_interval = weatherfeature
    t_labels = weatherlabel.astype(int)

    return t_interval, t_labels

def weather_source():
    # Olympia
    weatherlabel0 = pd.read_excel('Olympia.xlsx', usecols=[6])
    weatherfeature0 = pd.read_excel('Olympia.xlsx', usecols=[1, 2, 3, 4, 5])
    O_feature = np.array(weatherfeature0)

    weatherlabels0 = np.array(weatherlabel0)
    O_labels = weatherlabels0.reshape(weatherlabels0.shape[0], )

    for i in range(len(O_labels)):
        if O_labels[i] > 0:
            O_labels[i] = 1
        else:
            O_labels[i] = 0

    O_weather = [O_feature, O_labels]

    # Washing
    weatherlabel0 = pd.read_excel('Washing.xlsx', usecols=[7])
    weatherfeature0 = pd.read_excel('Washing.xlsx', usecols=[1, 2, 3, 5, 6])
    W_feature = np.array(weatherfeature0)

    weatherlabels0 = np.array(weatherlabel0)
    W_labels = weatherlabels0.reshape(weatherlabels0.shape[0], )

    for i in range(len(W_labels)):
        if W_labels[i] > 0:
            W_labels[i] = 1
        else:
            W_labels[i] = 0

    W_weather = [W_feature, W_labels]

    # Portland
    weatherlabel0 = pd.read_excel('Portland.xls', usecols=[6])
    weatherfeature0 = pd.read_excel('Portland.xls', usecols=[1, 2, 3, 4, 5])
    P_feature = np.array(weatherfeature0)

    weatherlabels0 = np.array(weatherlabel0)
    P_labels = weatherlabels0.reshape(weatherlabels0.shape[0], )

    for i in range(len(P_labels)):
        if P_labels[i] > 0:
            P_labels[i] = 1
        else:
            P_labels[i] = 0

    P_weather = [P_feature, P_labels]

    # London
    weatherlabel0 = pd.read_excel('London.xls', usecols=[6])
    weatherfeature0 = pd.read_excel('London.xls', usecols=[1, 2, 3, 4, 5])
    L_feature = np.array(weatherfeature0)

    weatherlabels0 = np.array(weatherlabel0)
    L_labels = weatherlabels0.reshape(weatherlabels0.shape[0], )

    for i in range(len(L_labels)):
        if L_labels[i] > 0:
            L_labels[i] = 1
        else:
            L_labels[i] = 0

    L_weather = [L_feature, L_labels]

    return O_weather, W_weather, P_weather, L_weather

def data_create_weather(s1_feature, s2_feature, s1_labels, s2_labels, t_interval, t_labels, batch_size, epsilon, sigma):
    source_1 = torch.utils.data.TensorDataset(torch.tensor(s1_feature, dtype=torch.float32), torch.tensor(s1_labels).long())
    source1_loader = torch.utils.data.DataLoader(source_1, batch_size, shuffle=True, drop_last=True)
    source_2 = torch.utils.data.TensorDataset(torch.tensor(s2_feature, dtype=torch.float32), torch.tensor(s2_labels).long())
    source2_loader = torch.utils.data.DataLoader(source_2, batch_size, shuffle=True, drop_last=True)
    source_loader = [source1_loader, source2_loader]

    e = epsilon
    s1_e_f = np.zeros((s1_feature.shape[0], num_inputs * 3))
    s2_e_f = np.zeros((s2_feature.shape[0], num_inputs * 3))
    for i in range(num_inputs):
        for j in range(s1_feature.shape[0]):
            s1_e_f[j, 3 * i] = s1_feature[j, i] - e
            s1_e_f[j, 3 * i + 1] = s1_feature[j, i]
            s1_e_f[j, 3 * i + 2] = s1_feature[j, i] + e
    for i in range(num_inputs):
        for j in range(s2_feature.shape[0]):
            s2_e_f[j, 3 * i] = s2_feature[j, i] - e
            s2_e_f[j, 3 * i + 1] = s2_feature[j, i]
            s2_e_f[j, 3 * i + 2] = s2_feature[j, i] + e
    source = [s1_e_f, s2_e_f]

    # select beta
    # B = np.arange(0, 1.02, 0.1)
    #
    # for q in range(len(B)):
    #     t_fuzzy = np.zeros((t_interval.shape[0], num_inputs * 3))
    #
    #     for i in range(num_inputs):
    #         for j in range(t_interval.shape[0]):
    #             t_fuzzy[j, 3 * i] = t_interval[j, 2 * i]
    #             t_fuzzy[j, 3 * i + 1] = B[q] * t_interval[j, 2 * i] + (1 - B[q]) * t_interval[j, 2 * i + 1]
    #             t_fuzzy[j, 3 * i + 2] = t_interval[j, 2 * i + 1]
    #     # correlation weight vector
    #     MMD = np.zeros(len(source))
    #     for i in range(len(source)):
    #         MMD[i] = mmd(source[i], t_fuzzy, sigma, 3)
    #
    #     if q == 0:
    #         best_mmd = np.sum(MMD)
    #         best_beta = B[q]
    #
    #     if np.sum(MMD) < best_mmd:
    #          best_mmd= np.sum(MMD)
    #          best_beta = B[q]
    best_beta = 1.0
    print('best_beta = ', best_beta)
    t_fuzzy = np.zeros((t_interval.shape[0], num_inputs * 3))
    for i in range(num_inputs):
        for j in range(t_interval.shape[0]):
            t_fuzzy[j, 3 * i] = t_interval[j, 2 * i]
            t_fuzzy[j, 3 * i + 1] = best_beta * t_interval[j, 2 * i] + (1 - best_beta) * t_interval[j, 2 * i + 1]
            t_fuzzy[j, 3 * i + 2] = t_interval[j, 2 * i + 1]
    MMD = np.zeros(len(source))
    for i in range(len(source)):
        MMD[i] = mmd(source[i], t_fuzzy, sigma, 3)
    MMD = 1 - MMD/(np.max(np.abs(MMD))*1.5)
    w_corr = MMD/np.sum(MMD)

    # fuzzy technique to extract crisp-valued
    t_df = DF_interval(t_interval, num_inputs, best_beta)
    target_data = torch.utils.data.TensorDataset(torch.tensor(t_df, dtype=torch.float32), torch.tensor(t_labels).long())
    target_loader = torch.utils.data.DataLoader(target_data, batch_size, shuffle=True, drop_last=True)

    return source_loader, target_loader, w_corr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FDIM-Net')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=str, default='OW', help="source domain names for real-world dataset", choices=['OW', 'OP', 'OL', 'WP', 'WL', 'PL'])
    parser.add_argument('--t', type=str, default='S', help="target domain name for real-world dataset")
    parser.add_argument('--batch_size', type=int, default=100, help="batch_size")
    parser.add_argument('--max_epoch', type=int, default=500, help="max epoch")
    parser.add_argument('--epoch_s', type=int, default=50, help="epoch for source model training")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='synthetic', choices=['synthetic', 'weather'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--epsilon', type=float, default=1e-3, help="hyperparameter")
    parser.add_argument('--sigma', type=float, default=0.5, help="hyperparameter")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cuda = True
    cudnn.benchmark = True

    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if args.dset == 'synthetic':
        num_inputs = 2
        class_number = 3
        sys_examples = 500
        source_loader, target_loader, w_corr = data_create(num_inputs = num_inputs, class_number = class_number, sys_examples = sys_examples,
                                                           batch_size = args.batch_size, epsilon = args.epsilon, sigma = args.sigma)
    elif args.dset == 'weather':
        num_inputs = 5
        class_number = 2
        t_interval, t_labels = weather_target()
        O_weather, W_weather, P_weather, L_weather = weather_source()
        if args.s == 'OW':
            s1_feature = O_weather[0]
            s1_labels = O_weather[1]
            s2_feature = W_weather[0]
            s2_labels = W_weather[1]
        elif args.s == 'OP':
            s1_feature = O_weather[0]
            s1_labels = O_weather[1]
            s2_feature = P_weather[0]
            s2_labels = P_weather[1]
        elif args.s == 'OL':
            s1_feature = O_weather[0]
            s1_labels = O_weather[1]
            s2_feature = L_weather[0]
            s2_labels = L_weather[1]
        elif args.s == 'WP':
            s1_feature = W_weather[0]
            s1_labels = W_weather[1]
            s2_feature = P_weather[0]
            s2_labels = P_weather[1]
        elif args.s == 'WL':
            s1_feature = W_weather[0]
            s1_labels = W_weather[1]
            s2_feature = L_weather[0]
            s2_labels = L_weather[1]
        elif args.s == 'PL':
            s1_feature = P_weather[0]
            s1_labels = P_weather[1]
            s2_feature = L_weather[0]
            s2_labels = L_weather[1]
        source_loader, target_loader, w_corr = data_create_weather(s1_feature, s2_feature, s1_labels, s2_labels, t_interval, t_labels,
                                                                   batch_size = args.batch_size, epsilon = args.epsilon, sigma = args.sigma)

    model_root = 'models'
    train_hist_FDIM = {}
    train_hist_FDIM['Total_loss'] = []
    train_hist_FDIM['disc_loss'] = []
    test_hist_FDIM = []

    if args.dset == 'synthetic':
        train_FDIM(source_loader, target_loader, args.lr, args.max_epoch, args.epoch_s, num_inputs, class_number, w_corr, train_hist_FDIM, test_hist_FDIM, cuda)
        score_FDIM = test_multi(target_loader, cuda, w_corr, 'corr')
        print('FDIM-Net with corr : %.4f' % (score_FDIM))
    elif args.dset == 'weather':
        train_FDIM1(source_loader, target_loader, args.lr, args.max_epoch, args.epoch_s, num_inputs, class_number, w_corr, train_hist_FDIM, test_hist_FDIM, cuda)
        score_FDIM = test_multi0(target_loader, cuda, w_corr, 'corr')
        print('FDIM-Net with corr : %.4f' % (score_FDIM))



      
        
