# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 19:11:26 2022

@author: 14025959_admin
"""
import torch

from torch import optim
from test2 import test_multi, test_DANN, test_Sonly, test_MSDA, test_multi0, test_CMSS
from utils import loss_cmss
import torch.nn.functional as F

import torch.backends.cudnn as cudnn

import model
import mmd
import msda
import sys
sys.path.append("..") 

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import numpy as np
import sys
from loss import Entropy, CrossEntropyLabelSmooth
sys.path.append("..")

def obtain_p(target_loader, F_net, C1_net, C2_net, cuda, feature_number):
    len_t_dataloader = len(target_loader)
    target_iter = iter(target_loader)
    pseudo_labels = []
    pseudo_target = torch.tensor([])
    for i in range(len_t_dataloader):
        data_target = target_iter.next()
        t_img, t_label = data_target
        batch_size = len(t_label)
        if cuda:
            t_img = t_img.cuda()

        feature = F_net(input_data=t_img)

        if cuda:
            feature = feature.cuda()

        label1 = C1_net(feature)
        label2 = C2_net(feature)
        pro1 = label1.data.max(1, keepdim=True)[0]
        pro2 = label2.data.max(1, keepdim=True)[0]
        pre1 = label1.data.max(1, keepdim=True)[1]
        pre2 = label2.data.max(1, keepdim=True)[1]
        t_img = t_img.cpu()
        for j in range(batch_size):
            if pre1[j].item() == pre2[j].item():
                pseudo_labels.append(pre1[j].item())
                pseudo_target = torch.cat((pseudo_target, t_img[j]), 0)
            elif pro1[j].item() >= 0.90:
                pseudo_labels.append(pre1[j].item())
                pseudo_target = torch.cat((pseudo_target, t_img[j]), 0)
            elif pro2[j].item() >= 0.90:
                pseudo_labels.append(pre2[j].item())
                pseudo_target = torch.cat((pseudo_target, t_img[j]), 0)
    pseudo_labels = torch.tensor(pseudo_labels).long()
    pseudo_target = pseudo_target.reshape(pseudo_labels.shape[0], feature_number)
    pseudo_t = torch.utils.data.TensorDataset(pseudo_target, pseudo_labels)
    pseudo_t_loader = torch.utils.data.DataLoader(pseudo_t, 200, drop_last=True)

    return pseudo_t, pseudo_t_loader

def train_FDIM1(source_loader, target_loader, lr, n_epoch, epoch_s, feature_number, class_number, w_corr, train_hist_FDIM, test_hist_FDIM, cuda):
    model_root = 'models'
    acc_max = 0
    loss_disc = torch.nn.L1Loss()
    loss_class = torch.nn.CrossEntropyLoss()
    F_net = model.Feature(feature_number)
    C1_net = model.Classifier(class_number)
    C2_net = model.Classifier(class_number)
    optimizer_F = optim.SGD(F_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_C1 = optim.SGD(C1_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_C2 = optim.SGD(C2_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)

    if cuda:
        F_net = F_net.cuda()
        C1_net = C1_net.cuda()
        C2_net = C2_net.cuda()
        loss_disc = loss_disc.cuda()

    for p in F_net.parameters():
        p.requires_grad = True
    for p in C1_net.parameters():
        p.requires_grad = True
    for p in C2_net.parameters():
        p.requires_grad = True

    for epoch0 in range(epoch_s):
        len_s1_dataloader = len(source_loader[0])
        len_s2_dataloader = len(source_loader[1])
        len_dataloader = max(len_s1_dataloader, len_s2_dataloader)
        source_iter1 = iter(source_loader[0])
        source_iter2 = iter(source_loader[1])

        for i in range(len_dataloader):

            if i % len_s1_dataloader == 0:
                source_iter1 = iter(source_loader[0])
            if i % len_s2_dataloader == 0:
                source_iter2 = iter(source_loader[1])

            data_source1 = source_iter1.next()
            s1_img, s1_label = data_source1
            data_source2 = source_iter2.next()
            s2_img, s2_label = data_source2

            F_net.zero_grad()
            if cuda:
                s1_img = s1_img.cuda()
                s2_img = s2_img.cuda()
                s1_label = s1_label.cuda()
                s2_label = s2_label.cuda()

            # extract common feature
            feature1 = F_net(input_data=s1_img)
            feature2 = F_net(input_data=s2_img)
            if cuda:
                feature1 = feature1.cuda()
                feature2 = feature2.cuda()

            C1_net.zero_grad()
            C2_net.zero_grad()

            class_labels1 = C1_net(feature1)
            class_labels2 = C2_net(feature2)

            err_class = CrossEntropyLabelSmooth(num_classes=class_number, epsilon=0.1)(class_labels1, s1_label) * w_corr[0] \
                        + CrossEntropyLabelSmooth(num_classes=class_number, epsilon=0.1)(class_labels2, s2_label) * w_corr[1]

            err_class.backward()
            optimizer_F.step()
            optimizer_C1.step()
            optimizer_C2.step()

    for epoch in range(n_epoch):
        len_t_dataloader = len(target_loader)
        target_iter = iter(target_loader)

        for i in range(len_t_dataloader):

            if i % len_t_dataloader == 0:
                target_iter = iter(target_loader)

            data_target = target_iter.next()
            t_img, t_label = data_target

            F_net.zero_grad()
            if cuda:
                t_img = t_img.cuda()

            # extract common feature
            feature_t = F_net(input_data=t_img)

            if cuda:
                feature_t = feature_t.cuda()

            C1_net.zero_grad()
            C2_net.zero_grad()
            p = float(i + epoch * len_t_dataloader) / n_epoch / len_t_dataloader
            Alpha = 2. / (1. + np.exp(-10 * p)) - 1

            class_labels01 = C1_net(feature_t)
            class_labels02 = C2_net(feature_t)

            err_disc = loss_disc(class_labels01, class_labels02)

            softmax_out0 = torch.nn.Softmax(dim=1)(class_labels01)
            entropy_loss0 = torch.mean(Entropy(softmax_out0))
            msoftmax0 = softmax_out0.mean(dim=0)
            entropy_loss0 -= torch.sum(-msoftmax0 * torch.log(msoftmax0 + 1e-5))

            softmax_out1 = torch.nn.Softmax(dim=1)(class_labels02)
            entropy_loss1 = torch.mean(Entropy(softmax_out1))
            msoftmax1 = softmax_out1.mean(dim=0)
            entropy_loss1 -= torch.sum(-msoftmax1 * torch.log(msoftmax1 + 1e-5))

            im_loss = entropy_loss0 * 1.0 * w_corr[0] + entropy_loss1 * 1.0 * w_corr[1]

            loss = Alpha * err_disc + im_loss

            loss.backward()
            optimizer_F.step()
            optimizer_C1.step()
            optimizer_C2.step()

        torch.save(F_net, '{0}/feature_model.pth'.format(model_root))
        torch.save(C1_net, '{0}/C1_model.pth'.format(model_root))
        torch.save(C2_net, '{0}/C2_model.pth'.format(model_root))
        acc = test_multi0(target_loader, cuda, w_corr, 'corr')
        sys.stdout.write('\r epoch: %d, loss_domain: %f, loss_disc: %f, acc_target: %f' \
                         % (epoch, im_loss.data.cpu().numpy(), err_disc.data.cpu().numpy(), acc))
        sys.stdout.flush()
        if acc > acc_max:
            acc_max = acc
        train_hist_FDIM['Total_loss'].append(loss.cpu().item())
        train_hist_FDIM['disc_loss'].append(err_disc.cpu().item())
        test_hist_FDIM.append(acc)

def train_FDIM(source_loader, target_loader, lr, n_epoch, epoch_s, feature_number, class_number, w_corr, train_hist_FDIM, test_hist_FDIM, cuda):
    model_root = 'models'
    acc_max = 0
    loss_disc = torch.nn.L1Loss()
    F_net = model.Feature(feature_number)
    C1_net = model.Classifier(class_number)
    C2_net = model.Classifier(class_number)
    C3_net = model.Classifier(class_number)
    optimizer_F = optim.SGD(F_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_C1 = optim.SGD(C1_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_C2 = optim.SGD(C2_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_C3 = optim.SGD(C3_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)

    if cuda:
        F_net = F_net.cuda()
        C1_net = C1_net.cuda()
        C2_net = C2_net.cuda()
        C3_net = C3_net.cuda()
        loss_disc = loss_disc.cuda()

    for p in F_net.parameters():
        p.requires_grad = True
    for p in C1_net.parameters():
        p.requires_grad = True
    for p in C2_net.parameters():
        p.requires_grad = True
    for p in C3_net.parameters():
        p.requires_grad = True

    for epoch0 in range(epoch_s):
        len_s1_dataloader = len(source_loader[0])
        len_s2_dataloader = len(source_loader[1])
        len_s3_dataloader = len(source_loader[2])
        len_dataloader = max(len_s1_dataloader, len_s2_dataloader, len_s3_dataloader)
        source_iter1 = iter(source_loader[0])
        source_iter2 = iter(source_loader[1])
        source_iter3 = iter(source_loader[2])

        for i in range(len_dataloader):

            if i % len_s1_dataloader == 0:
                source_iter1 = iter(source_loader[0])
            if i % len_s2_dataloader == 0:
                source_iter2 = iter(source_loader[1])
            if i % len_s3_dataloader == 0:
                source_iter3 = iter(source_loader[2])

            data_source1 = source_iter1.next()
            s1_img, s1_label = data_source1
            data_source2 = source_iter2.next()
            s2_img, s2_label = data_source2
            data_source3 = source_iter3.next()
            s3_img, s3_label = data_source3

            F_net.zero_grad()
            if cuda:
                s1_img = s1_img.cuda()
                s2_img = s2_img.cuda()
                s3_img = s3_img.cuda()
                s1_label = s1_label.cuda()
                s2_label = s2_label.cuda()
                s3_label = s3_label.cuda()

            # extract common feature
            feature1 = F_net(input_data=s1_img)
            feature2 = F_net(input_data=s2_img)
            feature3 = F_net(input_data=s3_img)
            if cuda:
                feature1 = feature1.cuda()
                feature2 = feature2.cuda()
                feature3 = feature3.cuda()

            C1_net.zero_grad()
            C2_net.zero_grad()
            C3_net.zero_grad()

            class_labels1 = C1_net(feature1)
            class_labels2 = C2_net(feature2)
            class_labels3 = C3_net(feature3)

            err_class = CrossEntropyLabelSmooth(num_classes=class_number, epsilon=0.1)(class_labels1, s1_label) * w_corr[0] \
                        + CrossEntropyLabelSmooth(num_classes=class_number, epsilon=0.1)(class_labels2, s2_label) * w_corr[1]\
                        + CrossEntropyLabelSmooth(num_classes=class_number, epsilon=0.1)(class_labels3, s3_label) * w_corr[2]

            err_class.backward()
            optimizer_F.step()
            optimizer_C1.step()
            optimizer_C2.step()
            optimizer_C3.step()

    for epoch in range(n_epoch):
        len_t_dataloader = len(target_loader)
        target_iter = iter(target_loader)

        for i in range(len_t_dataloader):

            if i % len_t_dataloader == 0:
                target_iter = iter(target_loader)

            data_target = target_iter.next()
            t_img, t_label = data_target

            F_net.zero_grad()
            if cuda:
                t_img = t_img.cuda()

            # extract common feature
            feature_t = F_net(input_data=t_img)

            if cuda:
                feature_t = feature_t.cuda()

            C1_net.zero_grad()
            C2_net.zero_grad()
            C3_net.zero_grad()
            p = float(i + epoch * len_t_dataloader) / n_epoch / len_t_dataloader
            Alpha = 2. / (1. + np.exp(-10 * p)) - 1

            class_labels01 = C1_net(feature_t)
            class_labels02 = C2_net(feature_t)
            class_labels03 = C3_net(feature_t)

            err_disc = (loss_disc(class_labels01, class_labels02)
            + loss_disc(class_labels01,class_labels03) + loss_disc(class_labels02, class_labels03)) / 3

            softmax_out0 = torch.nn.Softmax(dim=1)(class_labels01)
            entropy_loss0 = torch.mean(Entropy(softmax_out0))
            msoftmax0 = softmax_out0.mean(dim=0)
            entropy_loss0 -= torch.sum(-msoftmax0 * torch.log(msoftmax0 + 1e-5))

            softmax_out1 = torch.nn.Softmax(dim=1)(class_labels02)
            entropy_loss1 = torch.mean(Entropy(softmax_out1))
            msoftmax1 = softmax_out1.mean(dim=0)
            entropy_loss1 -= torch.sum(-msoftmax1 * torch.log(msoftmax1 + 1e-5))

            softmax_out2 = torch.nn.Softmax(dim=1)(class_labels03)
            entropy_loss2 = torch.mean(Entropy(softmax_out2))
            msoftmax2 = softmax_out2.mean(dim=0)
            entropy_loss2 -= torch.sum(-msoftmax2 * torch.log(msoftmax2 + 1e-5))

            im_loss = entropy_loss0 * 1.0 * w_corr[0] + entropy_loss1 * 1.0 * w_corr[1]+ entropy_loss2 * 1.0 * w_corr[2]

            loss = Alpha * err_disc + im_loss

            loss.backward()
            optimizer_F.step()
            optimizer_C1.step()
            optimizer_C2.step()
            optimizer_C3.step()

        torch.save(F_net, '{0}/feature_model.pth'.format(model_root))
        torch.save(C1_net, '{0}/C1_model.pth'.format(model_root))
        torch.save(C2_net, '{0}/C2_model.pth'.format(model_root))
        torch.save(C3_net, '{0}/C3_model.pth'.format(model_root))
        acc = test_multi0(target_loader, cuda, w_corr, 'corr')
        sys.stdout.write('\r epoch: %d, loss_domain: %f, loss_disc: %f, acc_target: %f' \
                         % (epoch, im_loss.data.cpu().numpy(), err_disc.data.cpu().numpy(), acc))
        sys.stdout.flush()
        if acc > acc_max:
            acc_max = acc
        train_hist_FDIM['Total_loss'].append(loss.cpu().item())
        train_hist_FDIM['disc_loss'].append(err_disc.cpu().item())
        test_hist_FDIM.append(acc)

def train_FRAT0(source_loader, target_loader, lr, n_epoch, feature_number, class_number, w_corr, train_hist_FRAT, test_hist_FRAT, cuda):
    model_root = 'models'
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()
    loss_disc = torch.nn.L1Loss()
    F_net = model.Feature(feature_number)
    C1_net = model.Classifier(class_number)
    C2_net = model.Classifier(class_number)
    D1_net = model.discriminator()
    D2_net = model.discriminator()
    optimizer_F = optim.Adam(F_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_C1 = optim.Adam(C1_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_C2 = optim.Adam(C2_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_D1 = optim.Adam(D1_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_D2 = optim.Adam(D2_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)

    if cuda:
        F_net = F_net.cuda()
        C1_net = C1_net.cuda()
        C2_net = C2_net.cuda()
        D1_net = D1_net.cuda()
        D2_net = D2_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()
        loss_disc = loss_disc.cuda()

    for p in F_net.parameters():
        p.requires_grad = True
    for p in C1_net.parameters():
        p.requires_grad = True
    for p in C2_net.parameters():
        p.requires_grad = True
    for p in D1_net.parameters():
        p.requires_grad = True
    for p in D2_net.parameters():
        p.requires_grad = True

    for epoch in range(n_epoch):
        len_dataloader = len(target_loader)
        source_iter1 = iter(source_loader[0])
        source_iter2 = iter(source_loader[1])
        target_iter = iter(target_loader)

        for i in range(len_dataloader):

            data_source1 = source_iter1.next()
            s1_img, s1_label = data_source1
            data_source2 = source_iter2.next()
            s2_img, s2_label = data_source2
            data_target = target_iter.next()
            t_img, t_label = data_target

            F_net.zero_grad()
            batch_size1 = len(s1_label)
            batch_size2 = len(s2_label)
            batch_size0 = len(t_label)
            if cuda:
                s1_img = s1_img.cuda()
                s2_img = s2_img.cuda()
                t_img = t_img.cuda()
                s1_label = s1_label.cuda()
                s2_label = s2_label.cuda()

            # extract common feature
            feature1 = F_net(input_data=s1_img)
            feature2 = F_net(input_data=s2_img)
            feature_t = F_net(input_data=t_img)

            if cuda:
                feature1 = feature1.cuda()
                feature2 = feature2.cuda()
                feature_t = feature_t.cuda()

            # align distribution
            D1_net.zero_grad()
            D2_net.zero_grad()
            domain_s1_label = torch.zeros(batch_size1).long()
            domain_s2_label = torch.zeros(batch_size2).long()
            domain_t_label = torch.ones(batch_size0).long()
            if cuda:
                domain_s1_label = domain_s1_label.cuda()
                domain_s2_label = domain_s2_label.cuda()
                domain_t_label = domain_t_label.cuda()

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            Alpha = 2. / (1. + np.exp(-10 * p)) - 1

            domain_labels1 = D1_net(feature1, Alpha)
            domain_labels2 = D2_net(feature2, Alpha)
            domain_labels01 = D1_net(feature_t, Alpha)
            domain_labels02 = D2_net(feature_t, Alpha)
            err_s1_domain = (loss_domain(domain_labels1, domain_s1_label) + loss_domain(domain_labels01, domain_t_label))*w_corr[0]
            err_s2_domain = (loss_domain(domain_labels2, domain_s2_label) + loss_domain(domain_labels02, domain_t_label))*w_corr[1]
            err_domain = err_s1_domain + err_s2_domain

            C1_net.zero_grad()
            C2_net.zero_grad()

            class_labels1 = C1_net(feature1)
            class_labels2 = C2_net(feature2)
            class_labels01 = C1_net(feature_t)
            class_labels02 = C2_net(feature_t)

            err_class = loss_class(class_labels1, s1_label)*w_corr[0] + loss_class(class_labels2, s2_label)*w_corr[1]
            err_disc = loss_disc(class_labels01, class_labels02)
            err_co = err_class + Alpha * err_disc

            loss = err_co + err_domain

            loss.backward()
            optimizer_F.step()
            optimizer_D1.step()
            optimizer_D2.step()
            optimizer_C1.step()
            optimizer_C2.step()

        torch.save(F_net, '{0}/feature_model.pth'.format(model_root))
        torch.save(C1_net, '{0}/C1_model.pth'.format(model_root))
        torch.save(C2_net, '{0}/C2_model.pth'.format(model_root))
        acc = test_multi0(target_loader, cuda, w_corr, 'corr')
        sys.stdout.write('\r epoch: %d, loss_domain: %f, loss_class: %f, loss_disc: %f, acc_target: %f' \
                         % (epoch, err_domain.data.cpu().numpy(), err_class.data.cpu().numpy(), err_disc.data.cpu().numpy(), acc))
        sys.stdout.flush()
        train_hist_FRAT['Total_loss'].append(loss.cpu().item())
        train_hist_FRAT['disc_loss'].append(err_disc.cpu().item())
        test_hist_FRAT.append(acc)

def train_DEMAT(source_loader, target_loader, lr, n_epoch, feature_number, class_number, alpha, w_corr, train_hist_DEMAT, test_hist_DEMAT, cuda):
    model_root = 'models'
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()
    loss_disc = torch.nn.L1Loss()
    F_net = model.Feature(feature_number)
    C1_net = model.Classifier(class_number)
    C2_net = model.Classifier(class_number)
    C3_net = model.Classifier(class_number)
    D1_net = model.discriminator()
    D2_net = model.discriminator()
    D3_net = model.discriminator()
    optimizer_F = optim.Adam(F_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_C1 = optim.Adam(C1_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_C2 = optim.Adam(C2_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_C3 = optim.Adam(C3_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_D1 = optim.Adam(D1_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_D2 = optim.Adam(D2_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_D3 = optim.Adam(D3_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # scheduler_F = optim.lr_scheduler.StepLR(optimizer_F, step_size=10, gamma=0.1)
    # scheduler_C1 = optim.lr_scheduler.StepLR(optimizer_C1, step_size=10, gamma=0.1)
    # scheduler_C2 = optim.lr_scheduler.StepLR(optimizer_C2, step_size=10, gamma=0.1)
    # scheduler_C3 = optim.lr_scheduler.StepLR(optimizer_C3, step_size=10, gamma=0.1)
    # scheduler_D1 = optim.lr_scheduler.StepLR(optimizer_C1, step_size=10, gamma=0.1)
    # scheduler_D2 = optim.lr_scheduler.StepLR(optimizer_C2, step_size=10, gamma=0.1)
    # scheduler_D3 = optim.lr_scheduler.StepLR(optimizer_C3, step_size=10, gamma=0.1)

    if cuda:
        F_net = F_net.cuda()
        C1_net = C1_net.cuda()
        C2_net = C2_net.cuda()
        C3_net = C3_net.cuda()
        D1_net = D1_net.cuda()
        D2_net = D2_net.cuda()
        D3_net = D3_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()
        loss_disc = loss_disc.cuda()

    for p in F_net.parameters():
        p.requires_grad = True
    for p in C1_net.parameters():
        p.requires_grad = True
    for p in C2_net.parameters():
        p.requires_grad = True
    for p in C3_net.parameters():
        p.requires_grad = True
    for p in D1_net.parameters():
        p.requires_grad = True
    for p in D2_net.parameters():
        p.requires_grad = True
    for p in D3_net.parameters():
        p.requires_grad = True

    for epoch in range(n_epoch):
        len_dataloader = len(target_loader)
        source_iter1 = iter(source_loader[0])
        source_iter2 = iter(source_loader[1])
        source_iter3 = iter(source_loader[2])
        target_iter = iter(target_loader)

        for i in range(len_dataloader):

            data_source1 = source_iter1.next()
            s1_img, s1_label = data_source1
            data_source2 = source_iter2.next()
            s2_img, s2_label = data_source2
            data_source3 = source_iter3.next()
            s3_img, s3_label = data_source3
            data_target = target_iter.next()
            t_img, t_label = data_target

            F_net.zero_grad()
            batch_size1 = len(s1_label)
            batch_size2 = len(s2_label)
            batch_size3 = len(s3_label)
            batch_size0 = len(t_label)
            if cuda:
                s1_img = s1_img.cuda()
                s2_img = s2_img.cuda()
                s3_img = s3_img.cuda()
                t_img = t_img.cuda()
                s1_label = s1_label.cuda()
                s2_label = s2_label.cuda()
                s3_label = s3_label.cuda()

            # extract common feature
            feature1 = F_net(input_data=s1_img)
            feature2 = F_net(input_data=s2_img)
            feature3 = F_net(input_data=s3_img)
            feature_t = F_net(input_data=t_img)

            if cuda:
                feature1 = feature1.cuda()
                feature2 = feature2.cuda()
                feature3 = feature3.cuda()
                feature_t = feature_t.cuda()

            # align distribution
            D1_net.zero_grad()
            D2_net.zero_grad()
            D3_net.zero_grad()
            domain_s1_label = torch.zeros(batch_size1).long()
            domain_s2_label = torch.zeros(batch_size2).long()
            domain_s3_label = torch.zeros(batch_size3).long()
            domain_t_label = torch.ones(batch_size0).long()
            if cuda:
                domain_s1_label = domain_s1_label.cuda()
                domain_s2_label = domain_s2_label.cuda()
                domain_s3_label = domain_s3_label.cuda()
                domain_t_label = domain_t_label.cuda()

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            Alpha = 2. / (1. + np.exp(-10 * p)) - 1

            domain_labels1 = D1_net(feature1, Alpha)
            domain_labels2 = D2_net(feature2, Alpha)
            domain_labels3 = D3_net(feature3, Alpha)
            domain_labels01 = D1_net(feature_t, Alpha)
            domain_labels02 = D2_net(feature_t, Alpha)
            domain_labels03 = D3_net(feature_t, Alpha)
            err_s1_domain = loss_domain(domain_labels1, domain_s1_label) + loss_domain(domain_labels01, domain_t_label)
            err_s2_domain = loss_domain(domain_labels2, domain_s2_label) + loss_domain(domain_labels02, domain_t_label)
            err_s3_domain = loss_domain(domain_labels3, domain_s3_label) + loss_domain(domain_labels03, domain_t_label)
            d_sum = err_s1_domain.data.cpu().numpy() + err_s2_domain.data.cpu().numpy() + err_s3_domain.data.cpu().numpy()
            d1 = err_s1_domain.data.cpu().numpy() / d_sum
            d2 = err_s2_domain.data.cpu().numpy() / d_sum
            d3 = err_s3_domain.data.cpu().numpy() / d_sum
            # err_domain = err_s1_domain * d1 + err_s2_domain * d2 + err_s3_domain * d3
            if d1 > alpha and d2 > alpha and d3 > alpha:
                err_domain = (err_s1_domain + err_s2_domain + err_s3_domain)/3
            elif d1 > alpha and d2 > alpha and d3 <= alpha:
                err_domain = (err_s1_domain + err_s2_domain)/2
            elif d1 > alpha and d2 <= alpha and d3 > alpha:
                err_domain = (err_s1_domain + err_s3_domain)/2
            elif d1 <= alpha and d2 > alpha and d3 > alpha:
                err_domain = (err_s2_domain + err_s3_domain)/2
            elif d1 > alpha and d2 <= alpha and d3 <= alpha:
                err_domain = err_s1_domain
            elif d1 <= alpha and d2 > alpha and d3 <= alpha:
                err_domain = err_s2_domain
            elif d1 <= alpha and d2 <= alpha and d3 > alpha:
                err_domain = err_s3_domain

            C1_net.zero_grad()
            C2_net.zero_grad()
            C3_net.zero_grad()

            class_labels1 = C1_net(feature1)
            class_labels2 = C2_net(feature2)
            class_labels3 = C3_net(feature3)
            class_labels01 = C1_net(feature_t)
            class_labels02 = C2_net(feature_t)
            class_labels03 = C3_net(feature_t)

            err_class = loss_class(class_labels1, s1_label) + loss_class(class_labels2, s2_label) + loss_class(class_labels3, s3_label)
            err_disc = (loss_disc(class_labels01, class_labels02) + loss_disc(class_labels01,class_labels03) + loss_disc(class_labels02, class_labels03)) / 3
            err_co = err_class + Alpha * err_disc

            loss = err_co + err_domain

            loss.backward()
            optimizer_F.step()
            optimizer_D1.step()
            optimizer_D2.step()
            optimizer_D3.step()
            optimizer_C1.step()
            optimizer_C2.step()
            optimizer_C3.step()

        torch.save(F_net, '{0}/feature_model.pth'.format(model_root))
        torch.save(C1_net, '{0}/C1_model.pth'.format(model_root))
        torch.save(C2_net, '{0}/C2_model.pth'.format(model_root))
        torch.save(C3_net, '{0}/C3_model.pth'.format(model_root))
        acc = test_multi(target_loader, cuda, w_corr, 'corr')
        sys.stdout.write('\r epoch: %d, loss_domain: %f, loss_class: %f, loss_disc: %f, acc_target: %f' \
                         % (epoch, err_domain.data.cpu().numpy(), err_class.data.cpu().numpy(), err_disc.data.cpu().numpy(), acc))
        sys.stdout.flush()
        train_hist_DEMAT['Total_loss'].append(loss.cpu().item())
        train_hist_DEMAT['disc_loss'].append(err_disc.cpu().item())
        test_hist_DEMAT.append(acc)

        # scheduler_F.step()
        # scheduler_C1.step()
        # scheduler_C2.step()
        # scheduler_C3.step()
        # scheduler_D1.step()
        # scheduler_D2.step()
        # scheduler_D3.step()

def train_DEMAT_fullD(source_loader, target_loader, lr, n_epoch, feature_number, class_number, cuda):
    model_root = 'models'
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()
    loss_disc = torch.nn.L1Loss()
    F_net = model.Feature(feature_number)
    C1_net = model.Classifier(class_number)
    C2_net = model.Classifier(class_number)
    C3_net = model.Classifier(class_number)
    D1_net = model.discriminator()
    D2_net = model.discriminator()
    D3_net = model.discriminator()
    optimizer_F = optim.Adam(F_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_C1 = optim.Adam(C1_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_C2 = optim.Adam(C2_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_C3 = optim.Adam(C3_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_D1 = optim.Adam(D1_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_D2 = optim.Adam(D2_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_D3 = optim.Adam(D3_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # scheduler_F = optim.lr_scheduler.StepLR(optimizer_F, step_size=10, gamma=0.1)
    # scheduler_C1 = optim.lr_scheduler.StepLR(optimizer_C1, step_size=10, gamma=0.1)
    # scheduler_C2 = optim.lr_scheduler.StepLR(optimizer_C2, step_size=10, gamma=0.1)
    # scheduler_C3 = optim.lr_scheduler.StepLR(optimizer_C3, step_size=10, gamma=0.1)
    # scheduler_D1 = optim.lr_scheduler.StepLR(optimizer_C1, step_size=10, gamma=0.1)
    # scheduler_D2 = optim.lr_scheduler.StepLR(optimizer_C2, step_size=10, gamma=0.1)
    # scheduler_D3 = optim.lr_scheduler.StepLR(optimizer_C3, step_size=10, gamma=0.1)

    if cuda:
        F_net = F_net.cuda()
        C1_net = C1_net.cuda()
        C2_net = C2_net.cuda()
        C3_net = C3_net.cuda()
        D1_net = D1_net.cuda()
        D2_net = D2_net.cuda()
        D3_net = D3_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()
        loss_disc = loss_disc.cuda()

    for p in F_net.parameters():
        p.requires_grad = True
    for p in C1_net.parameters():
        p.requires_grad = True
    for p in C2_net.parameters():
        p.requires_grad = True
    for p in C3_net.parameters():
        p.requires_grad = True
    for p in D1_net.parameters():
        p.requires_grad = True
    for p in D2_net.parameters():
        p.requires_grad = True
    for p in D3_net.parameters():
        p.requires_grad = True

    for epoch in range(n_epoch):
        len_dataloader = len(target_loader)
        source_iter1 = iter(source_loader[0])
        source_iter2 = iter(source_loader[1])
        source_iter3 = iter(source_loader[2])
        target_iter = iter(target_loader)

        for i in range(len_dataloader):

            data_source1 = source_iter1.next()
            s1_img, s1_label = data_source1
            data_source2 = source_iter2.next()
            s2_img, s2_label = data_source2
            data_source3 = source_iter3.next()
            s3_img, s3_label = data_source3
            data_target = target_iter.next()
            t_img, t_label = data_target

            F_net.zero_grad()
            batch_size1 = len(s1_label)
            batch_size2 = len(s2_label)
            batch_size3 = len(s3_label)
            batch_size0 = len(t_label)
            if cuda:
                s1_img = s1_img.cuda()
                s2_img = s2_img.cuda()
                s3_img = s3_img.cuda()
                t_img = t_img.cuda()
                s1_label = s1_label.cuda()
                s2_label = s2_label.cuda()
                s3_label = s3_label.cuda()

            # extract common feature
            feature1 = F_net(input_data=s1_img)
            feature2 = F_net(input_data=s2_img)
            feature3 = F_net(input_data=s3_img)
            feature_t = F_net(input_data=t_img)

            if cuda:
                feature1 = feature1.cuda()
                feature2 = feature2.cuda()
                feature3 = feature3.cuda()
                feature_t = feature_t.cuda()

            # align distribution
            D1_net.zero_grad()
            D2_net.zero_grad()
            D3_net.zero_grad()
            domain_s1_label = torch.zeros(batch_size1).long()
            domain_s2_label = torch.zeros(batch_size2).long()
            domain_s3_label = torch.zeros(batch_size3).long()
            domain_t_label = torch.ones(batch_size0).long()
            if cuda:
                domain_s1_label = domain_s1_label.cuda()
                domain_s2_label = domain_s2_label.cuda()
                domain_s3_label = domain_s3_label.cuda()
                domain_t_label = domain_t_label.cuda()

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            Alpha = 2. / (1. + np.exp(-10 * p)) - 1

            domain_labels1 = D1_net(feature1, Alpha)
            domain_labels2 = D2_net(feature2, Alpha)
            domain_labels3 = D3_net(feature3, Alpha)
            domain_labels01 = D1_net(feature_t, Alpha)
            domain_labels02 = D2_net(feature_t, Alpha)
            domain_labels03 = D3_net(feature_t, Alpha)
            err_s1_domain = loss_domain(domain_labels1, domain_s1_label) + loss_domain(domain_labels01, domain_t_label)
            err_s2_domain = loss_domain(domain_labels2, domain_s2_label) + loss_domain(domain_labels02, domain_t_label)
            err_s3_domain = loss_domain(domain_labels3, domain_s3_label) + loss_domain(domain_labels03, domain_t_label)
            err_domain = (err_s1_domain + err_s2_domain + err_s3_domain)/3

            C1_net.zero_grad()
            C2_net.zero_grad()
            C3_net.zero_grad()

            class_labels1 = C1_net(feature1)
            class_labels2 = C2_net(feature2)
            class_labels3 = C3_net(feature3)
            class_labels01 = C1_net(feature_t)
            class_labels02 = C2_net(feature_t)
            class_labels03 = C3_net(feature_t)

            err_class = loss_class(class_labels1, s1_label) + loss_class(class_labels2, s2_label) + loss_class(class_labels3, s3_label)
            err_disc = (loss_disc(class_labels01, class_labels02) + loss_disc(class_labels01,class_labels03) + loss_disc(class_labels02, class_labels03)) / 3
            err_co = err_class + Alpha * err_disc

            loss = err_co + err_domain

            loss.backward()
            optimizer_F.step()
            optimizer_D1.step()
            optimizer_D2.step()
            optimizer_D3.step()
            optimizer_C1.step()
            optimizer_C2.step()
            optimizer_C3.step()

        sys.stdout.write('\r epoch: %d, loss_domain: %f, loss_class: %f, loss_disc: %f' \
                         % (epoch, err_domain.data.cpu().numpy(), err_class.data.cpu().numpy(), err_disc))
        sys.stdout.flush()
        torch.save(F_net, '{0}/feature_model.pth'.format(model_root))
        torch.save(C1_net, '{0}/C1_model.pth'.format(model_root))
        torch.save(C2_net, '{0}/C2_model.pth'.format(model_root))
        torch.save(C3_net, '{0}/C3_model.pth'.format(model_root))

        # scheduler_F.step()
        # scheduler_C1.step()
        # scheduler_C2.step()
        # scheduler_C3.step()
        # scheduler_D1.step()
        # scheduler_D2.step()
        # scheduler_D3.step()

def train_MSDA(source_loader, target_loader, lr, n_epoch, feature_number, class_number, cuda):
    model_root = 'models'
    acc_max = 0
    loss_class = torch.nn.NLLLoss()
    loss_disc = torch.nn.L1Loss()
    F_net = model.Feature(feature_number)
    C1_net = model.Classifier0(class_number)
    C2_net = model.Classifier0(class_number)
    # optimizer_F = optim.Adam(F_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # optimizer_C1 = optim.Adam(C1_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # optimizer_C2 = optim.Adam(C2_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_F = optim.SGD(F_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_C1 = optim.SGD(C1_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_C2 = optim.SGD(C2_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)

    if cuda:
        F_net = F_net.cuda()
        C1_net = C1_net.cuda()
        C2_net = C2_net.cuda()
        loss_class = loss_class.cuda()
        loss_disc = loss_disc.cuda()

    for p in F_net.parameters():
        p.requires_grad = True
    for p in C1_net.parameters():
        p.requires_grad = True
    for p in C2_net.parameters():
        p.requires_grad = True

    for epoch in range(n_epoch):
        len_t_dataloader = len(target_loader)
        len_s1_dataloader = len(source_loader[0])
        len_s2_dataloader = len(source_loader[1])
        len_s3_dataloader = len(source_loader[2])
        len_dataloader = max(len_t_dataloader, len_s1_dataloader, len_s2_dataloader, len_s3_dataloader)
        source_iter1 = iter(source_loader[0])
        source_iter2 = iter(source_loader[1])
        source_iter3 = iter(source_loader[2])
        target_iter = iter(target_loader)

        for i in range(len_dataloader):

            if i % len_t_dataloader == 0:
                target_iter = iter(target_loader)
            if i % len_s1_dataloader == 0:
                source_iter1 = iter(source_loader[0])
            if i % len_s2_dataloader == 0:
                source_iter2 = iter(source_loader[1])
            if i % len_s3_dataloader == 0:
                source_iter3 = iter(source_loader[2])

            data_source1 = source_iter1.next()
            s1_img, s1_label = data_source1
            data_source2 = source_iter2.next()
            s2_img, s2_label = data_source2
            data_source3 = source_iter3.next()
            s3_img, s3_label = data_source3
            data_target = target_iter.next()
            t_img, t_label = data_target

            F_net.zero_grad()
            C1_net.zero_grad()
            C2_net.zero_grad()
            if cuda:
                s1_img = s1_img.cuda()
                s2_img = s2_img.cuda()
                s3_img = s3_img.cuda()
                t_img = t_img.cuda()
                s1_label = s1_label.cuda()
                s2_label = s2_label.cuda()
                s3_label = s3_label.cuda()

            # extract common feature
            feature1 = F_net(input_data=s1_img)
            feature2 = F_net(input_data=s2_img)
            feature3 = F_net(input_data=s3_img)
            feature_t = F_net(input_data=t_img)

            if cuda:
                feature1 = feature1.cuda()
                feature2 = feature2.cuda()
                feature3 = feature3.cuda()
                feature_t = feature_t.cuda()

            class_labels11 = C1_net(feature1)
            class_labels21 = C2_net(feature1)
            class_labels12 = C1_net(feature2)
            class_labels22 = C2_net(feature2)
            class_labels13 = C1_net(feature3)
            class_labels23 = C2_net(feature3)

            loss_s_c1 = loss_class(class_labels11, s1_label) + loss_class(class_labels12, s2_label) + loss_class(class_labels13, s3_label)
            loss_s_c2 = loss_class(class_labels21, s1_label) + loss_class(class_labels22, s2_label) + loss_class(class_labels23, s3_label)
            loss_msda = 0.0005 * msda.msda_regulizer(feature1, feature2, feature3, feature_t, 4)

            loss = loss_s_c1 + loss_s_c2 + loss_msda

            loss.backward()
            optimizer_F.step()
            optimizer_C1.step()
            optimizer_C2.step()

            F_net.zero_grad()
            C1_net.zero_grad()
            C2_net.zero_grad()

            feature1 = F_net(input_data=s1_img)
            feature2 = F_net(input_data=s2_img)
            feature3 = F_net(input_data=s3_img)
            feature_t = F_net(input_data=t_img)

            if cuda:
                feature1 = feature1.cuda()
                feature2 = feature2.cuda()
                feature3 = feature3.cuda()
                feature_t = feature_t.cuda()

            class_labels11 = C1_net(feature1)
            class_labels21 = C2_net(feature1)
            class_labels12 = C1_net(feature2)
            class_labels22 = C2_net(feature2)
            class_labels13 = C1_net(feature3)
            class_labels23 = C2_net(feature3)
            class_labels_t1 = C1_net(feature_t)
            class_labels_t2 = C2_net(feature_t)

            loss_s_c1 = loss_class(class_labels11, s1_label) + loss_class(class_labels12, s2_label) + loss_class(
                class_labels13, s3_label)
            loss_s_c2 = loss_class(class_labels21, s1_label) + loss_class(class_labels22, s2_label) + loss_class(
                class_labels23, s3_label)
            loss_msda = 0.0005 * msda.msda_regulizer(feature1, feature2, feature3, feature_t, 4)

            loss_dis = loss_disc(class_labels_t1, class_labels_t2)
            loss_s = loss_s_c1 + loss_s_c2 + loss_msda
            loss = loss_s - loss_dis
            loss.backward()
            optimizer_F.step()
            optimizer_C1.step()
            optimizer_C2.step()

            for i in range(4):
                F_net.zero_grad()
                C1_net.zero_grad()
                C2_net.zero_grad()
                feature_t = F_net(input_data=t_img)
                if cuda:
                    feature_t = feature_t.cuda()
                class_labels_t1 = C1_net(feature_t)
                class_labels_t2 = C2_net(feature_t)
                loss_dis = loss_disc(class_labels_t1, class_labels_t2)
                loss_dis.backward()
                optimizer_F.step()
                optimizer_C1.step()
                optimizer_C2.step()

        torch.save(F_net, '{0}/feature_model.pth'.format(model_root))
        torch.save(C1_net, '{0}/C1_model.pth'.format(model_root))
        torch.save(C2_net, '{0}/C2_model.pth'.format(model_root))
        acc = test_MSDA(target_loader, cuda)
        if acc > acc_max:
            acc_max = acc
        sys.stdout.write('\r epoch: %d, loss_s: %f, loss: %f, loss_disc: %f, acc: %f' \
                         % (epoch, loss_s.data.cpu().numpy(), loss.data.cpu().numpy(), loss_dis, acc))
        sys.stdout.flush()

def train_MSDA1(source_loader, target_loader, lr, n_epoch, feature_number, class_number, cuda):
    model_root = 'models'
    acc_max = 0
    loss_class = torch.nn.NLLLoss()
    loss_disc = torch.nn.L1Loss()
    F_net = model.Feature(feature_number)
    C1_net = model.Classifier0(class_number)
    C2_net = model.Classifier0(class_number)
    # optimizer_F = optim.Adam(F_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # optimizer_C1 = optim.Adam(C1_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # optimizer_C2 = optim.Adam(C2_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_F = optim.SGD(F_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_C1 = optim.SGD(C1_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_C2 = optim.SGD(C2_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)

    if cuda:
        F_net = F_net.cuda()
        C1_net = C1_net.cuda()
        C2_net = C2_net.cuda()
        loss_class = loss_class.cuda()
        loss_disc = loss_disc.cuda()

    for p in F_net.parameters():
        p.requires_grad = True
    for p in C1_net.parameters():
        p.requires_grad = True
    for p in C2_net.parameters():
        p.requires_grad = True

    for epoch in range(n_epoch):
        len_t_dataloader = len(target_loader)
        len_s1_dataloader = len(source_loader[0])
        len_s2_dataloader = len(source_loader[1])
        len_dataloader = max(len_t_dataloader, len_s1_dataloader, len_s2_dataloader)
        source_iter1 = iter(source_loader[0])
        source_iter2 = iter(source_loader[1])
        target_iter = iter(target_loader)

        for i in range(len_dataloader):

            if i % len_t_dataloader == 0:
                target_iter = iter(target_loader)
            if i % len_s1_dataloader == 0:
                source_iter1 = iter(source_loader[0])
            if i % len_s2_dataloader == 0:
                source_iter2 = iter(source_loader[1])

            data_source1 = source_iter1.next()
            s1_img, s1_label = data_source1
            data_source2 = source_iter2.next()
            s2_img, s2_label = data_source2
            data_target = target_iter.next()
            t_img, t_label = data_target

            F_net.zero_grad()
            C1_net.zero_grad()
            C2_net.zero_grad()
            if cuda:
                s1_img = s1_img.cuda()
                s2_img = s2_img.cuda()
                t_img = t_img.cuda()
                s1_label = s1_label.cuda()
                s2_label = s2_label.cuda()

            # extract common feature
            feature1 = F_net(input_data=s1_img)
            feature2 = F_net(input_data=s2_img)
            feature_t = F_net(input_data=t_img)

            if cuda:
                feature1 = feature1.cuda()
                feature2 = feature2.cuda()
                feature_t = feature_t.cuda()

            class_labels11 = C1_net(feature1)
            class_labels21 = C2_net(feature1)
            class_labels12 = C1_net(feature2)
            class_labels22 = C2_net(feature2)

            loss_s_c1 = loss_class(class_labels11, s1_label) + loss_class(class_labels12, s2_label)
            loss_s_c2 = loss_class(class_labels21, s1_label) + loss_class(class_labels22, s2_label)
            loss_msda = 0.0005 * msda.msda_regulizer(feature1, feature2, feature_t, 4)

            loss = loss_s_c1 + loss_s_c2 + loss_msda

            loss.backward()
            optimizer_F.step()
            optimizer_C1.step()
            optimizer_C2.step()

            F_net.zero_grad()
            C1_net.zero_grad()
            C2_net.zero_grad()

            feature1 = F_net(input_data=s1_img)
            feature2 = F_net(input_data=s2_img)
            feature_t = F_net(input_data=t_img)

            if cuda:
                feature1 = feature1.cuda()
                feature2 = feature2.cuda()
                feature_t = feature_t.cuda()

            class_labels11 = C1_net(feature1)
            class_labels21 = C2_net(feature1)
            class_labels12 = C1_net(feature2)
            class_labels22 = C2_net(feature2)
            class_labels_t1 = C1_net(feature_t)
            class_labels_t2 = C2_net(feature_t)

            loss_s_c1 = loss_class(class_labels11, s1_label) + loss_class(class_labels12, s2_label)
            loss_s_c2 = loss_class(class_labels21, s1_label) + loss_class(class_labels22, s2_label)
            loss_msda = 0.0005 * msda.msda_regulizer(feature1, feature2, feature_t, 4)

            loss_dis = loss_disc(class_labels_t1, class_labels_t2)
            loss_s = loss_s_c1 + loss_s_c2 + loss_msda
            loss = loss_s - loss_dis
            loss.backward()
            optimizer_F.step()
            optimizer_C1.step()
            optimizer_C2.step()

            for i in range(4):
                F_net.zero_grad()
                C1_net.zero_grad()
                C2_net.zero_grad()
                feature_t = F_net(input_data=t_img)
                if cuda:
                    feature_t = feature_t.cuda()
                class_labels_t1 = C1_net(feature_t)
                class_labels_t2 = C2_net(feature_t)
                loss_dis = loss_disc(class_labels_t1, class_labels_t2)
                loss_dis.backward()
                optimizer_F.step()
                optimizer_C1.step()
                optimizer_C2.step()

        torch.save(F_net, '{0}/feature_model.pth'.format(model_root))
        torch.save(C1_net, '{0}/C1_model.pth'.format(model_root))
        torch.save(C2_net, '{0}/C2_model.pth'.format(model_root))
        acc = test_MSDA(target_loader, cuda)
        if acc > acc_max:
            acc_max = acc
        sys.stdout.write('\r epoch: %d, loss_s: %f, loss: %f, loss_disc: %f, acc: %f' \
                         % (epoch, loss_s.data.cpu().numpy(), loss.data.cpu().numpy(), loss_dis, acc))
        sys.stdout.flush()

def train_MFSAN(source_loader, target_loader, lr, n_epoch, feature_number, class_number, cuda):
    model_root = 'models'
    acc_max = 0
    loss_class = torch.nn.NLLLoss()
    loss_disc = torch.nn.L1Loss()
    F_net = model.Feature(feature_number)
    F1_net = model.Feature_s()
    F2_net = model.Feature_s()
    # F3_net = model.Feature_s()
    C1_net = model.Classifier0(class_number)
    C2_net = model.Classifier0(class_number)
    # C3_net = model.Classifier(class_number)
    # optimizer_F = optim.Adam(F_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # optimizer_F1 = optim.Adam(F1_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # optimizer_F2 = optim.Adam(F2_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # optimizer_F3 = optim.Adam(F3_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # optimizer_C1 = optim.Adam(C1_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # optimizer_C2 = optim.Adam(C2_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # optimizer_C3 = optim.Adam(C3_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_F = optim.SGD(F_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_C1 = optim.SGD(C1_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_C2 = optim.SGD(C2_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    # optimizer_C3 = optim.SGD(C3_net.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)
    optimizer_F1 = optim.SGD(F1_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_F2 = optim.SGD(F2_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    # optimizer_F3 = optim.SGD(F3_net.parameters(), lr=lr, weight_decay=0.0005, momentum=0.9)

    if cuda:
        F_net = F_net.cuda()
        F1_net = F1_net.cuda()
        F2_net = F2_net.cuda()
        # F3_net = F3_net.cuda()
        C1_net = C1_net.cuda()
        C2_net = C2_net.cuda()
        # C3_net = C3_net.cuda()
        loss_class = loss_class.cuda()
        loss_disc = loss_disc.cuda()

    for p in F_net.parameters():
        p.requires_grad = True
    for p in C1_net.parameters():
        p.requires_grad = True
    for p in C2_net.parameters():
        p.requires_grad = True
    # for p in C3_net.parameters():
    #     p.requires_grad = True
    for p in F1_net.parameters():
        p.requires_grad = True
    for p in F2_net.parameters():
        p.requires_grad = True
    # for p in F3_net.parameters():
    #     p.requires_grad = True

    for epoch in range(n_epoch):
        len_t_dataloader = len(target_loader)
        len_s1_dataloader = len(source_loader[0])
        len_s2_dataloader = len(source_loader[1])
        # len_s3_dataloader = len(source_loader[2])
        len_dataloader = max(len_t_dataloader, len_s1_dataloader, len_s2_dataloader)
        source_iter1 = iter(source_loader[0])
        source_iter2 = iter(source_loader[1])
        # source_iter3 = iter(source_loader[2])
        target_iter = iter(target_loader)

        for i in range(len_dataloader):

            if i % len_t_dataloader == 0:
                target_iter = iter(target_loader)
            if i % len_s1_dataloader == 0:
                source_iter1 = iter(source_loader[0])
            if i % len_s2_dataloader == 0:
                source_iter2 = iter(source_loader[1])
            # if i % len_s3_dataloader == 0:
            #     source_iter3 = iter(source_loader[2])

            data_source1 = source_iter1.next()
            s1_img, s1_label = data_source1
            data_source2 = source_iter2.next()
            s2_img, s2_label = data_source2
            # data_source3 = source_iter3.next()
            # s3_img, s3_label = data_source3
            data_target = target_iter.next()
            t_img, t_label = data_target

            F_net.zero_grad()
            F1_net.zero_grad()
            F2_net.zero_grad()
            # F3_net.zero_grad()
            if cuda:
                s1_img = s1_img.cuda()
                s2_img = s2_img.cuda()
                # s3_img = s3_img.cuda()
                t_img = t_img.cuda()
                s1_label = s1_label.cuda()
                s2_label = s2_label.cuda()
                # s3_label = s3_label.cuda()

            # extract common feature
            feature1 = F_net(input_data=s1_img)
            feature2 = F_net(input_data=s2_img)
            # feature3 = F_net(input_data=s3_img)
            feature_t = F_net(input_data=t_img)

            if cuda:
                feature1 = feature1.cuda()
                feature2 = feature2.cuda()
                # feature3 = feature3.cuda()
                feature_t = feature_t.cuda()

            feature1 = F1_net(input_data=feature1)
            feature2 = F2_net(input_data=feature2)
            # feature3 = F3_net(input_data=feature3)
            feature_t1 = F1_net(input_data=feature_t)
            feature_t2 = F2_net(input_data=feature_t)
            # feature_t3 = F3_net(input_data=feature_t)

            if cuda:
                feature1 = feature1.cuda()
                feature2 = feature2.cuda()
                # feature3 = feature3.cuda()
                feature_t1 = feature_t1.cuda()
                feature_t2 = feature_t2.cuda()
                # feature_t3 = feature_t3.cuda()

            C1_net.zero_grad()
            C2_net.zero_grad()
            # C3_net.zero_grad()

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            class_labels1 = C1_net(feature1)
            class_labels2 = C2_net(feature2)
            # class_labels3 = C3_net(feature3)
            class_labels01 = C1_net(feature_t1)
            class_labels02 = C2_net(feature_t2)
            # class_labels03 = C3_net(feature_t3)

            err_class = loss_class(class_labels1, s1_label) + loss_class(class_labels2, s2_label)
            err_disc = loss_disc(class_labels01, class_labels02)

            # sigma = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e4, 1e6]
            sigma = [1e-1, 1, 5, 10]
            loss_mmd = (mmd.mix_rbf_mmd2(feature1, feature_t1, sigma) + mmd.mix_rbf_mmd2(feature2, feature_t2, sigma))/2
            loss = 0.5 * loss_mmd + err_class + alpha * err_disc

            loss.backward()
            optimizer_F.step()
            optimizer_F1.step()
            optimizer_F2.step()
            # optimizer_F3.step()
            optimizer_C1.step()
            optimizer_C2.step()
            # optimizer_C3.step()

        torch.save(F_net, '{0}/feature_model.pth'.format(model_root))
        torch.save(C1_net, '{0}/C1_model.pth'.format(model_root))
        torch.save(C2_net, '{0}/C2_model.pth'.format(model_root))
        # torch.save(C3_net, '{0}/C3_model.pth'.format(model_root))
        acc = test_multi0(target_loader, cuda, 0, 'average')
        if acc > acc_max:
            acc_max = acc
        sys.stdout.write('\r epoch: %d, loss_mmd: %f, loss_class: %f, loss_disc: %f, acc: %f' \
                         % (epoch, loss_mmd.data.cpu().numpy(), err_class.data.cpu().numpy(), err_disc, acc))
        sys.stdout.flush()

def train_CMSS(source_loader, target_loader, batch_size, lr, n_epoch, feature_number, class_number, cuda):
    model_root = 'models'
    acc_max = 0
    loss_class = torch.nn.NLLLoss()
    # loss_domain = torch.nn.CrossEntropyLoss()
    loss_s_domain = loss_cmss
    F_net = model.Feature(feature_number)
    F_cmss_net = model.Feature_cmss(feature_number, batch_size)
    C_net = model.Classifier0(class_number)
    D_net = model.discriminator0()
    # optimizer_F = optim.Adam(F_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # optimizer_F_cmss = optim.Adam(F_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # optimizer_C = optim.Adam(C_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # optimizer_D = optim.Adam(D_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_F = optim.SGD(F_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_C = optim.SGD(C_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_F_cmss = optim.SGD(F_cmss_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_D = optim.SGD(D_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)

    if cuda:
        F_net = F_net.cuda()
        F_cmss_net = F_cmss_net.cuda()
        C_net = C_net.cuda()
        D_net = D_net.cuda()
        loss_class = loss_class.cuda()
        # loss_domain = loss_domain.cuda()

    for p in F_net.parameters():
        p.requires_grad = True
    for p in F_cmss_net.parameters():
        p.requires_grad = True
    for p in C_net.parameters():
        p.requires_grad = True
    for p in D_net.parameters():
        p.requires_grad = True

    for epoch in range(n_epoch):
        len_t_dataloader = len(target_loader)
        len_s_dataloader = len(source_loader)
        len_dataloader = max(len_t_dataloader, len_s_dataloader)
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        for i in range(len_dataloader):

            if i % len_t_dataloader == 0:
                target_iter = iter(target_loader)
            if i % len_s_dataloader == 0:
                source_iter = iter(source_loader)

            data_source = source_iter.next()
            s_img, s_label = data_source
            data_target = target_iter.next()
            t_img, t_label = data_target

            F_net.zero_grad()
            F_cmss_net.zero_grad()
            C_net.zero_grad()
            D_net.zero_grad()
            if cuda:
                s_img = s_img.cuda()
                t_img = t_img.cuda()
                s_label = s_label.cuda()

            # extract common feature
            feature = F_net(input_data=s_img)
            feature_cmss = F_cmss_net(input_data=s_img)
            feature_t = F_net(input_data=t_img)

            if cuda:
                feature = feature.cuda()
                feature_cmss = feature_cmss.cuda()
                feature_t = feature_t.cuda()

            class_labels = C_net(feature)

            err_class = loss_class(class_labels, s_label)

            # batch_size = len(s_label)
            # batch_size0 = len(t_label)

            # align distribution
            # domain_s_label = torch.zeros(batch_size).long()
            # domain_t_label = torch.ones(batch_size0).long()
            # if cuda:
            #     domain_s_label = domain_s_label.cuda()
            #     domain_t_label = domain_t_label.cuda()

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            domain_labels = D_net(feature, alpha)
            domain_labels01 = D_net(feature_t, alpha)
            class_weight = F.softmax(feature_cmss, 1)
            class_weight = torch.sum(class_weight, 0)
            # if cuda:
            #     class_weight = class_weight.cuda()
            # loss_w_domain = torch.nn.CrossEntropyLoss(weight=class_weight)
            # if cuda:
            #     loss_w_domain = loss_w_domain.cuda()
            err_domain = loss_s_domain(domain_labels, domain_labels01, class_weight, cuda)
            loss = err_class + err_domain

            loss.backward()
            optimizer_F.step()
            optimizer_F_cmss.step()
            optimizer_C.step()
            optimizer_D.step()

        torch.save(F_net, '{0}/feature_model_CMSS.pth'.format(model_root))
        torch.save(C_net, '{0}/C_model_CMSS.pth'.format(model_root))
        acc = test_CMSS(target_loader, cuda)
        if acc > acc_max:
            acc_max = acc
        sys.stdout.write('\r epoch: %d, loss_domain: %f, loss_class: %f, acc: %f' \
                         % (epoch, err_domain.data.cpu().numpy(), err_class.data.cpu().numpy(), acc))
        sys.stdout.flush()


def train_DANN(source_loader, target_loader, lr, n_epoch, feature_number, class_number, cuda):
    model_root = 'models'
    acc_max = 0
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()
    F_net = model.Feature(feature_number)
    C_net = model.Classifier0(class_number)
    D_net = model.discriminator()
    # optimizer_F = optim.Adam(F_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # optimizer_C = optim.Adam(C_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # optimizer_D = optim.Adam(D_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_F = optim.SGD(F_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_C = optim.SGD(C_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_D = optim.SGD(D_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)

    if cuda:
        F_net = F_net.cuda()
        C_net = C_net.cuda()
        D_net = D_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    for p in F_net.parameters():
        p.requires_grad = True
    for p in C_net.parameters():
        p.requires_grad = True
    for p in D_net.parameters():
        p.requires_grad = True

    for epoch in range(n_epoch):
        len_t_dataloader = len(target_loader)
        len_s_dataloader = len(source_loader)
        len_dataloader = max(len_t_dataloader, len_s_dataloader)
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)

        for i in range(len_dataloader):

            if i % len_t_dataloader == 0:
                target_iter = iter(target_loader)
            if i % len_s_dataloader == 0:
                source_iter = iter(source_loader)

            data_source = source_iter.next()
            s_img, s_label = data_source
            data_target = target_iter.next()
            t_img, t_label = data_target

            F_net.zero_grad()
            C_net.zero_grad()
            D_net.zero_grad()
            if cuda:
                s_img = s_img.cuda()
                t_img = t_img.cuda()
                s_label = s_label.cuda()

            # extract common feature
            feature = F_net(input_data=s_img)
            feature_t = F_net(input_data=t_img)

            if cuda:
                feature = feature.cuda()
                feature_t = feature_t.cuda()

            class_labels = C_net(feature)

            err_class = loss_class(class_labels, s_label)

            batch_size = len(s_label)
            batch_size0 = len(t_label)

            # align distribution
            domain_s_label = torch.zeros(batch_size).long()
            domain_t_label = torch.ones(batch_size0).long()
            if cuda:
                domain_s_label = domain_s_label.cuda()
                domain_t_label = domain_t_label.cuda()

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            domain_labels = D_net(feature, alpha)
            domain_labels01 = D_net(feature_t, alpha)
            err_domain = loss_domain(domain_labels, domain_s_label) + loss_domain(domain_labels01, domain_t_label)
            loss = err_class + err_domain

            loss.backward()
            optimizer_F.step()
            optimizer_C.step()
            optimizer_D.step()

        torch.save(F_net, '{0}/feature_model_DANN.pth'.format(model_root))
        torch.save(C_net, '{0}/C_model_DANN.pth'.format(model_root))
        acc = test_DANN(target_loader, cuda)
        if acc > acc_max:
            acc_max = acc
        sys.stdout.write('\r epoch: %d, loss_domain: %f, loss_class: %f, acc: %f' \
                         % (epoch, err_domain.data.cpu().numpy(), err_class.data.cpu().numpy(), acc))
        sys.stdout.flush()


def train_Sonly(source_loader, target_loader, lr, n_epoch, feature_number, class_number, cuda):

    model_root = 'models'
    acc_max = 0
    loss_class = torch.nn.NLLLoss()
    F_net = model.Feature(feature_number)
    C_net = model.Classifier0(class_number)
    # optimizer_F = optim.Adam(F_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    # optimizer_C = optim.Adam(C_net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer_F = optim.SGD(F_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)
    optimizer_C = optim.SGD(C_net.parameters(), lr=lr, weight_decay=1e-08, momentum=0.9)

    if cuda:
        F_net = F_net.cuda()
        C_net = C_net.cuda()
        loss_class = loss_class.cuda()

    for p in F_net.parameters():
        p.requires_grad = True
    for p in C_net.parameters():
        p.requires_grad = True

    for epoch in range(n_epoch):
        len_dataloader = len(source_loader)
        source_iter = iter(source_loader)

        for i in range(len_dataloader):

            data_source = source_iter.next()
            s_img, s_label = data_source

            F_net.zero_grad()
            C_net.zero_grad()
            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()

            # extract common feature
            feature = F_net(input_data=s_img)

            if cuda:
                feature = feature.cuda()

            class_labels = C_net(feature)

            err_class = loss_class(class_labels, s_label)

            err_class.backward()
            optimizer_F.step()
            optimizer_C.step()

        torch.save(F_net, '{0}/feature_model_Sonly.pth'.format(model_root))
        torch.save(C_net, '{0}/C_model_Sonly.pth'.format(model_root))
        acc = test_Sonly(target_loader, cuda)
        if acc > acc_max:
            acc_max = acc
        sys.stdout.write('\r epoch: %d, loss_class: %f, acc: %f' \
                         % (epoch, err_class.data.cpu().numpy(), acc))
        sys.stdout.flush()

