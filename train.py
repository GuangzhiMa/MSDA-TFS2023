import torch

from torch import optim
from test2 import test_multi, test_multi0
import torch.nn.functional as F

import torch.backends.cudnn as cudnn

import model
import mmd
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

