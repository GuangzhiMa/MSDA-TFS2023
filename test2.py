import os
import torch.backends.cudnn as cudnn
import torch.utils.data


def test_multi(test_loader, cuda, w_corr, type):

    model_root = 'models'

    """ test """
    F_net = torch.load(os.path.join(
        model_root, 'feature_model.pth'
    ))
    F_net = F_net.eval()
    C1_net = torch.load(os.path.join(
        model_root, 'C1_model.pth'
    ))
    C1_net = C1_net.eval()
    C2_net = torch.load(os.path.join(
        model_root, 'C2_model.pth'
    ))
    C2_net = C2_net.eval()
    C3_net = torch.load(os.path.join(
        model_root, 'C3_model.pth'
    ))
    C3_net = C3_net.eval()

    if cuda:
        F_net = F_net.cuda()
        C1_net = C1_net.cuda()
        C2_net = C2_net.cuda()
        C3_net = C3_net.cuda()

    len_dataloader = len(test_loader)
    data_target_iter = iter(test_loader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        feature1 = F_net(input_data=t_img)
        feature2 = F_net(input_data=t_img)
        feature3 = F_net(input_data=t_img)
        
        if cuda:
            feature1 = feature1.cuda()
            feature2 = feature2.cuda()
            feature3 = feature3.cuda()

        if type == 'average':
            class_output = (C1_net(feature1) + C2_net(feature2) + C3_net(feature3))/3
        else:
            class_output = C1_net(feature1)*w_corr[0] + C2_net(feature2)*w_corr[1] + C3_net(feature3)*w_corr[2]
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu


def test_multi0(test_loader, cuda, w_corr, type):
    model_root = 'models'

    """ test """
    F_net = torch.load(os.path.join(
        model_root, 'feature_model.pth'
    ))
    F_net = F_net.eval()
    C1_net = torch.load(os.path.join(
        model_root, 'C1_model.pth'
    ))
    C1_net = C1_net.eval()
    C2_net = torch.load(os.path.join(
        model_root, 'C2_model.pth'
    ))
    C2_net = C2_net.eval()

    if cuda:
        F_net = F_net.cuda()
        C1_net = C1_net.cuda()
        C2_net = C2_net.cuda()

    len_dataloader = len(test_loader)
    data_target_iter = iter(test_loader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()

        feature1 = F_net(input_data=t_img)
        feature2 = F_net(input_data=t_img)

        if cuda:
            feature1 = feature1.cuda()
            feature2 = feature2.cuda()

        if type == 'average':
            class_output = (C1_net(feature1) + C2_net(feature2)) / 2
        else:
            class_output = C1_net(feature1) * w_corr[0] + C2_net(feature2) * w_corr[1]
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu
