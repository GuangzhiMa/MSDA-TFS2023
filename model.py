import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import ReverseLayerF
from torchvision import models
import numpy as np
# from easydl import *

class Feature(nn.Module):

    def __init__(self, input_size):
        super(Feature, self).__init__()
        self.feature_classifier = nn.Sequential()
        self.feature_classifier.add_module('f_fc1', nn.Linear(input_size, 100))
        self.feature_classifier.add_module('f_bn1', nn.BatchNorm1d(100))
        self.feature_classifier.add_module('f_relu1', nn.ReLU())
        self.feature_classifier.add_module('f_drop1', nn.Dropout(0.5))
        self.feature_classifier.add_module('f_fc2', nn.Linear(100, 100))
        self.feature_classifier.add_module('f_bn2', nn.BatchNorm1d(100))
        self.feature_classifier.add_module('f_relu2', nn.ReLU())
        self.feature_classifier.add_module('f_drop2', nn.Dropout(0.5))
        self.feature_classifier.add_module('f_fc3', nn.Linear(100, 100))
        self.feature_classifier.add_module('f_bn3', nn.BatchNorm1d(100))
        self.feature_classifier.add_module('f_relu3', nn.ReLU())
        self.feature_classifier.add_module('f_drop3', nn.Dropout(0.5))

    def forward(self, input_data):
        feature = self.feature_classifier(input_data)

        return feature

class Feature_s(nn.Module):

    def __init__(self):
        super(Feature_s, self).__init__()
        self.feature_classifier = nn.Sequential()
        self.feature_classifier.add_module('f_fc1', nn.Linear(100, 100))
        self.feature_classifier.add_module('f_bn1', nn.BatchNorm1d(100))
        self.feature_classifier.add_module('f_relu1', nn.ReLU())
        self.feature_classifier.add_module('f_drop1', nn.Dropout(0.5))

    def forward(self, input_data):
        feature = self.feature_classifier(input_data)

        return feature

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    
class Classifier(nn.Module):

    def __init__(self, class_number):
        super(Classifier, self).__init__()
        self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_fc1', nn.Linear(100, 100))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(100, class_number))
        self.class_classifier.apply(init_weights)

    def forward(self, feature):
        class_labels = self.class_classifier(feature)
       
        return class_labels


class Classifier0(nn.Module):

    def __init__(self, class_number):
        super(Classifier0, self).__init__()
        self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_fc1', nn.Linear(100, 100))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(100, class_number))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

    def forward(self, feature):
        class_labels = self.class_classifier(feature)

        return class_labels
    
class discriminator(nn.Module):

    def __init__(self):
        super(discriminator, self).__init__()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(100, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU())
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, feature, alpha):
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        output = self.domain_classifier(reverse_feature)

        return output

class discriminator0(nn.Module):

    def __init__(self):
        super(discriminator0, self).__init__()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(100, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU())
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))

    def forward(self, feature, alpha):
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        output = self.domain_classifier(reverse_feature)

        return output
