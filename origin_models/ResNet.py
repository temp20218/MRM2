#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
from sklearn.metrics import accuracy_score,f1_score
class ResNetTS(nn.Module):
    def __init__(self,input_features_d, n_feature_maps, nb_classes):
        super(ResNetTS, self).__init__()
        self.input_bn= nn.BatchNorm1d(input_features_d)

        self.layer1_c = nn.Conv1d(input_features_d, n_feature_maps, 8, 1, padding=4)
        self.layer1= nn.Sequential(
            # nn.Conv1d(input_features_d, n_feature_maps, 7, 1, padding=(7-1)//2),
            nn.BatchNorm1d(n_feature_maps),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feature_maps, n_feature_maps, 5, 1, padding=(5 - 1) // 2),
            nn.BatchNorm1d(n_feature_maps),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feature_maps, n_feature_maps, 3, 1, padding=(3 - 1) // 2),
            nn.BatchNorm1d(n_feature_maps)
        )
        is_expand_channels = not (input_features_d == n_feature_maps)
        if is_expand_channels:
            self.short_cut_layer1= nn.Sequential(
                nn.Conv1d(input_features_d, n_feature_maps, 1, 1, ),
                nn.BatchNorm1d(n_feature_maps)
            )
        else:
            self.short_cut_layer1= nn.BatchNorm1d(n_feature_maps)
        self.layer1_relu= nn.ReLU(inplace=True)

        self.layer2_c = nn.Conv1d(n_feature_maps, n_feature_maps * 2, 8, 1, padding=4)
        self.layer2= nn.Sequential(
            # nn.Conv1d(n_feature_maps, n_feature_maps * 2, 7, 1, padding=(7 - 1) // 2),
            nn.BatchNorm1d(n_feature_maps*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feature_maps*2, n_feature_maps*2, 5, 1, padding=(5 - 1) // 2),
            nn.BatchNorm1d(n_feature_maps*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feature_maps*2, n_feature_maps*2, 3, 1, padding=(3 - 1) // 2),
            nn.BatchNorm1d(n_feature_maps*2)
        )
        self.short_cut_layer2 = nn.Sequential(
            nn.Conv1d(n_feature_maps, n_feature_maps*2, 1, 1, ),
            nn.BatchNorm1d(n_feature_maps*2)
        )
        self.layer2_relu = nn.ReLU(inplace=True)

        self.layer3_c=nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, 8, 1, padding=4)
        self.layer3= nn.Sequential(
            # nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, 7, 1, padding=(7 - 1) // 2),
            nn.BatchNorm1d(n_feature_maps*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feature_maps*2, n_feature_maps*2, 5, 1, padding=(5 - 1) // 2),
            nn.BatchNorm1d(n_feature_maps*2),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feature_maps*2, n_feature_maps*2, 3, 1, padding=(3 - 1) // 2),
            nn.BatchNorm1d(n_feature_maps*2)
        )
        self.short_cut_layer3 = nn.BatchNorm1d(n_feature_maps*2)
        self.layer3_relu = nn.ReLU(inplace=True)

        #
        self.global_average_pool= nn.AdaptiveAvgPool1d((1))
        self.linear= nn.Linear(n_feature_maps*2,nb_classes)
        # self.softmax= nn.Softmax(1)

    def forward(self, x) :
        x= self.input_bn(x)
        # x= self.layer1_relu(self.layer1(x)+self.short_cut_layer1(x))
        # x= self.layer2_relu(self.layer2(x)+self.short_cut_layer2(x))
        # x= self.layer3_relu(self.layer3(x)+self.short_cut_layer3(x))
        x= self.layer1_relu(self.layer1(self.layer1_c(x)[:,:,:-1])+self.short_cut_layer1(x))
        x= self.layer2_relu(self.layer2(self.layer2_c(x)[:,:,:-1])+self.short_cut_layer2(x))
        x= self.layer3_relu(self.layer3(self.layer3_c(x)[:,:,:-1])+self.short_cut_layer3(x))

        return (self.linear(self.global_average_pool(x).squeeze(-1)))
