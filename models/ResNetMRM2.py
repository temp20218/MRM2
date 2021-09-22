#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
import math
from easydict import EasyDict as easydict
import json


class ResNetMRM2(nn.Module):
    def __init__(self, input_features_d, n_feature_maps, nb_classes):
        super(ResNetMRM2, self).__init__()
        self.input_bn = nn.BatchNorm1d(input_features_d)

        self.layer1_c = nn.Conv1d(input_features_d, n_feature_maps, 8, 1, padding=4)
        self.layer1 = nn.Sequential(
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
            self.short_cut_layer1 = nn.Sequential(
                nn.Conv1d(input_features_d, n_feature_maps, 1, 1, ),
                nn.BatchNorm1d(n_feature_maps)
            )
        else:
            self.short_cut_layer1 = nn.BatchNorm1d(n_feature_maps)
        self.layer1_relu = nn.ReLU(inplace=True)

        self.layer2_c = nn.Conv1d(n_feature_maps, n_feature_maps * 2, 8, 1, padding=4)
        self.layer2 = nn.Sequential(
            # nn.Conv1d(n_feature_maps, n_feature_maps * 2, 7, 1, padding=(7 - 1) // 2),
            nn.BatchNorm1d(n_feature_maps * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, 5, 1, padding=(5 - 1) // 2),
            nn.BatchNorm1d(n_feature_maps * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, 3, 1, padding=(3 - 1) // 2),
            nn.BatchNorm1d(n_feature_maps * 2)
        )
        self.short_cut_layer2 = nn.Sequential(
            nn.Conv1d(n_feature_maps, n_feature_maps * 2, 1, 1, ),
            nn.BatchNorm1d(n_feature_maps * 2)
        )
        self.layer2_relu = nn.ReLU(inplace=True)

        self.layer3_c = nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, 8, 1, padding=4)
        self.layer3 = nn.Sequential(
            # nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, 7, 1, padding=(7 - 1) // 2),
            nn.BatchNorm1d(n_feature_maps * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, 5, 1, padding=(5 - 1) // 2),
            nn.BatchNorm1d(n_feature_maps * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, 3, 1, padding=(3 - 1) // 2),
            nn.BatchNorm1d(n_feature_maps * 2)
        )
        self.short_cut_layer3 = nn.BatchNorm1d(n_feature_maps * 2)
        self.layer3_relu = nn.ReLU(inplace=True)

        self.global_average_pool = nn.AdaptiveAvgPool1d((1))
        self.linear = nn.Linear(n_feature_maps * 2, nb_classes)

        self.x_train_embeddings = None
        self.labels = None
        self.nb_classes = nb_classes

    def forward(self, x, labels=None):
        x = self.input_bn(x)  # batch *1 *seq_len
        # x= self.layer1_relu(self.layer1(x)+self.short_cut_layer1(x))
        # x= self.layer2_relu(self.layer2(x)+self.short_cut_layer2(x))
        # x= self.layer3_relu(self.layer3(x)+self.short_cut_layer3(x))
        x = self.layer1_relu(self.layer1(self.layer1_c(x)[:, :, :-1]) + self.short_cut_layer1(x))  # batch *64 * seq_len
        x = self.layer2_relu(self.layer2(self.layer2_c(x)[:, :, :-1]) + self.short_cut_layer2(x))  # batch *128* seq_len
        x = self.layer3_relu(self.layer3(self.layer3_c(x)[:, :, :-1]) + self.short_cut_layer3(x))  # batch *128* seq_len
        x_embeddings = self.global_average_pool(x).squeeze(-1)
        orig_out = (self.linear(x_embeddings))  # batch * nb_class
        knn_out = None
        if labels is not None:
            # knn_labels = labels[:]
            # knn_labels[knn_labels==-1]= orig_out[knn_labels==-1] # 将验证集中的样本置为pred
            is_test = (labels == -1)
            is_train = (labels != -1)
            x_train_embeddings, x_test_embeddings, orig_out_test = x_embeddings[is_train], x_embeddings[is_test],torch.softmax(orig_out[is_test],dim = 1)
            similar_x_test = torch.matmul(x_embeddings,x_test_embeddings.T)
            similar_x_test_stand = (similar_x_test-similar_x_test.mean(1,keepdim = True))/similar_x_test.std(1,keepdim = True)
            knn_out = torch.matmul((similar_x_test_stand),orig_out_test/orig_out_test.sum(dim = 0,keepdim = True))
        return {
            'tcrm2_out': orig_out,
            'x_embeddings': x_embeddings,
            'ttrm2_out': knn_out
        }

    def updaye(self, x_train_embeddings, labels):
        self.x_train_embeddings = x_train_embeddings
        self.labels = labels


if __name__ == '__main__':
    a = torch.ones((128, 1, 35))
    model = ResNetMRM2(1, 64, 4)
    out = model(a,torch.randint(0,2,(128,))*-1)
    for key in out:
        if out[key] is not None:
            print(out[key].shape)
