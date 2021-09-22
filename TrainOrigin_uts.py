#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from origin_models.OS_CNN import OS_CNN
from origin_models.FCN import FCNTS
from origin_models.FCN_LSTM import FCNLSTMTS
from origin_models.InceptionTime import InceptionTime
from origin_models.ResNet import ResNetTS
from OS_CNN_Structure_build import generate_layer_parameter_list
from utils import set_parser, readucr,SingleTSDataset
from Config import flist
import json

# data_dir='/media/spx/document/pythonproject/tsr/UCRArchive_2018'
# data_dir ='/media/spx/externdisk/python_workspace/tsr/UCRArchive_2018'
# data_dir = '/home/shipengxiang/jupyter/UCRArchive_2018'


# class SingleTSDataset(Dataset):
#     def __init__(self, data, labels):
#         super(SingleTSDataset, self).__init__()
#         self.data = torch.from_numpy(data)
#         self.labels = torch.from_numpy(labels)
#         self.len, self.seq_len = self.data.shape
#
#     def __getitem__(self, item):
#         return self.data[item].unsqueeze(0).type(torch.float32), self.labels[item].type(torch.LongTensor)
#
#     def __len__(self):
#         return self.len


if __name__ == '__main__':
    train_config = set_parser()
    result = {}
    result['Name'] = []
    result['val_loss'] = []
    result['val_acc'] = []
    max_epoch = train_config.max_epoch
    val_interval = 1
    data_dir = train_config.data_dir


    Max_kernel_size = 89
    paramenter_number_of_layer_list = [8 * 128, 5 * 128 * 256 + 2 * 256 * 128]
    start_kernel_size = 1
    out_dir = train_config.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for fid, fname in enumerate(flist):
        print("--------------start process the data set {}:{}--------".format(fid, fname))
        x_train, y_train = readucr(data_dir + '/' + fname + '/' + fname + '_TRAIN.tsv')
        x_test, y_test = readucr(data_dir + '/' + fname + '/' + fname + '_TEST.tsv')
        nb_classes = len(np.unique(y_test))
        if x_train.shape[1]>512:
            batch_size = 128
        else:
            batch_size = 512
        # batch_size = int(min(x_train.shape[0] / 10, 32))

        label2id = {}
        for i, key in enumerate(set(y_train)):
            label2id[key] = i
        y_train = np.array([label2id[key] for key in y_train])
        y_test = np.array([label2id[key] for key in y_test])

        train_dataset = SingleTSDataset(x_train, y_train,is_train=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=8, pin_memory=True)

        test_dataset = SingleTSDataset(x_test, y_test,is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=8, pin_memory=True)
        json.dump(train_config.to_dict(), open(os.path.join(out_dir, 'train_config.json'), 'w'), indent=4)
        device = torch.device('cuda:{}'.format(train_config.start_cuda_id)) \
            if train_config.gpu_nums > 0 and torch.cuda.is_available() else torch.device('cpu')

        if train_config.model_name.lower() == 'fcn':
            model = FCNTS(input_features_d=1,n_feature_maps=128,nb_classes=nb_classes).to(device)
        elif train_config.model_name.lower() =='fcn_lstm':
            model= FCNLSTMTS(1,128,nb_classes,8).to(device)
        elif train_config.model_name.lower()=='resnet':
            model = ResNetTS(1,64,nb_classes).to(device)
        elif train_config.model_name.lower()=='inceptiontime':
            model= InceptionTime(1,nb_classes=nb_classes).to(device)
        else:
            receptive_field_shape = min(int(x_train.shape[-1] / 4), Max_kernel_size)
            layer_parameter_list = generate_layer_parameter_list(start_kernel_size,
                                                                 receptive_field_shape,
                                                                 paramenter_number_of_layer_list,
                                                                 in_channel=1#int(x_train.shape[1])
                                                                 )
            model = OS_CNN(layer_parameter_list, nb_classes, False).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50,
                                                               verbose=True, min_lr=0.0001)
        best_loss = 1e9
        final_res = [0, 0]
        for epoch in range(max_epoch):
            model.train()
            running_loss = 0.0
            start = time.time()
            train_pred = []
            train_gt = []
            for i,(xs, labels, _, _) in (enumerate(train_loader, 0)):
                optimizer.zero_grad()
                xs, labels = xs.to(device), labels.to(device)
                out = model(xs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_pred.append(out.argmax(dim=-1).detach().to('cpu').numpy())
                train_gt.append(labels.detach().to('cpu').numpy())
            train_pred = np.concatenate(train_pred, axis=0)
            train_gt = np.concatenate(train_gt, axis=0)

            acc = accuracy_score(y_true=train_gt, y_pred=train_pred)
            running_loss = running_loss / (i + 1)
            # print('train: [%4d/ %5d] loss: %.6f, acc: %.6f,  time: %f s' %
            #       (epoch + 1, i, running_loss, acc, time.time() - start))
            model.eval()
            val_loss = 0.0
            pred = []
            gt = []
            for i, (xs, labels, _, _) in enumerate(test_loader, 0):
                xs, labels = xs.to(device), labels.to(device)
                out = model(xs)
                loss = criterion(out, labels)
                val_loss += loss.item()
                pred.append(out.argmax(dim=-1).detach().to('cpu').numpy())
                gt.append(labels.detach().to('cpu').numpy())
            gt = np.concatenate(gt, axis=0)
            pred = np.concatenate(pred, axis=0)
            val_loss = val_loss / (i + 1)
            acc_val = accuracy_score(y_true=gt, y_pred=pred)  # sum(gt==pred)/gt.shape[0]
            print('epoch [%4d/ %4d], test loss: %.6f, test acc %.6f, train loss: %.6f, train acc: %.6f, time: %f s' %
                  (epoch + 1, max_epoch, val_loss, acc_val, running_loss, acc, time.time() - start))
            scheduler.step(running_loss)

            if best_loss > running_loss:
                best_loss = running_loss
                final_res = [val_loss, acc_val]
                torch.save(model.state_dict(), os.path.join(out_dir, '{}_best_model.pth'.format(fname)))
        result["Name"].append(fname)
        result['val_acc'].append(final_res[1])
        result['val_loss'].append(final_res[0])
        df_result = pd.DataFrame(result)
        df_result.to_csv(os.path.join(out_dir, 'Result_{}.csv'.format(train_config.start_dataset_id)))
