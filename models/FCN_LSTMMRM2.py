#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class FCNLSTMMRM2(nn.Module):
    def __init__(self,input_features_d=1, n_feature_maps= 128, nb_classes= 2, NUM_CELLS=8):
        super(FCNLSTMMRM2, self).__init__()
        # self.input_bn= nn.BatchNorm1d(input_features_d)

        self.layer1_c = nn.Conv1d(input_features_d, n_feature_maps, 8, 1, padding=4)
        self.layer1= nn.Sequential(
            # nn.Conv1d(input_features_d, n_feature_maps, 7, 1, padding=(7-1)//2),
            nn.BatchNorm1d(n_feature_maps),
            nn.ReLU(inplace=True),
        )

        self.layer2_c = nn.Conv1d(n_feature_maps, n_feature_maps * 2, 5, 1, padding=2)
        self.layer2= nn.Sequential(
            # nn.Conv1d(n_feature_maps, n_feature_maps * 2, 7, 1, padding=(7 - 1) // 2),
            nn.BatchNorm1d(n_feature_maps*2),
            nn.ReLU(inplace=True),
        )

        self.layer3_c=nn.Conv1d(n_feature_maps * 2, n_feature_maps , 3, 1, padding=1)
        self.layer3= nn.Sequential(
            # nn.Conv1d(n_feature_maps * 2, n_feature_maps * 2, 7, 1, padding=(7 - 1) // 2),
            nn.BatchNorm1d(n_feature_maps),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(input_features_d,NUM_CELLS,batch_first=True)

        self.global_average_pool= nn.AdaptiveAvgPool1d((1))
        self.linear= nn.Linear(n_feature_maps+NUM_CELLS,nb_classes)
        # self.softmax= nn.Softmax(1)

    def forward(self, x,labels = None) :
        # batch *seq * channel
        y,_= self.lstm(x.permute(0,2,1))# batch * seq * channel
        y = y.permute((0,2,1))

        # x= self.input_bn(x)
        # x= self.layer1_relu(self.layer1(x)+self.short_cut_layer1(x))
        # x= self.layer2_relu(self.layer2(x)+self.short_cut_layer2(x))
        # x= self.layer3_relu(self.layer3(x)+self.short_cut_layer3(x))
        x= self.layer1(self.layer1_c(x)[:,:,:-1])
        x= self.layer2(self.layer2_c(x))
        x= self.layer3(self.layer3_c(x))

        xy = torch.cat([x,y],dim = 1)

        x_embeddings = self.global_average_pool(xy).squeeze(-1)
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

if __name__=='__main__':
    model = FCNLSTMMRM2(1)
    input_tensor = torch.zeros((3,1,100))
    model.eval()
    out = model(input_tensor)
    for key in out:
        if out[key] is not None:
            print(out[key].shape)