#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class InceptionModele(nn.Module):
    def __init__(self, channel_num, n_feature_maps= 32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41):
        super(InceptionModele, self).__init__()
        self.nb_filter = n_feature_maps
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernerl_size = kernel_size-1
        self.bottleneck_size = 32

        self.pre_module = None

        input_channel_num = channel_num

        if self.use_bottleneck and int(channel_num)>self.bottleneck_size:
            self.pre_module = (nn.Conv1d(channel_num,self.bottleneck_size,kernel_size=1,bias=False))
            input_channel_num= self.bottleneck_size
        kernel_size_s = [self.kernerl_size//(2**i)+1 for i in range(3)]

        self.inception_modules = nn.ModuleList()
        for i in range(len(kernel_size_s)):
            self.inception_modules.append(nn.Conv1d(input_channel_num,self.nb_filter,kernel_size=kernel_size_s[i],padding=(kernel_size_s[i]-1)//2,bias=False))

        self.inception_modules.append(nn.Sequential(
            nn.MaxPool1d(kernel_size=3,stride=1,padding=(3-1)//2),
            nn.Conv1d(input_channel_num,self.nb_filter,kernel_size=1,bias=False)
        ))

        self.bn_act= nn.Sequential(
            nn.BatchNorm1d(self.nb_filter*len(self.inception_modules)),
            nn.ReLU(inplace=True),
        )
    def forward(self, x) :
        input_inception = x
        if self.pre_module is not None:
            input_inception = self.pre_module(x)
        conv_list = []
        for i in range(len(self.inception_modules)):
            # print('layer_{}'.format(i))
            conv_list.append(self.inception_modules[i](input_inception))
        cat_x = torch.cat(conv_list,dim=1)
        return self.bn_act(cat_x)

class ShortCutLayer(nn.Module):
    def __init__(self,inputer_features_d,n_feature_maps):
        super(ShortCutLayer, self).__init__()
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels=inputer_features_d,out_channels=n_feature_maps,kernel_size=1,bias=False),
            nn.BatchNorm1d(n_feature_maps)
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self,in_tensor,out_tensor):
        x = self.shortcut(in_tensor)
        return self.act(x+out_tensor)



class InceptionTimeMRM2(nn.Module):
    def __init__(self,input_features_d=1, n_feature_maps= 32, nb_classes= 2, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41):
        super(InceptionTimeMRM2, self).__init__()
        self.nb_filter = n_feature_maps
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernerl_size = kernel_size-1
        self.bottleneck_size = 32
        kernel_size_s = [self.kernerl_size//(2**i)+1 for i in range(3)]
        in_channel_num = input_features_d
        in_channel_num_shortcut = input_features_d
        out_channel_num = self.nb_filter*(len(kernel_size_s)+1)

        self.inception_modules = nn.ModuleList()
        self.shortcut_modules= nn.ModuleList()
        for d in range(self.depth):
            self.inception_modules.append(InceptionModele(in_channel_num,))
            if self.use_residual and d%3==2:
                self.shortcut_modules.append(ShortCutLayer(in_channel_num_shortcut,out_channel_num))
                in_channel_num_shortcut = out_channel_num
            else:
                self.shortcut_modules.append(None)

            in_channel_num,out_channel_num = out_channel_num, self.nb_filter*(len(kernel_size_s)+1)




        self.global_average_pool= nn.AdaptiveAvgPool1d((1))
        self.linear= nn.Linear(in_channel_num,nb_classes)




    def forward(self, x,labels = None) :

        input_res = x

        for d in range(self.depth):
            x = self.inception_modules[d](x)
            if self.shortcut_modules[d] is not None:
                x = self.shortcut_modules[d](input_res,x)
                input_res = x

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
if __name__=='__main__':
    model = InceptionTimeMRM2(1)
    input_tensor = torch.zeros((3,1,100))
    model.eval()
    out = model(input_tensor)
    for key in out:
        if out[key] is not None:
            print(out[key].shape)