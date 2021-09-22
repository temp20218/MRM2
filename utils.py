import argparse
import pandas as pd
import time
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import os



class MixedTSDataset(Dataset):
    def __init__(self, data, labels, num_train):
        super(MixedTSDataset, self).__init__()
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)
        self.len, self.seq_len = self.data.shape
        self.is_train = torch.zeros(data.shape[0], dtype=torch.bool)
        self.is_train[:num_train] = 1
        self.id = torch.arange(0, self.data.shape[0]).long()

    def __getitem__(self, item):
        return self.data[item].unsqueeze(0).type(torch.float32), self.labels[item].type(torch.int64), self.is_train[
            item], self.id[item]

    def __len__(self):
        return self.len

class SingleTSDataset(Dataset):
    def __init__(self, data, labels, is_train=True):
        super(SingleTSDataset, self).__init__()
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)
        self.len, self.seq_len = self.data.shape
        self.is_train = is_train
        self.is_train_flag = torch.zeros(self.len, dtype=torch.bool)
        if self.is_train:
            self.is_train_flag = torch.ones(self.len, dtype=torch.bool)

        self.id = torch.arange(0, self.len).long()

    def __getitem__(self, item):
        return self.data[item].unsqueeze(0).type(torch.float32), \
               self.labels[item].type(torch.int64), \
               self.is_train_flag[item], \
               self.id[item]

    def __len__(self):
        return self.len

class MultiTSDataset(Dataset):
    def __init__(self, data, labels, is_train=True):
        super(MultiTSDataset, self).__init__()
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)
        self.len, self.channel,self.seq_len = self.data.shape
        self.is_train = is_train
        self.is_train_flag = torch.zeros(self.len, dtype=torch.bool)
        if self.is_train:
            self.is_train_flag = torch.ones(self.len, dtype=torch.bool)

        self.id = torch.arange(0, self.len).long()

    def __getitem__(self, item):
        return self.data[item].type(torch.float32), \
               self.labels[item].type(torch.int64), \
               self.is_train_flag[item], \
               self.id[item]

    def __len__(self):
        return self.len


def readuea(dir,fname,mode):
    X = np.load(os.path.join(dir,fname,"{}_{}.npy".format(fname,mode)))
    Y = np.load(os.path.join(dir,fname,"{}_{}_label.npy".format(fname,mode)))
    return X,Y


def readucr(filename):
    data = pd.read_csv(filename, sep='	', header=None)
    data= data.fillna(0.0).to_numpy()
    # data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

# def train_mrm2():

class TSCConfig(object):
    def __init__(self):
        # config for training
        self.data_dir = '/media/spx/document/pythonproject/tsr/UCRArchive_2018'
        self.out_dir = './out_'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.lr = 0.001
        self.max_epoch = 1000
        self.batch_size = 128

        self.nclass = None
        self.gpu_nums = 1
        self.start_cuda_id = 0
        self.weight_decay =0.01
        self.pretrain_model_dir=''
        self.start_dataset_id= 0
        self.model_name = ''

    def to_dict(self):
        return self.__dict__
    def update(self,dic):
        self.__dict__.update(dic)



def set_parser():
    config = TSCConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/media/spx/document/pythonproject/tsr/UCRArchive_2018',
                        type=str, required=False,help='the directory of UCRAchieve 2018')
    parser.add_argument('--out_dir', default=None, type=str, required=False,
                        help='The output directory where model predictions and checkpoints will be written. ')
    parser.add_argument('--max_epoch', default=500, type=int, required=False,
                        help='the max epoch for training the model')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate for the model')
    parser.add_argument('--gpu_nums', default=1, type=int,
                        help='the number of the gpu')
    parser.add_argument('--start_cuda_id', default=0, type=int,
                        help="if use gpu, start_cuda_id set the start gpu device id")
    parser.add_argument('--batch_size',default=128, type=int,
                        help='the number sample of a batch')
    parser.add_argument('--weight_decay',default=0.01,type= float,
                        help="")
    parser.add_argument('--pretrain_model_dir',default='',type=str,
                        help='the directory for the pretrain model')
    parser.add_argument('--start_dataset_id', default=0, type=int,
                        help='')
    parser.add_argument('--model_name',default='',type=str,
                        help="")
    args = parser.parse_args()
    config.data_dir= args.data_dir
    config.out_dir =args.out_dir
    config.lr =args.lr
    config.max_epoch= args.max_epoch
    config.batch_size = args.batch_size
    config.gpu_nums = args.gpu_nums
    config.start_cuda_id= args.start_cuda_id

    config.weight_decay= args.weight_decay
    config.pretrain_model_dir= args.pretrain_model_dir
    config.start_dataset_id= args.start_dataset_id
    config.model_name = args.model_name
    return config




class TSCMRM2Config(object):
    def __init__(self):
        # config for training
        self.data_dir = '/media/spx/document/pythonproject/tsr/UCRArchive_2018'
        self.out_dir = './out_'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.lr = 0.001
        self.max_epoch = 500
        self.batch_size =128

        self.nclass = None

        self.gpu_nums = 1
        self.start_cuda_id = 0
        self.weight_decay =0.01
        self.pretrain_model_dir=''
        self.start_dataset_id= 0
        self.lambda_loss = 0.5
        self.gamma_loss = 0.5
        self.model_name = ''

    def to_dict(self):
        return self.__dict__
    def update(self,dic):
        self.__dict__.update(dic)



def set_mrm2_parser():
    config = TSCMRM2Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/media/spx/document/pythonproject/tsr/UCRArchive_2018',
                        type=str, required=False,help='the directory of UCRAchieve 2018')
    parser.add_argument('--out_dir', default=None, type=str, required=False,
                        help='The output directory where model predictions and checkpoints will be written. ')
    parser.add_argument('--max_epoch', default=500, type=int, required=False,
                        help='the max epoch for training the model')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='learning rate for the model')
    parser.add_argument('--gpu_nums', default=1, type=int,
                        help='the number of the gpu')
    parser.add_argument('--start_cuda_id', default=0, type=int,
                        help="if use gpu, start_cuda_id set the start gpu device id")
    parser.add_argument('--batch_size',default=128, type=int,
                        help='the number sample of a batch')
    parser.add_argument('--weight_decay',default=0.01,type= float,
                        help="")
    parser.add_argument('--pretrain_model_dir',default='',type=str,
                        help='the directory for the pretrain model')
    parser.add_argument('--start_dataset_id', default=0, type=int,
                        help='')
    parser.add_argument('--lambda_loss',default=0.01,type= float,
                        help="")
    parser.add_argument('--gamma_loss',default=0.01,type= float,
                        help="")
    parser.add_argument('--model_name',default='',type=str,
                        help="")
    args = parser.parse_args()
    config.data_dir= args.data_dir
    config.out_dir =args.out_dir
    config.lr =args.lr
    config.max_epoch= args.max_epoch
    config.batch_size = args.batch_size
    config.gpu_nums = args.gpu_nums
    config.start_cuda_id= args.start_cuda_id

    config.weight_decay= args.weight_decay
    config.pretrain_model_dir= args.pretrain_model_dir
    config.start_dataset_id= args.start_dataset_id
    config.lambda_loss= args.lambda_loss
    config.gamma_loss= args.gamma_loss
    config.model_name = args.model_name

    return config


