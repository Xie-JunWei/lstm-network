"""
This is a test for classify the trajectories.
WE have three kinds of trajectories:  | L_D| c_aphi| c_h|
                         with label:  |  0 |   1   |  2 |

Try to use linear+LSTM+linear NET.

"""

from __future__ import print_function
import os
from utils.dataset import dataset,collate_fn
import torch.nn as nn
import torch.optim
import torch.utils.data as Data
import pandas as pd
from sklearn.model_selection import train_test_split
from Classify_net import Classify
from utils.d_process import *
from torch.optim import lr_scheduler
from train_util import train, trainlog
import logging
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyper Parameters
INPUT_SIZE = 6
OUTPUT_SIZE = 3
TIME_STEP = 1200
LINEAR1_OUT = 32    # D:通过6维输入，提取另外的特征 *2
HIDDEN_SIZE = 64    # 记录调试日志
BATCH_SIZE = 64    #NUM_LATERS = 2      #
LR = 0.01
N_TRAIN_POINTS = 100 #没有用到
EPOCH = 1000 #没有用到
MAX_LEN = 600

save_dir = r'F:\classi\model_test_1_16/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = '%s/trainlog.log'%save_dir
trainlog(logfile)
# 原始数据
rawdata_root = r'F:\classi\train/'
# 读标签
all_pd = pd.read_csv(r"F:\HXDD/dataset_train.csv", sep=",",
                   header=None,
                   names=["file_name","label"])[1:]   ##加r''的目的在于啥 ，dataset_train.csv好像有点不对

#print(all_pd.head())
# 训练数据和测试数据划分
train_pd, val_pd = train_test_split(all_pd, test_size=0.25, random_state=43,
                                    stratify=all_pd['label']) #去查下这个函数的用法 2020-10-20 有可能指的是标签只有1维
# print(val_pd.shape)

# 数据预测处理
data_process = {
    'train':data_process(max_len=MAX_LEN),
    'val':data_process(max_len=MAX_LEN)
}
data_set={}
data_set['train']=dataset(trajroot=rawdata_root, anno_pd=train_pd,
                          dprocess=data_process['train'])   #train_pd 应该调成 val_pd,原先为train_pd
data_set['val']=dataset(trajroot=rawdata_root, anno_pd=val_pd,
                          dprocess=data_process['val'])  #train_pd 应该调成 val_pd,原先为train_pd
# sklearn读取数据，数据打包
dataloader={}
dataloader['train']=torch.utils.data.DataLoader(data_set['train'],batch_size=BATCH_SIZE,
                                                shuffle=True,num_workers=0,collate_fn=collate_fn) #参数决定了由几个进程来处理

dataloader['val']=torch.utils.data.DataLoader(data_set['val'],batch_size=BATCH_SIZE,
                                                shuffle=True,num_workers=0,collate_fn=collate_fn)

'''model'''
model = Classify()
base_lr = 0.01
resume = None
# 第一次运行使用resume=None
# resume=None
if resume:
    # 加载已有模型
    model.eval()
    model.load_state_dict(torch.load(resume))
model.double() # cuda之前需要将数据转换
model.cuda()


optimizer = torch.optim.Adam(model.parameters(), lr=0.001,betas=(0.9, 0.999), eps= 1e-08, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)  # 设置学习率，lr*gama^(epoch/step_size)
if __name__ == '__main__':
    train(model,
          epoch_num=100,
          start_epoch=0,
          optimizer=optimizer,
          criterion=criterion,
          exp_lr_scheduler=exp_lr_scheduler,
          data_set=data_set,
          data_loader=dataloader,
          save_dir=save_dir,
          print_inter=50,
          val_inter=400)

