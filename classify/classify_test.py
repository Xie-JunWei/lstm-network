"""
This is a test for classify the trajectories.
WE have three kinds of trajectories:  | L_D| c_aphi| c_h|
                         with label:  |  0 |   1   |  2 |


Try to use linear+LSTM+linear NET.

"""
from __future__ import print_function
from utils.dataset import dataset,collate_fn
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from Classify_net import Classify
from utils.d_process import *
import time
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import torch.utils.data as Data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import pandas as pd
# INPUT_SIZE = 6
# OUTPUT_SIZE = 3
# TIME_STEP = 1200
# LINEAR1_OUT = 32    # D:通过6维输入，提取另外的特征 *2
# HIDDEN_SIZE = 64    # 记录调试日志
# BATCH_SIZE = 64     #
# NUM_LATERS = 2      #



# Hyper Parameters
INPUT_SIZE = 6
OUTPUT_SIZE = 3
TIME_STEP = 1200
LINEAR1_OUT = 32    # D:通过6维输入，提取另外的特征 *2
HIDDEN_SIZE = 128    # 记录调试日志
BATCH_SIZE = 64     #
NUM_LATERS = 2      #
LR = 0.01
N_TRAIN_POINTS = 100
EPOCH = 1
MAX_LEN = 600
# Hyper Parameters
# INPUT_SIZE = 6
#
# OUTPUT_SIZE = 3
# TIME_STEP = 1200
# LINEAR1_OUT = 32    # D:通过6维输入，提取另外的特征 *2
# HIDDEN_SIZE = 64    # 记录调试日志
# BATCH_SIZE = 64     #
# NUM_LATERS = 2      #
# LR = 0.01
# N_TRAIN_POINTS = 100
# EPOCH = 1

class Classify(nn.Module):
    def __init__(self):
        super(Classify, self).__init__()
        # 如果在最前面加了Linear，fe需要更改input_size
        self.lstm = nn.LSTM(
            input_size=LINEAR1_OUT,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LATERS,
            batch_first=True
        )  # 需要注意数据格式，在前期定义时设置batch_first = True
        self.input_layer = nn.Linear(in_features=INPUT_SIZE, out_features=LINEAR1_OUT) # Ⅰ：在输入前加一层线性层，做一次映射
        self.out_layer = nn.Linear(in_features=HIDDEN_SIZE, out_features=OUTPUT_SIZE) # Ⅱ：仅在输出加一层线性层，将隐含层特征映射为真实值

    def forward(self, input,h_n, c_n):

        mid = F.relu(self.input_layer(input), inplace=False)
        r_out, (h_n, c_n) = self.lstm(mid,(h_n, c_n))
        mid2 = F.relu(r_out[:, -1, :], inplace=False)
        out=self.out_layer(mid2)
        return out,h_n, c_n

if __name__ == "__main__":
    # set random seed to 0

    np.random.seed(0)
    torch.manual_seed(0)
    #加载模型
    resume = 'F:\classi\model/weights-90-50-[0.9700].pth'

    # load dataset
    rawdata_root = r'F:\classi\test'## 原始测试数据  怎么还是train的数据集呢，不应该是test的数据集吗？？？
    # 读测试集标签
    all_pd = pd.read_csv(r"F:\classi\test/dataset_test.csv", sep=",",
                         header=None,
                         names=["file_name", "label"])[1:]

    # 全部划给测试集
    train_pd, val_pd = train_test_split(all_pd, test_size=0.1, random_state=43)  # 去查下这个函数的用法 2020-10-20 有可能指的是标签只有1维
    # 数据预处理
    data_process = {
    'train':data_process(max_len=MAX_LEN),
    'val':data_process(max_len=MAX_LEN)
    }  #调用的是d_process中的dp函数（）
    data_set = {}
    data_set['train'] = dataset(trajroot=rawdata_root, anno_pd=train_pd, dprocess=data_process['train'])# train_pd 应该调成 val_pd,原先为train_pd
    data_set['val'] = dataset(trajroot=rawdata_root, anno_pd=val_pd, dprocess=data_process['val'])

    # dataloader = {}
    loader = Data.DataLoader(data_set['train'], batch_size=BATCH_SIZE,
                                                     shuffle=True, num_workers=0,
                                                     collate_fn=collate_fn)  # 参数决定了由几个进程来处理


   # print(data_set['train'])
    # Data Loader
    # # 把dataset放入Dataloader中
    # loader = Data.DataLoader(
    #     dataset=data_set['train'],
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=0,
    # )
    # build the model
    classify_net = Classify()
    if resume:
        classify_net.eval()
        state_dict = torch.load(resume)
        classify_net.load_state_dict(state_dict)
        classify_net.double()
        classify_net.cuda()
    print(classify_net)

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classify_net.parameters(), lr=LR)
    #
    # training the model
    accuracy_list = []
    precision_list =[]
    for epoch in range(EPOCH):
        save_count = 0
        n=0
        for step, (batch_x, batch_y, batch_z) in enumerate(loader):

    #         # 考虑 h_0,c_0  初始化
            h_n = torch.zeros(NUM_LATERS, batch_x.shape[0], HIDDEN_SIZE, dtype=torch.double).cuda()
            c_n = torch.zeros(NUM_LATERS, batch_x.shape[0], HIDDEN_SIZE, dtype=torch.double).cuda()
            print('batch_x',batch_x.shape)
            batch_y = [int(x) for x in batch_y]   #这句话啥意思？？？？
            batch_x = Variable(batch_x).cuda()
            batch_y = Variable(torch.tensor(batch_y))  #batch_y是标签，你的batch_y是标签吗？
            # print('train label',batch_y)
            timestart = time.time()
            test_output,h_n, c_n = classify_net(batch_x, h_n, c_n)
            pred_y = torch.max(test_output, 1)[1].cpu().data.numpy()
            end_time=time.time()
            time_consume=end_time-timestart
            batch_y=batch_y.cpu().data.numpy()
            print(type(batch_y))
            accuracy = float((pred_y == batch_y).astype(int).sum()) / float(batch_y.size)
            print('pred_type', pred_y)
            print('real_label',batch_y.astype(int))
            print('File_Name',batch_z)
            np_data = np.array([pred_y, batch_y, batch_z])
            np_data = np_data.transpose()
            pd_data = pd.DataFrame(np_data,columns=['predict', 'real', 'FileName'])
            print(pd_data)
            pd_data.to_csv(os.path.join('E:\desktop\classify\show_file', 'output.csv'), mode='a')
            print('time_consume', time_consume)
            accuracy_s = accuracy_score(batch_y, pred_y)
            print('ac_s', accuracy_s)
            print('ac',accuracy)
            accuracy_list.append(accuracy)
            # precision = precision_score(y_test[n*20:(n+1)*20].astype(int), pred_y)
            # precision_list.append(precision)
            print('Epoch: ', epoch, '| test accuracy: %.5f' % accuracy)
            # if step%350==0:
            def initplt():
                plt.figure(figsize=(30, 10))
                plt.title('Classify Test Accuracy', fontsize=30)
                plt.xlabel('step', fontsize=20)
                plt.ylabel('accuracy', fontsize=20)
                # plt.xticks([0,50,500],fontsize=20)
                # plt.yticks([0,1],fontsize=20)
                plt.xticks(np.arange(0, 500, step=50),fontsize=20)
                plt.yticks(np.arange(0, 1, step=0.05), fontsize=20)
                # plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],fontsize=20)
            def draw(acc):
                plt.plot(np.arange(len(acc)), acc,'k',linewidth=3.0)

        initplt()
        draw(accuracy_list)
        plt.show()
        print(accuracy_list)
        accu_overall=np.sum(accuracy_list)/np.size(accuracy_list)