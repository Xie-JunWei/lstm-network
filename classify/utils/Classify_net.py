from __future__ import print_function
import torch.nn as nn
import torch.nn.functional as F

# Hyper Parameters
INPUT_SIZE = 6   #这边当时也改过了，把网络参数修正过了
OUTPUT_SIZE = 3
TIME_STEP = 1200
LINEAR1_OUT = 32    # D:通过6维输入，提取另外的特征 *2
HIDDEN_SIZE = 128    # 记录调试日志
BATCH_SIZE = 64     #
NUM_LATERS = 2     #
LR = 0.01  #学习率为什么也调整了，你原先训练的学习率到底是0.001还是0.01？？？？
N_TRAIN_POINTS = 100
EPOCH = 1000

class Classify(nn.Module):
    def __init__(self):
        super(Classify, self).__init__()
        # 如果在最前面加了Linear，fe需要更改input_size
        self.lstm = nn.LSTM(
            input_size=LINEAR1_OUT,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LATERS,
            batch_first=True            #bidirectional=True
        )  # 需要注意数据格式，在前期定义时设置batch_first = True
        self.input_layer = nn.Linear(in_features=INPUT_SIZE, out_features=LINEAR1_OUT) # Ⅰ：在输入前加一层线性层，做一次映射
        self.out_layer = nn.Linear(in_features=HIDDEN_SIZE, out_features=OUTPUT_SIZE) # Ⅱ：仅在输出加一层线性层，将隐含层特征映射为真实值

    def forward(self, input, h_n, c_n):

        mid = F.relu(self.input_layer(input), inplace=False)
        r_out, (h_n, c_n) = self.lstm(mid,(h_n, c_n))
        mid2 = F.relu(r_out[:, -1, :], inplace=False)
        out=self.out_layer(mid2)
        # print('out',out.shape)
        return out,h_n, c_n
