# from __future__ import print_function
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

# x=np.arange(30)
# y1=np.zeros(10)
# y1=y1.astype(int)
# #print(y1)
# y2=np.ones(10)
# y2=y2.astype(int)
# #print(y2)
# y3=np.arange(10,dtype=np.float)
# y3=np.full_like(y3,2)
# y3=y3.astype(int)
# #print(y3)
# y=np.append(y1,y2)
# y=np.append(y,y3)
#
# x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=1,stratify=y)
# print(x)
# print(y)
# print(x_test)
# print(y_test)
#
# # 原始数据
# rawdata_root='F:\classi/train'
# # 读标签
# all_pd=pd.read_csv("F:\HXDD/dataset_train.csv", sep=",",
#                    header=None,
#                    names=["file_name","label"])
# print(all_pd.head())
# aa=all_pd['file_name']
# print(aa)
#
# train_pd, val_pd = train_test_split(all_pd, test_size=0.2, random_state=43,
#                                     stratify=all_pd['label']) #去查下这个函数的用法 2020-10-20 有可能指的是标签只有1维
# print(val_pd)
#
# step=0
# a=step % 2500
print('a=')
epoch_num = 100
start_epoch = 0
xr=range(start_epoch,epoch_num)
print(np.array(xr))