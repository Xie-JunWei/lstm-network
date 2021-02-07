from __future__ import division
import os
import torch
import torch.utils.data as data
from d_process import *
import numpy as np
import pandas as pd

N_DIM=6

class dataset(data.Dataset):
    def __init__(self, trajroot, anno_pd, dprocess=None):
        self.root_path = trajroot
        self.paths = anno_pd["file_name"].tolist()
        self.labels = anno_pd["label"].tolist()
        self.dprocess = dprocess

    def __len__(self):
        # print('len(self.paths)',len(self.paths))
        return len(self.paths)
    def __getitem__(self, item):
        # print('item',item)
        traj_path = os.path.join(self.root_path, self.paths[item])
        traj = np.loadtxt(traj_path)[:, 1:N_DIM+1]

        if self.dprocess is not None:
            traj = self.dprocess(traj)
        label = self.labels[item]
        FileName=self.paths[item]

        return torch.from_numpy(traj).double(), label

def collate_fn(batch):
    trajs = []
    label = []
    FileName=[]

    for sample in batch:
        trajs.append(sample[0])
        label.append(sample[1])
    return torch.stack(trajs, 0), label  # name
