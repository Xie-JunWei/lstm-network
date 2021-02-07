# 预处理 加随机噪声，归一化
import numpy as np

def dp(traj_,max_len_):
    x, x_v, y, y_v ,z ,z_v= [],[],[],[],[],[]

    ini = 0
    # count = 1201

    for line in traj_:
        if ini==0:
            x_0=line[0]
            y_0=line[1]
            z_0=line[2]
            ini+=1
        else:
            rand_r = np.random.normal(0, 145, 3)
            rand_v = np.random.normal(0, 23.2, 3)
            np.random.shuffle(rand_r)
            np.random.shuffle(rand_v)

            # 添加随机噪声
            # x.append((line[0]-x_0+rand_r[0])/10e3)
            # y.append((line[1]-y_0+rand_r[1])/10e3)
            # z.append((line[2]-z_0+rand_r[2])/10e3)
            # x_v.append(line[3]+rand_v[0])
            # y_v.append(line[4]+rand_v[1])
            # z_v.append(line[5]+rand_v[2])

            x.append((line[0] - x_0) / 10e5)
            y.append((line[1] - y_0) / 10e5)
            z.append((line[2] - z_0) / 10e5)
            x_v.append(line[3] / 10e2)
            y_v.append(line[4] / 10e2)
            z_v.append(line[5] / 10e2)


        max_len_=max_len_-1
        if max_len_<0:
            break
    xnew, ynew, znew, x_vnew, y_vnew ,z_vnew=np.array(x), np.array(y),np.array(z),\
                                              np.array(x_v),np.array(y_v),np.array(z_v)
    new = np.transpose(np.vstack((xnew,ynew,znew,x_vnew,y_vnew,z_vnew)))
    return new
class data_process(object):
    def __init__(self, max_len):
        self.max_len = max_len
    def __call__(self, traj):
        traj=dp(traj,self.max_len)
        return traj
