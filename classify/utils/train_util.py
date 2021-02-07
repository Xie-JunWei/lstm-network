from __future__ import division
import torch
import os, time, datetime
from torch.autograd import Variable
import logging
import numpy as np
from math import ceil
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LATERS = 2
HIDDEN_SIZE = 128  #Hidden_szie为128，不是256


def dt():
    return datetime.datetime.now().strftime('%y-%m-%d %H:%M:%S')

def trainlog(logfilepath, head='%(message)s'):
    logger = logging.getLogger('mylogger')
    logging.basicConfig(filename=logfilepath, level=logging.INFO, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def train(model,
          epoch_num,
          start_epoch,
          optimizer,
          criterion,
          exp_lr_scheduler,  # 学习率
          data_set,
          data_loader,
          save_dir,
          print_inter=200,
          val_inter=3500
          ):
    step = -1
    for epoch in range(start_epoch,epoch_num):
        # train phase
        exp_lr_scheduler.step(epoch)
        model.train(True)

        for batch_cnt, data in enumerate(data_loader['train']):  # 枚举方法，返回的是编号和对应值
            step += 1
            model.train(True)
            # print data
            inputs, labels = data
            # print('inputs0',inputs)
            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels).astype(np.int)).long().cuda())
            # print('labels',labels)
            # zero the parameter gradients
            optimizer.zero_grad()
            h_n = torch.zeros(NUM_LATERS, inputs.shape[0], HIDDEN_SIZE, dtype=torch.double).cuda()
            c_n = torch.zeros(NUM_LATERS, inputs.shape[0], HIDDEN_SIZE, dtype=torch.double).cuda()
            # print('inputs1',inputs)
            outputs,h_n, c_n  = model(inputs, h_n, c_n)  # model这个函数执行的是forward()函数？？？return out,h_n, c_n
            # print('outshape',outputs.shape)
            if isinstance(outputs, list):
                # print('outputs1',outputs[0])
                loss = criterion(outputs[0], labels)
                # print('outputs2', outputs[1])
                loss +=criterion(outputs[1], labels)
                outputs=outputs[0]
                # print('outputs',outputs)

            else:
                loss = criterion(outputs, labels)
                # print('outputs3',loss)
            _, preds = torch.max(outputs, 1)
            # print('preds',preds)
            loss.backward()
            optimizer.step()

            # batch loss
            if step % print_inter == 0:
                _, preds = torch.max(outputs, 1)

                # batch_corrects = torch.sum((preds == labels)).data[0]
                batch_corrects = torch.sum((preds == labels)).item()
                batch_acc = batch_corrects / (labels.size(0))

                logging.info('%s [%d-%d] | batch-loss: %.3f | acc@1: %.3f'
                             % (dt(), epoch, batch_cnt, loss.item(), batch_acc))
            if step % val_inter ==0:   # 应该是循环3500
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())
                # val phase
                model.train(False) # set model to evaluate mode

                val_loss = 0
                val_corrects = 0
                val_size = ceil(len(data_set['val']) / data_loader['val'].batch_size)

                t0 = time.time()

                for batch_cnt_val, data_val in enumerate(data_loader['val']):
                    # print data
                    inputs, labels = data_val

                    h_n = torch.zeros(NUM_LATERS, inputs.shape[0], HIDDEN_SIZE, dtype=torch.double).cuda()
                    c_n = torch.zeros(NUM_LATERS, inputs.shape[0], HIDDEN_SIZE, dtype=torch.double).cuda()

                    inputs = Variable(inputs.cuda())
                    labels = Variable(torch.from_numpy(np.array(labels).astype(np.int)).long().cuda())

                    # forward
                    outputs, h_n, c_n = model(inputs, h_n, c_n)
                    if isinstance(outputs, list):
                        loss = criterion(outputs[0], labels)
                        loss += criterion(outputs[1], labels)
                        outputs = outputs[0]
                    else:
                        loss = criterion(outputs, labels)
                    _,preds = torch.max(outputs, 1)

                    # statistics
                    val_loss += loss.item()
                    batch_corrects = torch.sum((preds == labels)).item()
                    val_corrects += batch_corrects
                val_loss = val_loss / val_size
                val_acc = 1.0 * val_corrects / len(data_set['val'])

                t1 = time.time()
                since = t1-t0
                logging.info('--'*30)
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())
                logging.info('%s epoch[%d]-val-loss: %.4f ||val-acc@1: %.4f ||time: %d'
                      % (dt(), epoch, val_loss, val_acc, since))
                pred = preds.cpu().data.numpy()
                print('prediction', pred)
                print('label',labels)

                save_path = os.path.join(save_dir,
                                         'weights-%d-%d-[%.4f].pth' % (epoch, batch_cnt, val_acc)) #不需要每次都存
                torch.save(model.state_dict(), save_path)
                logging.info('saved model to %s' % (save_path))
                logging.info('--'*30)








