from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import os
from time import time

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

from networks.TSPCS_Net import TSPCS_Net_
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder


import Constants
#import image_utils
savepath='intermediate_results/'
import numpy as np

def TSPCS_Net_Train():
    NAME = 'TSPCS-Net' + Constants.ROOT.split('/')[-1]
    solver = MyFrame(TSPCS_Net_, dice_bce_loss, 2e-4)
    batchsize = 4
    dataset = ImageFolder(root_path=Constants.ROOT, datasets='pavement')
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=4)

    solver.paraNum() 

    mylog = open('logs/' + NAME + '.log', 'w')
    tic = time()
    no_optim = 0
    total_epoch = Constants.TOTAL_EPOCH
    train_epoch_best_loss = Constants.INITAL_EPOCH_LOSS
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        index = 0

        for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss, pred = solver.optimize()
            train_epoch_loss += train_loss
            index = index + 1
        print('epoch:', epoch, '    time before imwrite:', int(time() - tic))
        show_image = (img + 1.6) / 3.2 * 255.
        cv2.imwrite(savepath+'img0-tanet-'+str(epoch)+'.png',np.transpose(show_image[0,:,:,:].cpu().detach().numpy(),(1,2,0)))
        cv2.imwrite(savepath+'mask0-tanet-'+str(epoch)+'.png',np.transpose(mask[0,:,:,:].cpu().detach().numpy()*255,(1,2,0)))
        cv2.imwrite(savepath+'pred0-tanet-'+str(epoch)+'.png',np.transpose(pred[0,:,:,:].cpu().detach().numpy()*255,(1,2,0)))
        train_epoch_loss = train_epoch_loss/len(data_loader_iter)
        print(mylog, '********')
        print(mylog, 'epoch:', epoch, '    time:', int(time() - tic))
        print(mylog, 'train_loss:', train_epoch_loss)
        print(mylog, 'SHAPE:', Constants.Image_size)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('totalNum in an epoch:',index)
        print('train_loss:', train_epoch_loss)
        print('SHAPE:', Constants.Image_size)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('./weights/' + NAME + '_plus_spatial_multi.th')
        if no_optim > Constants.NUM_EARLY_STOP:
            print(mylog, 'early stop at %d epoch' % epoch)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > Constants.NUM_UPDATE_LR:
            if solver.old_lr < 5e-7:
                break
            solver.load('./weights/' + NAME + '_plus_spatial_multi.th')
            solver.update_lr(2.0, factor=True, mylog=mylog)
        mylog.flush()

    print(mylog, 'Finish!')
    print('Finish!')
    mylog.close()


if __name__ == '__main__':
    print(torch.__version__)
    TSPCS_Net_Train()



