import torch
import torch.nn as nn
from torch.autograd import Variable as V
import cv2
import numpy as np

class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode=False):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()
        
    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id
        
    def test_one_img(self, img):
        pred,_ = self.net.forward(img)
        return pred    

    def test_batch(self):
        self.forward(volatile=True)
        mask =  self.net.forward(self.img).cpu().data.numpy().squeeze(1)
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask, self.img_id
    
    def test_one_img_from_path(self, path):
        img = cv2.imread(path)
        img = np.array(img, np.float32)/255.0 * 3.2 - 1.6
        img = V(torch.Tensor(img).cuda())
        
        mask = self.net.forward(img).squeeze().cpu().data.numpy()
        mask[mask>0.5] = 1
        mask[mask<=0.5] = 0
        
        return mask
        
    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred, prediction2, prediction3, prediction4 = self.net.forward(self.img)

        loss1 = self.loss(self.mask, pred)
        loss2 = self.loss(self.mask, prediction2)
        loss3 = self.loss(self.mask, prediction3)
        loss4 = self.loss(self.mask, prediction4)
        loss = 0.85 * loss1 + 0.1 * loss2 + 0.03 * loss3 + 0.02 * loss4
        loss.backward()
        self.optimizer.step()
        return loss.data, pred
        
    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path),strict=False)
    
    def paraNum(self):
        print("the network have {} paramerters in total".format(sum(x.numel() for x in self.net.parameters())))

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        mylog.write( 'update learning rate: %f -> %f\n' % (self.old_lr, new_lr))
        print ('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr
