'''
Unsupervised Learning of Image Segmentation Based on Differentiable Feature Clustering

Based on https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip
'''

#from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import cv2
import sys
import numpy as np
import torch.nn.init
import random
from copy import deepcopy
use_cuda = torch.cuda.is_available()





# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim, nChannel, nConv):
        super(MyNet, self).__init__()
        self.nChannel = nChannel
        self.nConv    = nConv


        self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(nConv-1):
            self.conv2.append( nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(nChannel) )
        self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

class DFC:
    def __init__(self, nChannel = 100, lr = 0.1, nConv = 2, minLabels = 3, stepsize_sim = 1, stepsize_con = 1, use_cuda = False):
        self.nChannel = nChannel
        self.lr       = lr
        self.nConv    = nConv
        self.stepsize_sim = stepsize_sim
        self.stepsize_con = stepsize_con
        self.use_cuda     = use_cuda
        self.minLabels    = minLabels

    def initialize_clustering(self, im):
        data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
        if use_cuda:
            data = data.cuda()
        self.data = Variable(data)
        self.im = im
        
        # train
        self.model = MyNet( data.size(1), self.nChannel, self.nConv)
        if self.use_cuda:
            self.model.cuda()
        self.model.train()

        # similarity loss definition
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # scribble loss definition
        self.loss_fn_scr = torch.nn.CrossEntropyLoss()

        # continuity loss definition
        self.loss_hpy = torch.nn.L1Loss(size_average = True)
        self.loss_hpz = torch.nn.L1Loss(size_average = True)

        self.HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], self.nChannel)
        self.HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, self.nChannel)
        if self.use_cuda:
            self.HPy_target = HPy_target.cuda()
            self.HPz_target = HPz_target.cuda()
            
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.label_colours = np.random.randint(255,size=(100,3))
    
    def step(self):
    
        self.optimizer.zero_grad()
        output = self.model( self.data )[ 0 ]
        response_map = output.clone().detach().numpy()

        output = output.permute( 1, 2, 0 ).contiguous().view( -1, self.nChannel )

        outputHP = output.reshape( (self.im.shape[0], self.im.shape[1], self.nChannel) )
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = self.loss_hpy(HPy,self.HPy_target)
        lhpz = self.loss_hpz(HPz,self.HPz_target)

        ignore, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        
        loss = self.stepsize_sim * self.loss_fn(output, target) + self.stepsize_con * (lhpy + lhpz)
            
        loss.backward()
        self.optimizer.step()

        print (' label num :', nLabels, ' | loss :', loss.item())

        if nLabels <= self.minLabels:
            print ("nLabels", nLabels, "reached minLabels", self.minLabels, ".")

        im_target_rgb = np.array([self.label_colours[ c % self.nChannel ] for c in im_target])
        return im_target_rgb.reshape( self.im.shape ).astype( np.uint8 ), response_map



if __name__ == "__main__":
    dfc = DFC(minLabels=10, nChannel=100, nConv=2, lr=0.01, stepsize_con=5)

    im = cv2.imread('PCiDS/sFCM/74.jpeg')

    h, w = 95, 95
    y, x = (im.shape[0] - h)//2, (im.shape[1] - w)//2

    im = im[y:y + h, x: x + w]

    dfc.initialize_clustering(im)

    for i in range(0, 100):
        im, r_map = dfc.step()
        cv2.imwrite('PCiDS/sFCM/gifs/{}.jpeg'.format(i), im)
        #cv2.imshow('{}'.format(i), im)
        #cv2.waitKey(10)
'''
# save output image
if not args.visualize:
    output = model( data )[ 0 ]
    output = output.permute( 1, 2, 0 ).contiguous().view( -1, args.nChannel )
    ignore, target = torch.max( output, 1 )
    im_target = target.data.cpu().numpy()
    im_target_rgb = np.array([label_colours[ c % args.nChannel ] for c in im_target])
    im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
cv2.imwrite( "output.png", im_target_rgb )
'''