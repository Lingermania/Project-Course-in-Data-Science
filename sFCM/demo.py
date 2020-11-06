#from __future__ import print_function
import argparse
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

use_cuda = torch.cuda.is_available()
'''
parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--scribble', action='store_true', default=False, 
                    help='use scribbles')
parser.add_argument('--nChannel', metavar='N', default=100, type=int, 
                    help='number of channels')
parser.add_argument('--maxIter', metavar='T', default=1000, type=int, 
                    help='number of maximum iterations')
parser.add_argument('--minLabels', metavar='minL', default=3, type=int, 
                    help='minimum number of labels')
parser.add_argument('--lr', metavar='LR', default=0.1, type=float, 
                    help='learning rate')
parser.add_argument('--nConv', metavar='M', default=2, type=int, 
                    help='number of convolutional layers')
parser.add_argument('--visualize', metavar='1 or 0', default=1, type=int, 
                    help='visualization flag')
parser.add_argument('--input', metavar='FILENAME',
                    help='input image file name', required=True)
parser.add_argument('--stepsize_sim', metavar='SIM', default=1, type=float,
                    help='step size for similarity loss', required=False)
parser.add_argument('--stepsize_con', metavar='CON', default=1, type=float, 
                    help='step size for continuity loss')
parser.add_argument('--stepsize_scr', metavar='SCR', default=0.5, type=float, 
                    help='step size for scribble loss')
args = parser.parse_args()
'''


# CNN model
class MyNet(nn.Module):
    def __init__(self,input_dim,nConv,nChannel):
        super(MyNet, self).__init__()
        self.nConv=nConv
        self.nChannel=nChannel
        self.conv1 = nn.Conv2d(input_dim, self.nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(self.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(self.nConv-1):
            self.conv2.append( nn.Conv2d(self.nChannel, self.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(self.nChannel) )
        self.conv3 = nn.Conv2d(self.nChannel, self.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(self.nChannel)
        

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


    def run(self,im_path,scribble=False,nChannel=100,maxIter=1000,minLabels=3,lr=0.1,nConv=2,visualize=1,verbose=1,stepsize_sim=1,stepsize_con=1,stepsize_scr=0.5,save_final_output=1):
        
        # load image
        im = cv2.imread(im_path)
        data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
        if use_cuda:
            data = data.cuda()
        data = Variable(data)

        # load scribble
        if scribble:
            mask = cv2.imread(im_path.replace('.'+im_path.split('.')[-1],'_scribble.png'),-1)
            mask = mask.reshape(-1)
            mask_inds = np.unique(mask)
            mask_inds = np.delete( mask_inds, np.argwhere(mask_inds==255) )
            inds_sim = torch.from_numpy( np.where( mask == 255 )[ 0 ] )
            inds_scr = torch.from_numpy( np.where( mask != 255 )[ 0 ] )
            target_scr = torch.from_numpy( mask.astype(np.int) )
            if use_cuda:
                inds_sim = inds_sim.cuda()
                inds_scr = inds_scr.cuda()
                target_scr = target_scr.cuda()
            target_scr = Variable( target_scr )
            # set minLabels
            minLabels = len(mask_inds)

        

        # similarity loss definition
        loss_fn = torch.nn.CrossEntropyLoss()

        # scribble loss definition
        loss_fn_scr = torch.nn.CrossEntropyLoss()

        # continuity loss definition
        loss_hpy = torch.nn.L1Loss(size_average = True)
        loss_hpz = torch.nn.L1Loss(size_average = True)

        HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], nChannel)
        HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, nChannel)
        if use_cuda:
            HPy_target = HPy_target.cuda()
            HPz_target = HPz_target.cuda()
            
        optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        label_colours = np.random.randint(255,size=(100,3))

        for batch_idx in range(maxIter):
            # forwarding
            optimizer.zero_grad()
            output = self( data )[ 0 ]
            output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )

            outputHP = output.reshape( (im.shape[0], im.shape[1], nChannel) )
            HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
            HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
            lhpy = loss_hpy(HPy,HPy_target)
            lhpz = loss_hpz(HPz,HPz_target)

            ignore, target = torch.max( output, 1 )
            im_target = target.data.cpu().numpy()
            nLabels = len(np.unique(im_target))
            if visualize:
                im_target_rgb = np.array([label_colours[ c % nChannel ] for c in im_target])
                im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
                cv2.imshow( "output", im_target_rgb )
                cv2.imwrite('pytorch-unsupervised-segmentation-tip/Output_brain/{}.jpeg'.format(batch_idx), im_target_rgb)
                cv2.waitKey(10)

            # loss 
            if scribble:
                loss = stepsize_sim * loss_fn(output[ inds_sim ], target[ inds_sim ]) + stepsize_scr * loss_fn_scr(output[ inds_scr ], target_scr[ inds_scr ]) + stepsize_con * (lhpy + lhpz)
            else:
                loss = stepsize_sim * loss_fn(output, target) + stepsize_con * (lhpy + lhpz)
                
            loss.backward()
            optimizer.step()

            if verbose:
                print (batch_idx, '/', maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())

            if nLabels <= minLabels:
                print ("nLabels", nLabels, "reached minLabels", minLabels, ".")
                break

        # save output image
        if not visualize:
            output = self( data )[ 0 ]
            output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
            ignore, target = torch.max( output, 1 )
            im_target = target.data.cpu().numpy()
            im_target_rgb = np.array([label_colours[ c % nChannel ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
        
        if save_final_output:
            cv2.imwrite( "output.png", im_target_rgb )
        
        return (None,im_target_rgb)


if __name__ == "__main__":
    run_MyNet.run('Pytorch-unsupervised-segmentation-tip/BSD500/101027.jpg', visualize=0)