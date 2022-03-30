# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:43:29 2022

@author: WYJ
"""

import torch
import os
import torch.autograd as autograd
from src.models import models
from src.utils import utils,train_utils
from torch import nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import imageio
from torch.autograd import Variable
from src.dataset import datasets
torch.backends.cudnn.enabled=False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

G = models.Generator()
G = G.to(device)

DATASET = 'fer2013'

# dataset
if DATASET == 'fer2013':
    fer2013_image_path='./data/datasets/fer2013/data/fer2013/train'

    dataloader = datasets.get_loader(fer2013_image_path, 48, 64,1, 'test')
if DATASET == 'expw':
    expw_train_file_path = './data/datasets/expw/train.lst'
    expw_img_path = './data/datasets/expw/ExpwCropped' 
    dataset_train = datasets.Resize_ExpW_dataset(expw_train_file_path,expw_img_path,64)
    dataloader = DataLoader(dataset_train,batch_size=1,shuffle=False,num_workers=1) 


# checkpoint
checkpoint_dir = './data/checkpoint/expression-synthesis-fer2013/Temp.pth.tar'
#checkpoint_dir = './data/checkpoint/expression-synthesis-expw/Temp.pth.tar'

state = torch.load(checkpoint_dir)
G.load_state_dict(state['model_G'])
G.eval()


# sampled images
fixed_x = []
real_c = []
for i, (images, labels) in enumerate(dataloader):
    fixed_x.append(images)
    real_c.append(labels)
    if i == 50:
        break

demo_dir = './generated_images/fer2013-debug'

if not os.path.exists(demo_dir):
    os.makedirs(demo_dir)

for i,x in enumerate(fixed_x):
#    print(fixed_x[0].shape)
    imageio.imsave(os.path.join(demo_dir, '%03d_real.png' % (i)),utils.img_cvt(x[0]))
    for j in range(7):
        c = train_utils.one_hot(torch.ones(x.size(0))*j,7)
        x = x.to(device)
        c = c.to(device)
        fake_image = G(x,c)
        imageio.imsave(os.path.join(demo_dir, '%03d_express_%01d.png' % (i,j)),utils.img_cvt(fake_image[0]))
