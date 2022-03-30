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
import argparse,os,yaml
from torch.autograd import Variable
from src.dataset import datasets
torch.backends.cudnn.enabled=False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 


parser = argparse.ArgumentParser()    
#big-bz-aug-dropout-fer
#data-augmentation-expw
#data-augmentation-fer
#aug-dropout-fer
#aug-dropout-expw
parser.add_argument('--cfg_file', type=str,default = './configs/big-bz-aug-dropout-expw.yaml')
args = parser.parse_args()

# load cfg
f = open(args.cfg_file)
cfg = yaml.load(f)
f.close()

work_dir = os.path.join('./data/checkpoint',cfg['eid'])


expw_test_file_path = './data/datasets/expw/test.lst'
#expw_val_file_path = './data/datasets/expw/val.lst'
expw_img_path = './data/datasets/expw/ExpwCropped' 

fer_file_path = './data/datasets/fer2013/fer2013.csv'

aug = False

if cfg['dataset'] == 'expw':
    dataset_test = datasets.ExpW_dataset(expw_test_file_path,expw_img_path,aug)

if cfg['dataset'] == 'fer2013':  
    dataset_test = datasets.Fer13_dataset(fer_file_path,'test',False)


# define model
use_dropout = False
if 'dropout' in cfg.keys() and cfg['dropout'] == True:
    use_dropout =True        

model = models.resnet_model(use_dropout=use_dropout).to(device)
model.eval()

'''load checkpoint'''
temp_checkpoint_file = os.path.join(work_dir,'Temp.pth.tar')
temp_checkpoint = torch.load(temp_checkpoint_file)
best_epoch = temp_checkpoint['epoch'] - cfg['early_stop_epoch']

best_checkpoint_file = os.path.join(work_dir,'epoch_%03d.pth.tar'%best_epoch)
checkpoint = torch.load(best_checkpoint_file)
model.load_state_dict(checkpoint['model'])


dataloader_test = DataLoader(dataset_test,batch_size=8,shuffle=False,num_workers=0,drop_last = False)

classes = [ 'anger', 'disgust','fear','happy','sad','surprised','normal']

y_true = np.array([])
y_pred = np.array([])

for i, data in enumerate(dataloader_test):
    batch_data, batch_label = data
    batch_data = batch_data.to(device)
    batch_label= batch_label.to(device)
    batch_label = batch_label.long().squeeze()
    
    
    batch_logit = model(batch_data)

    prediction = torch.argmax(batch_logit,1)    
    
    y_true = np.hstack([y_true,batch_label.cpu().numpy()])
    y_pred = np.hstack([y_pred,prediction.cpu().numpy()])




tick_marks = np.array(range(len(classes))) + 0.5
 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#import matplotlib.pyplot.cm as cm

def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):

    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = cm_normalized
    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')
    

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()
    
            
cm = confusion_matrix(y_true, y_pred)
result_dir = './classification'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
plot_confusion_matrix(cm, os.path.join(result_dir,cfg['eid']+'.png'), title='confusion matrix')
acc = (y_true == y_pred).sum() / len(y_pred)
with open(os.path.join(result_dir,cfg['eid']+'.txt'),'w') as f:
    f.writelines(str(acc))
