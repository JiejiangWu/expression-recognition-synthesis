import matplotlib
matplotlib.use('Agg')
import yaml
import argparse,os
#from src.dataset import datasets
#from src.training import training
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
torch.backends.cudnn.enabled=False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def smooth_avg(data,interval):
    result = data.copy()
    left = 0
    right = min(left+interval,len(data))
    while left < len(data):
        avg = np.mean(data[left:right])
        result[left:right] = [avg] * (right-left)
        left = right
        right = min(left+interval,len(data))
    return result
    

def expand(array,len1):
    idx1 = np.arange(0,1,1/len(array))[0:len(array)]
    idx2 = np.arange(0,1,1/len1)
    return np.interp(idx2,idx1,array)
    
def save_loss_png(losslogs,name,smooth_interval=100):
    color_list = ['b-','r-','g-','y-']
    
    nums = len(losslogs)
    

    '''concatenate all epoch's loss'''
    data = [[] for i in range(nums)]
    for n, loss in enumerate(losslogs):
        for epoch_loss in loss:
            data[n].append(epoch_loss.data)
        
        data[n] = np.concatenate(data[n],0)
            
    '''interplate loss data to make trian/val loss the same length in x axis'''
    maxLen = 0
    for data_ in data:
        if len(data_) > maxLen:
            maxLen = len(data_)
    
    
    fig = plt.figure(figsize=(7,5))
    for i,data_ in enumerate(data):
#        plt.plot(data_,color_list[i])
        data_ = expand(data_,maxLen)
        plt.plot(smooth_avg(data_,smooth_interval),color_list[i])
        plt.savefig(name)
    plt.close()

def main():
    a = [1,2,3,4,5,6,7,8,9]
    print(smooth_avg(a,4))
    checkpoint_dir = './data/checkpoint/expression-recongnition-expw/Temp.pth.tar'
    checkpoint = torch.load(checkpoint_dir)
#    model.load_state_dict(checkpoint['model'])
#    optimizer.load_state_dict(checkpoint['optimizer'])
#    epoch = checkpoint['epoch']+1
    trainlog = checkpoint['trainlog']
    vallog = checkpoint['vallog']
    train_loss = trainlog['loss']
    train_acc = trainlog['acc']
    val_loss = vallog['loss']
    val_acc = vallog['acc']
    
    save_loss_png([train_loss,val_loss],'expw-loss.png')
    save_loss_png([train_acc,val_acc],'expw-acc.png')
    
    save_loss_png([train_loss],'expw-train-loss.png')
    save_loss_png([train_acc],'expw-train-acc.png')
    save_loss_png([val_loss],'expw-val-loss.png')
    save_loss_png([val_acc],'expw-val-acc.png')
    
    checkpoint_dir = './data/checkpoint/expression-recongnition-fer2013/Temp.pth.tar'
    checkpoint = torch.load(checkpoint_dir)
#    model.load_state_dict(checkpoint['model'])
#    optimizer.load_state_dict(checkpoint['optimizer'])
#    epoch = checkpoint['epoch']+1
    trainlog = checkpoint['trainlog']
    vallog = checkpoint['vallog']
    train_loss = trainlog['loss']
    train_acc = trainlog['acc']
    val_loss = vallog['loss']
    val_acc = vallog['acc']
    
    save_loss_png([train_loss,val_loss],'fer2013-loss.png')
    save_loss_png([train_acc,val_acc],'fer2013-acc.png')
    
    save_loss_png([train_loss],'fer2013-train-loss.png')
    save_loss_png([train_acc],'fer2013-train-acc.png')
    save_loss_png([val_loss],'fer2013-val-loss.png')
    save_loss_png([val_acc],'fer2013-val-acc.png')
    
    
    checkpoint_dir = './data/checkpoint/aug-fer2013/Temp.pth.tar'
    checkpoint = torch.load(checkpoint_dir)
#    model.load_state_dict(checkpoint['model'])
#    optimizer.load_state_dict(checkpoint['optimizer'])
#    epoch = checkpoint['epoch']+1
    trainlog = checkpoint['trainlog']
    vallog = checkpoint['vallog']
    train_loss = trainlog['loss']
    train_acc = trainlog['acc']
    val_loss = vallog['loss']
    val_acc = vallog['acc']
    
    save_loss_png([train_loss,val_loss],'aug-fer2013-loss.png')
    save_loss_png([train_acc,val_acc],'aug-fer2013-acc.png')
    
    save_loss_png([train_loss],'aug-fer2013-train-loss.png')
    save_loss_png([train_acc],'aug-fer2013-train-acc.png')
    save_loss_png([val_loss],'aug-fer2013-val-loss.png')
    save_loss_png([val_acc],'aug-fer2013-val-acc.png')
    
    
    checkpoint_dir = './data/checkpoint/aug-expw/Temp.pth.tar'
    checkpoint = torch.load(checkpoint_dir)
#    model.load_state_dict(checkpoint['model'])
#    optimizer.load_state_dict(checkpoint['optimizer'])
#    epoch = checkpoint['epoch']+1
    trainlog = checkpoint['trainlog']
    vallog = checkpoint['vallog']
    train_loss = trainlog['loss']
    train_acc = trainlog['acc']
    val_loss = vallog['loss']
    val_acc = vallog['acc']
    
    save_loss_png([train_loss,val_loss],'aug-expw-loss.png')
    save_loss_png([train_acc,val_acc],'aug-expw-acc.png')
    
    save_loss_png([train_loss],'aug-expw-train-loss.png')
    save_loss_png([train_acc],'aug-expw-train-acc.png')
    save_loss_png([val_loss],'aug-expw-val-loss.png')
    save_loss_png([val_acc],'aug-expw-val-acc.png')
    
    
main()
