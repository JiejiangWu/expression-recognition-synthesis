import torch
import torch.autograd as autograd
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')


def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

def save_loss_png(losslog,name):
    fig = plt.plot(losslog.data)
    plt.savefig(name)
    plt.close()
    
def one_hot(labels, dim):
    """Convert label indices to one-hot vector"""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out