#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:50:04 2019

@author: magic
"""
import torch
import numpy as np

def img_cvt(images):
    return (255. * images).detach().cpu().numpy().clip(0, 255).astype('uint8').transpose(1, 2, 0)

def save_checkpoint(path, iteration, iterations_since_improvement, model,optimizer,
                    best_metric, recent_metric, Losses_during_iteration, metric_during_iteration,is_best):
    state = {'iteration': iteration,
             'iterations_since_improvement': iterations_since_improvement,
             'best_metric': best_metric,
             'recent_metric': recent_metric,
             'model': model,
             'optimizer':optimizer,
             'Losses_during_iteration':Losses_during_iteration,
             'metric_during_iteration':metric_during_iteration}
    
    filename = path + '/Temp.pth.tar'
    best_filename = path + '/Best.pth.tar'
    torch.save(state, filename)
#     If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, best_filename)    
        


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.data.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count