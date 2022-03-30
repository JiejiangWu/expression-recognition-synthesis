import os,shutil
import torch
from sklearn import cross_validation
import numpy as np
source_file = 'G:/projects/pyProjects/netease-homework/expression-recongnition/data/datasets/expw/label.lst'

train_file = 'G:/projects/pyProjects/netease-homework/expression-recongnition/data/datasets/expw/train.lst'
val_file = 'G:/projects/pyProjects/netease-homework/expression-recongnition/data/datasets/expw/val.lst'
test_file = 'G:/projects/pyProjects/netease-homework/expression-recongnition/data/datasets/expw/test.lst'


def split_ExpW(src_path,train_file,val_file,test_file):
    data = []
    names = {}
    with open(src_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            content = line.split(' ')
            meta_name = content[0]
            id_name = content[1]
            final_name = meta_name.split('.')[0] + '_'+str(id_name)+'.png'
            
#            data_count+=1
            if not final_name in names: 
                data.append([final_name, int(content[7])])
                names[final_name] = 0
            else:
                continue
    
    train_len = int(len(data)*0.7)
    val_len = int(len(data)*0.1)
    test_len = len(data)-train_len-val_len
    
    idx = np.array(range(len(data)))
    np.random.shuffle(idx)
    train_idx = idx[0:train_len]
    val_idx = idx[train_len:train_len+val_len]
    test_idx = idx[train_len+val_len:]
    
    train_idx = sorted(train_idx)
    val_idx = sorted(val_idx)
    test_idx = sorted(test_idx)  
    
    with open(train_file,'w') as f:
        for i in train_idx:
            f.write(data[i][0] + ' ' + str(data[i][1]))
            f.write('\n')

    with open(test_file,'w') as f:
        for i in test_idx:
            f.write(data[i][0] + ' ' + str(data[i][1]))
            f.write('\n')
            
    with open(val_file,'w') as f:
        for i in val_idx:
            f.write(data[i][0] + ' ' + str(data[i][1]))
            f.write('\n')
        
#    return data

split_ExpW(source_file,train_file,val_file,test_file)