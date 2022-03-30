# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:00:08 2022

@author: WYJ
"""
import pandas as pd
import numpy as np
import os#, cv2
from torch.utils.data import Dataset,DataLoader
import torch.utils 
from skimage.io import imread
import skimage.transform
from torchvision import transforms


def preprocess(image, input_size, augmentation=True):
    if augmentation:
        crop_transform = transforms.Compose([
            transforms.Resize(input_size // 4 * 5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomCrop(input_size),
            transforms.RandomRotation(10)])
    else:
        crop_transform = transforms.CenterCrop(input_size)

    result = transforms.Compose([
        transforms.ToPILImage(),
        crop_transform,
        transforms.ToTensor(),
    ])(image)
    return result

def preprocess2(image, output_size):
    crop_transform = transforms.Compose([
                transforms.Resize(output_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
#                transforms.Normalize(0)
                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
                ])
    

    result = transforms.Compose([
        transforms.ToPILImage(),
        crop_transform,
    ])(image)
    return result

def preprocess_resize(image, output_size):
    crop_transform = transforms.Compose([
                transforms.Resize(output_size),
                ])
    

    result = transforms.Compose([
        transforms.ToPILImage(),
        crop_transform,
    ])(image)
    return result

def preprocess3(image, output_size):
    crop_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
#                transforms.Normalize(0)
                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
                ])
    
    result = transforms.Compose([
#        transforms.ToPILImage(),
        crop_transform,
    ])(image)
    return result


def load_fer2013(file_path='G:/projects/pyProjects/netease-homework/expression-recongnition/data/datasets/fer2013.csv'):
    """ Load fer2013.csv dataset from csv file """
    df = pd.read_csv(file_path)
    train = df[df['Usage'] == 'Training']
    val = df[df['Usage'] == 'PublicTest']
    test = df[df['Usage'] == 'PrivateTest']
    return train, val, test



def parse_fer2013(data, target_size=(48, 48), target_channel=1):
    """ Parse fer2013 data to images with specified sizes,
        and one-hot vector as labels """
    real_image_size = (48, 48)
    real_image_channel = 1
    images = np.empty(shape=(len(data), *target_size, target_channel))
    labels = np.empty(shape=(len(data), 1))
    for i, idx in enumerate(data.index):
        img = np.fromstring(data.loc[idx, 'pixels'], dtype='uint8', sep=' ')
        img = np.reshape(img, (48, 48))
        if target_size != real_image_size:
            img = skimage.transform.resize(img,target_size)
#            img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        img = img[..., np.newaxis]
        if target_channel != real_image_channel:
            img = np.repeat(img, repeats=target_channel, axis=-1)
        label = data.loc[idx, 'emotion']
        images[i] = img
        labels[i] = label
    return images, labels



class Fer13_dataset(torch.utils.data.Dataset):
    def __init__(self,file_path,split_part = 'train',aug=False,img_size = (48,48),img_channel = 3):
        train,val,test = load_fer2013(file_path)
        if split_part == 'train':
            metadata = train
        elif split_part == 'val':
            metadata = val
        elif split_part == 'test':
            metadata = test
        
        images,labels = parse_fer2013(metadata,img_size,img_channel)
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.aug = aug
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,i):
        
        tmp_img = self.images[i]
        tmp_img = tmp_img.astype('uint8')
        
        tmp_img = preprocess(tmp_img,48,self.aug)

        return tmp_img,self.labels[i]

class Resize_Fer13_dataset(torch.utils.data.Dataset):
    def __init__(self,file_path,split_part = 'train',output_size=128):
        train,val,test = load_fer2013(file_path)
        if split_part == 'train':
            metadata = train
        elif split_part == 'val':
            metadata = val
        elif split_part == 'test':
            metadata = test
        img_size = (48,48)
        img_channel = 3
        images,labels = parse_fer2013(metadata,img_size,img_channel)
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.output_size = output_size
        
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,i):
        tmp_img = self.images[i]
        tmp_img = tmp_img.astype('uint8')
#        print(tmp_img.shape)
        tmp_img = preprocess2(tmp_img,self.output_size)
        return tmp_img,self.labels[i]
    

def load_ExpW(file_path,img_path):
    data = []
    txt_count = 0
    data_count = 0
    miss_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            txt_count+=1
            content = line.split(' ')
            path = os.path.join(img_path, content[0])

            if os.path.exists(path):
                data_count+=1
                data.append([path, int(content[1])])
            else:
                miss_count+=1
    
    print(txt_count)
    print(data_count)
    print(miss_count)
    return data


class ExpW_dataset(torch.utils.data.Dataset):
    def __init__(self,file_path,img_path,aug):
        self.data = load_ExpW(file_path,img_path)
        self.aug = aug
    def __len__(self):
        return len(self.data)
    def __getitem__(self,i):
        tmp_img = imread(self.data[i][0])
        tmp_img = preprocess(tmp_img,48,self.aug)   
        
        return tmp_img, self.data[i][1]
    

class Resize_ExpW_dataset(torch.utils.data.Dataset):
    def __init__(self,file_path,img_path,output_size):
        data = load_ExpW(file_path,img_path)
        self.images = []
        self.labels = []
        for i, tmp_data in enumerate(data):
            tmp_img = imread(tmp_data[0])
            tmp_img = preprocess_resize(tmp_img,output_size)
            self.images.append(tmp_img)
            self.labels.append(tmp_data[1])
        self.output_size = output_size
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,i):
#        tmp_img = imread(self.data[i][0])
        tmp_img = self.images[i]
#        tmp_img = preprocess2(tmp_img,self.output_size)   
        tmp_img = preprocess3(tmp_img,self.output_size)   

        return tmp_img,self.labels[i]
#        return tmp_img, self.data[i][1]    

#
#train, val, test = load_fer2013()
#images,labels = parse_fer2013(train)
        

from torchvision.datasets import ImageFolder
def get_loader(image_path, crop_size, image_size, batch_size, mode='train'):
    """Build and return data loader."""

    if mode == 'train':
        transform = transforms.Compose([
            #transforms.CenterCrop(crop_size),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            #transforms.CenterCrop(crop_size),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = ImageFolder(image_path, transform)
    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader    
    
def main():
    expw_file_path = 'G:/projects/pyProjects/netease-homework/expression-recongnition/data/datasets/expw/val.lst'
    expw_img_path = 'G:/projects/pyProjects/netease-homework/expression-recongnition/data/datasets/expw/ExpwCropped'
    
    fer_file_path = 'G:/projects/pyProjects/netease-homework/expression-recongnition/data/datasets/fer2013/fer2013.csv'
    
#    expw_dataset = ExpW_dataset(expw_file_path,expw_img_path,aug=True)
#    dataloader = DataLoader(expw_dataset,batch_size=4,shuffle=True,num_workers=0,drop_last = True)
 
#    fer_dataset = Fer13_dataset(fer_file_path)
#    dataloader = DataLoader(fer_dataset,batch_size=2,shuffle=True,num_workers=0,drop_last=True)
    
#    resize_fer_dataset = Resize_Fer13_dataset(fer_file_path,'train',64)
#    dataloader = DataLoader(resize_fer_dataset,batch_size=2,shuffle=True,num_workers=0,drop_last=True)
    
    resize_expw_dataset = Resize_ExpW_dataset(expw_file_path,expw_img_path,64)
    dataloader = DataLoader(resize_expw_dataset,batch_size=2,shuffle=True,num_workers=0,drop_last=True)
    
    for i, data in enumerate(dataloader):
        print(data[1])
    dataset_iter = iter(dataloader)
    data = next(dataset_iter)
    print(1)
if __name__ == '__main__':
    main()
