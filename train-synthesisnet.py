import matplotlib
matplotlib.use('Agg')
import yaml
import argparse,os
from src.dataset import datasets
from src.training import training
import torch
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled=False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--cfg_file', type=str,default = './configs/synthesis-fer.yaml')
    args = parser.parse_args()
    
    # load cfg
    f = open(args.cfg_file)
    cfg = yaml.load(f)
    f.close()
    
    work_dir = os.path.join('./data/checkpoint',cfg['eid'])
    

#    fer_file_path = './data/datasets/fer2013/fer2013.csv'

    
#    if cfg['dataset'] == 'expw':
#        dataset_train = datasets.Resize_ExpW_dataset(expw_train_file_path,expw_img_path,cfg['image_size'])
#    if cfg['dataset'] == 'fer2013':  
#        dataset_train = datasets.Resize_Fer13_dataset(fer_file_path,'train',cfg['image_size'])
    
    if cfg['dataset'] == 'fer2013':
        fer2013_image_path='./data/datasets/fer2013/data/fer2013/train'
    
        dataloader = datasets.get_loader(fer2013_image_path, 48, 64,32, 'train')
    if cfg['dataset'] == 'expw':
        expw_train_file_path = './data/datasets/expw/train.lst'
        expw_img_path = './data/datasets/expw/ExpwCropped' 
        dataset_train = datasets.Resize_ExpW_dataset(expw_train_file_path,expw_img_path,cfg['image_size'])
        dataloader = DataLoader(dataset_train,batch_size=cfg['batch_size'],shuffle=True,num_workers=1)     

#    dataset_val = datasets.Fer13_dataset(fer_file_path,'val',False)
    

    print('PID:'+ str(os.getpid()))
    training.train_synthesis_net(dataloader,work_dir,cfg)
       
main()
