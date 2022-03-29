import matplotlib
matplotlib.use('Agg')
import yaml
import argparse,os
from src.dataset import datasets
from src.training import training
import torch
torch.backends.cudnn.enabled=False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--cfg_file', type=str,default = './configs/expw.yaml')
    args = parser.parse_args()
    
    # load cfg
    f = open(args.cfg_file)
    cfg = yaml.load(f)
    f.close()
    
    work_dir = os.path.join('./data/checkpoint',cfg['eid'])
    
    
    expw_train_file_path = './data/datasets/expw/train.lst'
    expw_val_file_path = './data/datasets/expw/val.lst'
    expw_img_path = './data/datasets/expw/ExpwCropped' 
    
    fer_file_path = './data/datasets/fer2013/fer2013.csv'
    
    if 'augmentation' in cfg.keys():
        aug = cfg['augmentation']
    else:
        aug = False
    
    if cfg['dataset'] == 'expw':
        dataset_train = datasets.ExpW_dataset(expw_train_file_path,expw_img_path,aug)
        dataset_val = datasets.ExpW_dataset(expw_val_file_path,expw_img_path,False)

    if cfg['dataset'] == 'fer2013':  
        dataset_train = datasets.Fer13_dataset(fer_file_path,'train',aug)
        dataset_val = datasets.Fer13_dataset(fer_file_path,'val',False)
    

    print('PID:'+ str(os.getpid()))
    training.train_recognition_net(dataset_train,dataset_val,work_dir,cfg)
       
main()
