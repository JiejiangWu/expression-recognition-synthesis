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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 



def train_recognition_net(dataset,val_dataset,checkpoint_dir,cfg):
    use_dropout = False
    if 'dropout' in cfg.keys() and cfg['dropout'] == True:
        use_dropout =True        
    
#    print('start')
    model = models.resnet_model(use_dropout=use_dropout).to(device)
    optimizer = optim.Adam(model.parameters(),lr=cfg['lr'])
    criterion = nn.CrossEntropyLoss(reduction='elementwise_mean').to(device)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    '''load checkpoint'''
    temp_checkpoint_file = os.path.join(checkpoint_dir,'Temp.pth.tar')
    if os.path.exists(temp_checkpoint_file):
        checkpoint = torch.load(temp_checkpoint_file)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']+1
        trainlog = checkpoint['trainlog']
        vallog = checkpoint['vallog']
        train_loss = trainlog['loss']
        train_acc = trainlog['acc']
        val_loss = vallog['loss']
        val_acc = vallog['acc']
        best_val_acc = vallog['best_acc']
        epoches_since_last_improve = vallog['epoches_since_last_improve']
        
    else:
        epoch = 0
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []
        best_val_acc = [0]
        epoches_since_last_improve = [0]
        trainlog = {'loss':train_loss,
                    'acc':train_acc
                    } 
        vallog = {'loss':val_loss,
                  'acc':val_acc,
                  'best_acc': best_val_acc,
                  'epoches_since_last_improve': epoches_since_last_improve
                }
        


    
#    if len(train_loss) == epoch:
#        train_loss.append(tmp_train_loss)
#        val_loss.append(tmp_val_loss)
#        train_acc.append(tmp_train_acc)
#        val_acc.append(tmp_val_acc)
#    else:
#        train_loss[epoch] = tmp_train_loss
#        val_loss[epoch] = tmp_val_loss
#        train_acc[epoch] = tmp_train_acc
#        val_acc[epoch] = tmp_val_acc
    
    
    
    for e in range(epoch,cfg['epoches']):
        
        dataloader_train = DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=True,num_workers=0,drop_last = True)
        dataloader_val = DataLoader(val_dataset,batch_size=cfg['batch_size'],shuffle=True,num_workers=0,drop_last = True)
        
        tmp_train_loss = utils.AverageMeter()
        tmp_val_loss = utils.AverageMeter()
        tmp_train_acc = utils.AverageMeter()
        tmp_val_acc = utils.AverageMeter()
        '''train'''
        model.train()
#        correct = torch.zeros(1).to(device)
#        total = torch.zeros(1).to(device)
#        print(len(dataloader_train))
        for i, data in enumerate(dataloader_train):
        
            batch_data, batch_label = data
            batch_data = batch_data.to(device)
            batch_label= batch_label.to(device)
            batch_label = batch_label.long().squeeze()
            
            optimizer.zero_grad()
            
            batch_logit = model(batch_data)
            loss = criterion(batch_logit,batch_label)
            loss.backward()
                            
            optimizer.step()    

            prediction = torch.argmax(batch_logit,1)    
            tmp_acc = (prediction == batch_label).sum().float() / len(batch_label)

            tmp_train_loss.update(loss.item())
            tmp_train_acc.update(tmp_acc.item())
            
            if i % cfg['print_freq'] == 0:
                print('[epoch{0}][{1}/{2}]\t'
                  'train_loss:{loss.val:.3f} ({loss.avg:.3f})'
                  'train_acc:{acc.val:.3f} ({acc.avg:.3f})'.format(e,i,len(dataloader_train),loss=tmp_train_loss,acc=tmp_train_acc))
        
        '''val'''
        model.eval()
        for i, data in enumerate(dataloader_val):
        
            batch_data, batch_label = data
            batch_data = batch_data.to(device)
            batch_label= batch_label.to(device)
            batch_label = batch_label.long().squeeze()
            
            batch_logit = model(batch_data)
            loss = criterion(batch_logit,batch_label)

            prediction = torch.argmax(batch_logit,1)    
            tmp_acc = (prediction == batch_label).sum().float() / len(batch_label)

            tmp_val_loss.update(loss.item())
            tmp_val_acc.update(tmp_acc.item())
            
            if i % cfg['print_freq'] == 0:
                print('[epoch{0}][{1}/{2}]\t'
                  'val_loss:{loss.val:.3f} ({loss.avg:.3f})'
                  'val_acc:{acc.val:.3f} ({acc.avg:.3f})'.format(e,i,len(dataloader_val),loss=tmp_val_loss,acc=tmp_val_acc))        
        

        
        '''if for 'decay_epoch' epoches, acc not improve, decay the learning rate'''
        '''if for 'early_stop_epoch' epoches, acc not improve, early stop the training'''
        epoch_val_acc = tmp_val_acc.avg
        if epoch_val_acc < best_val_acc[0]:
            epoches_since_last_improve[0] += 1
        else:
            epoches_since_last_improve[0] = 0
            best_val_acc[0] = epoch_val_acc
            
        decay = False
        early_stop = False
        if epoches_since_last_improve[0] >= cfg['decay_epoch']:
            decay = True
        if epoches_since_last_improve[0] >= cfg['early_stop_epoch']:
            early_stop = True
            
        if decay:
            train_utils.adjust_learning_rate(optimizer,cfg['decay_rate'])
            
        '''save checkpoint'''
        tmp_model_path = os.path.join(checkpoint_dir,'Temp.pth.tar')
        epoch_model_path = os.path.join(checkpoint_dir,'epoch_%03d.pth.tar'%e)

        train_loss.append(tmp_train_loss)
        val_loss.append(tmp_val_loss)
        train_acc.append(tmp_train_acc)
        val_acc.append(tmp_val_acc)

        state = {'model':model.state_dict(),
                 'optimizer':optimizer.state_dict(),
                 'epoch': e,
                 'trainlog':trainlog,
                 'vallog':vallog
                }
        torch.save(state,tmp_model_path)
        torch.save(state,epoch_model_path)
        
        if early_stop:
            break
            
            
            
            

def train_synthesis_net(dataloader,checkpoint_dir,cfg):
    if cfg['model_type'] == 'origin':
        model_G = models.origin_Generator(c_dim=cfg['c_dim'],repeat_num=cfg['g_bottleneck_num']).to(device)
        model_D = models.origin_Discriminator(image_size=cfg['image_size'],c_dim=cfg['c_dim'],repeat_num=cfg['d_bottleneck_num']).to(device)
    else:
        model_G = models.Generator(c_dim=cfg['c_dim'],repeat_num=cfg['g_bottleneck_num']).to(device)
        model_D = models.Discriminator(image_size=cfg['image_size'],c_dim=cfg['c_dim'],repeat_num=cfg['d_bottleneck_num']).to(device)
    optimizer_G = optim.Adam(model_G.parameters(),cfg['g_lr'],[0.5,0.999])
    optimizer_D = optim.Adam(model_D.parameters(),cfg['d_lr'],[0.5,0.999])
    
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    demo_dir = os.path.join(checkpoint_dir,'demo')
    if not os.path.exists(demo_dir):
        os.makedirs(demo_dir)
    '''load checkpoint'''
    temp_checkpoint_file = os.path.join(checkpoint_dir,'Temp.pth.tar')
    if os.path.exists(temp_checkpoint_file):
        checkpoint = torch.load(temp_checkpoint_file)
        model_G.load_state_dict(checkpoint['model_G'])
        model_D.load_state_dict(checkpoint['model_D'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        
        epoch = checkpoint['epoch']+1
        trainlog = checkpoint['trainlog']
        D_loss = trainlog['D_loss']
        G_loss = trainlog['G_loss']

        D_real_loss = D_loss['real']
        D_fake_loss = D_loss['fake']
        D_classify_loss = D_loss['classify']
        D_gp_loss = D_loss['gp']
        
        G_rec_loss = G_loss['rec']
        G_fake_loss = G_loss['fake']
        G_classify_loss = G_loss['classify']
        
    else:
        epoch = 0
        D_real_loss = []
        D_fake_loss = []
        D_classify_loss = []
        D_gp_loss = []
        
        G_rec_loss = []
        G_fake_loss = []
        G_classify_loss = []       
        
        D_loss = {
                'real':D_real_loss,
                'fake':D_fake_loss,
                'classify':D_classify_loss,
                'gp':D_gp_loss
                }
        G_loss = {
                'fake':G_fake_loss,
                'rec':G_rec_loss,
                'classify':G_classify_loss
                }
        
        trainlog = {'D_loss':D_loss,
                    'G_loss':G_loss
                    } 

    dataloader_train = dataloader
    for e in range(epoch,cfg['epoches']):
#        dataloader_train = dataloder
        tmp_D_real_loss = utils.AverageMeter()
        tmp_D_fake_loss = utils.AverageMeter()
        tmp_D_classify_loss = utils.AverageMeter()
        tmp_D_gp_loss = utils.AverageMeter()
        tmp_G_fake_loss = utils.AverageMeter()
        tmp_G_rec_loss = utils.AverageMeter()
        tmp_G_classify_loss = utils.AverageMeter()
        
        '''train'''
        model_G.train()
        model_D.train()
        for i, (real_x, real_label) in enumerate(dataloader_train):
            # Generat fake labels randomly (target domain labels)
            rand_idx = torch.randperm(real_label.size(0))
            fake_label = real_label[rand_idx]

            real_c = train_utils.one_hot(real_label,7)
            fake_c = train_utils.one_hot(fake_label,7)

            # Convert tensor to variable
            real_x = real_x.to(device)
            real_c = real_c.to(device)           # input for the generator
            fake_c = fake_c.to(device)
            real_label = real_label.to(device)   # this is same as real_c if dataset == 'CelebA'
            fake_label = fake_label.to(device)
            
            # ================== Train D ================== #

            # Compute loss with real images
            out_src, out_cls, out_feats_real = model_D(real_x)
            d_loss_real = - torch.mean(out_src)

            d_loss_cls = F.cross_entropy(out_cls, real_label)


            # Compute loss with fake images
            fake_x = model_G(real_x, fake_c)
            fake_x = Variable(fake_x.data)
            out_src, out_cls, out_feats_fake = model_D(fake_x)
            d_loss_fake = torch.mean(out_src)

            # Backward + Optimize
            d_loss = d_loss_real + d_loss_fake + cfg['lambda_classify'] * d_loss_cls
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            d_loss.backward(retain_graph=True)
            optimizer_D.step()

            # Compute gradient penalty
            alpha = torch.rand(real_x.size(0), 1, 1, 1).to(device).expand_as(real_x)
            interpolated = Variable(alpha * real_x.data + (1 - alpha) * fake_x.data, requires_grad=True)
            out, out_cls, out_feats = model_D(interpolated)

            grad = torch.autograd.grad(outputs=out,
                                       inputs=interpolated,
                                       grad_outputs=torch.ones(out.size()).to(device),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)

            d_loss_gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()

            # Backward + Optimize
            d_loss = cfg['lambda_gp'] * d_loss_gp
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            d_loss.backward(retain_graph=True)
            optimizer_D.step()
            
            tmp_D_real_loss.update(d_loss_real.item())
            tmp_D_fake_loss.update(d_loss_fake.item())
            tmp_D_classify_loss.update(d_loss_cls.item())
            tmp_D_gp_loss.update(d_loss_gp.item())

            # ================== Train G ================== #
            if (i+1) % cfg['D_G_step'] == 0:

                # Original-to-target and target-to-original domain
                fake_x = model_G(real_x, fake_c)
                rec_x = model_G(fake_x, real_c)

                # Compute losses
                out_src, out_cls, out_feats_fake = model_D(fake_x)
                g_loss_fake = - torch.mean(out_src)

                g_loss_cls = F.cross_entropy(out_cls, fake_label)

                ### Discriminate for rec_x
                out_src, out_cls, out_feats_rec = model_D(rec_x)
                g_loss_rec = torch.mean(torch.abs(real_x - rec_x))

               

                # Backward + Optimize
                g_loss = g_loss_fake + cfg['lambda_classify'] * g_loss_cls + cfg['lambda_rec'] * g_loss_rec

                optimizer_D.zero_grad()
                optimizer_G.zero_grad()
                g_loss.backward(retain_graph=True)
                optimizer_G.step()

   
                tmp_G_fake_loss.update(g_loss_fake.item())
                tmp_G_rec_loss.update(g_loss_rec.item())
                tmp_G_classify_loss.update(g_loss_cls.item())

#                      
            if (i+1) % cfg['print_freq'] == 0:
                print('[epoch{0}][{1}/{2}]\t'
                  'd_real:{d_real.val:.3f} ({d_real.avg:.3f})'
                  'd_fake:{d_fake.val:.3f} ({d_fake.avg:.3f})'
                  'd_classify:{d_classify.val:.3f} ({d_classify.avg:.3f})'
                  'd_gp:{d_gp.val:.3f} ({d_gp.avg:.3f})'
                  
                  'g_fake:{g_fake.val:.3f} ({g_fake.avg:.3f})'
                  'g_rec:{g_rec.val:.3f} ({g_rec.avg:.3f})'
                  'g_classify:{g_classify.val:.3f} ({g_classify.avg:.3f})'.format(e,i,len(dataloader_train),
                              d_real = tmp_D_real_loss,
                              d_fake = tmp_D_fake_loss,
                              d_classify = tmp_D_classify_loss,
                              d_gp = tmp_D_gp_loss,
                              g_fake = tmp_G_fake_loss,
                              g_rec = tmp_G_rec_loss,
                              g_classify = tmp_G_classify_loss))   
               
            if (i+1) % cfg['demo_freq'] == 0:
                 imageio.imsave(os.path.join(demo_dir, '%03d_%03d_real.png' % (e,i)),utils.img_cvt(real_x[0]))
                 imageio.imsave(os.path.join(demo_dir, '%03d_%03d_fake.png' % (e,i)),utils.img_cvt(fake_x[0]))
                 imageio.imsave(os.path.join(demo_dir, '%03d_%03d_rec.png' % (e,i)),utils.img_cvt(rec_x[0]))

        if e >= cfg['decay_begin_epoch'] and e % cfg['decay_step_epoch'] == 0:
            train_utils.adjust_learning_rate(optimizer_D,cfg['decay_rate'])
            train_utils.adjust_learning_rate(optimizer_G,cfg['decay_rate'])

            
        '''save checkpoint'''
        if e % cfg['checkpoint_freq'] == 0:
            tmp_model_path = os.path.join(checkpoint_dir,'Temp.pth.tar')
            epoch_model_path = os.path.join(checkpoint_dir,'epoch_%03d.pth.tar'%e)
    
            
            D_real_loss.append(tmp_D_real_loss)
            D_fake_loss.append(tmp_D_fake_loss)
            D_classify_loss.append(tmp_D_classify_loss)
            D_gp_loss.append(tmp_D_gp_loss)
            
            G_fake_loss.append(tmp_G_fake_loss)
            G_rec_loss.append(tmp_G_rec_loss)
            G_classify_loss.append(tmp_G_classify_loss)
            
            state = {'model_D':model_D.state_dict(),
                     'model_G':model_G.state_dict(),
                     'optimizer_D':optimizer_D.state_dict(),
                     'optimizer_G':optimizer_G.state_dict(),
                     'epoch': e,
                     'trainlog':trainlog
                    }
            torch.save(state,tmp_model_path)
            torch.save(state,epoch_model_path)        
