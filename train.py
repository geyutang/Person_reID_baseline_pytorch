# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
# from model import ft_net, ft_net_dense, PCB, resnet_arc, dense_arc
from model import *
from random_erasing import RandomErasing
import yaml
from shutil import copyfile
from metrics import *
import OneCycle as OneCycle
from center_loss import CenterLoss
def main(ids, name, balanced_sample=False, backbone='resnet', loss='softmax',
         dataset='market',embedding=512, scale=30, margin=0.01, 
         weight_cent=1, lr=0.05, weight_lr=0.1, epochs=60, optimizer='SGD',
         weight_decay= 5e-4, scheduler_type='step'):
    version =  torch.__version__
    # args :
    # backbone=('resnet', 'resnetmid', 'dense')
    # loss = ('softmax', 'arcface', 'cosface', 'sphere', 'center')
    # dataset = ('market', 'duke', 'cuhk03')
    # optimizer = ('sgd', 'adam')
    #fp16
    # try:
    #     from apex.fp16_utils import *
    # except ImportError: # will be 3.x series
    #     print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
    ######################################################################
    # Options
    # --------
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids',default='1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
    parser.add_argument('--data_dir',default='/home/tanggeyu/Dataset/Market-1501/pytorch',type=str, help='training dir path')
    parser.add_argument('--train_all', action='store_true', help='use all training data' )
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
    parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
    parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
    parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
    ######################################################################
    # set default paramter
    
    opt = parser.parse_args()
    opt.train_all = True
    opt.gpu_ids = ids
    opt.lr = lr
    num_epochs = epochs
    opt.name = name
    fp16 = opt.fp16
    data_dir = opt.data_dir
    name = opt.name
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    # opt.use_dense = use_dense
    if dataset == 'market':
        data_dir = '/home/tanggeyu/Dataset/Market-1501/pytorch'
    if dataset == 'cuhk':
        data_dir = '/home/tanggeyu/Dataset/cuhk03-np/pytorch'
    if dataset == 'duke':
        data_dir = '/home/tanggeyu/Dataset/DukeMTMC-reID/pytorch'

    opt.data_dir = data_dir
    for str_id in str_ids:
        gid = int(str_id)
        if gid >=0:
            gpu_ids.append(gid)
    
    # set gpu ids
    if len(gpu_ids)>0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True
    ######################################################################
    # Load Data
    # ---------
    
    transform_train_list = [
            #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
            transforms.Resize((256,128), interpolation=3),
            transforms.Pad(10),
            transforms.RandomCrop((256,128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
    
    transform_val_list = [
            transforms.Resize(size=(256,128),interpolation=3), #Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
    
    if opt.PCB:
        transform_train_list = [
            transforms.Resize((384,192), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        transform_val_list = [
            transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
    
    if opt.erasing_p>0:
        transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]
    
    if opt.color_jitter:
        transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list
    
    print(transform_train_list)
    data_transforms = {
        'train': transforms.Compose( transform_train_list ),
        'val': transforms.Compose(transform_val_list),
    }
    
    
    train_all = ''
    if opt.train_all:
         train_all = '_all'
    #########################################################################
    # balanced sampler defination
    def make_weights_for_balanced_classes(images, nclasses):                        
        count = [0.] * nclasses                                                      
        for item in images:                                                         
            count[item[1]] += 1                                                     
        weight_per_class = [0.] * nclasses                                      
        N = float(sum(count))                                                   
        for i in range(nclasses):                                                   
            weight_per_class[i] = N/float(count[i])                                 
            # weight_per_class[i] = 1./float(count[i])
        weight = [0.] * len(images)                                              
        for idx, val in enumerate(images):                                          
            weight[idx] = weight_per_class[val[1]]                                  
        return weight

    image_datasets = {}

    image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                              data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                              data_transforms['val'])
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    if balanced_sample:
        weights = make_weights_for_balanced_classes(image_datasets['train'].imgs, len(image_datasets['train'].classes))
       
        # weights = torch.DoubleTensor(weights)
        weights = torch.DoubleTensor(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
        # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
        #                                              shuffle=True, sampler=sampler, num_workers=8, pin_memory=True) # 8 workers may work faster
        #               for x in ['train', 'val']}
        dataloaders = {}
        dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=opt.batchsize,
                                                      sampler=sampler,num_workers=8, pin_memory=True)
        dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=opt.batchsize,
                                                     shuffle=True,num_workers=8, pin_memory=True)
    else:
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                                     shuffle=True, num_workers=8, pin_memory=True) # 8 workers may work faster
                      for x in ['train', 'val']}
    
    class_names = image_datasets['train'].classes
    
    use_gpu = torch.cuda.is_available()
    
    since = time.time()
    inputs, classes = next(iter(dataloaders['train']))
    print(time.time()-since)

    ###########################################################
    # One cycle policy function
    def update_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def update_mom(optimizer, mom):
        for g in optimizer.param_groups:
            g['momentum'] = mom
    ############################################################

    ######################################################################
    # Training the model
    # ------------------
    #
    # Now, let's write a general function to train a model. Here, we will
    # illustrate:
    #
    # -  Scheduling the learning rate
    # -  Saving the best model
    #
    # In the following, parameter ``scheduler`` is an LR scheduler object from
    # ``torch.optim.lr_scheduler``.
    
    y_loss = {} # loss history
    y_loss['train'] = []
    y_loss['val'] = []
    y_err = {}
    y_err['train'] = []
    y_err['val'] = []
    
    def train_model(model, metric_fc, criterion, optimizer, optimizer_centloss, scheduler, num_epochs=25):
        since = time.time()
    
        #best_model_wts = model.state_dict()
        #best_acc = 0.0
        ################################################3
        # average the loss
        running_loss_cycle = 0.0
        avg_beta = 0.98
        use_cycle = False
        #################################################
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
    
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    #####################################
                    # sheduler
                    #####################################
                    if scheduler_type!='one_cycle':
                        scheduler.step()
                    model.train(True)  # Set model to training mode
                else:
                    model.train(False)  # Set model to evaluate mode
    
                running_loss = 0.0
                running_corrects = 0.0
                # Iterate over data.
                for i, (inputs, labels) in enumerate(dataloaders[phase]):
                    # get the inputs
                    # inputs, labels = data
                    now_batch_size,c,h,w = inputs.shape
                    if now_batch_size<opt.batchsize: # skip the last batch
                        continue
                    #print(inputs.shape)
                    # wrap them in Variable
                    if use_gpu:
                        inputs = Variable(inputs.cuda().detach())
                        labels = Variable(labels.cuda().detach())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)
                    # if we use low precision, input also need to be fp16
                    if fp16:
                        inputs = inputs.half()
                    ###############################################
                    # one cycle policy
                    if scheduler_type=='one_cycle':    
                        lr, mom = onecycle.calc()
                        update_lr(optimizer, lr)
                        update_mom(optimizer, mom)
                    ##############################################
                    # zero the parameter gradients
                    if loss != 'center':
                        optimizer.zero_grad()
                    else:
                        optimizer.zero_grad()
                        optimizer_centloss.zero_grad()
    
                    # forward
                    if phase == 'val':
                        with torch.no_grad():
                            outputs, features = model(inputs)

                    else:
                        outputs, features = model(inputs)
   
                    if not opt.PCB:
                        _, preds = torch.max(outputs.data, 1)
                        if loss == 'softmax':
                            train_loss = criterion(outputs, labels)
                        if loss=='arcface' or loss=='cosface' or loss=='sphere':
                            norm_output = metric_fc(features, labels)
                            train_loss = criterion(norm_output, labels)
                        if loss=='center':
                            loss_xent = criterion(outputs, labels)
                            loss_cent = criterion_cent(features, labels)
                            loss_cent *= weight_cent
                            train_loss = loss_xent + loss_cent
                        
                    else:
                        part = {}
                        sm = nn.Softmax(dim=1)
                        num_part = 6
                        for i in range(num_part):
                            part[i] = outputs[i]
    
                        score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
                        _, preds = torch.max(score.data, 1)
    
                        train_loss = criterion(part[0], labels)
                        for i in range(num_part-1):
                            train_loss += criterion(part[i+1], labels)
    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        if fp16: # we use optimier to backward loss
                            optimizer.backward(train_loss)
                        else:
                            train_loss.backward()
                        
                        if loss != 'center':
                            optimizer.step()
                        else:
                            optimizer.step()
                            # by doing so, weight_cent would not impact on the learning of centers
                            for param in criterion_cent.parameters():
                                param.grad.data *= (1. / weight_cent)
                            optimizer_centloss.step()
    
                    # statistics
                    if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                        running_loss += train_loss.item() * now_batch_size
                        ##################################################
                        # one cycle policy 
                        running_loss_cycle = avg_beta * running_loss + (1-avg_beta) *train_loss.item()
                        smoothed_loss_cycle = running_loss / (1 - avg_beta**(i+1))
                        ##########################################################3
                    else :  # for the old version like 0.3.0 and 0.3.1
                        running_loss += train_loss.data[0] * now_batch_size
                    running_corrects += float(torch.sum(preds == labels.data))
    
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]
                
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
                
                y_loss[phase].append(epoch_loss)
                y_err[phase].append(1.0-epoch_acc)            
                # deep copy the model
                if phase == 'val':
                    last_model_wts = model.state_dict()
                    if epoch%5 == 4:
                        save_network(model, epoch)
                    draw_curve(epoch)
    
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        #print('Best val Acc: {:4f}'.format(best_acc))
    
        # load best model weights
        model.load_state_dict(last_model_wts)
        save_network(model, 'last')
        return model
    
    
    ######################################################################
    # Draw Curve
    #---------------------------
    x_epoch = []
    fig = plt.figure()
    ax0 = fig.add_subplot(121, title="loss")
    ax1 = fig.add_subplot(122, title="top1err")
    def draw_curve(current_epoch):
        x_epoch.append(current_epoch)
        ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
        ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
        ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
        ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
        if current_epoch == 0:
            ax0.legend()
            ax1.legend()
        fig.savefig( os.path.join('./model',name,'train.jpg'))
    
    ######################################################################
    # Save model
    #---------------------------
    def save_network(network, epoch_label):
        save_filename = 'net_%s.pth'% epoch_label
        save_path = os.path.join('./model',name,save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            network.cuda(gpu_ids[0])
    
    
    ######################################################################
    # Finetuning the convnet
    # ----------------------
    #
    # Load a pretrainied model and reset final fully connected layer.
    #
    # base model
        
    if backbone=='resnet':
        model = ft_net(len(class_names), opt.droprate)
        # model = ft_net_dense(len(class_names), opt.droprate)
        ############################################################
    if backbone=='dense':
        # model = dense_arc(embedding, opt.droprate)
        model = ft_net_dense(len(class_names), opt.droprate)
    if backbone=='resnetmid':
        model = ft_net_middle(len(class_names), opt.droprate)

    # #####################################################################
    # # other metric learning loss
    # if backbone=='resnet' and (loss=='arcface' or loss=='cosface' or loss=='sphere'):
    #     model = resnet_metric(embedding, opt.droprate)
    # if backbone=='dense' and (loss=='arcface' or loss=='cosface' or loss=='sphere'):
    #     model = dense_metric(embedding, opt.droprate)
    # if backbone=='resnetmid' and (loss=='arcface' or loss=='cosface' or loss=='sphere'):
    #     model = resnetmiddle_metric(embedding, opt.droprate)

    # else:
    #    #  model = ft_net(len(class_names), opt.droprate)
    #     model = resnet_arc(embedding, opt.droprate)
        

    
    if opt.PCB:
        model = PCB(len(class_names))
 #################################################################################
 # define the metric_fc
 # loss = ('softmax', 'arcface', 'cosface', 'sphere', 'center')
    metric_fc = None
    if loss == 'arcface':
        metric_fc = ArcMarginProduct(embedding, len(class_names), s=scale, m=margin)
    if loss == 'cosface':
        metric_fc = AddMarginProduct(embedding, len(class_names), s=scale, m=margin)
    if loss == 'sphere':
        metric_fc = SphereProduct(embedding, len(class_names), m=margin)
    
    
    # print(model)
    
    # if not opt.PCB:
    #     ignored_params = list(map(id, model.model.fc.parameters() ))+list(map(id, model.classifier.parameters() ))
        
    #     base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    #     optimizer_ft = optim.SGD([
    #              {'params': base_params, 'lr': 0.1*opt.lr},
    #              {'params': model.model.fc.parameters(), 'lr': opt.lr},
    #              {'params': model.classifier.parameters(), 'lr': opt.lr}
    #              # {'params': model.addblock.parameters(), 'lr': opt.lr},
    #              # {'params': metric_fc.parameters(), 'lr': opt.lr}
    #          ], weight_decay=5e-4, momentum=0.9, nesterov=True)
######################################################################
# define a training criterion
    if loss!='center':
        criterion=nn.CrossEntropyLoss()
    if loss == 'center':
        criterion = nn.CrossEntropyLoss()
        criterion_cent = CenterLoss(num_classes=len(class_names), feat_dim=embedding, use_gpu=use_gpu)

    #################################################################
    # defination of the parameter update
    optimizer_centloss = None
    if loss=='softmax':
        ignored_params = list(map(id, model.model.fc.parameters() ))+list(map(id, model.classifier.parameters() ))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        if optimizer=='SGD':
            optimizer_ft = optim.SGD([
                     {'params': base_params, 'lr': weight_lr*opt.lr},
                     {'params': model.model.fc.parameters(), 'lr': opt.lr},
                     {'params': model.classifier.parameters(), 'lr': opt.lr}
                     # {'params': model.addblock.parameters(), 'lr': opt.lr},
                     # {'params': metric_fc.parameters(), 'lr': opt.lr}
                 ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        if optimizer=='ADAM':
            optimizer_ft = optim.Adam([
                     {'params': base_params, 'lr': weight_lr*opt.lr},
                     {'params': model.model.fc.parameters(), 'lr': opt.lr},
                     {'params': model.classifier.parameters(), 'lr': opt.lr}
                     # {'params': model.addblock.parameters(), 'lr': opt.lr},
                     # {'params': metric_fc.parameters(), 'lr': opt.lr}
                 ],betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    
    if loss=='arcface' or loss=='cosface'or loss=='sphere':
        ignored_params = list(map(id, model.model.fc.parameters() ))+list(map(id, model.classifier.parameters() ))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        if optimizer=='SGD':
            optimizer_ft = optim.SGD([
                     {'params': base_params, 'lr': weight_lr*opt.lr},
                     {'params': model.model.fc.parameters(), 'lr': opt.lr},
                     # {'params': model.classifier.parameters(), 'lr': opt.lr}
                     # {'params': model.addblock.parameters(), 'lr': opt.lr},
                     {'params': metric_fc.parameters(), 'lr': opt.lr}
                 ], weight_decay=5e-4, momentum=0.9, nesterov=True)
        if optimizer=='ADAM':
            optimizer_ft = optim.Adam([
                     {'params': base_params, 'lr': weight_lr*opt.lr},
                     {'params': model.model.fc.parameters(), 'lr': opt.lr},
                     # {'params': model.classifier.parameters(), 'lr': opt.lr}
                     # {'params': model.addblock.parameters(), 'lr': opt.lr},
                     {'params': metric_fc.parameters(), 'lr': opt.lr}
                 ],betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    if loss == 'center':
        if optimizer=='SGD':
            optimizer_ft = torch.optim.SGD(model.parameters(), lr=weight_lr*lr, weight_decay=5e-04, momentum=0.9, nesterov=True)
            optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=lr)
        if optimizer=='ADAM':
            optimizer_ft = torch.optim.Adam(model.parameters(), lr=weight_lr*lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
            optimizer_centloss = torch.optim.Adam(criterion_cent.parameters(), lr=lr)
        

    if opt.PCB :
        ignored_params = list(map(id, model.model.fc.parameters() ))
        ignored_params += (list(map(id, model.classifier0.parameters() )) 
                         +list(map(id, model.classifier1.parameters() ))
                         +list(map(id, model.classifier2.parameters() ))
                         +list(map(id, model.classifier3.parameters() ))
                         +list(map(id, model.classifier4.parameters() ))
                         +list(map(id, model.classifier5.parameters() ))
                         #+list(map(id, model.classifier6.parameters() ))
                         #+list(map(id, model.classifier7.parameters() ))
                          )
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
                 {'params': base_params, 'lr': 0.1*opt.lr},
                 {'params': model.model.fc.parameters(), 'lr': opt.lr},
                 {'params': model.classifier0.parameters(), 'lr': opt.lr},
                 {'params': model.classifier1.parameters(), 'lr': opt.lr},
                 {'params': model.classifier2.parameters(), 'lr': opt.lr},
                 {'params': model.classifier3.parameters(), 'lr': opt.lr},
                 {'params': model.classifier4.parameters(), 'lr': opt.lr},
                 {'params': model.classifier5.parameters(), 'lr': opt.lr},
                 #{'params': model.classifier6.parameters(), 'lr': 0.01},
                 #{'params': model.classifier7.parameters(), 'lr': 0.01}
             ], weight_decay=5e-3, momentum=0.9, nesterov=True)
    
    # Decay LR by a factor of 0.1 every 40 epochs
    if scheduler_type=='step':
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
    if scheduler_type=='one_cycle':
        ####################################################################3
        # change scheduler
        onecycle = OneCycle.OneCycle((dataset_sizes['train'] * num_epochs /opt.batchsize),
            max_lr=5e-3, prcnt=(num_epochs - 42) * 100/num_epochs,
             momentum_vals=(0.95, 0.8))
        exp_lr_scheduler = onecycle
    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # It should take around 1-2 hours on GPU. 
    #
    dir_name = os.path.join('./model',name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    #record every run
    copyfile('./train.py', dir_name+'/train.py')
    copyfile('./model.py', dir_name+'/model.py')
    
    # save opts
    with open('%s/opts.yaml'%dir_name,'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)
    
    # model to gpu
    model = model.cuda()
    if loss == 'sphere' or loss=='cosface' or loss=='arcface':
        metric_fc = metric_fc.cuda()
    if fp16:
        model = network_to_half(model)
        optimizer_ft = FP16_Optimizer(optimizer_ft, static_loss_scale = 128.0)
    
    
    
    model = train_model(model, metric_fc, criterion, optimizer_ft, 
            optimizer_centloss, exp_lr_scheduler,num_epochs)

if __name__ == '__main__':
    main()
