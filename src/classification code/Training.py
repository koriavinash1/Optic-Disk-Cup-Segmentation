import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import models, transforms, datasets
from torch.optim import lr_scheduler


from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm2

import torchnet as tnt
import time
import copy
from Networks import *
import DataGenerator
import pandas as pd


nnClassCount = 2
num_epochs   = 1
nnIsTrained  = True

nnArchitectureList = [{'name': 'densenet201', 'model' : DenseNet201()},
{'name': 'densenet169', 'model' : DenseNet169()},
{'name': 'densenet161', 'model' : DenseNet161()},
{'name': 'densenet121', 'model': DenseNet121()},
{'name': 'resnet152', 'model': ResNet152()},
{'name': 'resnet101', 'model': ResNet101()},
{'name': 'resnet50', 'model': ResNet50()},
{'name': 'resnet34', 'model': ResNet34()},
{'name': 'resnet18', 'model': ResNet18()}]


archs = []
acc_class1 = []
acc_class0 = []
lossss = []

def train_model(model, name, criterion, optimizer, scheduler, num_epochs=15):
    
    confusion_matrix = tnt.meter.ConfusionMeter(2, normalized= True)
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000.0
    best_acc  = 0.0
    for epoch in range(num_epochs):
        

        print(name + '  Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            confusion_matrix.reset()

            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                                                                                                                                                                                                                                                                                                                                        
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    confusion_matrix.add(preds.view(-1), labels.data.view(-1))

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            scheduler.step(epoch_loss)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_loss_model_wts = copy.deepcopy(model.state_dict())

                ## MODIFIED BY AVINASH
                model_name   = './Model/best_model_'+ str(i) + '_loss_.pth.tar'
                saved_params = {'model_state_dict': model.state_dict(),
                                'optimizer_state_dict':optimizer.state_dict(),
                                'epoch_id': epoch}
                torch.save(saved_params, model_name)
                print (".................best loss model saved..............")

                 # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_acc_model_wts = copy.deepcopy(model.state_dict())

                model_name   = './Model/best_model_'+ str(i) + '_acc_.pth.tar'
                saved_params = {'model_state_dict': model.state_dict(),
                                'optimizer_state_dict':optimizer.state_dict(),
                                'epoch_id': epoch}

                torch.save(saved_params, model_name)
                print (".................best acc model saved..............")

            cmc = confusion_matrix.conf
            print(cmc)
            print()
            if phase == 'val':
                archs.append(name +'_' + str(epoch))
                acc_class0.append(cmc[0, 0]/ (cmc[0].sum()))
                acc_class1.append(cmc[1, 1]/ (cmc[1].sum()))
                lossss.append(best_loss)
   
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:.4f}'.format(best_loss))

    # load best model weights
    model_acc = model
    model_loss= model
    model_acc.load_state_dict(best_acc_model_wts)
    model_loss.load_state_dict(best_loss_model_wts)
    return model_acc, model_loss




####### Loop over all the architectures
for j in range(len(nnArchitectureList)):

    ###### Ensembles
    for i in range(1, 6):

        with open('../LoadData/shuffle_' + str(i) + '/normalize_param.txt') as f:
            params = f.read().splitlines()

        param1 = np.array(params[:3]*7, dtype = 'float32')
        param2 = np.array(params[3:]*7, dtype = 'float32')


        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomCrop(500),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation((0,360)),
                ]),
            'val': transforms.Compose([ 
            ## Common transforms are done in GataGenerator 
            ]),        
            'test': transforms.Compose([
            ## Common transforms are done in GataGenerator 
            ]),
        }

        data_dir = '../LoadData/shuffle_' + str(i) + '/'
        image_datasets = {x: DataGenerator.DatasetGenerator(os.path.join(data_dir, x),param1, param2, data_transforms[x])\
                                                            for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                     shuffle=True, num_workers=4)
                      for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_ft = nnArchitectureList[j]['model'].to(device)
        
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(np.array([1/100,1/300]))).to(device)
        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model_acc, model_loss = train_model(model_ft, nnArchitectureList[j]['name'], criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs)
        torch.save(model_acc, './Model/Final/' + nnArchitectureList[j]['name'] + '_' + str(i) + 'best_acc.pth.tar')
        torch.save(model_loss, './Model/Final/' + nnArchitectureList[j]['name'] + '_' +str(i) + 'best_loss.pth.tar')


#### Saving Logs
sub = pd.DataFrame()
sub['arch'] = archs
sub['class0_acc'] = acc_class0
sub['class1_acc'] = acc_class1
sub['loss']  = lossss
sub.to_csv('./training_logs_MixDrishti.csv', index=False)