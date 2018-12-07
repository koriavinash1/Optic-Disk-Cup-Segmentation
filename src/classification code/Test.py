import os
from PIL import Image
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm2

import torchnet as tnt

import torchvision
from torchvision import models, transforms, datasets
import torch.nn.functional as F
import time
import pandas as pd

from Networks import *
import DataGenerator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(np.array([1/40,1/360])).to(device))
conf = np.zeros((2,2), dtype = 'int32')

save = pd.DataFrame()
softmx_normal_list = []
softmx_gl_list = []
out_gl_list = []
out_normal_list = []
pred_list = []
pth_list = []
conf_list = []
label_list = []
model_list = []

def testing(model):
	running_corrects = 0
	confusion_matrix = tnt.meter.ConfusionMeter(2, normalized= True)

	best_acc = 0.0            
	running_loss = 0.0
	running_corrects = 0

	model.eval()

	confusion_matrix.reset()

	for inputs, labels, pth in tqdm(dataloaders['test']):
		
		inputs = inputs.to(device)
		labels = labels.to(device)

		with torch.set_grad_enabled(False):
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)

			# loss = criterion(outputs, labels)
			confusion_matrix.add(preds.view(-1), labels.data.view(-1))
			# running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels.data)

		
		softmx = F.softmax(outputs, dim =1)
		softmx = softmx.cpu().numpy()
		outputs = outputs.cpu().numpy()
		
		# running_loss += loss.item() * inputs.size(0)
		softmx_gl_list.extend(softmx[:,0])
		softmx_normal_list.extend(softmx[:,1])
		out_gl_list.extend(outputs[:,0])
		out_normal_list.extend(outputs[:,1])
		pred_list.extend(preds.cpu().numpy())
		label_list.extend(labels.cpu().numpy())
		pth_list.extend(pth)
		model_list.append(j.split('/')[-1])	  ##########  Append Model list
	
		conf_list.append(str(confusion_matrix.conf))

	# epoch_loss = running_loss / dataset_sizes['test']
	acc = running_corrects.double() / dataset_sizes['test']
	# print('Loss :   {:.4f}'.format(epoch_loss))
	print('accuracy:   {:.4f}'.format(acc)) 
	print(confusion_matrix.conf)

	# return confusion_matrix.conf


for i in range(1,2):

	with open('../LoadData/OpticDiscRefuge/shuffle_'+ str(i) + '/normalize_param.txt') as f:
	    params = f.read().splitlines()

	param1 = np.array(params[:3]*7, dtype = 'float32')
	param2 = np.array(params[3:]*7, dtype = 'float32')
		

	data_dir = '../LoadData/OpticDiscRefuge/shuffle_' + str(i)

	image_datasets = {x: DataGenerator.DatasetGenerator(os.path.join(data_dir, x),param1, param2)\
	                                                    for x in ['test']}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
	                                             shuffle=False, num_workers=4)
	              for x in ['test']}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
	class_names = image_datasets['test'].classes

	model_dir = './Model/Final'
	
	models = [os.path.join(model_dir, p) for p in next(os.walk(model_dir))[2]]
	models.sort()

	for j in models:

		model = torch.load(j)
		print('______Testing Model_______', j)
		testing(model)

	save = pd.DataFrame()
	save['Model'] = model_list
	save['Img'] = pth_list
	save['Output_gl'] = out_gl_list
	save['Output_normal'] = out_normal_list
	save['Softmax_gl'] = softmx_gl_list
	save['Softmax_normal'] = softmx_normal_list
	save['Prediction'] = pred_list
	save['Ground_Truth'] = label_list
	save['Confusion'] = conf_list
	save.to_csv('./Test_Logs_Refuge_' + str(i) + '.csv', index = True)


