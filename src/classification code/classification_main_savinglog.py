import os
from PIL import Image
import os.path
import numpy as np 
import matplotlib.pyplot as plt
import cv2

import torch
from torchvision import models, transforms, datasets
import torch.nn.functional as F

from scipy import misc
import pandas as pd

from Networks import *
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def optic_disk(img, seg, bounding_box = 550):
	
	d = int(bounding_box/2)

	x_, y_ = np.where(seg<255)

	### Taking pts with Max and Min coordinates
	x_min = np.min(x_)
	y_min = np.min(y_)
	x_max = np.max(x_)
	y_max = np.max(y_)

	### Averaging it to get approx center of Optic Disc
	x = int((x_min + x_max)/2)
	y = int((y_min + y_max)/2)

	od = img[max(x-d,0):min(x+d,np.shape(img)[0]), max(y-d,0):min(y+d,min(x+d,np.shape(img)[1])),:]
	im_od = misc.imresize(od, (500, 500, 3))

	return im_od

### Clahe
def clahe_single(ori_img,clipLimit,tileGridSize):

	# ori_img = Image.open(pth)
	# bgr = cv2.imread(pth)
	lab = cv2.cvtColor(ori_img, cv2.COLOR_RGB2LAB)
	
	lab_planes = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit,tileGridSize)
	lab_planes[0] = clahe.apply(lab_planes[0])
	
	lab = cv2.merge(lab_planes)
	rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

	return rgb

def clahe_all(ori_img):

	rgb_1 = clahe_single(ori_img, 2.0 , (8,8))
	rgb_2 = clahe_single(ori_img, 10.0, (8,8))

	rgb_3 = clahe_single(ori_img, 2.0,  (100,100))
	rgb_4 = clahe_single(ori_img, 100.0, (100,100))

	rgb_5 = clahe_single(ori_img, 2.0, (300,300))

	rgb_6 = clahe_single(ori_img, 2.0,  (500,500))

	return np.concatenate( (ori_img, rgb_1,rgb_2,rgb_3,rgb_4,rgb_5,rgb_6), axis = -1 )


########### img is PIL image
def testing(model, inputs):


	model.eval()

	with torch.set_grad_enabled(False):
		bs, ncrops, c, h, w = inputs.size()
		result = model(inputs.view(-1, c, h, w)) # fuse batch size and ncrops
		outputs = result.view(bs, ncrops, -1).mean(1) # avg over crops
		# _, preds = torch.max(outputs, 1) #Getting predictions

		
	
	softmx = F.softmax(outputs, dim =1)
	softmx = softmx.cpu().numpy()
	# outputs = outputs.cpu().numpy()
	
	# softmx_gl_list.extend(softmx[:,0])
	# softmx_normal_list.extend(softmx[:,1])
	# out_gl_list.extend(outputs[:,0])
	# out_normal_list.extend(outputs[:,1])
	# pred_list.extend(preds.cpu().numpy())
	# label_list.extend(labels.cpu().numpy())
	# pth_list.extend(pth)
	# model_list.append(model)    ##########  Append Model list

	return softmx[:,0]  ## Logit for Glaucoma class


def predict_models(img):

	gl_softmx_list = []

	##### All tramnsforms in DataGenerator code
	img = Image.fromarray(img)
	img = transforms.TenCrop(500)(img)

	inputs =[]
	for i in img:
		inputs.append(clahe_all(np.array(i)))

	transformsList1 = []
	transformsList1.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
	transformsList1.append(transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(torch.FloatTensor(param1_1),\
																							torch.FloatTensor(param1_2))(crop) \
																							for crop in crops])))
	transformComp1 = transforms.Compose(transformsList1)
	inputs1 = transformComp1(inputs)
	inputs1 = inputs1.unsqueeze(0) ## To add batch size dimension
	inputs1 = inputs1.to(device)
	# print(inputs.size())

	transformsList2 = []
	transformsList2.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
	transformsList2.append(transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(torch.FloatTensor(param2_1),\
																							torch.FloatTensor(param2_2))(crop) \
																							for crop in crops])))
	transformComp2 = transforms.Compose(transformsList2)
	inputs2 = transformComp2(inputs)
	inputs2 = inputs2.unsqueeze(0) ## To add batch size dimension
	inputs2 = inputs2.to(device)
	# print(inputs.size())

	transformsList3 = []
	transformsList3.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
	transformsList3.append(transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(torch.FloatTensor(param3_1),\
																							torch.FloatTensor(param3_2))(crop) \
																							for crop in crops])))
	transformComp3 = transforms.Compose(transformsList3)
	inputs3 = transformComp3(inputs)
	inputs3 = inputs3.unsqueeze(0) ## To add batch size dimension
	inputs3 = inputs3.to(device)
	# print(inputs.size())


	transformsList4 = []
	transformsList4.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
	transformsList4.append(transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(torch.FloatTensor(param4_1),\
																							torch.FloatTensor(param4_2))(crop) \
																							for crop in crops])))
	transformComp4 = transforms.Compose(transformsList4)
	inputs4 = transformComp4(inputs)
	inputs4 = inputs4.unsqueeze(0) ## To add batch size dimension
	inputs4 = inputs4.to(device)
	
	transformsList5 = []
	transformsList5.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
	transformsList5.append(transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(torch.FloatTensor(param5_1),\
																							torch.FloatTensor(param5_2))(crop) \
																							for crop in crops])))
	transformComp5 = transforms.Compose(transformsList5)
	inputs5 = transformComp5(inputs)
	inputs5 = inputs5.unsqueeze(0) ## To add batch size dimension
	inputs5 = inputs5.to(device)
	# print(inputs.size())


	
	# print('_________Testing Model_________', j)
	gl_softmx_list.extend(testing(model1, inputs1))
	gl_softmx_list.extend(testing(model2, inputs2))
	gl_softmx_list.extend(testing(model3, inputs3))
	gl_softmx_list.extend(testing(model4, inputs4))
	gl_softmx_list.extend(testing(model5, inputs5))
	# print('_______________OK______________') 

	return gl_softmx_list


# def classification_main(ori_img, seg, bounding_box=550):
def classification_main(im_od):

	# im_od = optic_disk(ori_img, seg, bounding_box)

	# print(im_od.dtype)
	# print(im_od.shape)

	gl_softmx_list = []

	gl_softmx_list.extend(predict_models(im_od))
	# gl_softmx_list.extend(predict_models(im_od, './Model/OnlyRefuge'))


	gl_softmx_avg = np.mean(np.array(gl_softmx_list))
	# pred = np.round(gl_softmx_avg)

	return gl_softmx_avg, gl_softmx_list



if __name__ == '__main__':

	# seg_load_dir = '../DATA/REFUGE-Validation400/REFUGE-Validation400_segmentations/'
	# seg_pths = [os.path.join(seg_load_dir, p) for p in next(os.walk(seg_load_dir))[2]]
	# seg_pths.sort()

	img_load_dir = '../DATA/REFUGE-Validation400/OD_val/'
	img_pths = [os.path.join(img_load_dir, p) for p in next(os.walk(img_load_dir))[2]]
	img_pths.sort()

	# model1 = torch.load('./Model/densenet201_3best_loss.pth.tar')
	# model2 = torch.load('./Model/densenet201_3best_loss.pth.tar')
	# model3 = torch.load('./Model/densenet201_3best_loss.pth.tar')
	# model4 = torch.load('./Model/densenet201_4best_loss.pth.tar')
	# model5 = torch.load('./Model/densenet201_4best_loss.pth.tar')

	model1 = torch.load('./Model/resnet18_1best_loss.pth.tar')
	model2 = torch.load('./Model/resnet18_2best_loss.pth.tar')
	model3 = torch.load('./Model/resnet18_3best_loss.pth.tar')
	model4 = torch.load('./Model/resnet18_4best_loss.pth.tar')
	model5 = torch.load('./Model/resnet18_5best_loss.pth.tar')

	models_dir = './Model/'

	normalize_param_loc = os.path.join(models_dir, 'normalize_param_1.txt')
	with open(normalize_param_loc) as f:
		params = f.read().splitlines()

	param1_1 = np.array(params[:3]*7, dtype = 'float32')
	param1_2 = np.array(params[3:]*7, dtype = 'float32')

	normalize_param_loc = os.path.join(models_dir, 'normalize_param_2.txt')
	with open(normalize_param_loc) as f:
		params = f.read().splitlines()

	param2_1 = np.array(params[:3]*7, dtype = 'float32')
	param2_2 = np.array(params[3:]*7, dtype = 'float32')

	normalize_param_loc = os.path.join(models_dir, 'normalize_param_3.txt')
	with open(normalize_param_loc) as f:
		params = f.read().splitlines()

	param3_1 = np.array(params[:3]*7, dtype = 'float32')
	param3_2 = np.array(params[3:]*7, dtype = 'float32')

	normalize_param_loc = os.path.join(models_dir, 'normalize_param_4.txt')
	with open(normalize_param_loc) as f:
		params = f.read().splitlines()

	param4_1 = np.array(params[:3]*7, dtype = 'float32')
	param4_2 = np.array(params[3:]*7, dtype = 'float32')

	normalize_param_loc = os.path.join(models_dir, 'normalize_param_5.txt')
	with open(normalize_param_loc) as f:
		params = f.read().splitlines()

	param5_1 = np.array(params[:3]*7, dtype = 'float32')
	param5_2 = np.array(params[3:]*7, dtype = 'float32')

	filename = []
	glaucoma_risk = []
	all_gl_softmx = []
	for img_pth in tqdm(img_pths):
			
		# print(img_pth.split('/')[-1], seg_pth.split('/')[-1])

		img = np.array(Image.open(img_pth))
		# seg = np.array(Image.open(seg_pth))
		# print(type(seg))
		# od = optic_disk(img, seg, bounding_box=550)
		
		gl_softmx, gl_softmx_list= classification_main(img)

			
		filename.append(img_pth.split('/')[-1])
		glaucoma_risk.append(gl_softmx)
		all_gl_softmx.append(gl_softmx_list)
			
		# print(gl_softmx)
		
		# if len(filename)==3:
		# 	break

	csv_file = pd.DataFrame()
	csv_file['FileName'] = filename
	csv_file['Glaucoma Risk'] = glaucoma_risk
	print(np.sum(np.round(np.array(glaucoma_risk))))
	# print(np.sum(np.round(np.array(all_gl_softmx))))

	csv_file.to_csv('./classification_results.csv', index = False)


	models = ['resnet18_1best_loss', 'resnet18_2best_loss', 'resnet18_3best_loss', 'resnet18_4best_loss', 'resnet18_5best_loss']
	# models.extend([os.path.join('./Model/OnlyRefuge', p) for p in next(os.walk('./Model/OnlyRefuge'))[2] if p != 'normalize_param.txt'])
	# models.sort()

	all_gl_softmx = np.array(all_gl_softmx)
	for i in range(len(models)):
		csv_file[models[i]] = all_gl_softmx[:,i] 



	csv_file.to_csv('./test_logs_all_models_Resnet18.csv', index = False)






