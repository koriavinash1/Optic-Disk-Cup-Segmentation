import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
import cv2
import numpy as np
from model import FCDenseNet57

root_path = '/media/brats/Varghese/REFUGE_2018/data/Testing/Disc_Cup_Images'
test_imgs = os.listdir(root_path)

modelC_ckpt    = '/media/brats/Varghese/REFUGE_2018/segmentation_Codes/modelsclaheWC11/model-m-best_loss.pth.tar'

modelWC_ckpt   = '/media/brats/Varghese/REFUGE_2018/segmentation_Codes/models/model-m-25062018-184326-tramisu_2D_FC57_without_coordinate_loss = 3.8611210505167644_acc = 0.9975007267321571_best_acc.pth.tar'

modelWC = FCDenseNet57(3, 5)
modelC= FCDenseNet57(3, 11)

modelWC.load_state_dict(torch.load(modelWC_ckpt)['state_dict'])
modelC.load_state_dict(torch.load(modelC_ckpt)['state_dict'])

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
transformList.append(transforms.ToTensor())
transformList.append(normalize)
transformSequence=transforms.Compose(transformList)


def apply_coordinates(image):
    # image: h, h, c
    assert image.size[0] == image.size[1] 
    x, y  = image.size
    x = np.arange(x)/image.size[0]
    y = np.arange(y)/image.size[1]
    Xmat, Ymat = np.meshgrid(x, y)
    return Xmat, Ymat


def clahe_single(ori_img,clipLimit,tileGridSize):
    ori_img = np.uint8(ori_img)
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

    rgb_2 = clahe_single(ori_img, 2.0, (300,300))

    return Image.fromarray(rgb_1), Image.fromarray(rgb_2)

# compute dice
def background(self,data):
	return data==0

def opticdisk(self,data):
	return data==1

def opticcup(self,data):
	return data==2

def get_dice_score(self,prediction,ground_truth):
	masks=(self.background, self.opticdisk, self.opticcup)
	pred=torch.exp(prediction)
	p=np.uint8(np.argmax(pred.data.cpu().numpy(), axis=1))
	gt=np.uint8(ground_truth.data.cpu().numpy())
	b, od, oc=[2*np.sum(func(p)*func(gt)) / (np.sum(func(p)) + np.sum(func(gt))+1e-3) for func in masks]
	return b, od, oc


plt.ion()
for img in test_imgs:
    imagen = Image.open(os.path.join(root_path,img)).convert('RGB').resize((512,512))
    rgb1, rgb2 = clahe_all(np.array(imagen))
    
    image = transformSequence(imagen)
    rgb1= transformSequence(rgb1)
    rgb2= transformSequence(rgb2)
    image3 = torch.cat([image, rgb1, rgb2], 0)
    
    Xmat, Ymat = apply_coordinates(imagen)
    Xmat = torch.FloatTensor(Xmat).unsqueeze(0)
    Ymat = torch.FloatTensor(Ymat).unsqueeze(0)

    
    imageC  = torch.cat([image3, Xmat, Ymat], 0) # comment for normal model
    imageWC = torch.cat([image, Xmat, Ymat], 0)
 
    print (imageC.size())
    predC    = modelC(imageC.unsqueeze(0))
    _, predC = torch.max(predC, 1)
    predC    = predC.squeeze(0).detach().cpu().numpy()

    predWC   = modelWC(imageWC.unsqueeze(0))
    _, predWC=torch.max(predWC, 1)
    predWC   = predWC.squeeze(0).detach().cpu().numpy()

    plt.subplot(1,3,1)
    plt.imshow(imagen)
    plt.subplot(1,3,2)
    plt.imshow(predC)
    plt.xlabel("with 11 channels")
    plt.subplot(1,3,3)
    plt.imshow(predWC)
    plt.xlabel("with 5 channels")
    plt.pause(0.5)
    plt.show()
