import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np 
import torch

import matplotlib.pyplot as plt
from torchvision import transforms
import cv2


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


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir_):
    # dir_ image datapath
    images, labels = [], []
    gl_load_dir = os.path.join(dir_, 'Glaucoma')
    normal_load_dir = os.path.join(dir_, 'Non-Glaucoma')
    
    gl_images = [os.path.join(gl_load_dir, p) for p in next(os.walk(gl_load_dir))[2]]
    normal_images = [os.path.join(normal_load_dir, p) for p in next(os.walk(normal_load_dir))[2]]
                 
    images = sorted(gl_images) + sorted(normal_images)
    labe1s = [0]*len(gl_images) + [1]*len(normal_images)
    return images, labe1s


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class DatasetGenerator(data.Dataset):

    def __init__(self, root, param1, param2, transform=None,
                 loader=pil_loader):
        
        classes, class_to_idx = find_classes(root)
        imgs, labels = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.param2 = param2
        self.param1 = param1
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):

        img = self.loader(self.imgs[index])
        img = transforms.RandomHorizontalFlip(p=0.5)(img)
        img = transforms.RandomRotation((0,360))(img)
        img = transforms.TenCrop(500)(img)
        

        transformsList = []
        transformsList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformsList.append(transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(torch.FloatTensor(self.param1),torch.FloatTensor(self.param2))(crop) for crop in crops])))
        transformComp = transforms.Compose(transformsList)


        img_ =[]
        for i in img:
            img_.append(clahe_all(np.array(i)))

        img_ = transformComp(img_)
            
        label = self.labels[index]

        return img_, label, self.imgs[index].split('/')[-1]

    def __len__(self):
        return len(self.imgs)
