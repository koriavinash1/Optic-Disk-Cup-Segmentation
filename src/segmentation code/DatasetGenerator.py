import torch.utils.data as data

from PIL import Image
import os
import cv2
import os.path
import numpy as np 
import torch
import h5py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import skimage.morphology as morph
from skimage.feature import canny
import matplotlib.pyplot as plt
# from cv2 import 

from skimage.restoration import denoise_tv_chambolle
from skimage.restoration import denoise_tv_bregman
from skimage.transform import resize
from torchvision import transforms

import torchvision.transforms.functional as TF

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy','.hdf5',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

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

    rgb_2 = clahe_single(ori_img, 2.0, (300,300))

    return Image.fromarray(rgb_1), Image.fromarray(rgb_2)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir_ = '/media/brats/MyPassport/Refuge/Annotation-Training400/Disc_Cup_Images'):
    # dir_ image datapath
    images, segs = [], []
    for target in sorted(os.listdir(dir_)):
        d = os.path.join(dir_, target)
        if not os.path.exists(d):
            print (d)
            continue
        seg_path = d.replace('Disc_Cup_Images', 'Disc_Cup_Masks')
        seg_path = seg_path.replace('jpg', 'bmp')
        images.append(d)
        segs.append(seg_path)
    return images, segs

def multilabel_binarize(image_nD, nlabel):
    image_nD = np.array(image_nD)
    labels = range(nlabel)
    out_shape = (len(labels),) + image_nD.shape
    bin_img_stack = np.ones(out_shape, dtype='uint8')
    for label in labels:
        bin_img_stack[label] = np.where(image_nD == label, bin_img_stack[label], 0)
    return bin_img_stack


selem = morph.disk(1)
def getEdgeEnhancedWeightMap(label, label_ids =[0,1,2], scale=1, edgescale=1, assign_equal_wt=False):
    label = multilabel_binarize(label, len(label_ids))# convert to onehot vector
    # label = np.expand_dims(label, 0)
    shape = (0,)+label.shape[1:]
    weight_map = np.empty(shape, dtype='uint8')
    if assign_equal_wt:
        return np.ones_like(label)
    for i in range(label.shape[0]): 
        #Estimate weight maps:
        weights = np.ones(len(label_ids))
        slice_map = np.ones(label[i,:,:].shape)
        for _id in label_ids:
            class_frequency = np.sum(label[i,:,:] == label_ids[_id])
            if class_frequency:
                weights[label_ids.index(_id)] = scale*label[i,:,:].size/class_frequency
                slice_map[np.where(label[i,:,:]==label_ids.index(_id))] = weights[label_ids.index(_id)]
                edge = np.float32(morph.binary_dilation(
                    canny(np.float32(label[i,:,:]==label_ids.index(_id)),sigma=1), selem=selem))
                edge_frequency = np.sum(np.sum(edge==1.0))
                if edge_frequency:    
                    slice_map[np.where(edge==1.0)] += edgescale*label[i,:,:].size/edge_frequency
            # print (weights)
            # utils.imshow(edge, cmap='gray')
        # utils.imshow(weight_map, cmap='gray')
        weight_map = np.append(weight_map, np.expand_dims(slice_map, axis=0), axis=0)
    return np.sum(weight_map, 0)


def apply_coordinates(image):
    # image: h, h, c
    assert image.size[0] == image.size[1] 
    x, y  = image.size
    x = np.arange(x)/image.size[0]
    y = np.arange(y)/image.size[1]
    Xmat, Ymat = np.meshgrid(x, y)
    return Xmat, Ymat


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB').crop((0, 0, min(img.size), min(img.size))).resize((512,512))

def seg_loader(path):
    seg = np.array(Image.open(path).convert('L'))
    seg = np.uint8(resize(seg, (512, 512), order=0)*255)
    seg[seg <= 50] = 1
    seg[(seg > 50)*(seg <= 200)] = 2
    seg[seg > 200] = 0
    seg = seg[0:min(seg.shape), 0:min(seg.shape)]
    return Image.fromarray(np.uint8(seg))

def numpy_loader(path):
    x=np.load(path)
    x=np.swapaxes(x,0,2) # convert to c, h,w
    x= np.swapaxes(x,0,1)
    return x


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class DatasetGenerator(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, seg_loader=seg_loader):
        imgs, segs = make_dataset(root)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.segs = segs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.segLoader = seg_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        img = self.loader(self.imgs[index])
        target = self.segLoader(self.segs[index])
        rgb1, rgb2 = clahe_all(np.array(img))


        if np.random.randn() > 0.5:
            img = TF.hflip(img)
            rgb1 = TF.hflip(rgb1)
            rgb2 = TF.hflip(rgb2)
            target = TF.hflip(target)

        if np.random.randn() > 0.5:
            img = TF.vflip(img)
            rgb1 = TF.vflip(rgb1)
            rgb2 = TF.vflip(rgb2)
            target = TF.vflip(target)
        
    # if np.random.randn() > 0.5:
        
        wtmap  = getEdgeEnhancedWeightMap(target)

        Xmat, Ymat = apply_coordinates(img)
        Xmat = torch.FloatTensor(Xmat).unsqueeze(0)
        Ymat = torch.FloatTensor(Ymat).unsqueeze(0)

        if self.transform is not None:
            img = self.transform(img)
            rgb1= self.transform(rgb1)
            rgb2= self.transform(rgb2)
            img = torch.cat([img, rgb1, rgb2], 0)
            img = torch.cat([img, Xmat, Ymat], 0)

        if self.target_transform is not None:
            target = self.target_transform(target)
            # target = torch.cat([target, Xmat, Ymat], 0)
            # print (target.size())

        return img, np.uint8(target), wtmap, self.imgs[index]

    def __len__(self):
        return len(self.imgs)

if __name__=='__main__':

    
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transformList = []
    transformList.append(transforms.ToTensor())
    transformList.append(normalize)
    transformSequence=transforms.Compose(transformList)

    dl=DatasetGenerator('/media/brats/MyPassport/Refuge/Annotation-Training400/Disc_Cup_Images',transformSequence)
    a,b,c,d= next(iter(dl))


    seg=Image.open('/media/brats/MyPassport/Refuge/Annotation-Training400/Disc_Cup_Masks/g0010.bmp').convert('L')
    ts= transformSequence(seg)

