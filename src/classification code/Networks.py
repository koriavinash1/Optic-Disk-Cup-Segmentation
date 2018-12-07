import torch
from torchvision import models
from torch import nn

class DenseNet201(nn.Module):

    def __init__(self, num_channel = 21, classCount=2, isTrained = True):
        
        super(DenseNet201, self).__init__()
        
        self.first_conv  = nn.Sequential(nn.Conv2d(num_channel, 3, kernel_size=3, padding=1),nn.BatchNorm2d(3))
        densenet = models.densenet201(pretrained= isTrained)
        self.features    = densenet.features
        kernelCount = densenet.classifier.in_features
        self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x= self.first_conv(x)
        x= self.features(x)
        x= nn.functional.adaptive_avg_pool2d(x,(1,1)).view(x.size(0),-1)
        x= self.classifier(x)
        return x

class DenseNet169(nn.Module):

    def __init__(self, num_channel = 21, classCount=2, isTrained = True):
        
        super(DenseNet169, self).__init__()
        
        self.first_conv  =nn.Sequential(nn.Conv2d(num_channel, 3, kernel_size=3, padding=1),nn.BatchNorm2d(3))
        densenet = models.densenet169(pretrained= isTrained)
        self.features    = densenet.features
        kernelCount = densenet.classifier.in_features
        self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x= self.first_conv(x)
        x= self.features(x)
        x= nn.functional.adaptive_avg_pool2d(x,(1,1)).view(x.size(0),-1)
        x= self.classifier(x)
        return x

class DenseNet161(nn.Module):

    def __init__(self, num_channel = 21, classCount=2, isTrained = True):
        
        super(DenseNet161, self).__init__()
        
        self.first_conv  =nn.Sequential(nn.Conv2d(num_channel, 3, kernel_size=3, padding=1),nn.BatchNorm2d(3))
        densenet = models.densenet161(pretrained= isTrained)
        self.features    = densenet.features
        kernelCount = densenet.classifier.in_features
        self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x= self.first_conv(x)
        x= self.features(x)
        x= nn.functional.adaptive_avg_pool2d(x,(1,1)).view(x.size(0),-1)
        x= self.classifier(x)
        return x


class DenseNet121(nn.Module):

    def __init__(self, num_channel = 21, classCount=2, isTrained = True):
        
        super(DenseNet121, self).__init__()
        
        self.first_conv  =nn.Sequential(nn.Conv2d(num_channel, 3, kernel_size=3, padding=1),nn.BatchNorm2d(3))
        densenet = models.densenet121(pretrained= isTrained)
        self.features    = densenet.features
        kernelCount = densenet.classifier.in_features
        self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x= self.first_conv(x)
        x= self.features(x)
        x= nn.functional.adaptive_avg_pool2d(x,(1,1)).view(x.size(0),-1)
        x= self.classifier(x)
        return x

#===============================================================================================================

class ResNet152(nn.Module):

    def __init__(self, num_channel = 21, classCount=2, isTrained = True):
        
        super(ResNet152, self).__init__()
        
        self.first_conv  =nn.Sequential(nn.Conv2d(num_channel, 3, kernel_size=3, padding=1),nn.BatchNorm2d(3))
        self.resnet = models.resnet152(pretrained= isTrained)
        self.resnet.avgpool=nn.AdaptiveAvgPool2d((1,1))
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x= self.first_conv(x)
        x= self.resnet(x)
        return x

class ResNet50(nn.Module):

    def __init__(self, num_channel = 21, classCount=2, isTrained = True):
        
        super(ResNet50, self).__init__()
        
        self.first_conv  =nn.Sequential(nn.Conv2d(num_channel, 3, kernel_size=3, padding=1),nn.BatchNorm2d(3))
        self.resnet = models.resnet50(pretrained= isTrained)
        self.resnet.avgpool=nn.AdaptiveAvgPool2d((1,1))
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x= self.first_conv(x)
        x= self.resnet(x)
        return x

class ResNet34(nn.Module):

    def __init__(self, num_channel = 21, classCount=2, isTrained = True):
        
        super(ResNet34, self).__init__()
        
        self.first_conv  =nn.Sequential(nn.Conv2d(num_channel, 3, kernel_size=3, padding=1),nn.BatchNorm2d(3))
        self.resnet = models.resnet34(pretrained= isTrained)
        self.resnet.avgpool=nn.AdaptiveAvgPool2d((1,1))
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x= self.first_conv(x)
        x= self.resnet(x)
        return x

class ResNet18(nn.Module):

    def __init__(self, num_channel = 21, classCount=2, isTrained = True):
        
        super(ResNet18, self).__init__()
        
        self.first_conv  =nn.Sequential(nn.Conv2d(num_channel, 3, kernel_size=3, padding=1),nn.BatchNorm2d(3))
        self.resnet = models.resnet18(pretrained= isTrained)
        self.resnet.avgpool=nn.AdaptiveAvgPool2d((1,1))
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x= self.first_conv(x)
        x= self.resnet(x)
        return x


class ResNet101(nn.Module):

    def __init__(self, num_channel = 21, classCount=2, isTrained = True):
        
        super(ResNet101, self).__init__()
        
        self.first_conv  =nn.Sequential(nn.Conv2d(num_channel, 3, kernel_size=3, padding=1),nn.BatchNorm2d(3))
        self.resnet = models.resnet101(pretrained= isTrained)
        self.resnet.avgpool=nn.AdaptiveAvgPool2d((1,1))
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        x= self.first_conv(x)
        x= self.resnet(x)
        return x
