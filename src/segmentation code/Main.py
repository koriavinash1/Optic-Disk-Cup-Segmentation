import os
import numpy as np
import time
import sys
import pandas as pd

from model import FCDenseNet103, FCDenseNet57, FCDenseNet67
from Trainer import Trainer

import torch
from torch.autograd import Variable
import json

# from Inference import Inference

Trainer = Trainer()


nclasses = 3

#--------------------------------------------------------------------------------

def main (nnClassCount=nclasses):
	# "Define Architectures and run one by one"

	nnArchitectureList = [
							{
								'name': 'tramisu_2D_FC57',
								'model' : FCDenseNet57(n_classes = nclasses),
								'TrainPath': '/home/dlguru/Varghese/Refuge/data/Training/Disc_Cup_Images',
								'ValidPath': '/home/dlguru/Varghese/Refuge/data/Validation/Disc_Cup_Images',
								'ckpt' : None
							}

						]

	for nnArchitecture in nnArchitectureList:
		runTrain(nnArchitecture=nnArchitecture)



def getDataPaths(path):
 	data = pd.read_csv(path)
 	imgpaths = data['Paths'].as_matrix()
 	return imgpaths

#--------------------------------------------------------------------------------

def runTrain(nnArchitecture = None):

	timestampTime = time.strftime("%H%M%S")
	timestampDate = time.strftime("%d%m%Y")
	timestampLaunch = timestampDate + '-' + timestampTime

	TrainPath = nnArchitecture['TrainPath']
	ValidPath = nnArchitecture['ValidPath']

	nnClassCount = nclasses

	#---- Training settings: batch size, maximum number of epochs
	trBatchSize = 4
	trMaxEpoch = 60*4

	print ('Training NN architecture = ', nnArchitecture['name'])
	info_dict = {
				'batch_size': trBatchSize,
				'architecture':nnArchitecture['name'] ,
				'number of epochs':trMaxEpoch,
				'train path':TrainPath, 
				'valid_path':ValidPath,
				'number of classes':nclasses,
				'Date-Time':	timestampLaunch
	} 
	if not os.path.exists('../modelsclaheWC11'): os.mkdir('../modelsclaheWC11')
	with open('../modelsclaheWC11/config.txt','w') as outFile:
		json.dump(info_dict, outFile)
	

	Trainer.train(TrainPath,  ValidPath, nnArchitecture, nnClassCount, trBatchSize, trMaxEpoch, timestampLaunch, nnArchitecture['ckpt'])


#--------------------------------------------------------------------------------

if __name__ == '__main__':
	main()
	# runTest()
