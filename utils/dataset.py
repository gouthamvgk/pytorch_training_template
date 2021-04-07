import os
import torch
import numpy as np
import cv2
import pycocotools.coco as coco
from torch.utils.data import Dataset
from pathlib import Path

class my_dataset(Dataset):
	def __init__(self, dataset_params, typ="train"):
		super(my_dataset, self).__init__()
		self.config = dataset_params
		self.images = []


	def __len__(self):
		return len(self.images)

	def __getitem__(self, index: int):
		#########Image augmentation code
		#########Image processing code
		image = None#get image code here
		return image