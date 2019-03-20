import torchvision.datasets.voc as VOC
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt 
import torchvision.transforms as transforms
import numpy as np
import torch


dataset = VOC.VOCDetection("../data/",
                 year='2012',
                 image_set='val',
                 download=True,
                 transform=None,
                 target_transform=None)
