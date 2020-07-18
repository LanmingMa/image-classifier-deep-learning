import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import os, random
import glob
import argparse

import model_utilities

ap = argparse.ArgumentParser(
    description='predict-file')
ap.add_argument('checkpoint', default='/home/workspace/paind-project/checkpoint.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = ap.parse_args()
number_of_outputs = pa.top_k
power = pa.gpu
path = pa.checkpoint
img_set = np.random.randint(low = 1, high = 10)

trainloader, validloader, testloader = model_utilities.load_data(image_folder_path)
model = model_utilities.load_checkpoint('checkpoint.pth')
image_path = model_utilities.random_select_img(img_set)
top_probs, top_classes = predict(image_path, model)

i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(top_probs[i], top_classes[i]))
    i += 1