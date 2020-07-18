import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
import model_utilities

ap = argparse.ArgumentParser(description='Train.py')
# Command Line ardguments
ap.add_argument('data_dir', nargs='*', action="store", default="ImageClassifier/flowers")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default = 12)
ap.add_argument('--arch', dest="arch", action="store", default="densenet121", type = str)

pa = ap.parse_args()
image_folder_path = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
epochs = pa.epochs

train_data, valid_data, test_data, trainloader , validloader, testloader = model_utilities.load_data(image_folder_path)
# model, optimizer, criterion = model_utilities.NN_setup(structure, lr)
# model_utilities.train_NN(model, optimizer, criterion, epochs, trainloader, validloader)
model_utilities.save_checkpoint('checkpoint.pth', train_data)
print("The Model has been successfully trained")
