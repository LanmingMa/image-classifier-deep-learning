# Imports packages
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
from torch.autograd import Variable
from PIL import Image
import glob, os
import random
import json
import argparse


def load_data(image_folder_path):
    '''
    Arguments : the datas' path
    Returns : The loaders for the train, validation and test datasets
    This function receives the location of the image files, applies the necessery transformations
    (rotations,flips,normalizations and crops) and converts the images to tensor in order to be able to be fed into the neural
    network
    '''
    data_dir = image_folder_path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
                                          
    return train_data, valid_data, test_data, trainloader, validloader, testloader
                                          
def NN_setup(structure = 'densenet121', lr = 0.001):
    '''
    Arguments: The architecture for the model is densenet121, the hyperparameters for the network which is learning rate and whether to use gpu or not.
    Returns: The set up model, along with the criterion and the optimizer for the Training.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained = True)
    
    # Turn off gradients for the model
    for param in model.parameters():
        param.requires_grad = False                                      
    # Define the new classifier
    classifier = nn.Sequential(nn.Linear(1024, 782),
                          nn.ReLU(),
                          nn.Dropout(p=0.2),
                          nn.Linear(782, 512),
                          nn.ReLU(),
                          nn.Dropout(p=0.4),
                          nn.Linear(512, 210),
                          nn.ReLU(),
                          nn.Dropout(p=0.5),
                          nn.Linear(210, 90),
                          nn.ReLU(),
                          nn.Dropout(p=0.2),
                          nn.Linear(90, 102),
                          nn.LogSoftmax(dim=1))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    model.cuda()
    return model, optimizer, criterion

def train_NN(model, optimizer, criterion, epochs, trainloader, validloader):
    steps = 0 # Track the number of train steps
    running_loss = 0 # Track loss
    # Number of steps before printing out the validation loss
    print_every = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                          
    print("--------------Start training Neural Networks------------- ")                                     
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1 #increment steps here
            images, labels = images.to(device), labels.to(device) # Move input and label tensors to the default device
            optimizer.zero_grad()

            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step() #optimizer take a step

            running_loss += loss.item() #track loss
            # Do validation on the test set
            if steps % print_every == 0: 
                valid_loss = 0
                accuracy = 0
                model.eval() #into evalution inference mode, turns off dropout
                with torch.no_grad():    
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1) #first largest value
                        equality = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor))

                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every), 
                      #every time it's printed out, take the average
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)), 
                      #sum up all the loss divided by batch size
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                running_loss = 0
                model.train()
    print("------------- Training Finished-----------------------")
    print("Model has been successfully trained. It went through")
    print("----------Epochs: {}------------------------------------".format(epochs))
    print("----------Steps: {}-----------------------------".format(steps))

def save_checkpoint(checkpoint_path, train_data, model):
    '''
    Arguments: The saving path and the hyperparameters of the network
    Returns: Nothing
    This function saves the model to checkpoint.pth
    '''
    model.class_to_idx = train_data.class_to_idx
    model.cpu()
    # Build a dictionary with the information to rebuild the model.
    checkpoint = {'input_size': 1024,
                  'output_size': 102,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, checkpoint_path)
                                          
def load_checkpoint(checkpoint_path):
    '''
    Arguments: The checkpoint file path
    Returns: The Neural Netowrk model
    '''
    checkpoint = torch.load(checkpoint_path)
    model.input_size = checkpoint['input_size']
    model.output_size = checkpoint['output_size']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
                                          
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    ''' 
    # Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image_path)
    img_pil.thumbnail((256,256), Image.ANTIALIAS)
    img_pil = img_pil.crop((0, 0, 224, 224))
    img_array = np.array(img_pil)
    img_array = img_array/255
    
    # Normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean)/std
                                          
    img_array = img_array.transpose((2, 0, 1))
    
    return img_array

def random_select_img(img_set):
    parent_image_path = random.choice(os.listdir(data_dir + '/test/' + img_set + '/'))
    image_path = data_dir + '/test/' + img_set + '/' + parent_image_path
    return image_path
                                          
def predict(image_path, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)
                                          
    img_array = process_image(image_path) 
    image = torch.from_numpy(img_array).type(torch.FloatTensor) 
    image.unsqueeze_(0)
                                          
    # Turn off gradients to speed up this part
    with torch.no_grad():
        logps = model.forward(image)

    # Output of the network are logits, need to take softmax for probabilities
    probs = torch.exp(logps)
    
    # Top probs
    top_probs, top_indexes = probs.topk(5)
#     top_probs,  top_indexes = top_probs.cpu(), top_indexes.cpu()
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_indexes = top_indexes.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [cat_to_name[idx_to_class[idx]] for idx in top_indexes]
    
    return top_probs, top_classes