# Imports here
import matplotlib.pyplot as plt
import torch
from torch import nn as nn
from torch import optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import numpy as np
from collections import OrderedDict
from workspace_utils import active_session
from PIL import Image
import seaborn as sb
import pandas as pd
import time


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# For the train transform:
# Randomly rotate the images
# Randomly resize and crop
# Randomly flip the image
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# For the validation and test transforms:
# Resize and crop
valid_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


test_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])


# Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

batch_size=32
# Using the image datasets and the trainforms, define the dataloaders
train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)
test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size)


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
print(cat_to_name)




# Load a pretrained network (densenet161)
# Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout


# Load a pretrained model (densenet)
model = models.vgg19(pretrained=True)
# Freeze the parameters so we dont backpropagate through them
for param in model.parameters():
    param.requires_grad = False

# Create our classifier to replace the current one in the model
model.classifier = nn.Sequential(nn.Linear(2208, 1000),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(1000, 102),
                                 nn.LogSoftmax(dim=1))
#model.classifier = nn.Sequential(nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
#                                 nn.ReLU(),
#                                 nn.MaxPool2d(kernel_size=2, stride=2),
#                                 nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
#                                 nn.ReLU(),
#                                 nn.MaxPool2d(kernel_size=2, stride=2),
#                                 nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
#                                 nn.ReLU(),
#                                 nn.MaxPool2d(kernel_size=2, stride=2),
#                                 nn.Dropout(),
#                                 nn.Linear(18432,10000),
#                                 nn.ReLU(),
#                                 nn.Dropout(),
#                                 nn.Linear(10000,5000),
#                                 nn.ReLU(),
#                                 nn.Dropout(),
#                                 nn.Linear(5000, 102),
#                                 nn.LogSoftmax(dim=1))
#class Net(nn.Module):

#    def __init__(self):
#        super(Net, self).__init__()
#        # 3 channels (RGB), kernel=5
#        self.conv1 = nn.Conv2d(3, 6, 5)
#        self.conv2 = nn.Conv2d(6, 16, 5)
        
#        self.pool = nn.MaxPool2d(2, 2)

        
#        self.fc1 = nn.Linear(44944, 5000)
#        self.fc2 = nn.Linear(5000, 1000)
#        self.fc3 = nn.Linear(1000, 102)

#    def forward(self, x):
#        x = self.pool(F.relu(self.conv1(x)))
#        x = self.pool(F.relu(self.conv2(x)))
#        x = x.view(x.size(0),  -1)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = F.log_softmax(self.fc3(x), dim=1)
#        return x
#model = Net();
print(model)



# Train the classifier layers using backpropagation using the pre-trained network to get the features
# Track the loss and accuracy on the validation set to determine the best hyperparameters

# Use the GPU if its available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the loss function
criterion = nn.NLLLoss()

# Define the learning rate
learning_rate = .01

# Define the optimizer (only train the classifier parameters leaving the feature parameters frozen)
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

#optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define the training epochs
epochs = 30

# Set the model to the device for training
model.to(device)

start_time = time.time()

# Keep a record of the test vs validation losses to graph the learning curve
train_losses, test_losses = [], []
# With an active session train our model
with active_session():
    
    # Create the training loop
    for e in range(epochs):
        # Define the training loss for each epoch
        training_loss = 0
        
        for images, labels in train_dataloaders:            
            # Move the image and label tensors to the default device
            images, labels = images.to(device), labels.to(device)
            
            # Zero the gradients since they accumulate
            optimizer.zero_grad()
            
            #image = images.view(images.shape[0], -1)
            
            #print("test")

            
            # Get the log probability from the model
            logps = model(images)

            # Get the loss
            loss = criterion(logps, labels)

            # Backpropagate
            loss.backward()

            # Gradient Descent
            optimizer.step()

            # Keep track of the training loss
            training_loss += loss.item()
        else:
            # Keep track of the validation loss and accuracy
            validation_loss = 0
            accuracy = 0
            
            #print("validation")
            
            # Set the model to evaluation mode. This will turn off the dropout functionality
            model.eval()
            
            # Turn off the gradients for validation
            with torch.no_grad():
                # Create the validation loop
                for images, labels in valid_dataloaders:
                    # Move the image and label tensors to the default device
                    images, labels = images.to(device), labels.to(device)
                    
                    #images.resize_(images.shape[0], -1)
                    
                    # Get the log probability 
                    logps = model(images)
                    
                    # Get the loss
                    loss = criterion(logps, labels)
                    
                    # Get probability from the model
                    ps = torch.exp(logps)
                    
                    # Get the top class from the predictions
                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)
                    
                    # Get the accuracy for the prediction
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    # Keep track of the validation loss
                    validation_loss += loss.item()
                    
            # Set the model back to train mode
            model.train()
            elapsed_time = time.time() - start_time  
            
            # Update the training and validation losses to graph the learning curve
            train_losses.append(training_loss/len(train_dataloaders))
            test_losses.append(validation_loss/len(valid_dataloaders))
            
            print("Epoch: {}\n".format(e),
                  "Training Loss: {}\n".format(training_loss/len(train_dataloaders)),
                  "Validation Loss: {}\n".format(validation_loss/len(valid_dataloaders)),
                  "Accuracy: {}\n".format(accuracy/len(valid_dataloaders) * 100),
                  "Time: {}\n".format(elapsed_time))  