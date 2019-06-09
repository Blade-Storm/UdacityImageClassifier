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
#from workspace_utils import active_session
from PIL import Image
import seaborn as sb
import pandas as pd
import time
import nnModel
import helpers.JsonLoader
import helpers.DataLoader

# Define the batch_size and create the data loaders
batch_size = 32
train_dataloaders, valid_dataloaders, test_dataloaders, train_datasets = helpers.DataLoader.load_image_data(batch_size)

# Get the contents of the cat_to_name json file
cat_to_name = helpers.JsonLoader.load_json('cat_to_name.json')
    
# Create the model
model = nnModel.create_model()

# Define the loss function, learning rate, optimizer, and epochs to train with
criterion = nn.NLLLoss()
learning_rate = .001
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
epochs = 10

# Train the model with validation
nnModel.train_model_validation(model, train_dataloaders, valid_dataloaders, criterion, optimizer,epochs)

# Perform the sanity check
nnModel.sanity_check(cat_to_name, 'flowers/test/101/image_07949.jpg', model, 101)

# Save the model
nnModel.save_model(model, train_datasets, learning_rate, batch_size, epochs, criterion, optimizer)
