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

batch_size = 32
train_dataloaders, valid_dataloaders, test_dataloaders, train_datasets = nnModel.get_data(batch_size)


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
#print(cat_to_name)



model = nnModel.create_model()

criterion = nn.NLLLoss()
learning_rate = .005
optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)
epochs = 5

nnModel.train_model(model, train_dataloaders, valid_dataloaders, criterion, optimizer,epochs)


nnModel.save_model(model, train_datasets, learning_rate, batch_size, epochs, criterion, optimizer)
