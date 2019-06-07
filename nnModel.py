import torch
from torch import nn as nn
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from torch import optim as optim
import torch.nn.functional as F
import time


def create_model():
    # Load a pretrained network (densenet161)
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout


    # Load a pretrained model (densenet)
    model = models.vgg19(pretrained=True)
    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # Create our classifier to replace the current one in the model
    model.classifier = nn.Sequential(nn.Linear(25088, 200),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(200, 102),
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
    return model

def get_data(batch_size):
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


    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size, shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size)

    return train_dataloaders, valid_dataloaders, test_dataloaders, train_datasets


def train_model(model, train_dataloaders, valid_dataloaders, criterion, optimizer, epochs):
    # Train the classifier layers using backpropagation using the pre-trained network to get the features
    # Track the loss and accuracy on the validation set to determine the best hyperparameters

    # Use the GPU if its available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the loss function
    criterion = criterion # nn.NLLLoss()

    # Define the learning rate
    #learning_rate = learning_rate

    # Define the optimizer (only train the classifier parameters leaving the feature parameters frozen)
    optimizer = optimizer #optim.SGD(model.classifier.parameters(), lr=learning_rate)

    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Define the training epochs
    epochs = epochs

    # Set the model to the device for training
    model.to(device)

    start_time = time.time()

    # Keep a record of the test vs validation losses to graph the learning curve
    train_losses, test_losses = [], []
    # With an active session train our model
    #with active_session():
        
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

    print("Done training model")


def save_model(model, train_datasets, learning_rate, batch_size, epochs, criterion, optimizer):
    # TODO: Save the checkpoint
    # Before saving the model set it to cpu to aviod loading issues later
    device = torch.device('cpu')
    model.to(device)

    # Save the train image dataset
    model.class_to_idx = train_datasets[0]

    # Save other hyperparamters
    # TODO: Pass in the input and output sizes
    checkpoint = {'input_size': 25088,
                'output_size': 102,
                'arch': 'densenet',
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'classifier' : model.classifier,
                'epochs': epochs,
                'criterion': criterion,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, 'myModelCheckpoint.pth')
    print("Done saving model")


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(filepath, model):
    # Load the model and force the tensors to be on the CPU
    checkpoint = torch.load(filepath,  map_location=lambda storage, loc: storage)
        
    model.load_state_dict(checkpoint['state_dict'])
    model.learning_rate = checkpoint['learning_rate']
    
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    #optimizer.load_state_dict(checkpoint['optimizer'])

    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    #return model, optimizer
    return model