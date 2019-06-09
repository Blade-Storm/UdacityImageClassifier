import torch
from torch import nn as nn
import torchvision
from torchvision import models
from torch import optim as optim
import torch.nn.functional as F
import time
import helpers.ProcessImage
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict





def create_model():
    # Load a pretrained network (densenet161)
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout


    # Load a pretrained model (densenet)
    model = models.vgg19(pretrained=True)
    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # Create our classifier to replace the current one in the model
    model.classifier = nn.Sequential(nn.Linear(25088,500),
                                     nn.Dropout(0.5),
                                     nn.ReLU(),                                     
                                     nn.Linear(500,102),
                                     nn.LogSoftmax(dim=1))
    #model.classifier = nn.Sequential(nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
    #                                 nn.ReLU(),
    #                                 nn.MaxPool2d(kernel_size=2, stride=2),
    #                                 nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
    #                                nn.ReLU(),
    #                                 nn.MaxPool2d(kernel_size=2, stride=2),
    #                                 nn.Dropout(),
    #                                 nn.Linear(44944,5000),
    #                                 nn.ReLU(),
    #                                 nn.Dropout(),
    #                                 nn.Linear(5000,1000),
    #                                 nn.ReLU(),
    #                                 nn.Dropout(),
    #                                 nn.Linear(1000, 102),
    #                                 nn.LogSoftmax(dim=1))
    #class Net(nn.Module):

    #    def __init__(self):
    #        super(Net, self).__init__()
            # 3 channels (RGB), kernel=5
    #        self.conv1 = nn.Conv2d(3, 6, 5)
    #        self.conv2 = nn.Conv2d(6, 16, 5)
            
    #        self.pool = nn.MaxPool2d(2, 2)

            
    #        self.fc1 = nn.Linear(44944, 5000)
    #        self.fc2 = nn.Linear(5000, 1000)
    #        self.fc3 = nn.Linear(1000, 1000)
    #        self.fc4 = nn.Linear(1000, 102)

    #    def forward(self, x):
    #        x = self.pool(F.relu(self.conv1(x)))
    #        x = self.pool(F.relu(self.conv2(x)))
    #        x = x.view(x.size(0),  -1)
    #        x = F.dropout(F.relu(self.fc1(x)), 0.5)
    #        x = F.dropout(F.relu(self.fc2(x)), 0.5)
    #        x = F.dropout(F.relu(self.fc3(x)), 0.5)
    #        x = F.log_softmax(self.fc4(x), dim=1)
    #        return x
    #model = Net()
    print(model)
    return model



def train_model_validation(model, train_dataloaders, valid_dataloaders, criterion, optimizer, epochs):
    # Train the classifier layers using backpropagation using the pre-trained network to get the features
    # Track the loss and accuracy on the validation set to determine the best hyperparameters

    # Use the GPU if its available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the model to the device for training
    model.to(device)

    start_time = time.time()

    # Keep a record of the test vs validation losses to graph the learning curve
    #train_losses, test_losses = [], []
    # With an active session train our model
    #with active_session():
        
        # Create the training loop
    for e in range(epochs):
        # Set the model back to train mode
        model.train()
        # Define the training loss for each epoch
        training_loss = 0
        
        for images, labels in train_dataloaders:            
            # Move the image and label tensors to the default device
            images, labels = images.to(device), labels.to(device)
            
            # Zero the gradients since they accumulate
            optimizer.zero_grad()
            
            # Get the log probability from the model
            logps = model.forward(images)

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
                    logps = model.forward(images)
                    
                    # Get the loss
                    loss = criterion(logps, labels)
                    
                    # Get probability from the model
                    ps = torch.exp(logps)
                    
                    # Get the top class from the predictions
                    top_p, top_class = ps.topk(1, dim=1)

                    # For debugging purposes print the actual vs predicted label
                    # so we can visually see if the model is doing well... (it is)
                    print("Actual Label: {}".format(labels[0]))
                    print("Predicted Label: {}\n".format(top_class[0][0]))

                    equals = top_class == labels.view(*top_class.shape)
                    
                    # Get the accuracy for the prediction
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    # Keep track of the validation loss
                    validation_loss += loss.item()
                    
            
            elapsed_time = time.time() - start_time  
            
            # Update the training and validation losses to graph the learning curve
            #train_losses.append(training_loss/len(train_dataloaders))
            #test_losses.append(validation_loss/len(valid_dataloaders))
            
            print("Epoch: {}\n".format(e),
                    "Training Loss: {}\n".format(training_loss/len(train_dataloaders)),
                    "Validation Loss: {}\n".format(validation_loss/len(valid_dataloaders)),
                    "Accuracy: {}\n".format(accuracy/len(valid_dataloaders) * 100),
                    "Time: {}\n".format(elapsed_time))  

    print("Done training model")



def save_model(model, train_datasets, learning_rate, batch_size, epochs, criterion, optimizer):
    # TODO: Save the checkpoint
    print("Saving the model...")
    # Before saving the model set it to cpu to aviod loading issues later
    device = torch.device('cpu')
    model.to(device)

    # Save the train image dataset
    model.class_to_idx = train_datasets[0]

    # Save other hyperparamters
    # TODO: Pass in the input and output sizes
    checkpoint = {'input_size': 25088,
                'output_size': 102,
                #'hidden_layers': [each.out_features for each in model.hidden_layers],
                'arch': 'vgg',
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
    print("Loading the model...")
    # Load the model and force the tensors to be on the CPU
    checkpoint = torch.load(filepath,  map_location=lambda storage, loc: storage)
   
    model.classifier = nn.Sequential(nn.Linear(checkpoint['input_size'], 500),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(500, checkpoint['output_size']),
                            nn.LogSoftmax(dim=1))

    model.load_state_dict(checkpoint['state_dict'])
    model.learning_rate = checkpoint['learning_rate']
    
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    #optimizer.load_state_dict(checkpoint['optimizer'])

    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    print("Done loading the model")
    #return model, optimizer
    return model    




def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Use the GPU if its available
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to('cpu')
    
    #Switch the model to evaluation mode to turn off dropout
    model.eval()
    
    with torch.no_grad():
    # Implement the code to predict the class from an image file    
        # Processs the image
        image = helpers.ProcessImage.process_image(image_path)

        # We need a tensor for the model so change the image to a np.Array and then a tensor
        image = torch.from_numpy(np.array([image])).float()
        image.to('cpu')

        # Use the model to make a prediction
        logps = model.forward(image)
        ps = torch.exp(logps)

        # Get the top 5 probabilities and classes. This is returned as a tenosr of lists
        p, classes = ps.topk(topk, dim=1)
        
        # Get the first items in the tensor list to get the list of probs and classes
        top_p = p.tolist()[0]
        top_classes = classes.tolist()[0]
    
        return top_p, top_classes


def sanity_check(cat_to_name, file_path, model, index):
    # Display an image along with the top 5 classes
    # Create a plot that will have the image and the bar graph
    fig = plt.figure(figsize = [10,5])

    # Create the axes for the flower image 
    ax = fig.add_axes([.5, .4, .225, .225])

    # Process the image and show it
    result = helpers.ProcessImage.process_image(file_path)
    ax = helpers.ProcessImage.imshow(result, ax)
    ax.axis('off')

    ax.set_title(cat_to_name[str(index)])


    # Make a prediction on the image
    predictions, classes = predict(file_path, model)

    # Get the lables from the json file
    labels = []
    for c in classes:
        labels.append(cat_to_name[str(c)])

    # Make a bar graph
    # Create the axes for the bar graph
    ax1 = fig.add_axes([.5, .1, .225, .225])

    # Get the range for the probabilities
    y_pos = np.arange(len(labels))

    # Plot as a horizontal bar graph
    plt.barh(y_pos, predictions, align='center', alpha=0.5)
    plt.yticks(y_pos, labels)
    plt.xlabel('probabilities')
    plt.show()