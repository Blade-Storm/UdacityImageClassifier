import torch
from torch import nn as nn
from torch.autograd import Variable
import torchvision
from torchvision import models
import torch.nn.functional as F
import time
import helpers.ProcessImage
import matplotlib.pyplot as plt
import numpy as np


def create_model(arch, hidden_units):
    '''
        Creates a pretrained model using VGG19 or Densenet161 and returns the model
        
        Inputs:
        arch - The architecture to be used. Either 'vgg19' or 'densenet161'
        hidden_units - The number of units in the hidden layer
        
        Outputs:
        model - The created (loaded) pretrained model
    '''
    # Load a pretrained network (vgg19 or densenet161)
    # Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    print("Creating the model...")

    # Load a pretrained model
    if arch.lower() == "vgg19":
        model = models.vgg19(pretrained=True)
        input_features = 25088
    elif arch.lower() == "densenet161":
        model = models.densenet161(pretrained=True)
        input_features = 2208
    else:
        # We dont support the entered model architecture so return to start over
        print("Model architecture: {}, is not supported. \n Try vgg19 or densenet161".format(arch.lower()))
        return 0
    
    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    # Create our classifier to replace the current one in the model
    model.classifier = nn.Sequential(nn.Linear(input_features,hidden_units),
                                     nn.ReLU(),         
                                     nn.Dropout(0.5),                            
                                     nn.Linear(hidden_units,102),
                                     nn.LogSoftmax(dim=1))

    print("Done creating the model\n")
    return model



def train_model(model, train_dataloaders, valid_dataloaders, criterion, optimizer, epochs, use_gpu):
    '''
        Trains a model using a given loss function, optimizer, dataloaders, epochs, and whether or not to use the GPU. Outputs loss and accuracy numbers
        
        Inputs:
        model - The model to train
        train_dataloaders - The data for the training
        valid_dataloaders - The data for the validation
        criterion - The loss function 
        optimizer - The optimizer
        epochs - The number of epochs to run the training for
        use_gpu - Whether or not to train with the GPU
        
        Outputs:
        Prints out the training and validation losses and accuracies
    '''
    # Train the classifier layers using backpropagation using the pre-trained network to get the features
    # Track the loss and accuracy on the validation set to determine the best hyperparameters
    print("Training the model...\n")

    # Use the GPU if its available
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Set the model to the device for training
    model.to(device)

    # Capture the current time for tracking purposes
    start_time = time.time()

    # Keep track of the losses and accuracies for training and validation
    train_losses, validation_losses, training_accuracies, validation_accuracies = [], [], [], []

    # Create the training loop
    for e in range(epochs):
        # Set the model back to train mode
        model.train()
        # Define the training loss for each epoch
        training_loss = 0

        training_accuracy = 0
        
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

            # Get the top class from the predictions
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            
            # Get the accuracy for the prediction
            training_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
        else:
            # Keep track of the validation loss and accuracy
            validation_loss = 0
            validation_accuracy = 0
            
            # Set the model to evaluation mode. This will turn off the dropout functionality
            model.eval()

            # Turn off the gradients for validation
            with torch.no_grad():
                # Create the validation loop
                for images, labels in valid_dataloaders:
                    # Move the image and label tensors to the default device
                    images, labels = images.to(device), labels.to(device)
                                        
                    # Get the log probability 
                    logps = model.forward(images)
                    
                    # Get the loss
                    loss = criterion(logps, labels)
                    
                    # Get probability from the model
                    ps = torch.exp(logps)
                    
                    # Get the top class from the predictions
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    
                    # Get the accuracy for the prediction
                    validation_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                    # Keep track of the validation loss
                    validation_loss += loss.item()
                    
        # Get the total time that has elapsed
        elapsed_time = time.time() - start_time  

        # Update the training and validation losses to graph the learning curve
        train_losses.append(training_loss/len(train_dataloaders))
        validation_losses.append(validation_loss/len(valid_dataloaders))
        training_accuracies.append(training_accuracy/len(train_dataloaders) * 100)
        validation_accuracies.append(validation_accuracy/len(valid_dataloaders) * 100)

        # Print out the statistical information
        print("Training Epoch: {}\n".format(e),
                "Training Loss: {}\n".format(training_loss/len(train_dataloaders)),
                "Training Accuracy: {}\n".format(training_accuracy/len(train_dataloaders) * 100),
                "Validation Loss: {}\n".format(validation_loss/len(valid_dataloaders)),
                "Validation Accuracy: {}\n".format(validation_accuracy/len(valid_dataloaders) * 100),
                "Total Time: {}\n".format(elapsed_time))  

    
    plt.plot(train_losses, label='Training loss')
    plt.plot(validation_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()
    plt.plot(training_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.legend(frameon=False)
    plt.show()

    print("\nDone training the model \n")



def save_model(model, train_datasets, learning_rate, batch_size, epochs, criterion, optimizer, hidden_units, arch):
    '''
        Saves a model to a checkpoint file with the learning rate, batch size, epochs, loss function, optimizer, hidden units, and architecture used in training
        
        Inputs:
        model - The model to train
        train_datasets - The dataset for the training. This is used to get the classes to indexes
        learning_rate - The learning rate used for training
        criterion - The loss function 
        optimizer - The optimizer
        epochs - The number of epochs to run the training for
        hidden_units - The hidden layers unit size
        arch - The architecture used
        save_directory - The directory to save the pth file to
        
        Outputs:
        Saves the checkpoint.pth file to the given directory
    '''
    print("Saving the model...")

    # Save the train image dataset
    model.class_to_idx = train_datasets.class_to_idx

    if arch.lower() == "vgg19":
        input_features = 25088
    elif arch.lower() == "densenet161":
        input_features = 2208

    # Save other hyperparamters
    checkpoint = {'input_size': input_features,
                'output_size': 102,
                'hidden_units': hidden_units,
                'arch': arch,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'classifier' : model.classifier,
                'epochs': epochs,
                'criterion': criterion,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}


    torch.save(checkpoint, 'checkpoint.pth')
    print("Done saving the model")


def load_model(checkpoint_file):
    '''
        Loads a model using a checkpoint.pth file
        
        Inputs:
        checkpoint_file - The file path and name for the checkpoint
        
        Outputs:
        model - Returns the loaded model
    '''
    print("Loading the model...")
    # Load the model and force the tensors to be on the CPU
    checkpoint = torch.load(checkpoint_file)
   
    if(checkpoint['arch'].lower() == 'vgg19' or checkpoint['arch'].lower() == 'densenet161'):
        model = getattr(torchvision.models, checkpoint['arch'])(pretrained = True)


    model.classifier = checkpoint['classifier'] 
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    # Freeze the parameters so we dont backpropagate through them
    for param in model.parameters():
        param.requires_grad = False

    print("Done loading the model")
    return model    




def predict(categories, image_path, model, use_gpu, topk):
    '''
        Predict the class (or classes) of an image using a trained deep learning model.
        
        Inputs:
        categories - The categories json file that maps the names of the flowers
        image_path - The path and file name to the image to predict
        use_gpu - Whether or not to use the gpu for inference
        topk - The top n restults of the inference
        
        Outputs:
        top_p - The probabilities for the predictions
        labels - The class labels for the predictions
    '''
    # Use the GPU if its available
    #device = torch.device('cuda' if use_gpu else 'cpu')
    model.to('cpu')
    
    #Switch the model to evaluation mode to turn off dropout
    model.eval()
    
    with torch.no_grad():
        # Implement the code to predict the class from an image file    
        # Processs the image
        image_tensor = helpers.ProcessImage.process_image(image_path)

        # We need a tensor for the model so change the image to a np.Array and then a tensor        
        image = torch.from_numpy(np.array([image_tensor])).float()
        #image_tensor = image_tensor.unsqueeze_(0)
        #image = Variable(image_tensor)
        image.to('cpu')

        # Use the model to make a prediction
        logps = model.forward(image)
        ps = torch.exp(logps)

        # Get the top 5 probabilities and index of classes. This is returned as a tensor of lists
        p, classes = ps.topk(topk)

        # Get the first items in the tensor list to get the list of probs and classes
        top_p = p.tolist()[0]
        top_classes = classes.tolist()[0]

        # Reverse the categories dictionary
        idx_to_class = {v:k for k, v in model.class_to_idx.items()}

        # Get the lables from the json file
        labels = []
        for c in top_classes:
            labels.append(categories[idx_to_class[c]])

    
        output = list(zip(top_p, labels))

        print("Top Probabilities and their Classes: {}".format(output))
        return top_p, labels


def sanity_check(cat_to_name, file_path, model, index):
    '''
        A sanity check that shows the flower image that we are trying to infer and the outcome of the prediction
        
        Inputs:
        cat_to_name - The categories json file that maps the names of the flowers
        file_path - The path and file name to the image to predict
        model - The model to use for inference
        index - The index of the flower we are trying to infer
        
        Outputs:
        Prints the image of the flower we are trying to predict and a bar graph of the predicted labels and their probabilities
    '''
    # Display an image along with the top classes
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
    predictions, classes = predict(cat_to_name, file_path, model, True, 5)

    # Make a bar graph
    # Create the axes for the bar graph
    fig.add_axes([.5, .1, .225, .225])

    # Get the range for the probabilities
    y_pos = np.arange(len(classes))

    # Plot as a horizontal bar graph
    plt.barh(y_pos, predictions, align='center', alpha=0.5)
    plt.yticks(y_pos, classes)
    plt.xlabel('probabilities')
    plt.show()
