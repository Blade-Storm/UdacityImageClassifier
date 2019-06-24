# UdacityImageClassifier
Final Project of the Udacity AI Programming with Python Nanodegree.
An image classifier built with pytorch that predicts the names of flowers. The classifier currently has training, validation, and test data for 102 flowers and uses transfer learning with either VGG19 or Densenet161 to train and infer with.

## Downloads and Installations
These instructions assume two things: 
1. You have git installed on your machine. If you don't you can click [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to set that up.
2. You have a package manager installed. These installation instructions use [conda](https://docs.conda.io/en/latest/) but [pip](https://pypi.org/project/pip/) works just as well to install the nessesary packages.


You can clone the repository with git:

`git clone https://github.com/Blade-Storm/UdacityImageClassifier.git`

Open a terminal and change directory into the repository

`cd ../UdacityImageClassifier`

Install the following packages if you dont already have them:
1. Python 3.7: `conda install python==3.7` 
2. Numpy 1.16: `conda install numpy==1.16`
3. Pytorch 1.1: `conda install pytorch=1.1`
4. Torchvision 0.3: `conda install torchvision=0.3`
5. Matplotlib 3.0: `conda install matplotlib==3.0`


## How to use
First, you will need to train a model on the training and validtion data and then load the trained model to make predictions using the test data.

### Training a model
Once you are in the working directory of the repository to train a model (either VGG19 or Densenet161) you can run the `train.py` file like so:

`python train.py './flowers'` 

The `./flowers` file path is the data directory that includes the train and valid folders that contain the training and validation images. By default the VGG19 model will be used along with other pre-set hyperparameters. Here is a list of all of the arguments you can use in training along with some examples:

1. data_directory: 
- The relative path to the image files to train on. It should include two folders: 'train' and 'valid' for training.
2. --save_dir: 
- The relative path to the directory you wish to save the trained model's checkpoint to. This file path must exist prior to training
- Default is the current directory '/'
3. --arch: 
- The architacture you wish to use for the training. This can either be 'vgg19' or 'densenet161'
- Default is vgg19
4. --learning_rate: 
- The learning rate for the training process
- Default is 0.001
5. --hidden_units: 
- The number of units used in the hidden layer. NOTE: There is only one hidden layer used in this project and thus only one hidden_unit required for training
- Default is 512
6. --epochs: 
- The amount of epochs to train for.
- Default is 15
7. --batch_size: 
- The size of the image batches you want to use for training.
- Default is 32
8. --gpu: 
- A boolean value for if you would like to use the GPU for training.
- Default is false

Example use of all arguments:

`python train.py './flowers' --save_dir './checkpoints' --arch 'densenet161' --learning_rate 0.01 --hidden_units 500 --epochs 10 --batch_size 64 --gpu`


### Using a trained model for inference


## Licence
