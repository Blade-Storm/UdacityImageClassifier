# UdacityImageClassifier
Final Project of the Udacity AI Programming with Python Nanodegree.
An image classifier built with pytorch that predicts flower names. The classifier currently has training, validation, and test data for 102 flowers and uses transfer learning with either VGG19 or Densenet161 to train and infer with.

## Downloads and Installations
These instructions assume two things: 
1. You have git installed on your machine. If you don't you can click [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to do that
2. You have a package manager installed. These installation instructions use [conda](https://docs.conda.io/en/latest/) but [pip](https://pypi.org/project/pip/) works just as well to install the nessesary packages.


You can clone the repository with git:

`git clone https://github.com/Blade-Storm/UdacityImageClassifier.git`

Open a terminal and change directory into the repository

Install the following packages if you dont already have them:
1. Python 3.7: `conda install python==3.7` 
2. Numpy 1.16: `conda install numpy==1.16`
3. Pytorch 1.1: `conda install pytorch=1.1`
4. Torchvision 0.3: `conda install torchvision=0.3`
5. Matplotlib 3.0: `conda install matplotlib==3.0`


## How to use
First you will need to train a model on the training and validtion data and then load the trained model to make predictions using the test data.

### Training a model
To train a model (either VGG19 or Densenet161) you can run the `train.py` file like so:

`python train.py './flowers'` 

There are many arguments that can be used to 


### Using a trained model for inference


## Licence
