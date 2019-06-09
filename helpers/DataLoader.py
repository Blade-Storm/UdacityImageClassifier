import torch
import torchvision
from torchvision import datasets, transforms, models

def load_image_data(batch_size):
    data_dir = './flowers'
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
    valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size)
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size)

    return train_dataloaders, valid_dataloaders, test_dataloaders, train_datasets
