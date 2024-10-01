"""
In this project, we are going to train az AI algorithm to classify different types of fishes

Dataset Source: https://www.kaggle.com/datasets/markdaniellampa/fish-dataset

"""

import os
import torch
import torch.utils
from torchsummary import summary
import torchvision #It is for utils functions and it includes datasets
import torchvision.transforms as transforms #It is useful for image transformation
import torch.nn as nn #It creates neural networks
import torch.nn.functional as F #Functional api for layers
from torch.utils.data import Subset
import torch.optim as optim #for optimalization
import matplotlib.pyplot as plt #This is for visualization
import numpy as np #for basic array operations
import copy
import random

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()

        # First convolutional layer with BatchNorm and Dropout
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )

        # Second convolutional layer with BatchNorm and Dropout
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )

        # Third convolutional layer with BatchNorm and Dropout
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.25)
        )

        # Fully connected layers with Dropout
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.flatten(out, 1)  # flatten all dimensions except batch
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out




def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)



    ################Processing of the Dataset

    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=(5, 50), translate=(0.1, 0.3), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) #It Transforms our data for us


    #Here we load the Fish image train dataset, to work with
    fishDatasettrain = torchvision.datasets.ImageFolder(root="./FishImgDataset/train", transform=transform)
    fishDatasetval = torchvision.datasets.ImageFolder(root="./FishImgDataset/val", transform=transform)
    #Define how many samples per batch to load - It means we load x number of picture to the network at once
    batch_size: int = 16


    ################# It has meaning when we have not pre processed the data into tran-test-val folders, because in that case we would be doing it with this
    #Generate  a list of indices from 0 to len(fishDataset) -1
    indices = list(range(0, len(fishDatasettrain)))
    #Here we set the random seed so we can reproducate the end
    random.seed(42)
    random.shuffle(indices) # We shuffle them so the images are not in the original order
    #####################################


    # Create a DataLoader for the subset training dataset
    #trainloader = torch.utils.data.DataLoader(fishDatasettrain, batch_size = batch_size, shuffle = True, num_workers = 0)
    #valloader = torch.utils.data.DataLoader(fishDatasetval, batch_size = batch_size, shuffle = False, num_workers = 0)

    #Here we are going to use a different transform for the test dataset
    test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
    fishDatasettest = torchvision.datasets.ImageFolder(root="./FishImgDataset/test", transform=test_transform)   
    #testloader = torch.utils.data.DataLoader(fishDatasettest, batch_size = batch_size, shuffle = False, num_workers = 0)
    
    testloader = iter(fishDatasettest)
    trainloader = iter(fishDatasettrain)
    valloader = iter(fishDatasetval)

    #Get the class names into a tuple
    root = "./FishImgDataset/test"
    classes = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
    

    #Lets see some of the attributes
    print("Number of Images in the Training set:\n",len(fishDatasettrain))
    print("Number of Images in the Validation set:\n",len(fishDatasetval))
    #print("Test set shape (Number of Images, Height, Width, Number of channels):\n", testset.data.shape)
    print("Available classes: ", classes)
    
    ################### Create and train the Convolutional Neural Network (CNN)
    
    #Here we create the convolutional network with x len and than we print out the network type
    convnet = ConvNet(len(classes))
    print(convnet)


    ############### Add the loss function and the optimizers
    """
    SGD - Stochastic Gradient Descent and the Adaptive Movement optimizer are some of the most well-known optimizers
    """
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.SGD(convnet.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(convnet.parameters(), lr=0.01, betas=[0.9, 0.99])
    

    n_epochs = 10
    hist_train=[]
    hist_valid=[]
    best_loss=float('inf')
    best_model_wts = copy.deepcopy(convnet.state_dict())
    early_stop_tolerant_count=0
    early_stop_tolerant=10





    for epoch in range(n_epochs):  # loop over the dataset multiple times
        train_loss = 0.0
        convnet.train()
        for i, (inputs, labels) in enumerate(trainloader):
            #inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + loss + backward + optimize
            outputs = convnet(inputs)
            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(trainloader)  # Divide by number of batches
        hist_train.append(train_loss)

        # Validation
        convnet.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = convnet(inputs)
                loss = criteria(outputs, labels)
                valid_loss += loss.item()

        valid_loss /= len(valloader)  # Divide by number of batches
        hist_valid.append(valid_loss)

        early_stop_tolerant_count += 1
        if valid_loss < best_loss:
            early_stop_tolerant_count = 0
            best_loss = valid_loss
            best_model_wts = copy.deepcopy(convnet.state_dict())

        if early_stop_tolerant_count >= early_stop_tolerant:
            break

        print(f"Epoch {epoch:04d}: train loss {train_loss:.4f}, valid loss {valid_loss:.4f}")

    print('Finished Training')

if __name__ == "__main__":
    main()