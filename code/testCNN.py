import glob

import numpy as np
import os
import utils
import time

import digitFeatures
import linearModel
from CNNModel import CNN
from torch import optim
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import pdb

# There are three versions of MNIST dataset
dataTypes = ['digits-normal.mat', 'digits-scaled.mat', 'digits-jitter.mat']
# dataTypes = ['digits-jitter.mat']

# Accuracy placeholder
accuracy = np.zeros(len(dataTypes))
trainSet = 1
testSet = 3

criterion = nn.CrossEntropyLoss()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.images = x
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


for i in range(len(dataTypes)):
    dataType = dataTypes[i]

    #Load data
    path = os.path.join('..', 'data', dataType)
    data = utils.loadmat(path)
    print('+++ Loading dataset: {} ({} images)'.format(dataType, data['x'].shape[2]))

    # Organize into numImages x numChannels x width x height
    x = data['x'].transpose([2,0,1])
    x = np.reshape(x,[x.shape[0], 1, x.shape[1], x.shape[2]])
    y = data['y']
    # Convert data into torch tensors
    x = torch.tensor(x).float()
    y = torch.tensor(y).long() # Labels are categorical

    x = x.to(device)
    y = y.to(device)

    # Define the model
    model = CNN()
    model.to(device)
    model.train()
    # print(model)

    # Define loss function and optimizer
    # Implement this

    file = glob.glob(os.getcwd() + f"/CNN*{dataType}")

    if file:
        model.load_state_dict(torch.load(file[0]))
    else:
        # Start training
        xTrain = x[data['set']==trainSet,:,:,:]
        yTrain = y[data['set']==trainSet]

        # Loop over training data in some batches
        # Implement this

        batch_size = 32
        epochs = 200
        num_batches = xTrain.shape[0] // batch_size
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        for epoch in range(epochs):
            train_dataloader_iter = iter(DataLoader(CustomDataset(xTrain, yTrain), batch_size=batch_size, shuffle=True))

            for step in range(num_batches):
                image, labels = next(train_dataloader_iter)
                output = model(image)

                loss = criterion(output, labels)

                loss.backward()
                optimizer.step()

                optimizer.zero_grad()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch[{epoch + 1}/{epochs}]: loss = {loss.item():.8f}")

    # Test model
    xTest = x[data['set']==testSet,:,:,:]
    yTest = y[data['set']==testSet]

    model.eval() # Set this to evaluation mode
    # Loop over xTest and compute labels (implement this)
    with torch.no_grad():
        output = model(xTest)
        yPred = torch.max(output, 1)[1].cpu().numpy()

    # Map it back to numpy to use our functions
    yTest = yTest.cpu().numpy()
    (acc, conf) = utils.evaluateLabels(yTest, yPred, False)
    print('Accuracy [testSet={}] {:.2f} %\n'.format(testSet, acc*100))
    accuracy[i] = acc

    if file:
        best_acc = file[0].split('_')[2]
        if float(best_acc) < acc:
            print(f"Got better model for {dataType} !")
            torch.save(model.state_dict(), os.getcwd() + f"/CNN_{acc}_{dataType}")
    else:
        torch.save(model.state_dict(), os.getcwd() + f"/CNN_{acc}_{dataType}")

# Print the results in a table
print('+++ Accuracy Table [trainSet={}, testSet={}]'.format(trainSet, testSet))
print('--------------------------------------------------')
print('dataset\t\t\t', end="")
print('{}\t'.format('cnn'), end="")
print()
print('--------------------------------------------------')
for i in range(len(dataTypes)):
    print('{}\t'.format(dataTypes[i]), end="")
    print('{:.2f}\t'.format(accuracy[i]*100))

# Once you have optimized the hyperparameters, you can report test accuracy
# by setting testSet=3. Do not optimize your hyperparameters on the
# test set. That would be cheating.
