#Create a linear neural network pytorch model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import torch
import torch.utils.data as data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cells = 100
        self.model = nn.Sequential(
            nn.Linear(self.cells, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100,4)
        )
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss.to(self.device)

    def forward(self, x):
        return self.model(x)

    def train(self, x, y):
        x = torch.from_numpy(x).float().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = self.loss(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        return self.model(x).detach().cpu().numpy()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

#Create a pytorch custom dataset with the data from a text file for each class
#The data is in a textfile for each class, each line is a sample
#The first value is the label, the rest are the features


class Dataset(data.Dataset):
    def __init__(self, path1, path2, path3, path4):
        self.x = []
        self.y = []
        numclass = 0
        data = []
        data2 = []
        for path in [path1, path2, path3, path4]:
            Label = True
            with open(path, 'r') as f:
                for line in f:
                    if Label == True :
                        numclass = int(line)
                        print(numclass)
                        Label  = False
                    else :
                        linedata = line.split(' ')
                        for i in range(len(linedata)):
                            if i != 0 and i != 40:
                                data.append(linedata[i][:len(linedata[i])//2])
                                data.append(linedata[i][len(linedata[i])//2:])
                                print(data)
                            else:
                                data.append(linedata[i])
                
                        self.y.append(numclass)
                        self.x.append([float(x) for x in data])
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]



#load the dataset in a dataloader
#the dataset is split in 80% train and 20% test
#the train dataset is split in 80% train and 20% validation
#the dataloaders are shuffled and batched

def load_data(path1, path2, path3, path4, batch_size):
    dataset = Dataset(path1, path2, path3, path4)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])
    train_size = int(0.8 * len(train_dataset))
    validation_size = len(train_dataset) - train_size
    train_dataset, validation_dataset = data.random_split(train_dataset, [train_size, validation_size])
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, validation_loader, test_loader

#Train the model for a number of epochs
#The model is saved after each epoch if the validation loss is lower than the previous epoch
#The model is saved after the last epoch

def train(model, train_loader, validation_loader, epochs):
    best_loss = 100000
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.numpy()
            y = np.zeros((len(y), 4))
            y[np.arange(len(y)), y] = 1
            loss = model.train(x, y)
        model.eval()
        with torch.no_grad():
            for x, y in validation_loader:
                x = x.numpy()
                y = np.zeros((len(y), 4))
                y[np.arange(len(y)), y] = 1
                loss = model.loss(model.predict(x), y).item()
                if loss < best_loss:
                    best_loss = loss
                    model.save('model.pth')
        print('Epoch: {}, loss: {}'.format(epoch, loss))
    model.save('model.pth')

#Test the model on the test dataset
#The model is loaded from the last epoch

def test(model, test_loader):
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.numpy()
            y = np.zeros((len(y), 4))
            y[np.arange(len(y)), y] = 1
            loss = model.loss(model.predict(x), y).item()
            print('Test loss: {}'.format(loss))



#Train and test the model with tqdm 
#The progress bar shows the loss for each epoch

from tqdm import tqdm   
import time

def train_tqdm(model, train_loader, validation_loader, epochs):
    best_loss = 100000
    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(train_loader):
            x = x.numpy()
            y = np.zeros((len(y), 4))
            y[np.arange(len(y)), y] = 1
            loss = model.train(x, y)
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(validation_loader):
                x = x.numpy()
                y = np.zeros((len(y), 4))
                y[np.arange(len(y)), y] = 1
                loss = model.loss(model.predict(x), y).item()
                if loss < best_loss:
                    best_loss = loss
                    model.save('model.pth')
        print('Epoch: {}, loss: {}'.format(epoch, loss))
    model.save('model.pth')


if __name__ == '__main__':
    train_loader, validation_loader, test_loader = load_data('/Users/hugo/ArcticProject/ArcticGlove/position[NOMOVE].txt', '/Users/hugo/ArcticProject/ArcticGlove/position[SLIDE-LEFTtoRIGHT].txt', '/Users/hugo/ArcticProject/ArcticGlove/position[SLIDE-RIGHTtoLEFT]].txt', '/Users/hugo/ArcticProject/ArcticGlove/position[3SECTOUCH2FINGERS].txt', 64)
    model = Net(6, 4, 64, 0.01)
    train_tqdm(model, train_loader, validation_loader, 10)
    test(model, test_loader)

