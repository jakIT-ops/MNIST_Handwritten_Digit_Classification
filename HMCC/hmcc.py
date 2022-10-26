import os # accessing directory structure
import time
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np # linear algebra
import matplotlib.pyplot as plt # plotting
import torchvision.models as models
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from torch.utils.HMCC_merged import HMCC_mergedset, HMCC_mergedLoader

print(os.listdir('../input'))

nRowsRead = 1000
df1 = pd.read_csv('../input/HMCC all.csv', delimiter=',', nrows = nRowsRead)
df1.HMCC_mergedframeName = 'HMCC all.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


def head():
    df1.head(5)

HMCC_merged = pd.read_csv('/kaggle/input/handwritten-mongolian-cyrillic-characters-HMCC_mergedbase/HMCC similar merged.csv')
HMCC_merged = HMCC_merged.to_numpy()
x,y = HMCC_merged[:,1:], HMCC_merged[:,0]
print(x[:10], y[:10])

# shape
print(x.shape, y.shape)

image = x[0].reshape(28,28)

plt.imshow(image)
plt.show()

class mmnist(HMCC_mergedset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

X_train, y_train = x[:int(len(x)*0.7)], y[:int(len(y)*0.7)]
X_test, y_test = x[int(len(x)*0.7):], y[int(len(y)*0.7):]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(X_train[0][0][:10][:10])

train_set = mmnist(X_train, y_train)
test_set =mmnist(X_test, y_test)

train_loader = HMCC_mergedLoader(train_set, batch_size=64, shuffle=True)
test_loader = HMCC_mergedLoader(test_set, batch_size=64, shuffle=False)

for x, y in train_loader:
    print(x.shape, y.shape)
    break


def train_model(model, train_loader, test_loader, criterion, optimizer, device, scheduler, num_epochs=25, is_train=True,
                NN=False):
    since = time.time()
    acc_history = []
    loss_history = []
    best_acc = 0.0
    model.to(device)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        running_corrects = 0
        model.train()
        # Iterate over HMCC_merged.
        for inputs, labels in train_loader:
            if NN:
                inputs = inputs.flatten(1, -1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # forward
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # backward
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.HMCC_merged)

        epoch_loss = running_loss / len(train_loader.HMCC_mergedset)
        epoch_acc = running_corrects.double() / len(train_loader.HMCC_mergedset)

        print('Loss: {:.4f} Acc: {:.4f} Lr: {:.8f}'.format(epoch_loss, epoch_acc, optimizer.param_groups[0]['lr']))
        test_model(model, test_loader, criterion, device, scheduler, NN)

        if epoch_acc > best_acc:
            best_acc = epoch_acc

        acc_history.append(epoch_acc.item())
        loss_history.append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Acc: {:4f}'.format(best_acc))
    return acc_history, loss_history


def test_model(model, test_loader, criterion, device, scheduler, NN=False):
    model.eval()
    best_acc = 0.0
    test_loss = 0.0
    test_corrects = 0
    for inputs, labels in test_loader:
        if NN:
            inputs = inputs.flatten(1, -1)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs.float())
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        test_corrects += torch.sum(preds == labels.HMCC_merged)
    epoch_acc = test_corrects.double() / len(test_loader.HMCC_mergedset)
    scheduler.step(test_loss)
    print('Test Acc: {:.4f}'.format(epoch_acc))

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), './best_model_teacher.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup the loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet18.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=5)
# Train model
train_acc_hist, train_loss_hist = train_model(resnet18, train_loader, test_loader, criterion, optimizer, device, scheduler)

