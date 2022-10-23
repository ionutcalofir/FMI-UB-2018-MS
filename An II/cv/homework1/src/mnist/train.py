import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from scipy.special import softmax
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from keras.datasets import mnist
from torchvision import models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model():
    model = Net()

    return model

class MNISTDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.transform = self.get_transform()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        img = np.copy(self.x[idx])
        label = self.y[idx]

        img = self.transform(img)

        return img, int(label)

    def get_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
        ])

def predict_number_box(image):
    net = Net()
    PATH = 'mnist/mnist_net.pth'
    net.load_state_dict(torch.load(PATH))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    to_tensor = transforms.ToTensor()
    image = to_tensor(image)
    image = image.to(device)
    image = image.unsqueeze(0)

    output = net(image)
    result = softmax(output[0].detach().cpu().numpy())
    number = np.argmax(result)

    return number

def predict_number_grade(image):
    net = Net()
    PATH = 'mnist/mnist_net.pth'
    net.load_state_dict(torch.load(PATH))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    to_tensor = transforms.ToTensor()
    image = to_tensor(image)
    image = image.to(device)
    image = image.unsqueeze(0)

    output = net(image)
    result = softmax(output[0].detach().cpu().numpy())
    prob = np.max(result)
    number = np.argmax(result)

    return number, prob

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    model = load_model()
    ds = MNISTDataset(x_train, y_train)
    dataloader = DataLoader(ds, batch_size=32, shuffle=True)
    ds_test = MNISTDataset(x_test, y_test)
    dataloader_test = DataLoader(ds_test, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model = model.to(device)
    for epoch in range(3):
        running_loss = 0.
        for i_batch, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i_batch % 200 == 199:    # print every 2000 mini-batches
                print('[{}, {}] loss: {}'.format(epoch + 1, i_batch + 1, running_loss / 200))
                running_loss = 0.0


        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader_test:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Acc on test set at ep {}: {}'.format(epoch + 1, 100 * correct / total))

    PATH = './mnist_net.pth'
    torch.save(model.state_dict(), PATH)

if __name__ == '__main__':
    train()
