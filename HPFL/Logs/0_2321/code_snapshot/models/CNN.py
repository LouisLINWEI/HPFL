import torch.nn as nn
import torch.nn.functional as F

class LeNet_Cifar10(nn.Module):

    '''Written by Yi Zhang'''
    def __init__(self):
        super(LeNet_Cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
		
class LeNet_Cifar100(nn.Module):

    '''Written by Yi Zhang'''
    def __init__(self):
        super(LeNet_Cifar100, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# follow the Federated Learning paper but still at low accuracy
class CNN_Cifar10(nn.Module):

    def __init__(self):
        super(CNN_Cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(1024, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        """input: minibatch x in_channels x iH x iW"""
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

# CNN for Mnist following the Federated Learning Paper
class CNN_Mnist(nn.Module):

    def __init__(self):
        super(CNN_Mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        """input: minibatch x in_channels x iH x iW"""
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)
# Mimic CNN_Mnist but low accuracy
# class CNN_Cifar10(nn.Module):
# 
#     def __init__(self):
#         super(CNN_Cifar10, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
#         self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
#         self.fc1 = nn.Linear(4096, 512)
#         self.fc2 = nn.Linear(512, 10)
# 
#     def forward(self, x):
#         """input: minibatch x in_channels x iH x iW"""
#         x = self.conv1(x)
#         x = F.max_pool2d(x, 2, 2)
#         x = self.conv2(x)
#         x = F.max_pool2d(x, 2, 2)
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return F.log_softmax(x, dim=1)

class CNN_KWS(nn.Module):

    def __init__(self):
        super(CNN_Kws, self).__init__()
        self.conv1 = nn.Conv2d(1, 28, kernel_size=(10, 4), stride=(1, 1))
        self.conv2 = nn.Conv2d(28, 30, kernel_size=(10, 4), stride=(1, 2))
        #self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1920, 16)
        self.fc2 = nn.Linear(16, 128)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(self.conv2(x))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)    
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# CNN model from Zhang Yi
class CNN_Mnist_Yi(nn.Module):

    def __init__(self):
        super(CNN_Mnist_Yi, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.fc1 = nn.Linear(4*4*64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.p = 0.1

    def forward(self, x):
        """input: minibatch x in_channels x iH x iW"""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 64)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.p)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
		
class CNN_HAR(nn.Module):
    def __init__(self):
        super(CNN_HAR, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=32, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 9)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=64 * 26, out_features=1000),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=1000, out_features=500),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=500, out_features=6)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.reshape(-1, 64 * 26)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = F.softmax(out, dim=1)
        return out

class CNN_EMnist(nn.Module):

    def __init__(self):
        super(CNN_EMnist, self).__init__()
        # Input data size: (batch_size, channels, height, width)
        # (16, 1, 28, 28)

        # inputs: (16, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # outputs: (16, 10, 24, 24)

        # inputs: (16, 10, 24, 24)
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        # outputs: (16, 10, 12, 12)

        # inputs: (16, 10, 12, 12)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # outputs: (16, 20, 8, 8)

        # inputs: (16, 1280)
        self.fc1 = nn.Linear(1280, 256)
        # outputs: (16, 256)

        # inputs: (16, 256)
        self.fc2 = nn.Linear(256, 62)
        # inputs: (16, 62)


    def forward(self, x):
        # print("Shape1: ", x.shape)
        x = self.conv1(x)
        x = F.relu(self.mp1(x))
        # print("Shape2: ", x.shape)
        x = F.relu(self.conv2(x))
        # print("Shape3: ", x.shape)
        x = x.view(-1, 1280)
        # print("Jape: ", x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


