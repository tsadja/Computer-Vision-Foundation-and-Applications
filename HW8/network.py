import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ### YOUR CODE HERE
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0) #output: 6x28x28
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #output: 6x14x14
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0) #output: 16x10x10
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #output: 16x5x5
        
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        ### END YOUR CODE

    def forward(self, x):
        ### YOUR CODE HERE
        x = self.relu(self.pool1(self.conv1(x)))
        x = self.relu(self.pool2(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        ### END YOUR CODE
        return x
    
    
class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        ### YOUR CODE HERE
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1) #output: 32x32x32
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1) #output: 32x32x32
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #output: 32x16x16
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) #output: 64x16x16
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1) #output: 64x16x16
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) #output: 64x8x8
        
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) #output: 128x8x8
        self.bn5 = nn.BatchNorm2d(128)
        
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1) #output: 128x8x8
        self.bn6 = nn.BatchNorm2d(128)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2) #output: 128x4x4
        
        self.fc1 = nn.Linear(128*4*4, 10)
        ### END YOUR CODE

    def forward(self, x):
        ### YOUR CODE HERE
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        x = self.relu(self.bn5(self.conv5(x)))
        
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.pool6(x)
        
        x = x.view(-1,128*4*4)
        x = self.fc1(x)
        ### END YOUR CODE
        return x
    

### NOT AS GOOD...
class NetPart6(nn.Module):
    def __init__(self):
        super().__init__()
        ### YOUR CODE HERE
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1) #output: 8x32x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #output: 8x16x16
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1) #output: 16x16x16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #output: 16x8x8
        
        self.fc1 = nn.Linear(16*8*8, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 10)
        ### END YOUR CODE

    def forward(self, x):
        ### YOUR CODE HERE
        x = self.relu(self.pool1(self.conv1(x)))
        x = self.relu(self.pool2(self.conv2(x)))
        x = x.view(-1,16*8*8)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        ### END YOUR CODE
        return x