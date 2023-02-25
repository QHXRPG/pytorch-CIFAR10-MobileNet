from torchvision.datasets import CIFAR10
import torch
import torch.nn as nn
import numpy as np
import time
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
data = CIFAR10(root="./")
x = data.data
y = data.targets
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8)
x_train,x_test,y_train,y_test = torch.tensor(x_train,dtype=torch.float32),\
                                torch.tensor(x_test,dtype=torch.float32),\
                                torch.tensor(y_train,dtype=torch.long),\
                                torch.tensor(y_test,dtype=torch.long)
x_train,x_test = x_train.permute(0,3,1,2),x_test.permute(0,3,1,2)
train_data = Data.TensorDataset(x_train,y_train)
data_loader = Data.DataLoader(dataset=train_data,batch_size=100)

#%%
#深度可分离卷积
class DP_Conv(nn.Module):
    def __init__(self,indim,outdim,ksize,stride,padding):
        super(DP_Conv, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels=indim,out_channels=indim,kernel_size=ksize,
                                        stride=stride,padding=padding,groups=indim)
        self.point_conv = nn.Conv2d(in_channels=indim,out_channels=outdim,kernel_size=1)
        self.BN_1 = nn.BatchNorm2d(indim)
        self.relu=nn.ReLU()
        self.BN_2 = nn.BatchNorm2d(outdim)
    def forward(self,x:torch.Tensor):
        x = self.depthwise_conv(x)
        x = self.BN_1(x)
        x = self.relu(x)
        x = self.point_conv(x)
        return self.relu(x)

class Mobilenet(nn.Module):
    def __init__(self):
        super(Mobilenet, self).__init__()
        self.model1 = nn.Sequential(nn.Conv2d(3,32,3,2,1),  #16
                                   DP_Conv(32,32,3,1,1),  #16
                                   DP_Conv(32,64,3,2,1),#8
                                   DP_Conv(64,128,3,1,1),#8
                                   DP_Conv(128,256,3,2,1),#4
                                   DP_Conv(256,256,3,1,1),#4
                                   DP_Conv(256,512,3,2,1))#2
        self.model2 = nn.Sequential(DP_Conv(512,512,3,1,1),#2
                                    DP_Conv(512,512,3,1,1),#2
                                    DP_Conv(512,512,3,1,1),#2
                                    DP_Conv(512,512,3,1,1),#2
                                    DP_Conv(512,512,3,1,1),)#2
        self.pool = nn.AvgPool2d(2,2)#
        self.classfier = nn.Sequential(nn.Linear(512,100),#
                                       nn.Sigmoid(),
                                       nn.Linear(100,10))#
    def forward(self,x):
        x = self.model1(x)
        x = self.model2(x)
        x = self.pool(x)
        x=x.view(-1,512)
        x = self.classfier(x)
        return x
model = Mobilenet()
opt = torch.optim.Adam(model.parameters(),lr=0.05)
loss = nn.CrossEntropyLoss()

for i in range(50):
    for j,(x,y) in enumerate(data_loader):
        y_p = model(x)
        l = loss(y_p,y)
        opt.zero_grad()
        l.backward()
        opt.step()
    print(l.item())