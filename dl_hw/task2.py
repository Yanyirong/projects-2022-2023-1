
import jittor as jt  # 将 jittor 引入
from jittor import nn,transform, Module  # 引入相关的模块
import numpy as np
import sys, os
import random
import math 
from jittor.models import Resnet18
from jittor import init
if jt.has_cuda:
    jt.flags.use_cuda = 1 # jt.flags.use_cuda 表示是否使用 gpu 训练。
from PIL import Image
# 如果 jt.flags.use_cuda=1，表示使用GPU训练 如果 jt.flags.use_cuda = 0 表示使用 CPU
from jittor.dataset.cifar import CIFAR10
# #import matplotlib.pyplot as plt
# #import pylab as pl # 用于绘制 Loss 曲线 和 MNIST 数据

class BasicBlock(Module):
    def __init__(self,in_channels,out_channels,stride=[1,1],padding=1) -> None:
        super(BasicBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), 
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def execute(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = nn.relu(out)
        return out

# # 采用bn的网络中，卷积层的输出并不加偏置
class ResNet18(Module):
    def __init__(self, BasicBlock, num_classes=10) -> None:
        super(ResNet18, self).__init__()
        self.in_channels = 64
        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock,64,[[1,1],[1,1]])
        # conv3_x
        self.conv3 = self._make_layer(BasicBlock,128,[[2,1],[1,1]])
        # conv4_x
        self.conv4 = self._make_layer(BasicBlock,256,[[2,1],[1,1]])
        # conv5_x
        self.conv5 = self._make_layer(BasicBlock,512,[[2,1],[1,1]])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    #这个函数主要是用来，重复同一个残差块
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    def execute(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        # out = F.avg_pool2d(out,7)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out

def train(model, train_loader, optimizer, epoch, losses, losses_idx):
    model.train()
    lens = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        outputs = model(inputs)
        loss = nn.cross_entropy_loss(outputs, targets) 
        optimizer.step(loss)
        losses.append(loss.numpy()[0])
        losses_idx.append(epoch * lens + batch_idx)
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader) ,
                100. * batch_idx * batch_size/ len(train_loader), loss.numpy()[0]))

def val(model, val_loader, epoch):
    model.eval()
    total_acc = 0
    total_num = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.numpy(), axis=1)
        acc = np.sum(targets.numpy()==pred)
        total_acc += acc
        total_num += batch_size
        acc = acc / batch_size  	
    print('Test Acc =', total_acc / total_num)
    # f = open('acc2_2_1_100_64.txt','a')
    # f.write(str(epoch)+":"+str(total_acc/total_num)+'\n')
    # f.close()
    
batch_size = 128
learning_rate = 0.001
epochs = 50
losses = []
losses_idx = []

train_loader = CIFAR10(train = True,transform=transform.ToTensor())
# data = train_loader.data
# targets = train_loader.targets
# train_loader.data = []
# train_loader.targets = []
# l0 =l1 = l2 = l3 = l4 =0

# for idx,label in enumerate(targets) :
#     if label >= 5:
#         train_loader.targets.append(label)
#         train_loader.data.append(data[idx])
#     elif label == 4:
#         if l4 % 10 ==0:
#             train_loader.targets.append(label)
#             train_loader.data.append(data[idx])
#         # if l4 % 10 ==0:
#         #     for i in range(10):
#         #         flip = transform.RandomHorizontalFlip()
#         #         image = Image.fromarray(np.uint8(data[idx]))
#         #         over_img = flip(image)
#         #         crop = transform.RandomCrop(20)
#         #         randum = random.random()
#         #         if randum < 0.5:
#         #             over_img = crop(over_img)
#         #             over_img = transform.resize(over_img,32)
#         #         image = np.uint8(over_img)
#         #         train_loader.data.append(image)
#         #         train_loader.targets.append(label)
#         l4 += 1
#     elif label == 3:
#         if l3 % 10 ==0:
#             train_loader.targets.append(label)
#             train_loader.data.append(data[idx])
#         # if l3 % 10 ==0:
#         #     for i in range(10):
#         #         flip = transform.RandomHorizontalFlip()
#         #         image = Image.fromarray(np.uint8(data[idx]))
#         #         over_img = flip(image)
#         #         crop = transform.RandomCrop(20)
#         #         randum = random.random()
#         #         if randum < 0.5:
#         #             over_img = crop(over_img)
#         #             over_img = transform.resize(over_img,32)
#         #         image = np.uint8(over_img)
#         #         train_loader.data.append(image)
#         #         train_loader.targets.append(label)
#         l3 += 1
#     elif label == 2:
#         if l2 % 10 ==0:
#             train_loader.targets.append(label)
#             train_loader.data.append(data[idx])
#         # if l2 % 10 ==0:
#         #     for i in range(10):
#         #         flip = transform.RandomHorizontalFlip()
#         #         image = Image.fromarray(np.uint8(data[idx]))
#         #         over_img = flip(image)
#         #         crop = transform.RandomCrop(20)
#         #         randum = random.random()
#         #         if randum < 0.5:
#         #             over_img = crop(over_img)
#         #             over_img = transform.resize(over_img,32)
#         #         image = np.uint8(over_img)
#         #         train_loader.data.append(image)
#         #         train_loader.targets.append(label)
#         l2 += 1
#     elif label == 1:
#         if l1 % 10 ==0:
#             train_loader.targets.append(label)
#             train_loader.data.append(data[idx])
#         # if l1 % 10 ==0:
#         #     for i in range(10):
#         #         flip = transform.RandomHorizontalFlip()
#         #         image = Image.fromarray(np.uint8(data[idx]))
#         #         over_img = flip(image)
#         #         crop = transform.RandomCrop(20)
#         #         randum = random.random()
#         #         if randum < 0.5:
#         #             over_img = crop(over_img)
#         #             over_img = transform.resize(over_img,32)
#         #         image = np.uint8(over_img)
#         #         train_loader.data.append(image)
#         #         train_loader.targets.append(label)
#         l1 += 1
#     else :
#         if l0 % 10 ==0:
#             train_loader.targets.append(label)
#             train_loader.data.append(data[idx])
#         # if l0 % 10 ==0:
#         #     for i in range(10):
#         #         flip = transform.RandomHorizontalFlip()
#         #         image = Image.fromarray(np.uint8(data[idx]))
#         #         over_img = flip(image)
#         #         crop = transform.RandomCrop(20)
#         #         randum = random.random()
#         #         if randum < 0.5:
#         #             over_img = crop(over_img)
#         #             over_img = transform.resize(over_img,32)
#         #         image = np.uint8(over_img)
#         #         train_loader.data.append(image)
#         #         train_loader.targets.append(label)
#         l0 += 1


train_loader.set_attrs(batch_size = batch_size,shuffle = True)
val_loader = CIFAR10(root='./',train = False,transform=transform.ToTensor())
model = ResNet18(BasicBlock)
optimizer = nn.Adam(model.parameters(), learning_rate)
for epoch in range(epochs):
    print(epoch,":")
    train(model, train_loader, optimizer, epoch, losses, losses_idx)
    val(model, val_loader,epoch)

#model_path = './Resnet_model_base_Resnet18.pkl'
#model.save(model_path)
#filename = open('Resnet_model_base_loss.txt', 'w')  
# for loss in losses:  
#      filename.write(str(loss)+'\n') 
# filename.close() 