import jittor as jt  # 将 jittor 引入
from jittor import nn,transform, Module  # 引入相关的模块
from jittor.dataset import Dataset
import numpy as np
import sys, os
import random
import math 
import pygmtools
# import cv2
from PIL import Image
from jittor.models import Resnet18
from jittor import init
if jt.has_cuda:
    jt.flags.use_cuda = 1 # jt.flags.use_cuda 表示是否使用 gpu 训练。
# 如果 jt.flags.use_cuda=1，表示使用GPU训练 如果 jt.flags.use_cuda = 0 表示使用 CPU
from jittor.dataset.cifar import CIFAR10
pygmtools.BACKEND = 'jittor'
#the new data has a more dimension,so we need to set a new dataset
class NewSet(Dataset):
    def __init__(self,data,targets,transform=None):
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = [Image.fromarray(img[0]),Image.fromarray(img[1]),Image.fromarray(img[2]),Image.fromarray(img[3])]
        if self.transform is not None:
            img = np.array([self.transform(img[0]),self.transform(img[1]),self.transform(img[2]),self.transform(img[3])])
        return img,target

    def __len__(self):
        return len(self.data)

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
        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def execute(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = nn.relu(out)
        return out

class ResNet18(Module):
    def __init__(self, BasicBlock, num_classes=10,pretrain = False) -> None:
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
        self.fc1 = nn.Linear(2048,4096)
        self.fc2 = nn.Linear(4096,16)
        self.relu = nn.ReLU()
        self.pretrain = pretrain

    #这个函数主要是用来，重复同一个残差块
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def pretrain_true(self):
        self.pretrain = True
    def execute(self, x):
        if self.pretrain == False:
            out = []
            for i in range(4):
                output = self.conv1(x[:,i])
                output = self.conv2(output)
                output = self.conv3(output)
                output = self.conv4(output)
                output = self.conv5(output)
                output = self.avgpool(output)
                output = output.reshape(x.shape[0], -1)
                out.append(output)
            out = jt.concat(out,dim = 1)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out).view(-1,4,4)
            out = pygmtools.sinkhorn(out)
        if self.pretrain == True:
            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.conv5(out)
            out = self.avgpool(out)
            out = out.reshape(x.shape[0], -1)
            out = self.fc(out)
        return out

def train(model, train_loader, optimizer, epoch, losses, losses_idx):
    model.train()
    lens = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        outputs = model(inputs)
        loss = nn.mse_loss(outputs, targets) * 16
        optimizer.step(loss)
        losses.append(loss.numpy()[0])
        losses_idx.append(epoch * lens + batch_idx)
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader) ,
                100. * batch_idx * batch_size/ len(train_loader), loss.numpy()[0]))

def val(model, val_loader,epoch):
    model.eval()    
    total_acc = 0
    total_num = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(np.array(outputs), axis=2)
        targets = np.argmax(np.array(targets),axis=2)
        #acc = (targets[:,0]==pred[:,0])+(targets[:,1]==pred[:,1])+(targets[:,2]==pred[:,2])+(targets[:,3]==pred[:,3])
        #acc = np.sum(acc)
        #acc = np.sum((targets[:,0]==pred[:,0]),(targets[:,1]==pred[:,1]),(targets[:,2]==pred[:,2]),(targets[:,3]==pred[:,3]))
        #acc = np.sum((targets[:,0]==pred[:,0]) and (targets[:,1]==pred[:,1]) and (targets[:,2]==pred[:,2]) and (targets[:,3]==pred[:,3]))
        # if (targets[:,0]==pred[:,0]) and (targets[:,1]==pred[:,1]) and (targets[:,2]==pred[:,2]) and (targets[:,3]==pred[:,3]):
        #     acc = 1
        # else:
        #     acc = 0
        acc = (targets[:,0]==pred[:,0])&(targets[:,1]==pred[:,1])&(targets[:,2]==pred[:,2])&(targets[:,3]==pred[:,3])
        acc = np.sum(acc)
        total_acc += acc
        total_num += batch_size
    print('test_acc:',total_acc/total_num)
    f = open('task3_acc.txt','a')
    f.write(str(epoch)+":"+str(total_acc/total_num)+'\n')
    f.close()

def transform_the_data_to_dic(file):
    import pickle
    with open(file, 'rb') as f:
        dic = pickle.load(f, encoding='bytes')
    return dic

batch_size = 128
learning_rate = 0.001
epochs = 50
losses = []
losses_idx = []

train_data = []
train_target = []
for i in range(1,6):
    train_dataset = transform_the_data_to_dic('/home/dl2022/yyr/cifar-10-batches-py/data_batch_'+str(i))
    train_data.append(train_dataset[b'data'])
    train_target.extend(train_dataset[b'labels'])
train_data = np.vstack(train_data).reshape(-1, 3, 32, 32)

val_data = []
val_target = []
val_dataset = transform_the_data_to_dic('/home/dl2022/yyr/cifar-10-batches-py/test_batch')
val_data.append(val_dataset[b'data'])
val_target.extend(val_dataset[b'labels'])
val_data = np.vstack(val_data).reshape(-1, 3, 32, 32)

train_loader = NewSet([],[],transform=transform.ToTensor())
train_loader.set_attrs(batch_size=batch_size,shuffle = True)
# cut the data
for index,data in enumerate(train_data):
    data = np.array([data[:,0:16,0:16],data[:,0:16,16:32],data[:,16:32,0:16],data[:,16:32,16:32]])
    order = np.array([0,1,2,3])
    target = jt.zeros((4,4))
    np.random.shuffle(order)
    target[0,order[0]] = 1
    target[1,order[1]] = 1
    target[2,order[2]] = 1
    target[3,order[3]] = 1
    data = data[order]
    train_loader.data.append(data.transpose(0,2,3,1))
    train_loader.targets.append(target)

val_loader = NewSet([],[],transform=transform.ToTensor()) 
for idx,data in enumerate(val_data):
    data = np.array([data[:,0:16,0:16],data[:,0:16,16:32],data[:,16:32,0:16],data[:,16:32,16:32]])
    order = np.array([0,1,2,3])
    target = jt.zeros((4,4))
    np.random.shuffle(order)
    target[0,order[0]] = 1
    target[1,order[1]] = 1
    target[2,order[2]] = 1
    target[3,order[3]] = 1
    data = data[order]
    val_loader.data.append(data.transpose(0,2,3,1))
    val_loader.targets.append(target)

# model = ResNet18(BasicBlock)
# optimizer = nn.Adam(model.parameters(), learning_rate)
# for epoch in range(epochs):
#     print(epoch,":")
#     train(model, train_loader, optimizer, epoch, losses, losses_idx)
#     val(model, val_loader,epoch)

# model_path = './model_hw3.pkl'
# model.save(model_path)
# filename = open('loss_task3.txt', 'w')  
# for loss in losses:  
#      filename.write(str(loss)+'\n') 
# filename.close() 

model = ResNet18(BasicBlock,10,True)
model.load_parameters(jt.load('./model_hw3.pkl'))

pretrain_train_loader = CIFAR10(train = True,transform=transform.ToTensor())
pretrain_train_loader.set_attrs(batch_size = batch_size,shuffle = True)
pretrain_val_loader = CIFAR10(root='./',train = False,transform=transform.ToTensor())
optimizer = nn.Adam(model.parameters(), learning_rate)
def pretrain_train(model, train_loader, optimizer, epoch, losses, losses_idx):
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

def pretrain_val(model, val_loader, epoch):
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
for epoch in range(epochs):
    print(epoch,":")
    pretrain_train(model, pretrain_train_loader, optimizer, epoch, losses, losses_idx)
    pretrain_val(model, pretrain_val_loader,epoch)

