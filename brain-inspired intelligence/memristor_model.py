import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
 
import torchvision.datasets as dsets
import torchvision.transforms as transforms
 
import matplotlib.pyplot as plt
import numpy as np
 
#matplotlib inline
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True
 
# 超参数
image_size = 28 # 图像分辨率28*28
num_classes = 10
num_epochs = 50
num_workers = 0
batch_size = 128
 
train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.Compose([transforms.ToTensor(),
                                                          transforms.RandomHorizontalFlip(), # 图像的一半概率翻转，一半不翻
                                                          transforms.Normalize(mean=0.5, std=0.5)
                                                         ]), 
                            download=True)
test_dataset = dsets.MNIST(root='./data',
                           train=False, 
                           transform=transforms.Compose([transforms.ToTensor(),
                                                          transforms.Normalize(mean=0.5, std=0.5)
                                                         ]),
                          download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
 
# 测试数据分成两部分，一部分作为校验数据，一部分作为测试数据
indices = range(len(test_dataset))
indices_val = indices[:4000] # 校验集
indices_test = indices[4000:] # 测试集
 
# 采样器随机从原始数据集中抽样数据，生成任意一个下标重排，从而利用下标来提取数据集中数据
sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)
val_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=sampler_val, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, sampler=sampler_test, num_workers=num_workers)


class ConvNet(nn.Module):
    # 构造函数，每当类ConvNet被具体化一个实例时就会被调用
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, stride=1, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1, stride=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1, stride=1, bias=True)
        self.fc1 = nn.Linear(image_size // 4 * image_size // 4 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(self.pool(x)))
        x = F.relu(self.conv3(x))
        #x = F.relu(self.conv4(self.pool(x)))
        #x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, image_size // 4 * image_size // 4 * 8)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=0.4)
        x = F.log_softmax(self.fc2(x), dim=1)
        return x
    
    # 提取特征图，返回前两层卷积层的特征图
    def retrieve_features(self, x):
        feature_map1 = F.relu(self.conv1(x))
        x = self.pool(feature_map1)
        feature_map2 = F.relu(self.conv2(x))
        return (feature_map1, feature_map2)

net = ConvNet()
# 采用多GPU训练
if torch.cuda.device_count() > 1:
    net = nn.DataParallel(net, device_ids=[0, 1])
net.to(device)
print(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=net.parameters(), lr=0.001)
#optimizer = optim.RMSprop(params=net.parameters(), lr=0.0001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
#optimizer = optim.NAdam(params=net.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004)

record = [] # 记录准确率等数值
#weights = [] # 每若干步就记录一次卷积核
 
def rightness(output, target):
    # torch.max函数返回输入张量给定维度上每行的最大值，并同时返回每个最大值的位置索引
    preds = output.data.max(dim=1, keepdim=True)[1] # keepdim保持输出的维度
    return preds.eq(target.data.view_as(preds)).sum(), len(target) # 返回数值为：（正确样例数，总样本数）
 
def add_noise_to_weights(mean, std, model):
    gassian_kernel = torch.distributions.Normal(mean, std)
    with torch.no_grad():
        for param in model.parameters():
            noise = gassian_kernel.sample(param.shape).to(param.device)
            noise = torch.exp(noise)
            param.mul_(noise)



best_acc = 0.0 # 最优准确率
best_epoch = 0 # 最优轮次
save_path = './ConvNet.pth'
x = []
train_acc_data = []
val_acc_data = []
loss_data = []
for epoch in range(num_epochs):
    # 训练
    train_rights = [] # 每轮次训练得到的准确数量
    net.train() # 把所有的dropout层打开
    # enumerate起到枚举器的作用，在train_loader循环时，枚举器会自动输出一个数字指示循环的次数，并记录在batch_idx中
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target) # data：手写数字图像，target：该图像对应标签
        output = net(data.to(device))
        loss = criterion(output, target.to(device))
        optimizer.zero_grad() # 清空所有被优化变量的梯度信息
        loss.backward()
        optimizer.step() # 进行单次优化，更新所有的参数
        train_rights.append(rightness(output, target.to(device)))
        
    # 校验
    # add_noise_to_weights(0,0.3,net)
    if epoch % 2 == 0:
        add_noise_to_weights(0,0.3,net)
    net.eval() # 把所有的dropout层关闭
    val_rights = [] # 每轮次校验得到的准确数量
    with torch.no_grad():
        for (data, target) in val_loader:
            data, target = Variable(data), Variable(target)
            output = net(data.to(device))
            val_rights.append(rightness(output, target.to(device)))
 
    train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
    val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
    train_acc = 1.0 * train_r[0] / train_r[1]
    val_acc = 1.0 * val_r[0] / val_r[1]
    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch + 1
        torch.save(net.state_dict(), save_path)
    x.append(epoch+1)
    train_acc_data.append(100 * train_acc)
    val_acc_data.append(100 * val_acc)
    loss_data.append(loss.item())
    print("[epoch {}] loss:{:.6f},train_acc：{:.2f}%，val_acc：{:.2f}%".format(
        epoch + 1, loss.item(),
        100 * train_acc, 100 * val_acc
    ))
    record.append((1 - train_acc, 1- val_acc))
    #weights.append([net.module.conv1.weight.data.clone(), net.module.conv1.bias.data.clone(), 
    #                net.module.conv2.weight.data.clone(), net.module.conv2.bias.data.clone()])
print("best epoch: %d,best val_acc: %.2f" %(best_epoch, best_acc * 100))

fig = plt.figure()
plt.plot(x, train_acc_data, x,val_acc_data)
plt.show()
fig.savefig("mem_deep_acc.png")

fig = plt.figure()
plt.plot(x, loss_data)
plt.show()
fig.savefig("mem_deep_loss.png")