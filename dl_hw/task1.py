import jittor as jt
import numpy as np
from jittor import nn, Module #, init
import matplotlib.pyplot as plt

class Model(Module):
    def __init__(self):
        self.layer1 = nn.Linear(1, 30)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(30, 1)
    def execute (self,x) :
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

np.random.seed(0)
jt.set_seed(3)
n = 1000
batch_size = 50

def get_data(n):
    for i in range(n):
        x = 4 * (0.5-np.random.rand(batch_size, 1))
        y = x*x
        yield jt.float32(x), jt.float32(y)

model = Model()
learning_rate = 0.2
optim = nn.SGD (model.parameters(), learning_rate)

for i,(x,y) in enumerate(get_data(800)):
    pred_y = model(x)
    loss = jt.sqr(pred_y - y)
    loss_mean = loss.mean()
    optim.step (loss_mean)
    if i % 50 ==0:
        print(f"train_step {i}, loss = {loss_mean.numpy().sum()}")

x_data = jt.array(np.linspace(-2,2,200)).reshape(200,1)
y_data = model(x_data)    
y_origin = x_data ** 2
fig = plt.figure()
x_data = np.array(x_data)
y_data = np.array(y_data)
y_origin = np.array(y_origin)
plt.plot(x_data, y_data, x_data,y_origin)
plt.show()
fig.savefig("plot.png")

assert loss_mean.numpy() < 0.005