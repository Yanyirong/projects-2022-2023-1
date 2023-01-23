import jittor as jt
from jittor import models, nn, lr_scheduler
import pygmtools as pygm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dataset import mydataset
from model import GMNet
from tqdm import tqdm
import json
import argparse
pygm.BACKEND = "jittor"
_ = jt.seed(1)
jt.flags.use_cuda = jt.has_cuda

pygm.BACKEND = "jittor"
vgg16_cnn = models.vgg16_bn(True)
model = GMNet(vgg16_cnn)
lr = 0.0001
epochs = 50
batch_size = 10

# optimizer = nn.Adam(model.parameters(), lr, eps=1e-8, betas=(0.9, 0.999))
# optimizer = nn.SGD(model.parameters(), lr)
optimizer = nn.Adam([{"params":model.cnn.parameters(),"lr":lr*0.1},{"params":model.gm_net.parameters(),"lr":lr}],lr)
    
#scheduler = lr_scheduler.ReduceLROnPlateau(optimizer)
scheduler = lr_scheduler.ExponentialLR(optimizer, 0.9, last_epoch=- 1)
    
train = mydataset("WillowObject").set_attrs(batch_size = batch_size, shuffle = True)
test = mydataset("WillowObject",train = False).set_attrs(batch_size = 1, shuffle = False)

filename = "result.json"
val_interval = 5
log_interval = 10


best_f1 = 0
train_loss = []
epoch_loss = []
test_acc = []
for epoch in range(epochs):
    epoch_train_loss = 0
    model.train()
    for batch_idx, (jt_img1, jt_img2, img1, img2, k1, k2, A1, A2, gt) in enumerate(tqdm(train,total=len(train)//batch_size+1)):
        X = model(jt_img1, jt_img2, k1, k2, A1, A2)
        loss = pygm.utils.permutation_loss(X, gt)
        epoch_train_loss += loss.numpy()
        train_loss.extend(loss.numpy())
        if (batch_idx+1) % log_interval == 0:
            print(loss.numpy()[0])
        optimizer.step(loss)
    epoch_train_loss /= (batch_idx+1)
    epoch_loss.extend(epoch_train_loss)


    with open("train_loss.txt","w") as t_loss:
        for i in train_loss:
            t_loss.write(str(i)+'\n')
    with open("epoch_loss.txt","w") as t_loss:
        for i in epoch_loss:
            t_loss.write(str(i)+'\n')
    if epoch % val_interval == 0 or epoch+1 == epochs:
        valid_list = []
        model.eval()
        print("eval done")
        for batch_idx, (jt_img1, jt_img2, img1, img2, k1, k2, A1, A2, id, cls) in enumerate(tqdm(test)):
            
            with jt.no_grad():
                X = model(jt_img1, jt_img2, k1, k2, A1, A2)
            tmp = dict()
            tmp['ids'] = (id[0][0],id[1][0])
            tmp['perm_mat'] = np.array(X[0])
            tmp['cls'] = cls[0]
            valid_list.append(tmp)
        result = test.benchmark.eval(valid_list, test.benchmark.classes,verbose=True)
        test_acc.extend(result['mean']['f1'])

        with open("test_acc.txt","w") as t_acc:
            for i in test_acc:
                t_acc.write(str(i)+'\n')
        with open(str(epoch)+'_'+filename, 'w') as file_obj:
            json.dump(result, file_obj)
        # if result['mean']['f1'] > best_f1:
        #     best_f1 = result['mean']['f1']
            # jt.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': loss,
            #     'f1_socre': best_f1
            #     },str(best_f1)+str(epoch)+"snapshot.pkl") 
    scheduler.step()
    # scheduler.step(epoch_train_loss)
    print("epoch "+str(epoch)+" done")

