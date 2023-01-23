import pygmtools as pygm
import jittor as jt
import numpy as np
from jittor import nn, Module, init, models
from dataset import mydataset
from tqdm import tqdm
from jittor import Var,nn

pygm.BACKEND = 'pytorch'

def local_response_norm(input: Var, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.0) -> Var:
    dim = input.ndim

    if input.numel() == 0:
        return input

    div = input.multiply(input).unsqueeze(1)
    if dim == 3:
        div = nn.pad(div, (0, 0, size // 2, (size - 1) // 2))
        div = nn.avg_pool2d(div, (size, 1), stride=1).squeeze(1)
    else:
        sizes = input.size()
        div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
        div = nn.pad(div, (0, 0, 0, 0, size // 2, (size - 1) // 2))
        div = nn.AvgPool3d((size, 1, 1), stride=1)(div).squeeze(1)
        div = div.view(sizes)
    div = div.multiply(alpha).add(k).pow(beta)
    return input / div

def l2norm(node_feat):
    return local_response_norm(
        node_feat, node_feat.shape[1] * 2, alpha=node_feat.shape[1] * 2, beta=0.5, k=0)

class CNNNet(Module):
    def __init__(self, vgg16_module):
        super(CNNNet, self).__init__()
        # The naming of the layers follow ThinkMatch convention to load pretrained models.
        self.node_layers = jt.nn.Sequential(*[_ for _ in list(vgg16_module.features)[:31]])
        self.edge_layers = jt.nn.Sequential(*[_ for _ in list(vgg16_module.features)[31:38]])  

    def execute(self, inp_img):
        feat_local = self.node_layers(inp_img)
        feat_global = self.edge_layers(feat_local)
        return feat_local, feat_global
    
class GMNet(Module):
    def __init__(self,vgg16_cnn):
        super(GMNet, self).__init__()
        self.gm_net = pygm.utils.get_network(pygm.pca_gm, pretrain="willow") # fetch the network object
        self.cnn = CNNNet(vgg16_cnn)

    def execute(self, img1, img2, kpts1, kpts2, A1, A2):
        # CNN feature extractor layers
        feat1_local, feat1_global = self.cnn(img1)
        feat2_local, feat2_global = self.cnn(img2)
        feat1_local = l2norm(feat1_local)
        feat1_global = l2norm(feat1_global)
        feat2_local = l2norm(feat2_local)
        feat2_global = l2norm(feat2_global)
        obj_resize = tuple(img1.shape[2:4])

        # obj_resize
        # upsample feature map
        feat1_local_upsample = jt.nn.interpolate(feat1_local, obj_resize, mode='bilinear')
        feat1_global_upsample = jt.nn.interpolate(feat1_global, obj_resize, mode='bilinear')
        feat2_local_upsample = jt.nn.interpolate(feat2_local, obj_resize, mode='bilinear')
        feat2_global_upsample = jt.nn.interpolate(feat2_global, obj_resize, mode='bilinear')
        feat1_upsample = jt.concat((feat1_local_upsample, feat1_global_upsample), dim=1)
        feat2_upsample = jt.concat((feat2_local_upsample, feat2_global_upsample), dim=1)

        # assign node features
        rounded_kpts1 = jt.round(kpts1).long()
        rounded_kpts2 = jt.round(kpts2).long()
        node1 = []
        node2 = []
        for i in range(feat1_upsample.shape[0]):
            node1.append(feat1_upsample[i, :, rounded_kpts1[i][1], rounded_kpts1[i][0]].t())
            node2.append(feat2_upsample[i, :, rounded_kpts2[i][1], rounded_kpts2[i][0]].t()) 
        node1 = jt.stack(node1,dim = 0)
        node2 = jt.stack(node2,dim = 0)
        #node1 = feat1_upsample[0, :, rounded_kpts1[0], rounded_kpts1[1]].t()  # shape: NxC
        #node2 = feat2_upsample[0, :, rounded_kpts2[0], rounded_kpts2[1]].t()  # shape: NxC

        # PCA-GM matching layers
        # X = pygm.pca_gm(node1, node2, A1, A2, network=self.gm_net) # the network object is reused
        X = pygm.ipca_gm(node1, node2, A1, A2, network=self.gm_net)
        return X