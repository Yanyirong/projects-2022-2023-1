import jittor as jt
import pygmtools as pygm
from jittor.dataset import Dataset
import jittor.transform as trans
import numpy as np
import itertools
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.spatial as spa
pygm.BACKEND = "jittor"

classes = ["Car","Duck","Face","Motorbike","Winebottle"]

def delaunay_triangulation(kpt):
    d = spa.Delaunay(kpt.numpy().transpose())
    A = jt.zeros((len(kpt[0]), len(kpt[0])))
    for simplex in d.simplices:
        for pair in itertools.permutations(simplex, 2):
            A[pair] = 1
    return A

class mydataset(Dataset):
    def __init__(self,name, train = True, batch_size = 4, shuffle = False, transform = None, 
                obj_resize=(256, 256), problem='2GM', filter='intersection'):
        # assert name == 'PascalVOC' or name == 'SPair71k' or name == 'WillowObject' or name == 'IMC_PT_SparseGM' or name == 'CUB2011', 'No match found for dataset {}'.format(name)
        # assert problem == '2GM' or problem == 'MGM' or problem == 'MGM3', 'No match found for problem {}'.format(problem)
        # assert filter == 'intersection' or filter == 'inclusion' or filter == 'unfiltered', 'No match found for filter {}'.format(filter)
        
        super().__init__()
        self.name = name
        self.is_train = train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transform = transform
        if self.is_train:
            self.benchmark = pygm.Benchmark(name,"train",obj_resize,problem,filter)
        else:
            self.benchmark = pygm.Benchmark(name,"test",obj_resize,problem,filter)
        id_list, self.size = self.benchmark.get_id_combination()
        self.ids = []
        for id in id_list:
            self.ids.extend(id)
        self.set_attrs(total_len = self.size)

    def __getitem__(self, index):
        if self.is_train:
            data_list, perm_mat_dict, ids = self.benchmark.get_data(list(self.ids[index]))
        else:
            data_list, ids = self.benchmark.get_data(list(self.ids[index]),test = True,shuffle=False)
        # load the data
        img1 = Image.fromarray(data_list[0]['img']).convert('RGB')
        img2 = Image.fromarray(data_list[1]['img']).convert('RGB')
        kpts1 = [[],[]]
        kpts2 = [[],[]]
        for kp in data_list[0]['kpts']:
            kpts1[0].append(kp['x'])
            kpts1[1].append(kp['y'])
        for kp in data_list[1]['kpts']:
            kpts2[0].append(kp['x'])
            kpts2[1].append(kp['y'])
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        jt_img1 = jt.float32(trans.to_tensor(img1))
        jt_img2 = jt.float32(trans.to_tensor(img2))
        kpts1 = jt.float32(kpts1)
        kpts2 = jt.float32(kpts2)
        A1 = delaunay_triangulation(kpts1)
        A2 = delaunay_triangulation(kpts2)
        if self.is_train:
            gt = jt.float32(np.array(perm_mat_dict[(0,1)].todense()))
            return jt_img1, jt_img2, img1 , img2, kpts1, kpts2, A1, A2, gt
        else:
            return jt_img1, jt_img2, img1 , img2, kpts1, kpts2, A1, A2, ids, data_list[0]['cls']
        
    def __len__(self):
        return self.size