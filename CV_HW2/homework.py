# -*- coding: utf-8 -*-
import pickle
from PIL import Image
import cv2
import numpy as np
import os
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import random
from keras import backend as K

images = []
labels = []
IMAGE_SIZE = 200


# 按照指定图像大小调整尺寸
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    return cv2.resize(image, (height, width))


def read_path(path_name):
    for dir_item in os.listdir(path_name):
        full_path = os.path.abspath(os.path.join(path_name, dir_item))

        if os.path.isdir(full_path):  # 如果是文件夹，继续递归调用
            read_path(full_path)
        else:  # 文件
            if dir_item.endswith('.jpg') or dir_item.endswith('.JPG') or dir_item.endswith('.png'):
                image = cv2.imread(full_path)
                image = resize_image(image)
                images.append(image)
                labels.append(path_name)

    return images, labels


def load_dataset(path_name):
    images, labels = read_path(path_name)

    images = np.array(images)
    print(images.shape)
    category = []
    for i in labels:
        category.append(i.split('/')[-1])
    temp = list(set(category))
    dic = {}
    for i in range(len(temp)):
        dic[temp[i]] = i
    for i in range(len(category)):
        labels[i] = dic[category[i]]
    labels = np.array(labels)
    print(labels.shape)
    return images, labels


class Dataset:
    def __init__(self, path_name):
        # 训练集
        self.train_images = None
        self.train_lb = None

        # 测试集
        self.test_images = None
        self.test_lb = None

        # 数据集加载路径
        self.path_name = path_name

        # 当前库采用的维度顺序
        self.input_shape = None

    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=3, nb_classes=102):
        # 加载数据集到内存
        images, labels = load_dataset(self.path_name)

        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.3,
                                                                                random_state=random.randint(0, 100))

        # 当前的维度顺序如果为'th'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
        if K.image_data_format() == 'channels_first':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

            # 输出训练集、验证集、测试集的数量
            print(train_images.shape[0], 'train samples')
            print(test_images.shape[0], 'test samples')

            self.train_lb = train_labels
            self.test_lb = test_labels

            # 像素数据浮点化以便归一化
            train_images = train_images.astype('float32')
            test_images = test_images.astype('float32')

            # 将其归一化,图像的各像素值归一化到0~1区间
            train_images /= 255
            test_images /= 255

            self.train_images = train_images
            self.test_images = test_images


data = Dataset('E:\edge_download\AI3604_HW2\AI3604_HW2\caltech-101\caltech-101')
data.load()
print(data.test_images.shape)
print(data.train_images.shape)


# TODO 利用SIFT从训练图像中提取特征
# 如果有需要，你也可以在pass之外的地方填写相关代码，请自便，下同。
# vec_dict 第i项： i为类别，对应的字典为所有属于该类的sift特征点的信息。注意：kp与des一一对应。
vec_dict = {i:{'kp':[], 'des':[]} for i in range(102)}

sift = cv2.SIFT_create()
for i in range(data.train_images.shape[0]):
    tep = cv2.normalize(data.train_images[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    kp_vector, des_vector = sift.detectAndCompute(tep, None)

    #####
    vec_dict[data.train_lb[i]]['kp'].extend(kp_vector)
    vec_dict[data.train_lb[i]]['des'].extend(des_vector)
    #####

# 统计最少特征点的类别
bneck_value = float("inf")
for i in range(102):
    if len(vec_dict[i]['kp']) < bneck_value:
        bneck_value = len(vec_dict[i]['kp'])

#这里的逻辑我实在没有看懂，我个人认为要同时对kp和des打包进行排列，而不是单独排列某个属性
for i in range(102):
    kp_list = vec_dict[i]['kp'] = sorted((vec_dict[i]['kp']),
                                         key=lambda x: x.response,
                                         reverse=True)


# TODO 为每个类别选择同样多的特征点用于聚类。特征点个数bneck_value

vec_list = vec_dict[0]['des'][0:bneck_value]
for i in range(1, 102):
    #####
    tmp = vec_dict[i]['des'][0:bneck_value]
    vec_list = np.append(vec_list, tmp, axis=0)
    #vec_list.extend(vec_dict[i]['des'][0:bneck_value])
    #####
vec_list = np.float64(vec_list)
print(vec_list)



# TODO 对提取出的特征点使用Kmeans聚类，设定合适的聚类中心个数
from sklearn.cluster import KMeans
#####
N_clusters = 1000
kmeans = KMeans(n_clusters=N_clusters,max_iter = 100, random_state=0).fit(vec_list)
pass
#####


# TODO 利用直方图统计每张图像中的特征点所属聚类中心的个数，将直方图归一化后便得到图像的特征向量。
num_images = data.train_images.shape[0]
hist_vector = np.zeros((num_images, N_clusters))
for i in range(num_images):
    tep = cv2.normalize(data.train_images[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    #####
    kp_vector, des_vector = sift.detectAndCompute(tep, None)  
    for j in range(len(kp_vector)):
        hist_vector[i][kmeans.predict(np.float64(des_vector[j]).reshape(1, -1))] += 1
    hist_vector[i] =hist_vector[i] / len(kp_vector)
    #####


# 使用SVM构建分类器
# 你可以自行构建分类器，也可以使用SVM
from sklearn import svm
classifier = svm.SVC(probability=True)
classifier.fit(hist_vector, data.train_lb)

# TODO 构建测试集并计算模型准确率
num_test_images = data.test_images.shape[0]
hist_test_vector = np.zeros((num_test_images, N_clusters))
for i in range(num_test_images):
    tep = cv2.normalize(data.test_images[i], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    #####
    kp_vector, des_vector = sift.detectAndCompute(tep, None)
    for j in range(len(kp_vector)):
        hist_test_vector[i][kmeans.predict(np.float64(des_vector[j]).reshape(1, -1))] += 1     
    hist_test_vector[i] =hist_test_vector[i] / len(kp_vector)
    #####


acc = classifier.predict(hist_test_vector)-data.test_lb
tep = len(acc[acc==0])
print('accuracy', tep/len(data.test_lb))