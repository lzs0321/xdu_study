import matplotlib.pyplot as plt
from scipy.cluster.vq import vq
import scipy.cluster.vq as cluster
import numpy as np
import cv2

image_nums = 1000

def SIFT(img):
    I = cv2.imread(img)
    detector = cv2.xfeatures2d.SIFT_create()
    (_, features) = detector.detectAndCompute(I, None)  # 返回的第一个值是关键点信息，不需要
    return features

descriptors = SIFT('corel/0/0.jpg')
desc = []
desc.append(descriptors)
for i in range(10):
    for j in range(100):
        temp_des = SIFT('corel/'+str(i)+'/'+str(j+i*100)+'.jpg')                    # (1,128)向量
        desc.append(temp_des)                             # desc以数组形式保存所有描述子
        descriptors = np.vstack((descriptors, temp_des))  # 将所有的特征点描述子堆叠为矩阵(n,128)
descriptors = descriptors[1211:,:]
desc = desc[1:]
descriptors.shape

