import numpy as np
import scipy.cluster.vq as cluster
from K_means import words_num
from SIFT import image_nums


# 计算一幅图的直方图向量
def project(descriptors, words_num, voc): # 描述子向量，视觉词汇数，视觉词典
    imhist = np.zeros(words_num)   # 初始化直方图向量
    cls, _ = vq(descriptors, voc)  # 将该图的各个描述子分配到离它最近的聚类中心代表的类中
    for c in cls:
        imhist[c] += 1
    return imhist
imhists = np.zeros((image_nums, words_num))    # 100/1.3s  500/6.5s  1000/8.6s 2000/17.3s  5000/26.1s 10000/49s  15000/1m12s
for i in range(image_nums):
    imhists[i] = project(desc[i], words_num, voc)
# 这里是TF-IDF优化，后来发现效果不好就不用了
occurence_num = np.sum(imhists > 0, axis=0)       # 计算各视觉词汇在1000张图片中出现了几次，1x1000
IDF = np.log((image_nums) / (occurence_num + 1))  # 逆文档频率 = log(总图片数/(出现次数+1))，1x1000
for i in range(image_nums):
    imhists[i] = imhists[i] / np.sum(imhists[i]) * IDF  # imhists[i] / np.sum(imhists[i])是TF，即各视觉词汇在一幅图中出现的频率