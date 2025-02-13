import numpy as np
import matplotlib.pyplot as plt

from SIFT import image_nums
from bof import imhists


def cosin_dist(x, imhists):
    dists = []
    for i in range(image_nums):
        dists.append(np.dot(x, imhists[i]) / (np.linalg.norm(x, ord=2) * np.linalg.norm(imhists[i], ord=2)))
    return np.sort(dists)[::-1], np.argsort(dists)[::-1]


def euclid_dist(x, imhists):
    dists = []
    for i in range(image_nums):
        dists.append(np.sqrt(np.sum((x - imhists[i]) ** 2)))
    return np.sort(dists), np.argsort(dists)


id = 720
input = imhists[id]
print('输入图像如下：')
img = plt.imread('corel/' + str(int(id / 100)) + '/' + str(id) + '.jpg')
plt.imshow(img)
plt.show()

dists, idxs = cosin_dist(input, imhists)
print('结果图像如下：')
ids = idxs[1:11]
plt.subplots(2, 5, figsize=(35, 10))
for i, id in enumerate(ids):
    img = plt.imread('corel/' + str(int(id / 100)) + '/' + str(id) + '.jpg')
    plt.subplot(2, 5, i + 1)
    plt.title(str(i + 1) + ': ' + str(id))
    plt.imshow(img)

plt.show()
# 0 2 3 4 5 7 准确率较高
# 1(海滩)图片容易和5(大象)/7(马)混淆
# 6(花)一旦背景有绿叶，容易和其他类别混
# 8(山峰/雪山)容易和其他类别混
# 9(食物)容易和1(非洲部落)/5(大象)/7(马)混