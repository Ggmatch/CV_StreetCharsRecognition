import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

'''
1. 认识数据
'''
# json.load将已编码的JSON字符串解码为Python字典对象{pic_id:{top:xx,...},...}
train_json = json.load(open(r'E:\Datas\StreetCharsRecognition\mchar_train.json'))

# 数据标注处理
def parse_json(d):
    arr = np.array([
        d['top'], d['height'], d['left'], d['width'], d['label']
    ])
    arr = arr.astype(int)
    return arr

arr = parse_json(train_json['000000.png']) #转化成np.array类型
img = cv2.imread(r'E:\Datas\StreetCharsRecognition\mchar_train\000000.png')

# 画图：共2行，原图占第一行，每个字符的图在第二行排列
plt.figure(figsize=(7, 5)) #创建画布
grid = plt.GridSpec(2, arr.shape[1] + 1, hspace=0.5, wspace=0.2) #子图数目=原图与每个字符的图
plt.subplot(grid[0,:])
plt.imshow(img)
plt.title("orginal")
plt.xticks([])
plt.yticks([])

for idx in range(arr.shape[1]):
    plt.subplot(grid[1,idx])
    plt.imshow(img[arr[0, idx]:arr[0, idx] + arr[1, idx], arr[2, idx]:arr[2, idx] + arr[3, idx]])
    plt.title(arr[4, idx])
    plt.xticks([])
    plt.yticks([])

plt.show()
os.system("pause")
plt.close("all")

'''
2. 统计标签信息：每张图片字符个数分布情况
'''
