import glob
import json

import torch
import numpy as np

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from data_processing import SVHNDataset
from utils import Tools


# 自定义模型
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
        # CNN提取特征模块
        self.cnn = nn.Sequential(
            # 输入图像channels*height*width:(3,64,128)
            nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1),
            # 卷积过后变成(16,64,128)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 池化过后变成(16,32,64)
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            # 卷积过后变成(32,32,64)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 池化过后变成(32,16,32)
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            # 卷积过后变成(64,16,32)
            nn.ReLU(),
            nn.MaxPool2d(2)
            # 池化过后变成(64,8,16)
        )
        # fc层，直接从向量64*8*16映射到0-10的标签输出上
        self.fc1 = nn.Linear(64 * 8 * 16, 11)
        self.fc2 = nn.Linear(64 * 8 * 16, 11)
        self.fc3 = nn.Linear(64 * 8 * 16, 11)
        self.fc4 = nn.Linear(64 * 8 * 16, 11)
        self.fc5 = nn.Linear(64 * 8 * 16, 11)
        self.fc6 = nn.Linear(64 * 8 * 16, 11)

    def forward(self, img):
        feat = self.cnn(img)
        # 拉伸为向量
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)
        return c1, c2, c3, c4, c5, c6

    def mytraining(self, train_loader, criterion, optimizer, device=torch.device('cpu')):
        # 切换模型为训练模式
        self.train()
        train_loss = []

        for i, (data, label) in enumerate(train_loader):
            c0, c1, c2, c3, c4, c5 = self(data.to(device))
            label = label.long().to(device)
            loss = criterion(c0, label[:, 0]) + \
                   criterion(c1, label[:, 1]) + \
                   criterion(c2, label[:, 2]) + \
                   criterion(c3, label[:, 3]) + \
                   criterion(c4, label[:, 4]) + \
                   criterion(c5, label[:, 5])
            loss /= 6
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(np.mean(train_loss),4)

    def myvalidating(self, val_loader, criterion, device=torch.device('cpu')):
        # 切换模型为预测模型
        self.eval()
        val_loss = []

        # 不记录模型梯度信息
        with torch.no_grad():
            for i, (data, label) in enumerate(val_loader):
                c0, c1, c2, c3, c4, c5 = self(data.to(device))
                label = label.long().to(device)
                loss = criterion(c0, label[:, 0]) + \
                       criterion(c1, label[:, 1]) + \
                       criterion(c2, label[:, 2]) + \
                       criterion(c3, label[:, 3]) + \
                       criterion(c4, label[:, 4]) + \
                       criterion(c5, label[:, 5])
                loss /= 6
                val_loss.append(loss.item())

        return round(np.mean(val_loss),4)

    def myPredicting(self, test_loader, device=torch.device('cpu')):
        # 切换模型为预测模型
        self.eval()
        is_init = True

        # 不记录模型梯度信息
        with torch.no_grad():
            for i, (data, label) in enumerate(test_loader):
                c0, c1, c2, c3, c4, c5 = self(data)
                l0 = np.reshape(c0.numpy().argmax(axis=1),(-1,1)) #100x11->100x1
                l1 = np.reshape(c1.numpy().argmax(axis=1),(-1,1)) #100x11->100x1
                l2 = np.reshape(c2.numpy().argmax(axis=1),(-1,1)) #100x11->100x1
                l3 = np.reshape(c3.numpy().argmax(axis=1),(-1,1)) #100x11->100x1
                l4 = np.reshape(c4.numpy().argmax(axis=1),(-1,1)) #100x11->100x1
                l5 = np.reshape(c5.numpy().argmax(axis=1),(-1,1)) #100x11->100x1
                # 合并->100x6
                tmp = np.concatenate((l0,l1,l2,l3,l4,l5),axis=1)
                if is_init:
                    pred_labels=tmp
                    is_init=False
                else:
                    pred_labels = np.concatenate((pred_labels,tmp),axis=0)

        return pred_labels


# 继承预训练好的模型
class SVHN_Model2(nn.Module):
    def __init__(self):
        super(SVHN_Model2, self).__init__()

        # 继承resnet18
        model_conv = models.resnet18(pretrained=True)
        # 将resnet18的最后一个池化层修改为自适应的全局平均池化层
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        # 微调，把fc层删除
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv
        # 自定义fc层
        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)
        self.fc6 = nn.Linear(512, 11)

    def forward(self, img):
        feat = self.cnn(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)
        return c1, c2, c3, c4, c5, c6

    def mytraining(self, train_loader, criterion, optimizer, device=torch.device('cpu')):
        # 切换模型为训练模式
        self.train()
        train_loss = []

        for i, (data, label) in enumerate(train_loader):
            c0, c1, c2, c3, c4, c5 = self(data.to(device))
            label = label.long().to(device)
            loss = criterion(c0, label[:, 0]) + \
                   criterion(c1, label[:, 1]) + \
                   criterion(c2, label[:, 2]) + \
                   criterion(c3, label[:, 3]) + \
                   criterion(c4, label[:, 4]) + \
                   criterion(c5, label[:, 5])
            loss /= 6
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(np.mean(train_loss),4)

    def myvalidating(self, val_loader, criterion, device=torch.device('cpu')):
        # 切换模型为预测模型
        self.eval()
        val_loss = []

        # 不记录模型梯度信息
        with torch.no_grad():
            for i, (data, label) in enumerate(val_loader):
                c0, c1, c2, c3, c4, c5 = self(data.to(device))
                label = label.long().to(device)
                loss = criterion(c0, label[:, 0]) + \
                       criterion(c1, label[:, 1]) + \
                       criterion(c2, label[:, 2]) + \
                       criterion(c3, label[:, 3]) + \
                       criterion(c4, label[:, 4]) + \
                       criterion(c5, label[:, 5])
                loss /= 6
                val_loss.append(loss.item())

        return round(np.mean(val_loss),4)

    def myPredicting(self, test_loader, device=torch.device('cpu')):
        # 切换模型为预测模型
        self.eval()
        is_init = True

        # 不记录模型梯度信息
        with torch.no_grad():
            for i, (data, label) in enumerate(test_loader):
                c0, c1, c2, c3, c4, c5 = self(data)
                l0 = np.reshape(c0.numpy().argmax(axis=1),(-1,1)) #100x11->100x1
                l1 = np.reshape(c1.numpy().argmax(axis=1),(-1,1)) #100x11->100x1
                l2 = np.reshape(c2.numpy().argmax(axis=1),(-1,1)) #100x11->100x1
                l3 = np.reshape(c3.numpy().argmax(axis=1),(-1,1)) #100x11->100x1
                l4 = np.reshape(c4.numpy().argmax(axis=1),(-1,1)) #100x11->100x1
                l5 = np.reshape(c5.numpy().argmax(axis=1),(-1,1)) #100x11->100x1
                # 合并->100x6
                tmp = np.concatenate((l0,l1,l2,l3,l4,l5),axis=1)
                if is_init:
                    pred_labels=tmp
                    is_init=False
                else:
                    pred_labels = np.concatenate((pred_labels,tmp),axis=0)

        return pred_labels

# test
if __name__ == '__main__':
    train_path = glob.glob(r'E:\Datas\StreetCharsRecognition\mchar_train\*.png')
    train_path.sort()
    train_json = json.load(open(r'E:\Datas\StreetCharsRecognition\mchar_train.json'))
    train_label = [train_json[x]['label'] for x in train_json]

    print("扩增前数据集大小", ":", len(train_path))

    # Tip1:测试时，训练集可以调小点
    train_loader = torch.utils.data.DataLoader(
        SVHNDataset(train_path[:20], train_label[:20],
                    transforms.Compose([
                        transforms.Resize((64, 128)),
                        transforms.ColorJitter(0.3, 0.3, 0.2),
                        transforms.RandomRotation(5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])),
        batch_size=10,  # 每批样本个数
        shuffle=False,  # 是否打乱顺序
        num_workers=5,  # 进程个数
    )
    # 模型生成
    model = SVHN_Model2()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器，这里优化模型的所有参数
    optimizer = torch.optim.Adam(model.parameters(), 0.005)

    loss_plot, c0_plot = [], []
    # Tip2:epochs设置小点，这里是3
    for epoch in range(3):
        for data in train_loader:
            c0, c1, c2, c3, c4, c5 = model(data[0])
            data[1] = data[1].long()
            loss = criterion(c0, data[1][:, 0]) + \
                   criterion(c1, data[1][:, 1]) + \
                   criterion(c2, data[1][:, 2]) + \
                   criterion(c3, data[1][:, 3]) + \
                   criterion(c4, data[1][:, 4]) + \
                   criterion(c5, data[1][:, 5])
            loss /= 6 #计算loss
            # 反向传播，优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_plot.append(loss.item())
            c0_plot.append((c0.argmax(1) == data[1][:, 0]).sum().item() * 1.0 / c0.shape[0])

        # Tips3:打印个东西，表示迭代过程没出问题
        # 一般都是打印epoch/train_loss/train_acc/val_loss/val_acc
        # 这里就打印epoch，凑合用
        print(epoch)
        sorted()
