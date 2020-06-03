from model import *
from data_processing import SVHNDataset
import torchvision.transforms as transforms
from utils import Tools
import torch
from torch import nn

if __name__=='__main__':
    # 配置环境
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 初始化参数
    train_img_path = r'E:\Datas\StreetCharsRecognition\mchar_train\*.png'
    train_label_path = r'E:\Datas\StreetCharsRecognition\mchar_train.json'
    val_img_path = r'E:\Datas\StreetCharsRecognition\mchar_val\*.png'
    val_label_path = r'E:\Datas\StreetCharsRecognition\mchar_val.json'
    test_img_path = r'E:\Datas\StreetCharsRecognition\mchar_test_a\*.png'
    demo_submit_path = r'E:\Datas\StreetCharsRecognition\mchar_sample_submit_A.csv'
    batch_size = 100
    epochs = 20
    lr = .001
    is_predicting = False #默认is_predicting=False, 表明为训练过程

    # 训练过程
    if not is_predicting:
        # 加载数据
        train_path,train_label = Tools.dataFromPath(train_img_path,train_label_path)
        train_dataset = SVHNDataset(train_path, train_label,
                    transforms.Compose([
                        transforms.Resize((64, 128)),
                        transforms.ColorJitter(0.3, 0.3, 0.2),
                        transforms.RandomRotation(5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]))
        val_path,val_label = Tools.dataFromPath(val_img_path,val_label_path)
        val_dataset = SVHNDataset(val_path, val_label,
                    transforms.Compose([
                        transforms.Resize((64, 128)),
                        transforms.ColorJitter(0.3, 0.3, 0.2),
                        transforms.RandomRotation(5),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ]))
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=5,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=5,
        )

        # 创建模型
        model = SVHN_Model2().to(device)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        best_epoch, best_loss, best_acc = -1, 1000.0, 0

        # 模型训练，并保存最优参数
        for epoch in range(epochs):
            train_loss = model.mytraining(train_loader, criterion, optimizer, device)
            val_loss = model.myvalidating(val_loader, criterion, device)
            # 记录下验证集精度
            if val_loss < best_loss:
                best_epoch, best_loss = epoch, val_loss
                # 保存model可学习参数
                torch.save(model.state_dict(), 'Model/model.pt')
            # 打印相关信息
            Tools.printInfo(epoch, train_loss, val_loss,
                            best_epoch, best_loss)
    else:
        # 预测过程
        test_path, test_label = Tools.dataFromPath(test_img_path)
        test_dataset = SVHNDataset(test_path, test_label,
                                    transforms.Compose([
                                        transforms.Resize((64, 128)),
                                        transforms.ColorJitter(0.3, 0.3, 0.2),
                                        transforms.RandomRotation(5),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]))
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=5,
        )
        model = SVHN_Model1()
        model.load_state_dict(torch.load("Model/model.pt", map_location='cpu'))
        pred_labels = model.myPredicting(test_loader)
        Tools.submit(demo_submit_path,pred_labels)