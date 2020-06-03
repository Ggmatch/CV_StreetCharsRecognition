from PIL import Image
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from utils import Tools


class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        # just handle one data
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # 定长字符识别策略，填充的字符为10，这样不会与有效字符0-9发生碰撞
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl) + (6 - len(lbl)) * [10]

        return img, torch.from_numpy(np.array(lbl))

    def __len__(self):
        return len(self.img_path)



# Test SVHNDataset
if __name__ == '__main__':
    # train_path = glob.glob(r'E:\Datas\StreetCharsRecognition\mchar_train\*.png')
    # train_path.sort()
    # train_json = json.load(open(r'E:\Datas\StreetCharsRecognition\mchar_train.json'))
    # train_label = [train_json[x]['label'] for x in train_json]
    train_path,train_label = Tools.dataFromPath(
        r'E:\Datas\StreetCharsRecognition\mchar_train\*.png',
        r'E:\Datas\StreetCharsRecognition\mchar_train.json'
    )

    print("扩增前数据集大小",":",len(train_path))

    train_loader = torch.utils.data.DataLoader(
        SVHNDataset(train_path, train_label,
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
    for data in train_loader:
        print(data[0][0])
        break
