import glob,json
import pandas as pd
import numpy as np

class Tools:
    @staticmethod
    def dataFromPath(img_path,label_path=None):
        imgs = glob.glob(img_path)
        imgs.sort()
        if label_path:
            label_json = json.load(open(label_path))
            labels = [label_json[x]['label'] for x in label_json]
        else: #制作假的测试集标签
            labels = [[10]]*len(imgs)
        return imgs,labels


    @staticmethod
    def calAcc(pred_label,true_label):
        length = len(true_label)
        count = 0
        for i in range(length):
            for j in range(len(true_label[i])):
                if true_label[i][j]==pred_label[i][j] or true_label[i][j]==10:
                    if true_label[i][j]==10:
                        count+=1
                        break
                else:
                    break
        return round(count/length,4)*100

    @staticmethod
    def printInfo(epoch,train_loss,val_loss,
                  best_epoch,best_val_loss,
                  train_acc='--',val_acc='--',best_val_acc='--'):
        print("epoch {}: train_loss {}, train_acc {}; val_loss {}, val_acc {}; " 
              "(best_epoch,best_val_loss,best_val_acc):({},{},{})".format(
            epoch,train_loss,train_acc,val_loss,val_acc,best_epoch,best_val_loss,best_val_acc))

    @staticmethod
    def submit(demo_submit_path,pred_labels,out_path='Submit_files/'):
        submit = pd.read_csv(demo_submit_path)
        pred_result = []
        for label in pred_labels:
            tmp = []
            for char in label:
                if char!=10:
                    tmp.append(char)
                else:
                    break
            # 意外情况，没有有效字符，默认填充0
            if not tmp:
                tmp.append(0)
            pred_result.append("".join(map(str,tmp)))
        # 填充到pd表格
        submit['file_code'] = pred_result
        # 保存为文件submit.csv
        out_path += "submit.csv"
        submit.to_csv(out_path,index=False)

#test
if __name__=='__main__':
    # Tools.printInfo(1,10.0,100,10.0,100,5,10.0,100)
    pred_labels = np.array([i for i in range(40000*6)]).reshape((-1,6))
    Tools.submit(r'E:\Datas\StreetCharsRecognition\mchar_sample_submit_A.csv',pred_labels)
