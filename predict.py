from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import numpy as np
from tqdm import tqdm
import os
import glob
from PIL import Image

def img2vec(filename):
    dataVec = np.zeros((1,800))
    img = Image.open(filename)
    img = img.crop((0,0,20,40))
    img = np.array(img)
    for i in range(40):
        for j in range(20):
            dataVec[0,20*i+j] = img[i][j]
    return dataVec

def make_dataset(parent_dir,sub_dirs,file_ext="*.jpg"):
    feature = []
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir,sub_dir,file_ext))):
            imgdata = img2vec(fn)
            feature.extend([imgdata])
    return feature

def svm_pred(root_dir,sub_dirs,data="*.jpg"):
    img_total = 0
    acc_total = 0
    clf = joblib.load("./model_save/svm.pkl")
    data = make_dataset(root_dir,sub_dirs)
    data = np.array(data)
    X = np.vstack(data[:, 0])
    scaler = StandardScaler()
    x_std = scaler.fit_transform(X)  # 标准化
    pred = clf.predict(x_std)
    k = 0
    for sub_dir in sub_dirs:
        acc = 0
        sub_path = os.path.join(root_dir,sub_dir)
        num = len(os.listdir(sub_path))
        img_total+=num
        for i in range(num):
            if pred[k] == sub_dir:
                acc+=1
                k+=1
            else:
                k+=1
        print("当前识别数字为：",sub_dir, "    识别正确个数为：",acc, "总数为：",num, "识别准确率为：",float(acc/num))
        acc_total+=acc
    print("训练集总数：",img_total, "正确识别总数：",acc_total, "准确率：",float(acc_total/img_total))

def cnn_pred(root_dir,sub_dirs):
    model = load_model("./model_save/base_cnn_model")
    img_total = 0
    acc_total = 0
    for sub_dir in sub_dirs:
        acc = 0
        sub_path = os.path.join(root_dir,sub_dir)
        num = len(os.listdir(sub_path))
        img_total+=num
        imgfile = [os.path.join(sub_path,x) for x in os.listdir(sub_path)]
        for img in imgfile:
            pred_img = img2vec(img)
            pred_img = pred_img.reshape(-1,40,20,1)
            predlabel = np.argmax(model.predict(pred_img))
            if str(predlabel) == sub_dir:
                acc+=1
        print("当前识别数字为：",sub_dir, "    识别正确个数为：",acc, "总数为：",num, "识别准确率为：",float(acc/num))
        acc_total+=acc
    print("训练集总数：", img_total, "正确识别总数：", acc_total, "准确率：", float(acc_total / img_total))

def main():
    i = input("请输入识别类别1.train   2.test")
    if i == '1':
        root_dir = "./data/train"
        sub_dirs = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        print("svm+++++++Train++++++++")
        svm_pred(root_dir,sub_dirs)
        print("cnn+++++++train+++++++")
        cnn_pred(root_dir,sub_dirs)
    if i == '2':
        root_dir = "./data/test"
        sub_dirs = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        print("svm+++++++TEST++++++++")
        svm_pred(root_dir,sub_dirs)
        print("cnn+++++++TEST+++++++")
        cnn_pred(root_dir,sub_dirs)

if __name__ == "__main__":
    main()
