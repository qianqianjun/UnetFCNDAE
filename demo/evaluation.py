import requests
import base64
import json
import time
import torch
import torch.nn as nn
from tools.utils import batchDataTransform,getBaseGrid
from setting.parameter import parameter as parm
from model.UnetDAE import Encoders,Decoders
import numpy as np
class image():
    def __init__(self, coders:str, image_type:str="BASE64", face_type:str="LIVE"):
        self.image=coders
        self.image_type=image_type
        self.face_type=face_type
        self.liveness_control="NONE"
    def toJSON(self):
        temp={
            "image":self.image,
            "image_type":self.image_type,
            "face_type":self.face_type,
            "liveness_control":self.liveness_control
        }
        return temp
class Data:
    def __init__(self,path1,path2):
        self.images=[]
        with open(path1,"rb") as f:
            encoder=base64.b64encode(f.read())
            text=encoder.decode()
            self.images.append(image(text))
        with open(path2,"rb") as f:
            encoder=base64.b64encode(f.read())
            text=encoder.decode()
            self.images.append(image(text))
    def toJSON(self):
        return json.dumps([item.toJSON() for item in self.images])
from PIL import Image
import torch.utils.data as data
import os
import cv2
import pickle as pk
class Datas(data.Dataset):
    def __init__(self,abs_img_dir:str,length:int,resizeTo:int=64,shuffle:bool=False):
        self.abs_dir=abs_img_dir
        self.resizeTo=resizeTo
        self.image_paths=os.listdir(abs_img_dir)
        self.image_paths=np.array(self.image_paths)
        if shuffle:
            index=np.random.permutation(len(self.image_paths))
            self.image_paths=self.image_paths[index]
        self.image_paths=self.image_paths[:length]
    def __getitem__(self, item):
        return self.image_paths[item],self.fileReader(item)
    def __len__(self):
        return len(self.image_paths)
    def fileReader(self,item):
        img=Image.open(os.path.join(self.abs_dir,self.image_paths[item]))
        img.convert('RGB')
        img.resize((self.resizeTo, self.resizeTo), Image.ANTIALIAS)
        return np.array(img)
def prepare_image():
    dataset_image_path="/home/qianqianjun/CODE/DataSets/DaeDatasets"
    save_dir="/home/qianqianjun/CODE/实验结果/重建结果/改进编码器"
    os.makedirs(save_dir,exist_ok=True)
    length=5000
    dataset=Datas(dataset_image_path,length)
    loader=torch.utils.data.DataLoader(dataset,batch_size=25,shuffle=False,num_workers=4)
    encoders,decoders=Encoders(parm),Decoders(parm)
    device=torch.device("cpu")
    nn.Module.load_state_dict(encoders,torch.load("../CelebA_out/2020-03-28_23:13:37/checkpoints/encoders.pth",map_location=device))
    nn.Module.load_state_dict(decoders,torch.load("../CelebA_out/2020-03-28_23:13:37/checkpoints/decoders.pth",map_location=device))
    if parm.useCuda:
        encoders=encoders.cuda()
        decoders=decoders.cuda()
    for name,data in loader:
        batch_data=batchDataTransform(data,3)
        base=getBaseGrid(imgSize=parm.imgSize,Inbatch=True,batchSize=batch_data.shape[0])
        if parm.useCuda:
            batch_data = batch_data.cuda()
            base = base.cuda()
        z,zI,zW,intersI,intersW=encoders(batch_data)
        texture,warp,out,_=decoders(zI,zW,intersI,intersW,base)
        images=out.detach().cpu().numpy()
        images=np.transpose(images,[0,2,3,1])
        images=np.array(images*255,dtype=np.uint8)
        for index,img in enumerate(images):
            temp=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(save_dir,name[index]),temp)

def evaluate():
    def index(score):
        ind=0
        if score>40 and score<=80:
            ind=1
        if score>80 and score<=90:
            ind=2
        if score>90:
            ind=3
        return ind
    path1 = "/home/qianqianjun/CODE/实验结果/重建结果/原始编码器"
    path2 = "/home/qianqianjun/CODE/实验结果/重建结果/改进编码器"
    path="/home/qianqianjun/CODE/DataSets/DaeDatasets"
    names=os.listdir(path1)
    request_url = "https://aip.baidubce.com/rest/2.0/face/v3/match"
    access_token = "24.7e8b426b9641416edee23272ea75dd2f.2592000.1588578630.282335-19253731"
    request_url = "{}?access_token={}".format(request_url, access_token)
    headers = {'content-type': 'application/json'}
    origin=[0,0,0,0,0]
    improve=[0,0,0,0,0]
    origin_scores=[]
    improve_scores=[]
    for name in names:
        data1=Data(os.path.join(path,name),os.path.join(path1,name))
        data2=Data(os.path.join(path,name),os.path.join(path2,name))
        params1 = data1.toJSON()
        params2=data2.toJSON()
        response1 = requests.post(request_url, data=params1, headers=headers)
        if response1:
            try:
                score1 = response1.json()["result"]["score"]
                origin[index(score1)] += 1
                origin_scores.append(score1)
            except:
                print(response1.json())
                origin[-1]+=1
        time.sleep(0.6)
        response2=requests.post(request_url,data=params2,headers=headers)
        if response2:
            try:
                score2=response2.json()["result"]["score"]
                improve[index(score2)] += 1
                improve_scores.append(score2)
            except:
                print(response2.json())
                improve[-1]+=1
        time.sleep(0.6)
    print(origin)
    print(improve)
    with open("origin","wb") as f:
        pk.dump(origin,f)
    with open("improve","wb") as f:
        pk.dump(improve,f)
    with open("origin_scores","wb") as f:
        pk.dump(origin_scores,f)
    with open("improve_scores","wb") as f:
        pk.dump(improve_scores,f)

import matplotlib.pyplot as plt
def showRect():
    with open("origin","rb") as f:
        origin=pk.load(f,encoding="utf-8")
        before=[origin[-1]]+ origin[:-1]
    with open("improve","rb") as f:
        improve=pk.load(f,encoding="utf-8")
        after=[improve[-1]]+ improve[:-1]
    x=range(len(before))
    label_list=['找不到人脸','可能性极低', '可能性较低', '可能性较高', '可能性极高']
    rects1=plt.bar(x,height=before,width=0.4,alpha=0.8,color="red",label="改进前")
    rects2=plt.bar([i+0.4 for i in x],width=0.4,height=after,color="blue",label="改进后")
    plt.ylabel("图片数量")
    plt.xticks([index + 0.2 for index in x], label_list)
    plt.xlabel("重建图像与原图同一人可能性级别")
    plt.title("重建图像质量评估")
    plt.legend()
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    plt.savefig(fname="rects.png",figsize=(10,10))
    plt.show()
def showHist():
    with open("origin_scores", "rb") as f:
        origin = pk.load(f, encoding="utf-8")
    with open("improve_scores", "rb") as f:
        improve = pk.load(f, encoding="utf-8")
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.hist(origin, bins=40, density=0, facecolor="red", edgecolor="black", alpha=0.7)
    plt.xlabel("相似度区间%")
    plt.ylabel("相似度频率")
    plt.title("改进前相似度分布直方图")
    plt.subplot(1,2,2)
    plt.hist(improve,bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("相似度区间%")
    #plt.ylabel("相似度频率")
    plt.title("改进后相似度分布直方图")
    plt.savefig(fname="hist.png",figsize=(8,4))
    plt.show()
# 调用训练好的模型准备重建图像
# prepare_image()
# 使用百度api得到人脸相似度
# evaluate()
# 绘制质量评估图
showRect()
# 绘制相似度直方图
showHist()