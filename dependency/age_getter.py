import os

import cv2
import torch

from model.UnetDAE import Encoders
from setting.parameter import parameter as parm
from tools.utils import batchDataTransform

young_path="/home/qianqianjun/下载/AGFW_cropped/cropped/128/male/age_15_19"
# old_path="/home/qianqianjun/下载/AGFW_cropped/cropped/128/male/老人脸"
old_path="/home/qianqianjun/下载/AGFW_cropped/cropped/128/male/age_40_44"
path1=os.listdir(young_path)
path2=os.listdir(old_path)
young_imgs=[cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(young_path,item)),cv2.COLOR_BGR2RGB),(parm.imgSize,parm.imgSize))
            for item in path1[:100]]
old_imgs=[cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(old_path,item)),cv2.COLOR_BGR2RGB),(parm.imgSize,parm.imgSize))
          for item in path2[:100]]

neg_tensor=batchDataTransform(torch.tensor(young_imgs,dtype=torch.float32),channel=3)
pos_tensor=batchDataTransform(torch.tensor(old_imgs,dtype=torch.float32),channel=3)

encoders=Encoders(parm)
# 加载模型参数
model_path="../AGFW_out/2020-04-01_19:49:07/checkpoints/encoders.pth"
torch.nn.Module.load_state_dict(encoders,torch.load(model_path,map_location=torch.device("cpu")))
parm.useCuda=False
if parm.useCuda:
    neg_tensor=neg_tensor.cuda()
    pos_tensor=pos_tensor.cuda()
    encoders=torch.nn.Module.cuda(encoders)
_,pos_tcode,pos_wcode,pos_tinters,pos_winters=encoders(pos_tensor)
_,neg_tcode,neg_wcode,neg_tinters,neg_winters=encoders(neg_tensor)
attr_tcode=torch.mean(pos_tcode,dim=0) - torch.mean(neg_tcode,dim=0)
attr_wcode=torch.mean(pos_wcode,dim=0) - torch.mean(neg_wcode,dim=0)
attr_tinters=[]
for pos,neg in zip(pos_tinters,neg_tinters):
    attr_tinters.append(
        torch.mean(pos,dim=0)-torch.mean(neg,dim=0)
    )
attr_winters=[]
for pos,neg in zip(pos_winters,neg_winters):
    attr_winters.append(
        torch.mean(pos,dim=0)-torch.mean(neg,dim=0)
    )
# 保存估计的属性向量
age={}
age["texture_code"]=attr_tcode
age["warp_code"]=attr_wcode
age["texture_inter_outs_attribute"]=attr_tinters
age["warp_inter_outs_attribute"]=attr_winters
import pickle as pk
with open("age","wb") as f:
    pk.dump(age,f)