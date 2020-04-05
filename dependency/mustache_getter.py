import os

import torch

from model.UnetDAE import Encoders, Decoders
from setting.parameter import parameter as parm
from tools.utils import batchDataTransform

model_path="../CelebA_out/2020-03-28_23:13:37/checkpoints"

parm.useCuda=False

# 导入模型
encoders=Encoders(parm)
decoders=Decoders(parm)
device=torch.device("cpu")
torch.nn.Module.load_state_dict(encoders,torch.load(os.path.join(model_path,"encoders.pth"),map_location=device))
torch.nn.Module.load_state_dict(decoders,torch.load(os.path.join(model_path,"decoders.pth"),map_location=device))

# 导入数据
pos_path="/home/qianqianjun/桌面/胡子人脸"
neg_path="/home/qianqianjun/桌面/男星"
import cv2
pos_imgs=[
    cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(pos_path,name)),cv2.COLOR_BGR2RGB),(parm.imgSize,parm.imgSize))
    for name in os.listdir(pos_path)
]
neg_imgs=[
    cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(neg_path,name)),cv2.COLOR_BGR2RGB),(parm.imgSize,parm.imgSize))
    for name in os.listdir(neg_path)
]

# 制作tensor
pos_tensor=batchDataTransform(torch.tensor(pos_imgs,dtype=torch.float32),3)
neg_tensor=batchDataTransform(torch.tensor(neg_imgs,dtype=torch.float32),3)

# 提取属性中间
_,pos_texture_code, pos_warp_code, pos_texture_inters, pos_warp_inters=encoders(pos_tensor)
_,neg_texture_code,neg_warp_code,neg_texture_inters,neg_warp_inters=encoders(neg_tensor)

texture_code_attribute=torch.mean(pos_texture_code,dim=0)-torch.mean(neg_texture_code,dim=0)
warp_code_attribute=torch.mean(pos_warp_code,dim=0)-torch.mean(neg_warp_code,dim=0)

texture_inter_outs_attribute = []
for pos_inter, neg_inter in zip(pos_texture_inters, neg_texture_inters):
    texture_inter_outs_attribute.append(
        torch.mean(pos_inter,dim=0)- torch.mean(neg_inter, dim=0)
    )

warp_inter_outs_attribute = []
for pos_inter, neg_inter in zip(pos_warp_inters, neg_warp_inters):
    warp_inter_outs_attribute.append(
        torch.mean(pos_inter,dim=0) - torch.mean(neg_inter, dim=0)
    )

attribute={}
attribute["texture_code"]=texture_code_attribute
attribute["warp_code"]=warp_code_attribute
attribute["texture_inter_outs_attribute"]=texture_inter_outs_attribute
attribute["warp_inter_outs_attribute"]=warp_inter_outs_attribute
import pickle as pk
with open("mustache","wb") as f:
    pk.dump(attribute,f)