import os
import cv2
from tools.utils import saveIntermediateImage
from setting.parameter import parameter as parm
from tools.utils import batchDataTransform,getBaseGrid
from model.UnetDAE import Encoders,Decoders
import torch
n_sample,n_row=25,5
dataset_dir="/home/qianqianjun/CODE/DataSets/DaeDatasets"
with open("mustache.txt","r") as f:
    content=f.readlines()
    paths=[os.path.join(dataset_dir,name.strip()) for name in content]
    imgs_path=[]
    test_list=[0,1,2,4,6,8,9,12,13,14,16,17,22,24]
    for i in range(len(test_list)):
        imgs_path.append(paths[test_list[i]])
imgs=[cv2.resize(cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB),(parm.imgSize,parm.imgSize)) for path in imgs_path]
batch_data=batchDataTransform(torch.tensor(imgs,dtype=torch.float32),3)

model_dir="../CelebA_out/2020-03-28_23:13:37/checkpoints"
encoders,decoders=Encoders(parm),Decoders(parm)
torch.nn.Module.load_state_dict(encoders,torch.load(os.path.join(model_dir,"encoders.pth")))
torch.nn.Module.load_state_dict(decoders,torch.load(os.path.join(model_dir,"decoders.pth")))
if parm.useCuda:
    encoders=encoders.cuda()
    decoders=decoders.cuda()
    batch_data=batch_data.cuda()
_,test_texture_code,test_warp_code,test_texture_inters,test_warp_inters=encoders(batch_data)

attr_path="../dependency/mustache"
import pickle as pk
with open(attr_path,"rb") as f:
    attr=pk.load(f,encoding="utf-8")
texture_code=attr["texture_code"]
warp_code=attr["warp_code"]
texture_inters=attr["texture_inter_outs_attribute"]
warp_inters=attr["warp_inter_outs_attribute"]

lambda_texture_inter_out=[2,2,4,4]
lambda_warp_inter_out=[2,2,4,4]
lambda_deepest_texture=4
lambda_deepest_warp=4
if parm.useCuda:
    texture_code=texture_code.cuda()
    warp_code=warp_code.cuda()
    for i in range(len(texture_inters)):
        texture_inters[i]=texture_inters[i].cuda()
    for i in range(len(warp_inters)):
        warp_inters[i]=warp_inters[i].cuda()

texture_code_interpolation_result= test_texture_code + lambda_deepest_texture * texture_code.unsqueeze(0).repeat(test_texture_code.size()[0], 1)
texture_inter_interpolation_result=[]
for i in range(len(test_texture_inters)):
    texture_inter_interpolation_result.append(
        test_texture_inters[i] + lambda_texture_inter_out[i] *
        texture_inters[i].repeat(test_texture_inters[i].size()[0], 1, 1, 1)
    )
### 变形信息插值
warp_code_interpolation_result= test_warp_code + lambda_deepest_warp * warp_code.unsqueeze(0).repeat(test_warp_code.size()[0], 1)
warp_inter_interpolation_result=[]
for i in range(len(test_warp_inters)):
    warp_inter_interpolation_result.append(
        test_warp_inters[i] + lambda_warp_inter_out[i] *
        warp_inters[i].repeat(test_warp_inters[i].size()[0], 1, 1, 1)
    )
# 结果图像重建
base=getBaseGrid(parm.imgSize,True,batch_data.size()[0])
if parm.useCuda:
    base=base.cuda()

texture_result,warp_result,result,_=decoders(texture_code_interpolation_result,
                                             warp_code_interpolation_result,
                                             texture_inter_interpolation_result,
                                             warp_inter_interpolation_result,base)


result1=torch.cat([batch_data[:7],result[:7]],dim=0)
result2=torch.cat([batch_data[7:],result[7:]],dim=0)
res=torch.cat([result1,result2],dim=0)
saveIntermediateImage(
    img_list=res.data.clone(),
    output_dir="out",
    filename="胡须插值结果",n_sample=28,nrow=7,padding=4)