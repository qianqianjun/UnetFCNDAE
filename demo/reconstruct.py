from model.UnetDAE import Encoders,Decoders
from setting.parameter import parameter as parm
from tools.utils import ImgSeleter,batchDataTransform,saveIntermediateImage,getBaseGrid
import torch
import os
import torch.nn as nn
model_path="../CelebA_out/2020-03-28_23:13:37/checkpoints"
attr_file_path="/home/qianqianjun/CODE/DataSets/CelebA/Anno/list_attr_celeba.txt"
image_data_path="/home/qianqianjun/CODE/DataSets/DaeDatasets"
encoders=Encoders(parm)
decoders=Decoders(parm)
device=torch.device("cpu")
nn.Module.load_state_dict(encoders,torch.load(os.path.join(model_path,"encoders.pth"),map_location=device))
nn.Module.load_state_dict(decoders,torch.load(os.path.join(model_path,"decoders.pth"),map_location=device))

selecter=ImgSeleter(attr_file_path,image_data_path)
imgs=selecter.getImagesWithoutAttribute("Mustache",["Male"],amount=207,shuffle=False)[200:]

tensor=batchDataTransform(torch.tensor(imgs,dtype=torch.float32),channel=3)

z,zI,zW,inters_I,inters_W=encoders(tensor)
base=getBaseGrid(parm.imgSize,True,tensor.shape[0])
texture,res_warp,out,warp=decoders(zI,zW,inters_I,inters_W,base)

result=torch.cat([tensor,texture,out],dim=0)
saveIntermediateImage(result.data.clone(),n_sample=21,nrow=7,filename="重建结果",output_dir="out")