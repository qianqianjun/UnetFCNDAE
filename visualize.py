import os

import cv2
import torch
import torch.nn as nn

from model.UnetDAE import Encoders, Decoders
from setting.parameter import parameter as parm
from tools.utils import getBaseGrid, batchDataTransform,saveIntermediateImage

encoders=Encoders(parm)
decoders=Decoders(parm)
model_path="out/other/2020-04-06_14:04:27/checkpoints"
device=torch.device("cpu")
nn.Module.load_state_dict(encoders,torch.load(os.path.join(model_path,"encoders.pth"),map_location=device))
nn.Module.load_state_dict(decoders,torch.load(os.path.join(model_path,"decoders.pth"),map_location=device))

img_dir="/home/qianqianjun/桌面/杨洋男"
names=os.listdir(img_dir)
imgs_path=[os.path.join(img_dir,name) for name in names]
batch_data=batchDataTransform(
    torch.tensor(
        [
            cv2.cvtColor(cv2.resize(cv2.imread(img_path),(parm.imgSize,parm.imgSize)),cv2.COLOR_BGR2RGB)
            for img_path in imgs_path[:9]
        ],
        dtype=torch.float32
    ),
    channel=3
)
base = getBaseGrid(imgSize=parm.imgSize, Inbatch=True, batchSize=batch_data.shape[0])
if parm.useCuda:
    batch_data=batch_data.cuda()
    encoders=encoders.cuda()
    decoders=decoders.cuda()
    base=base.cuda()

print("开始")
import time
start=time.time()
z,zI,zW,inters_t,inters_w=encoders(batch_data)
texture,warp,out,_=decoders(zI,zW,inters_t,inters_w,base)
end=time.time()
print(end-start)

### 绘制变形场
import matplotlib.pyplot as plt
import numpy as np
def fun(baseGrid:torch.Tensor,W:torch.Tensor,n_sample:int=9,n_row:int=3):
    assert n_row**2==n_sample
    plt.figure(figsize=(8,8))
    for i in range(n_row):
        for j in range(n_row):
            base=baseGrid.detach().cpu().numpy()[i*n_row +j]
            warp=W.detach().cpu().numpy()[i*n_row +j]
            res = base + warp

            plt.subplot(n_row,n_row,i*n_row+j+1)
            res=np.transpose(res,axes=[1,2,0])
            res=np.concatenate([res],axis=2)
            for row in range(res.shape[0]):
                plt.plot(res[res.shape[0]-1-row,:,0],-res[res.shape[0]-1-row,:,1],color='red')
            for col in range(res.shape[1]):
                plt.plot(res[:,res.shape[1]-1-col, 0], -res[:,res.shape[1]-1-col, 1], color='green')
    plt.savefig(fname="haha.png")
    plt.show()
fun(base,warp)

print(warp.shape)

saveIntermediateImage(img_list=batch_data.data.clone(),output_dir=".",n_sample=9,nrow=3,padding=2,filename="原图")
saveIntermediateImage(out.data.clone(),output_dir=".",n_sample=9,nrow=3,filename="重建",padding=2)
saveIntermediateImage(texture.data.clone(),output_dir=".",n_sample=9,filename="纹理",nrow=3,padding=2)