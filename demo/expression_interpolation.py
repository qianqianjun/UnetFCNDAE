import os
import pickle as pk
from tools.utils import saveIntermediateImage
from setting.parameter import parameter as parm
from tools.utils import getBaseGrid,batchDataTransform
from model.UnetDAE import Encoders,Decoders
import torch
import numpy as np
import cv2
def interpolation(attr_path, batch_data, decoders,
                  test_texture_code, test_warp_code,
                  test_texture_inters,test_warp_inters,**lambda_params):
    # 属性向量加载
    with open(attr_path, "rb") as f:
        attribute = pk.load(f, encoding="utf-8")
        texture_code = attribute["texture_code"]
        warp_code = attribute["warp_code"]
        texture_inters = attribute["texture_inter_outs_attribute"]
        warp_inters = attribute["warp_inter_outs_attribute"]
    # 插值系数
    lambda_texture_inter_out = lambda_params["lambda_texture_inter_out"]
    lambda_warp_inter_out = lambda_params["lambda_warp_inter_out"]
    lambda_deepest_texture = lambda_params["lambda_deepest_texture"]
    lambda_deepest_warp = lambda_params["lambda_deepest_warp"]

    base = getBaseGrid(parm.imgSize, True, batch_data.size()[0])
    if parm.useCuda:
        decoders = torch.nn.Module.cuda(decoders)
        batch_data = batch_data.cuda()
        texture_code = texture_code.cuda()
        warp_code = warp_code.cuda()
        for i in range(len(texture_inters)):
            texture_inters[i] = texture_inters[i].cuda()
        for i in range(len(warp_inters)):
            warp_inters[i] = warp_inters[i].cuda()
        base = base.cuda()

    # 纹理信息插值
    texture_code_interpolation_result = test_texture_code + lambda_deepest_texture * texture_code.unsqueeze(0).repeat(
        test_texture_code.size()[0], 1)
    texture_inter_interpolation_result = []
    for i in range(len(test_texture_inters)):
        texture_inter_interpolation_result.append(
            test_texture_inters[i] + lambda_texture_inter_out[i] *
            texture_inters[i].repeat(test_texture_inters[i].size()[0], 1, 1, 1)
        )
    ### 变形信息插值
    warp_code_interpolation_result = test_warp_code + lambda_deepest_warp * warp_code.unsqueeze(0).repeat(
        test_warp_code.size()[0], 1)
    warp_inter_interpolation_result = []
    for i in range(len(test_warp_inters)):
        warp_inter_interpolation_result.append(
            test_warp_inters[i] + lambda_warp_inter_out[i] *
            warp_inters[i].repeat(test_warp_inters[i].size()[0], 1, 1, 1)
        )
    # 结果图像重建
    texture_result, warp_result, result, _ = decoders(texture_code_interpolation_result,
                                                      warp_code_interpolation_result,
                                                      texture_inter_interpolation_result,
                                                      warp_inter_interpolation_result, base)
    return result
n_sample,n_row=8,4
data_dir="/home/qianqianjun/下载/jaffe"
# 模型加载
model_dir="../JAFFE_out/2020-03-27_21:38:52/checkpoints"
encoders,decoders=Encoders(parm),Decoders(parm)
torch.nn.Module.load_state_dict(encoders,torch.load(os.path.join(model_dir,"encoders.pth")))
torch.nn.Module.load_state_dict(decoders,torch.load(os.path.join(model_dir,"decoders.pth")))
# 加载数据
paths=os.listdir(data_dir)
test_imgs=[]
display_index=[1,2,3,4,6,7,9]
for index,i in enumerate(paths):
    if not i.endswith(".tiff"):
        continue
    if i.find("NE") !=-1:
        test_imgs.append(
            cv2.resize(
                cv2.imread(os.path.join(data_dir,i)),
                (parm.imgSize, parm.imgSize)))
test_imgs=np.array(test_imgs)
test_imgs=test_imgs[display_index]

# 测试图像编码
batch_data=batchDataTransform(torch.tensor(test_imgs,dtype=torch.float32),channel=3)
if parm.useCuda:
    encoders=encoders.cuda()
    batch_data=batch_data.cuda()
_,test_texture_code,test_warp_code,test_texture_inters,test_warp_inters=encoders(batch_data)
# 微笑插值
smile=interpolation(
    "../dependency/smile",batch_data,decoders,
    test_texture_code,test_warp_code,
    test_texture_inters,test_warp_inters,
    lambda_texture_inter_out=[0,4,8,0],lambda_warp_inter_out=[0,4,8,0],
    lambda_deepest_texture=0,lambda_deepest_warp=0
)
sad=interpolation(
    "../dependency/sad",batch_data,decoders,
    test_texture_code,test_warp_code,
    test_texture_inters,test_warp_inters,
    lambda_texture_inter_out=[1,4,4,0],lambda_warp_inter_out=[1,4,4,0],
    lambda_deepest_texture=1,lambda_deepest_warp=1
)
angry=interpolation(
    "../dependency/angry",batch_data,decoders,
    test_texture_code,test_warp_code,
    test_texture_inters,test_warp_inters,
    lambda_texture_inter_out=[1,2,4,1],lambda_warp_inter_out=[1,2,4,1],
    lambda_deepest_texture=1,lambda_deepest_warp=1
)
surprise=interpolation(
    "../dependency/surprise",batch_data,decoders,
    test_texture_code,test_warp_code,
    test_texture_inters,test_warp_inters,
    lambda_texture_inter_out=[0,2,8,0],lambda_warp_inter_out=[0,2,8,0],
    lambda_deepest_texture=0,lambda_deepest_warp=0
)
disgust=interpolation(
    "../dependency/disgust",batch_data,decoders,
    test_texture_code,test_warp_code,
    test_texture_inters,test_warp_inters,
    lambda_texture_inter_out=[0,4,4,0],lambda_warp_inter_out=[0,4,4,0],
    lambda_deepest_texture=0,lambda_deepest_warp=0
)
fear=interpolation(
    "../dependency/fear",batch_data,decoders,
    test_texture_code,test_warp_code,
    test_texture_inters,test_warp_inters,
    lambda_texture_inter_out=[0,4,4,0],lambda_warp_inter_out=[0,2,4,0],
    lambda_deepest_texture=0,lambda_deepest_warp=0
)
result1=torch.cat([batch_data,smile,sad,angry,surprise,disgust,fear],dim=0)
saveIntermediateImage(
    img_list=result1.data.clone(),
    output_dir="out",
    filename="表情插值结果",n_sample=28,nrow=7,padding=2)