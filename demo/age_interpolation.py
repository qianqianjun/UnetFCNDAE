from model.UnetDAE import Encoders,Decoders
from setting.parameter import parameter as parm
from tools.utils import getBaseGrid,batchDataTransform,saveIntermediateImage
import torch
import pickle as pk
import os
import cv2

def interpolation(attr_path, decoders,batch_data,
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
# 加载模型
data_dir="/home/qianqianjun/下载/AGFW_cropped/cropped/128/male/age_15_19"
model_dir="../AGFW_out/2020-04-01_19:49:07/checkpoints"
attr_path="../dependency/age"
# start=217
encoders=Encoders(parm)
decoders=Decoders(parm)
device=torch.device("cpu")
torch.nn.Module.load_state_dict(encoders,torch.load(os.path.join(model_dir,"encoders.pth"),map_location=device))
torch.nn.Module.load_state_dict(decoders,torch.load(os.path.join(model_dir,"decoders.pth"),map_location=device))
# 加载数据
with open("age.txt","r") as f:
    contents=f.readlines()
    display_list=[os.path.join(data_dir,line.strip()) for line in contents]
test_imgs=[
    cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(data_dir,item)),cv2.COLOR_BGR2RGB),(parm.imgSize,parm.imgSize))
    for item in display_list
]
batch_data=batchDataTransform(torch.tensor(test_imgs,dtype=torch.float32),channel=3)
if parm.useCuda:
    batch_data=batch_data.cuda()
    encoders=encoders.cuda()
_,test_texture_code,test_warp_code,test_texture_inters,test_warp_inters=encoders(batch_data)
result=interpolation(
    attr_path,decoders,batch_data,
    test_texture_code,test_warp_code,
    test_texture_inters,test_warp_inters,
    lambda_texture_inter_out=[0.5,1,4,0],lambda_warp_inter_out=[0.5,1,4,0],
    lambda_deepest_texture=0,lambda_deepest_warp=0
)
res1=torch.cat([batch_data[0:7],result[0:7]],dim=0)
res2=torch.cat([batch_data[7:14],result[7:14]],dim=0)
res3=torch.cat([batch_data[14:21],result[14:21]],dim=0)

res=torch.cat([res1,res2,res3],dim=0)
saveIntermediateImage(
    img_list=res,output_dir="out",n_sample=42,
    filename="年龄插值结果",nrow=7,padding=2
)