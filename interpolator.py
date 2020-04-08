"""
那一年，我背井离乡。
从此，乡亲们再也没有喝过一口井水
write by qianqianjun
2020.03.07
深度特征插值过程
"""
from tools.utils import *

# 加载预训练的模型
parm.useCuda=False
dataset_name="JAFFE"
encoders,decoders,model_dir=load_module(dataset_name,os.path.abspath("."))
# 读取实验数据集
neg_imgs,pos_imgs,test_imgs=test_AGFW(estimate_num=100,start_index=0,test_number=25,gender="male")
# neg_imgs,pos_imgs,test_imgs=test_CelebA(target_attr="Mustache",addition_attribute=["Male"],test_num=5,
#                                        start_index=0,estimate_num=50,estimate_from_index=0)
# neg_imgs,pos_imgs,test_imgs=test_JAFFE()


print("加载数据完毕")
texture_code, warp_code, texture_inters, warp_inters=getEstimateAttributeLantent(neg_imgs, pos_imgs, encoders)
# 读取测试文件
batch_data=batchDataTransform(torch.tensor(test_imgs,dtype=torch.float32),3)
# 获得测试文件隐空间
if parm.useCuda:
    batch_data=batch_data.cuda()
_, test_texture_code, test_warp_code, test_texture_inters, test_warp_inters=encoders(batch_data)
print("属性估计完成")
# 特征插值
# 设置级别
lambda_texture_inter_out=[0.5,3,8,16]
lambda_warp_inter_out=[0.5,3,8,16]
lambda_deepest_texture=16
lambda_deepest_warp=16

### 纹理插值
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
print("开始重建图像")
# 重建图片
test_texture, test_warp, test_out, _=decoders(test_texture_code, test_warp_code, test_texture_inters,test_warp_inters,base)

import matplotlib.pyplot as plt
import numpy as np
def fun(baseGrid:torch.Tensor,W:torch.Tensor):
    plt.figure(figsize=(8,8))
    for i in range(3):
        for j in range(3):
            base=baseGrid.detach().cpu().numpy()[i*3 +j]
            warp=W.detach().cpu().numpy()[i*3 +j]
            res = base + warp

            plt.subplot(3,3,i*3+j+1)
            res=np.transpose(res,axes=[1,2,0])
            res=np.concatenate([res],axis=2)
            for row in range(res.shape[0]):
                plt.plot(res[res.shape[0]-1-row,:,0],res[res.shape[0]-1-row,:,1],color='red')
            for col in range(res.shape[1]):
                plt.plot(res[:,res.shape[1]-1-col, 0], res[:,res.shape[1]-1-col, 1], color='green')

    plt.show()
fun(base,test_warp)

# 纹理和形状同时插值
texture_all_interpolation, warp_all_interpolation, out_all_interpolation, _Ip=\
    decoders(texture_code_interpolation_result, warp_code_interpolation_result,
             texture_inter_interpolation_result,warp_inter_interpolation_result,base)
# 单独纹理插值
texture_onlyT_interpolation,warp_onlyT_interpolation,out_onlyT_interpolation,_=\
    decoders(texture_code_interpolation_result,test_warp_code,
             texture_inter_interpolation_result,test_warp_inters,base)
# 单独变形插值
texture_onlyW_interpolation,warp_onlyW_interpolation,out_onlyW_interpolation,_=\
    decoders(test_texture_code,warp_code_interpolation_result,
             test_texture_inters,warp_inter_interpolation_result,base)
# 保存图像结果
print("图像结果保存")
n_sample=9
n_row=3
saveIntermediateImage(
    img_list=batch_data.data.clone(),
    output_dir=os.path.join(model_dir,parm.dirTestOutput),
    filename="01原图",n_sample=n_sample,nrow=n_row)
saveIntermediateImage(
    img_list=test_out.data.clone(),
    output_dir=os.path.join(model_dir,parm.dirTestOutput),
    filename="02无插值重建结果",n_sample=n_sample,nrow=n_row)
saveIntermediateImage(
    img_list=out_onlyT_interpolation.data.clone(),
    output_dir=os.path.join(model_dir,parm.dirTestOutput),
    filename="03只插值纹理结果",n_sample=n_sample,nrow=n_row)
saveIntermediateImage(
    img_list=out_onlyW_interpolation.data.clone(),
    output_dir=os.path.join(model_dir,parm.dirTestOutput),
    filename="04只插值形状结果",n_sample=n_sample,nrow=n_row)
saveIntermediateImage(
    img_list=out_all_interpolation.data.clone(),
    output_dir=os.path.join(model_dir,parm.dirTestOutput),
    filename="05组合插值结果",n_sample=n_sample,nrow=n_row)