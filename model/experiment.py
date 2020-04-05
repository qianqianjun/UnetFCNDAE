# 插值实验用到的方法
import os

import cv2
import torch

from model.UnetDAE import Encoders, Decoders
from setting.parameter import parameter as parm
from tools.utils import ImgSeleter, batchDataTransform


# 加载模型的函数
def load_module(dataset_name:str,prefix_path):
    paths = os.listdir(os.path.join(prefix_path,dataset_name + "_"+parm.result_out))
    if len(paths) == 0:
        exit("没有找到预训练的模型，请先训练网络")
    model_dir = os.path.abspath(os.path.join(prefix_path,dataset_name+"_"+parm.result_out, paths[-1]))
    print("加载模型地址：{}".format(model_dir))
    encoders = Encoders(parm)
    decoders = Decoders(parm)
    if parm.useCuda:
        torch.nn.Module.load_state_dict(encoders,torch.load(os.path.join(model_dir,parm.dirCheckpoints,"encoders.pth")))
        torch.nn.Module.load_state_dict(decoders,torch.load(os.path.join(model_dir,parm.dirCheckpoints,"decoders.pth")))
        encoders=encoders.cuda()
        decoders=decoders.cuda()
    else:
        device = torch.device("cpu")
        # 加载预训练的模型
        torch.nn.Module.load_state_dict(encoders, torch.load(os.path.join(model_dir, parm.dirCheckpoints, "encoders.pth"),
                                                       map_location=device))
        torch.nn.Module.load_state_dict(decoders, torch.load(os.path.join(model_dir, parm.dirCheckpoints, "decoders.pth"),
                                                       map_location=device))

    return encoders,decoders,model_dir

# 加载数据集相关的函数
def test_AGFW(estimate_num:int=100,test_number:int=100,start_index:int=0,
              estimate_select_from=0,gender:str="male",levels=None):
    """
    提取用于属性估计的图片数据集以及测试的数据集
    :param estimate_num: 用于估计属性的图片集图片数量
    :param test_number:  要进行插值测试的图片的数目
    :param start_index:  测试图片开始索引
    :param estimate_select_from: 估计属性的图片开始的索引
    :param gender:  性别信息
    :param levels:  两个年龄的级别 ，默认是 age_15_19 和 age_40_44
    :return: 用户估计属性的两个数据集，以及需要测试的图像
    """
    dataset_path=os.path.join("/home/qianqianjun/下载/AGFW_cropped/cropped/128",gender)
    if levels is None:
        age1="age_15_19"
        age2="age_40_44"
    else:
        age1=levels[0]
        age2=levels[1]
    neg_imgs=[]
    pos_imgs=[]
    test_imgs=[]

    all_neg_images_path=os.listdir(os.path.join(dataset_path,age1))
    all_pos_images_path=os.listdir(os.path.join(dataset_path,age2))
    assert len(all_neg_images_path)>= estimate_select_from+estimate_num + start_index + test_number
    assert len(all_pos_images_path)>= estimate_select_from+estimate_num

    for name in all_neg_images_path[estimate_select_from:estimate_select_from+estimate_num]:
        img = cv2.imread(os.path.join(dataset_path,age1, name))
        img = cv2.resize(img, (parm.imgSize, parm.imgSize))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        neg_imgs.append(img)
    for name in all_pos_images_path[estimate_select_from:estimate_select_from+estimate_num]:
        img = cv2.imread(os.path.join(dataset_path,age2, name))
        img = cv2.resize(img, (parm.imgSize, parm.imgSize))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pos_imgs.append(img)

    begin=estimate_select_from+estimate_num+start_index
    for name in all_neg_images_path[begin:begin+test_number]:
        img=cv2.imread(os.path.join(dataset_path,age1,name))
        img=cv2.resize(img,(parm.imgSize,parm.imgSize))
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        test_imgs.append(img)

    return neg_imgs,pos_imgs,test_imgs

def test_JAFFE():
    pos_imgs=[]
    neg_imgs=[]
    test_imgs=[]
    images_path="/home/qianqianjun/下载/jaffe"
    all_images_path=os.listdir("/home/qianqianjun/下载/jaffe")
    for name in all_images_path:
        if not name.endswith("tiff"):
            continue
        if name.find("HA")!=-1:
            pos_imgs.append(
                cv2.resize(
                    cv2.imread(os.path.join(images_path,name)),
                    (parm.imgSize,parm.imgSize)))
        if name.find("NE") != -1:
            neg_imgs.append(
                cv2.resize(
                    cv2.imread(os.path.join(images_path,name)),
                    (parm.imgSize,parm.imgSize)))
    for item in neg_imgs:
        test_imgs.append(item)
    return neg_imgs[:30],pos_imgs[:30],test_imgs

def test_CelebA(target_attr:str,addition_attribute=None,test_num=100,
                estimate_from_index=0,estimate_num=100,start_index=0):
    if addition_attribute is None:
        addition_attribute=[]
    # target_attr="Mustache"
    # target_attr="Eyeglasses"
    # target_attr="Mouth_Slightly_Open"
    # target_attr="Smiling"
    # target_attr="Young"
    neg_imgs = []
    pos_imgs = []
    test_imgs = []
    seleter=ImgSeleter(parm.attr_path,parm.dataset_image_path)
    pos_imgs_paths=seleter.getImagePathsByCondition([target_attr],True,addition_attributes=["Male"])
    neg_imgs_paths=seleter.getImagePathsByCondition([target_attr],False,addition_attributes=["Male"])

    assert len(neg_imgs_paths)>=estimate_from_index+estimate_num+test_num+start_index
    assert len(pos_imgs_paths)>=estimate_from_index+estimate_num
    for name in pos_imgs_paths[estimate_from_index:estimate_from_index+estimate_num]:
        pos_imgs.append(cv2.cvtColor(cv2.resize(
                    cv2.imread(name),(parm.imgSize,parm.imgSize)),cv2.COLOR_BGR2RGB))
    for name in neg_imgs_paths[estimate_from_index:estimate_from_index+estimate_num]:
        neg_imgs.append(cv2.cvtColor(cv2.resize(
            cv2.imread(name),(parm.imgSize,parm.imgSize)
        ),cv2.COLOR_BGR2RGB))
    begin=estimate_from_index+estimate_num
    for name in neg_imgs_paths[begin:begin+start_index+test_num]:
        test_imgs.append(cv2.cvtColor(cv2.resize(
            cv2.imread(name),(parm.imgSize,parm.imgSize)
        ),cv2.COLOR_BGR2RGB))

    return neg_imgs,pos_imgs,test_imgs

# 获取估计的属性向量的函数
def getEstimateAttributeLantent(neg_imgs:list,pos_imgs:list,encoders):
    neg_tensor = batchDataTransform(torch.tensor(neg_imgs, dtype=torch.float32), channel=3)
    pos_tensor = batchDataTransform(torch.tensor(pos_imgs, dtype=torch.float32), channel=3)
    # 获取要插值的数据
    if parm.useCuda:
        neg_tensor=neg_tensor.cuda()
        pos_tensor=pos_tensor.cuda()
    zpos, zIpos, zWpos, pos_texture_inter_outs,pos_warp_inter_outs = encoders(pos_tensor)
    zneg, zIneg, zwneg, neg_texture_inter_outs,neg_warp_inter_outs = encoders(neg_tensor)

    # 纹理属性向量的提取
    zI_attribute = torch.mean(zIpos - zIneg, dim=0)
    # 变形属性向量的提取
    zW_attribute = torch.mean(zWpos - zwneg, dim=0)

    texture_inter_outs_attribute = []
    for pos_inter, neg_inter in zip(pos_texture_inter_outs, neg_texture_inter_outs):
        texture_inter_outs_attribute.append(
            torch.mean(pos_inter - neg_inter, dim=0)
        )

    warp_inter_outs_attribute = []
    for pos_inter, neg_inter in zip(pos_warp_inter_outs, neg_warp_inter_outs):
        warp_inter_outs_attribute.append(
            torch.mean(pos_inter - neg_inter, dim=0)
        )

    return zI_attribute,zW_attribute,texture_inter_outs_attribute,warp_inter_outs_attribute
