"""
某日，花木兰的战友来木兰家探望，看到木兰正在刺绣，
所绣人物与木兰的母亲非常神似，但战友又不确定所绣之人是不是木兰的母亲，
遂问道：你秀你妈呢？

write by qianqianjun
2020.02.05
工具方法
"""
import datetime as date
import os
import pickle as pk
import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.utils as Util

from model.UnetDAE import Encoders, Decoders
from setting.parameter import Parameter
from setting.parameter import parameter as parm


def init_model(parm:Parameter,dataset_name:str,path_prefix:str):
    """
    训练前初始化模型。
    在Fine-tune 模式下，自动加载预训练模型进行微调训练
    Fine-tune 为 False 或者没有预训练的模型，从头开始训练
    :param parm:
    :param dataset_name:  训练使用的数据集名称
    :param path_prefix:  路径前缀，使用绝对路径
    :return:
    """
    encoders=Encoders(parm)
    decoders=Decoders(parm)
    if parm.useCuda:
        encoders=nn.Module.cuda(encoders)
        decoders=nn.Module.cuda(decoders)
    model_path=os.path.join(path_prefix,parm.result_out,dataset_name)
    if os.path.exists(model_path) and len(os.listdir(model_path)) >=1:
        path=os.path.join(model_path,os.listdir(model_path)[-1],parm.dirCheckpoints)
        nn.Module.load_state_dict(encoders,torch.load(
            os.path.join(path,"encoders.pth")
        ))
        nn.Module.load_state_dict(decoders,torch.load(
            os.path.join(path,"decoders.pth")
        ))
        if not os.path.exists(os.path.join(path,"info")):
            start_epoch=0
            learning_rate=parm.learning_rate
        else:
            with open(os.path.join(path,"info"),"rb") as f:
                info=pk.load(f,encoding="utf-8")
            start_epoch=info.get("epoch")
            learning_rate=info.get("lr")
    else:
        nn.Module.apply(encoders,weight_init)
        nn.Module.apply(decoders,weight_init)
        start_epoch=0
        learning_rate=parm.learning_rate
    return start_epoch,learning_rate,encoders,decoders

def init_train_env(parm,dataset_name:str,path_prefix):
    """
    初始化训练环境
    :param parm:
    :param path_prefix: 训练文件所在的绝对目录
    :return:
    """
    save_time_path=date.datetime.strftime(date.datetime.now(),"%Y-%m-%d_%H:%M:%S")
    # 实验可重复性设置（设置随机种子）
    if parm.randomSeed is None:
        parm.randomSeed = random.randint(1, 100000000)
    print("本次实验使用的随机种子", parm.randomSeed)

    random.seed(parm.randomSeed)
    np.random.seed(parm.randomSeed)
    torch.manual_seed(parm.randomSeed)

    # 强制使用 GPU 加速
    if torch.cuda.is_available() and not parm.useCuda:
        print("当前GPU 设备已就绪，请使用GPU加速: --useCuda=True")
        exit(0)

    # 创建实验用到的目录环境
    public=os.path.join(path_prefix,parm.result_out,dataset_name,save_time_path)
    os.makedirs(os.path.join(public,parm.dirCheckpoints),exist_ok=True)
    os.makedirs(os.path.join(public,parm.dirImageOutput),exist_ok=True)
    os.makedirs(os.path.join(public, parm.dirTestOutput),exist_ok=True)
    return public

# 此处有 bug ！
def weight_init(model):
    """
    用于初始化模型参数
    :param model: 调用这个方法进行初始化的模型
    :return: None
    """
    cname=model.__class__.__name__
    if cname.find("Conv") !=-1:
        model.weight.data.normal_(0.0,0.02)
    elif cname.find("BatcNorm") !=-1:
        model.weight.data.normal_(1.0,0.02)
        model.bias.data.fill_(0)


def batchDataTransform(batch_image:torch.Tensor,channel:int):
    """
    对批数据进行处理，包括数据归一化和维度顺序变换
    :param batch_image: 批图片 tensor
    :param channel: 通道数目
    :return:
    """
    batch_image=batch_image.float() / 255.0 # 数据进行标准化
    # 如果是灰度图形，需要增加一个维度。
    if channel==1:
        batch_image=batch_image.unsqueeze(3)

    # 将数据转换为 【N,C,H,W】 模式
    batch_image=batch_image.permute(0,3,1,2).contiguous()
    return batch_image

def getBaseGrid(imgSize:int=64,Inbatch:bool=False,batchSize:int=1,normalize=True):
    """
    get基础变形场
    :param imgSize: 图片大小
    :param Inbatch:  是否是批量数据
    :param batchSize:  批量数据的大小
    :param normalize:  是否进行归一化
    :return:
    """
    a=torch.arange(-(imgSize-1 ),imgSize,2) # 这里有bug
    if normalize:
        a=a /(imgSize-1.0) # 修改一个大 bug。
    x=a.repeat(imgSize,1) #在行的维度上重复imgsize次，在列上重复 1 次
    y=x.t()

    grid=torch.cat((x.unsqueeze(0),y.unsqueeze(0)),0)

    if Inbatch:
        grid=grid.unsqueeze(0).repeat(batchSize,1,1,1)
    #grid=torch.tensor(grid,requires_grad=False)
    grid=grid.clone().detach().requires_grad_(False)

    return grid

def saveIntermediateImage(img_list:torch.Tensor, output_dir:str, n_sample:int=4,
                          id_samele=None, dim:int=-1, filename:str="myimage", nrow:int=2, normalize:bool=False,
                          padding=2):
    """
    用于保存和可视化图片
    :param img_list:  图片 tensor
    :param output_dir:  图片输出目录
    :param n_sample:  有多少个样例
    :param id_samele:  样例的下标
    :param dim:  根据维度信息来决定是否进行维度增加
    :param filename:  保存的文件名称
    :param nrow:
    :param normalize: 是否进行规范化操作
    :return:
    """
    if id_samele is None:
        images=img_list[0:n_sample,:,:,:]
    else:
        images=img_list[id_samele,:,:,:]
    if dim > 0:
        images=images[:,dim,:,:].unsqueeze(1)

    Util.save_image(images,"{}/{}.png".format(output_dir,filename),nrow=nrow,normalize=normalize,padding=padding)
class ImgSeleter:
    def __init__(self, attr_file_path, image_path):
        self.attr_file_path = attr_file_path
        self.image_path = image_path
        assert os.path.exists(self.attr_file_path), "属性说明文件缺失"
        assert os.path.exists(self.image_path), "数据集图片文件缺失"

        # 读取属性列文件
        self.datafram = pd.read_csv(self.attr_file_path, delim_whitespace=True, index_col=0, header=1)
        # 获取所有属性
        self.attributes = [attribute for attribute in self.datafram.keys()]

    def getImagesWithAttribute(self, attribute: str, additions: list = None, amount: int = None, shuffle: bool = True):
        """
        获取具有某一属性的一些图片的地址
        :param attribute: 需要具有的属性
        :param additions:  其它需要满足的附加属性，可以为空。
        :param amount: 取得的数量
        :param shuffle: 是否进行顺序随机化，默认是随机的
        :return: 返回一些具有些属性和附加属性的图片地址
        """
        # 过滤附加属性
        result = self.datafram
        if additions is not None:
            for addition in additions:
                result = result[result[addition] > 0]

        result = result[result[attribute] > 0]
        # 读取图片的地址
        image_absoult_paths=np.array(
            [os.path.join(self.image_path,name) for name in result.index])
        assert len(image_absoult_paths) > 0,"没有满足条件的图片"
        # 打乱顺序，洗牌
        if shuffle:
            order=np.random.permutation(len(image_absoult_paths))
            image_absoult_paths=image_absoult_paths[order]
        # 读取数量
        if amount is not None:
            assert amount <=len(image_absoult_paths)
            image_absoult_paths=image_absoult_paths[:amount]
        # 读取图片数据
        imgs = np.array([cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB) for path in image_absoult_paths])
        return imgs

    def getImagesWithoutAttribute(self, attribute: str, additions: list = None, amount: int = None,
                                  shuffle: bool = True):
        """
        找到没有某一属性的图片
        :param attribute:
        :param additions:
        :param amount:
        :param shuffle:
        :return:
        """
        # 过滤附加属性
        result = self.datafram
        if additions is not None:
            for addition in additions:
                result = result[result[addition] > 0]

        result = result[result[attribute] < 0]
        # 读取图片的地址
        image_absoult_paths = np.array(
            [os.path.join(self.image_path, name) for name in result.index])
        assert len(image_absoult_paths) > 0, "没有满足条件的图片"
        # 打乱顺序，洗牌
        if shuffle:
            order = np.random.permutation(len(image_absoult_paths))
            image_absoult_paths = image_absoult_paths[order]
        # 读取数量
        if amount is not None:
            assert amount <= len(image_absoult_paths)
            image_absoult_paths = image_absoult_paths[:amount]
        # 读取图片数据
        imgs = np.array([cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB) for path in image_absoult_paths])
        return imgs

    def getImagePathsByAttributes(self,attributes:list,contain:bool=True,shuffle:bool=True):
        assert all(attribute in self.attributes for attribute in attributes) ,\
            "属性列表中出现非法属性"

        csv=self.datafram
        if contain:
            for attribute in attributes:
                csv=csv[ csv[attribute] > 0 ]
        else:
            for attribute in attributes:
                csv=csv[ csv[attribute] <0 ]

        indexs=csv.index
        assert len(indexs)>0
        files_path=[os.path.join(self.image_path,name) for name in indexs]
        files_path=np.array(files_path)
        if shuffle:
            permutation=np.random.permutation(len(files_path))
            files_path=files_path[permutation]
        return files_path

    def getImagePathsByCondition(self,attributes:list,contain:bool=True,
                                 addition_attributes:list=None,shuffle:bool=False):
        assert all(attribute in self.attributes for attribute in attributes), \
            "属性列表中出现非法属性"

        csv = self.datafram
        if contain:
            for attribute in attributes:
                csv = csv[csv[attribute] > 0]
        else:
            for attribute in attributes:
                csv = csv[csv[attribute] < 0]

        # 附加属性筛选
        if addition_attributes is not None:
            for attribute in addition_attributes:
                csv =csv[ csv[attribute] >0 ]

        indexs = csv.index
        assert len(indexs) > 0
        files_path = [os.path.join(self.image_path, name) for name in indexs]
        files_path = np.array(files_path)
        if shuffle:
            permutation = np.random.permutation(len(files_path))
            files_path = files_path[permutation]
        return files_path


################  插值操作函数  #######################
# 加载模型的函数
def load_module(dataset_name:str,prefix_path):
    paths = os.listdir(os.path.join(prefix_path,parm.result_out,dataset_name))
    if len(paths) == 0:
        exit("没有找到预训练的模型，请先训练网络")
    model_dir = os.path.abspath(os.path.join(prefix_path,parm.result_out,dataset_name, paths[-1]))
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
    dataset_image_path="/home/qianqianjun/CODE/DataSets/DaeDatasets"
    seleter=ImgSeleter(parm.attr_path,dataset_image_path)
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