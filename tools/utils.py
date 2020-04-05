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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.utils as Util
from PIL import Image

from setting.parameter import Parameter


def init_model(parm:Parameter,dataset_name:str,encoders:nn.Module,decoders:nn.Module,path_prefix:str):
    model_path=os.path.join(path_prefix,dataset_name+"_"+parm.result_out)
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


    return start_epoch,learning_rate

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
    os.makedirs(
        os.path.join(path_prefix, dataset_name + "_" + parm.result_out, save_time_path, parm.dirCheckpoints),
        exist_ok=True)
    os.makedirs(
        os.path.join(path_prefix, dataset_name + "_" + parm.result_out, save_time_path, parm.dirImageOutput),
        exist_ok=True)
    os.makedirs(os.path.join(path_prefix, dataset_name + "_" + parm.result_out, save_time_path, parm.dirTestOutput),
                exist_ok=True)
    return os.path.join(path_prefix,dataset_name+"_"+parm.result_out,save_time_path)

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

### 编码器，解码器模型工具类
from setting.parameter import Parameter
import torch.nn.functional as F
################ 工具类  ####################################
class Mixer(nn.Module):
    def __init__(self,parm:Parameter,in_channel:int,out_channel:int):
        """
        建立一个全连接网络
        :param parm:  超参数集合
        :param in_channel:  输入通道数目
        :param out_channel:  输出通道数目
        """
        super(Mixer, self).__init__()
        self.net=nn.Sequential(
            nn.Linear(in_channel,out_channel),
            nn.Sigmoid()
        )

    def forward(self,x:torch.Tensor):
        """
        :param x:
        :return:
        """
        out=self.net(x)
        return out

class Warper(nn.Module):
    def __init__(self,parm:Parameter):
        """
        将纹理图像进行空间变形
        :param parm:  超参数集合
        """
        super(Warper, self).__init__()
        self.parm=parm
        self.batchSize=parm.batchSize
        self.imgSize=parm.imgSize

    def forward(self,image_tensor:torch.Tensor,input_grid:torch.Tensor):
        """
        :param image_tensor: 输入图片的 tensor
        :param input_grid:  变形场
        :return:  经过变形场变换的最终图像
        """
        self.warp=input_grid.permute(0,2,3,1)
        # 不清楚 align_corners 干什么用的， 但是不加上这个参数会有 warning
        self.output=F.grid_sample(image_tensor,self.warp,align_corners=True)
        # torch.nn.functional.grid_sample()

        return self.output

class GridSpatialIntegral(nn.Module):
    def __init__(self,parm:Parameter):
        """
        变形场空间积分运算
        :param parm: 超参数集合
        """
        super(GridSpatialIntegral, self).__init__()
        self.parm=parm
        self.w=parm.imgSize

        self.filterx=torch.tensor(np.ones(shape=(1,1,1,self.w)),dtype=torch.float32,requires_grad=False)
        self.filtery=torch.tensor(np.ones(shape=(1,1,self.w,1)),dtype=torch.float32,requires_grad=False)

        if parm.useCuda:
            self.filterx=self.filterx.cuda()
            self.filtery=self.filtery.cuda()

    def forward(self, input_diffgrid:torch.Tensor):
        """
        :param input_diffgrid: 差分变形场 tensor
        :return:
        """
        x=F.conv_transpose2d(input=input_diffgrid[:,0,:,:].unsqueeze(1),weight=self.filterx,stride=1,padding=0)
        y=F.conv_transpose2d(input=input_diffgrid[:,1,:,:].unsqueeze(1),weight=self.filtery,stride=1,padding=0)
        output_grid=torch.cat((x[:,:,0:self.w,0:self.w],y[:,:,0:self.w,0:self.w]),1)

        return output_grid

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
        imgs = np.array([np.array(Image.open(path)) for path in image_absoult_paths])
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
        imgs = np.array([np.array(Image.open(path)) for path in image_absoult_paths])
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