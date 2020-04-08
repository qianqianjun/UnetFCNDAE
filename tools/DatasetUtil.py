"""
武则天证明成功和性别无关
姜子牙证明成功和年龄无关
我比他们都厉害，我证明了成功与我无关 【捂脸】
write by qianqianjun
2020.01.20
准备数据集
"""
import cv2
import torch.utils.data as data


def train_test_split(images_address,total=None,train:int=None):
    """
    分割训练和测试数据集
    :param total:  数据集总数
    :param train:  训练数据所占比重
    :return:  训练集和测试集图片的地址列表
    """
    length=len(images_address)
    if total==None:
        total=length
    if train==None:
        train=length/2
    assert total<=length ,"共有图片{0}张,不够{1}张！".format(length,total)
    train_num=train
    test_num=total-train_num

    train_set=images_address[:train_num]
    test_set=images_address[train_num:train_num+test_num]

    return train_set,test_set
class DataSet(data.Dataset):
    def __init__(self,files,resizeTo:int=64):
        """
        创建数据集
        :param files: 图片文件地址列表
        :param resizeTo:  将图片转换为多大
        """
        self.files=files
        self.resizeTo=resizeTo
        self.length=len(files)

    def __getitem__(self, index):
        """
        重载父类的方法
        :param index:
        :return:
        """
        filepath=self.files[index]
        image=self.imageFileReader(filepath,resizeTo=self.resizeTo)
        return  image

    def __len__(self):
        return self.length

    def imageFileReader(self,filepath:str,resizeTo:int):
        """
        图片文件读取方法
        :param filepath: 图片文件地址
        :param resizeTo: 像图片转换到多大（边长）
        :return: 返回图片的 np.array() 数组表示
        """
        return cv2.cvtColor(cv2.resize(cv2.imread(filepath),(resizeTo,resizeTo)),cv2.COLOR_BGR2RGB)