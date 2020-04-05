"""
不想学习的时候我喜欢照照镜子，然后就想学习了。
毕竟都长成这逼样了，还有什么理由不好好学习？
write by qianqianjun
2020.02.02
定义损失函数
"""

import numpy as np
import torch
import torch.nn as nn

from setting.parameter import Parameter


class TotalVaryLoss(nn.Module):
    def __init__(self,parm:Parameter):
        """
        自定义损失函数
        :param parm:
        """
        super(TotalVaryLoss, self).__init__()
        self.parm=parm

    def forward(self, x:torch.Tensor,weight=1):
        w=np.zeros(shape=1)
        w.fill(weight)
        w=torch.tensor(w,dtype=torch.float32,requires_grad=False)
        if self.parm.useCuda:
            w=w.cuda()
        # 第一个 abs 计算最后一维度的前 len-1 个分量 与 最后一维度 后 len-1 个分量的差值
        self.loss=w * (
                torch.sum(
                    torch.abs(x[:,:,:,:-1]-x[:,:,:,1:])
                ) +
                torch.sum(
                    torch.abs(x[:,:,:-1,:]-x[:,:,1:,:])
                )
        )
        return self.loss

class BiasReduceLoss(nn.Module):
    def __init__(self,parm:Parameter):
        """
        消除拟合过程引入的系统偏差，例如平均模板变小，数据版本变形（详见论文）
        :param parm:超参数集合
        """
        super(BiasReduceLoss, self).__init__()
        self.criterion=nn.MSELoss()
        self.parm=parm

    def forward(self, x:torch.Tensor,y:torch.Tensor,weight:float=1):
        """
        前向传播函数
        :param x:
        :param y:
        :param weight: 默认 w 值
        :return:
        """
        w=np.zeros(shape=1)
        w.fill(weight)
        w=torch.tensor(w,dtype=torch.float32,requires_grad=False)
        if self.parm.useCuda:
            w=w.cuda()

        self.avg=torch.mean(x,0).unsqueeze(0)
        # print(self.avg.dtype)
        # print(y.dtype)
        self.loss=w * self.criterion(self.avg,y)
        return self.loss

class SelfSmoothLoss2(nn.Module):
    def __init__(self,pram:Parameter):
        """
        平滑度损失函数
        :param pram: 全局参数集集合
        """
        super(SelfSmoothLoss2, self).__init__()
        self.parm=pram

    def forward(self, x:torch.Tensor,weight=1):
        """
        :param x:
        :param weight:
        :return:
        """
        w=np.ones(shape=1)
        w.fill(weight)
        w=torch.tensor(w,dtype=torch.float32,requires_grad=False)
        if self.parm.useCuda:
            w=w.cuda()
        self.x_diff=x[:,:,:,:-1]-x[:,:,:,1:]
        self.y_diff=x[:,:,:-1,:]-x[:,:,1:,:]
        self.loss=torch.sum(
            torch.mul(self.x_diff,self.x_diff)
        ) + torch.sum(
            torch.mul(self.y_diff,self.y_diff)
        )

        self.loss=w * self.loss
        return self.loss