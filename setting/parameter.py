"""
write by qianqianjun
2020.01.20
命令行程序
设置超参数，运行参数，运行环境等
"""

import argparse
import torch.backends.cudnn as cudnn

class Parameter(argparse.ArgumentParser):
    def __init__(self):
        """
        设置训练过程的超参数
        """
        super(Parameter, self).__init__()
        self.add_argument("--workers", type=int, help="数据加载线程的个数，默认4 线程", default=4)
        self.add_argument("--batchSize", type=int, help="输入批数据的大小，默认1", default=25)
        self.add_argument("--lr", type=float, default=0.0002, help="学习率，默认大小是 0.0002")

        self.add_argument("--dirCheckpoints", default="checkpoints", help="存放训练好的模型的目录", type=str)
        self.add_argument("--dirImageOutput", default="trainout", help="图片输出目录", type=str)
        self.add_argument("--dirTestOutput", default="testout", help="测试 结果或者图像 的目录", type=str)
        self.add_argument("--result_out",default="out",type=str,help="输出结果所在的目录")

        self.add_argument("--modelPath", type=str, default="", help="存放训练好的模型的目录")
        self.add_argument("--useCuda", default=True, help="是否使用 GPU 加速，默认不使用", type=bool)
        self.add_argument("--randomSeed", help="用于实验可重复性验证的随机种子", default=None, type=int)
        self.add_argument("--beta1", default=0.5, type=float, help="Adam 优化器需要使用的参数")
        self.add_argument("--epochs", default=300, type=int, help="对整个数据集训练多少次")

        self.add_argument("--texture_gate_channels",type=list,help="纹理短连接的通道数目",default=[4, 8, 16, 32])
        self.add_argument("--warp_gate_channels",type=list,help="变形短连接通道数目",default=[8,16,16,32])
        self.readCmd(self.parse_args(args=[]))
    def readCmd(self, args):
        """
        设置参数可以方便外部访问
        :param args: 命令行传过来的参数列表
        :return: None
        """
        self.workers=args.workers
        self.batchSize=args.batchSize
        self.learning_rate=args.lr
        self.dirCheckpoints=args.dirCheckpoints
        self.dirImageOutput=args.dirImageOutput
        self.dirTestOutput=args.dirTestOutput
        self.result_out=args.result_out
        self.useCuda=args.useCuda
        self.randomSeed=args.randomSeed
        self.modelPath=args.modelPath
        self.beta1=args.beta1
        self.epochs=args.epochs

        self.texture_gate_channels=args.texture_gate_channels
        self.warp_gate_channels=args.warp_gate_channels
    def setImageInfo(self,imgSize=64,ngf=32,ndf=32,channel=3):
        """
        设置图片信息相关的参数
        :param imgSize:  图片的大小（这里为边长，图片选用方形的）
        :param ngf:  起始卷积核个数（用于创建网络层结构）
        :param ndf:  结束卷积核个数（用于创建网络层结构）
        :param channel:  图片的通道数目
        :return: None
        """
        self.imgSize=imgSize
        self.ngf=ngf
        self.ndf=ndf
        self.channel=channel

    def setLantentInfo(self,idim=16,wdim=128,zdim=128):
        """
        设置潜在空间的相关信息
        :param idim:  纹理空间的维度
        :param wdim:  变形场空间的维度
        :param zdim:  通用纹理空间的维度
        :return:  None
        """
        self.idim=idim
        self.wdim=wdim
        self.zdim=zdim

parameter=Parameter()

# 224 * 224
parameter.setImageInfo()

# idim=32,wdim=256,zdim=256
parameter.setLantentInfo()
# 加速设置
cudnn.benchmark=True