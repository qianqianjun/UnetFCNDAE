import torch
import torch.nn as nn

# 不带瓶颈层的基本层
class BasicLayer(nn.Module):
    def __init__(self,in_channels,out_channels,down_sample=True,norm_method=nn.InstanceNorm2d):
        super(BasicLayer, self).__init__()
        if down_sample:
            Conv=nn.Conv2d
        else:
            Conv=nn.ConvTranspose2d
        self.net=nn.Sequential(
            norm_method(in_channels),
            nn.ReLU(inplace=True),
            Conv(in_channels=in_channels,out_channels=out_channels,kernel_size=3,
                      stride=1,padding=1,bias=False)
        )

    def forward(self, x:torch.Tensor):
        out=self.net(x)
        return torch.cat((x,out),dim=1)

# 带瓶颈层的 DenseNet 基本层
class BottleneckLayer(nn.Module):
    def __init__(self,in_channels,out_channels,down_sample=True,norm_method=nn.InstanceNorm2d):
        super(BottleneckLayer, self).__init__()
        if down_sample:
            Conv=nn.Conv2d
        else:
            Conv=nn.ConvTranspose2d
        inter_channels= out_channels * 4

        self.net=nn.Sequential(
            norm_method(in_channels),
            nn.ReLU(inplace=True),
            Conv(in_channels=in_channels,out_channels=inter_channels,kernel_size=1,
                 stride=1,padding=0,bias=False),

            norm_method(inter_channels),
            nn.ReLU(inplace=True),
            Conv(in_channels=inter_channels,out_channels=out_channels,kernel_size=3,
                 stride=1,padding=1,bias=False)
        )

    def forward(self, x):
        out=self.net(x)
        return torch.cat((x,out),dim=1)

# 密集连接块
class DenseBlock(nn.Module):
    def __init__(self,in_channels,layer_number,growth_rate,down_sample=True,use_bottleneck=False,
                 norm_method=nn.InstanceNorm2d):
        """
        用于创建密集网络块 编码器体系结构
        :param channels_num:  通道数量
        :param convs_num:  卷积的数量
        :param activation:  激活函数
        :param args:  其他参数
        """
        super(DenseBlock, self).__init__()

        assert layer_number > 0, "密集块的卷积层个数不可以小于 1"

        if use_bottleneck:
            Block=BottleneckLayer
        else:
            Block=BasicLayer

        layers=[]
        for i in  range(layer_number):
            layers.append(Block(in_channels=in_channels+i*growth_rate,out_channels=growth_rate,
                                down_sample=down_sample,norm_method=norm_method))
        self.net=nn.Sequential(*layers)


    def forward(self,inputs):
        """
        前向传播函数
        :param inputs: 输入的 tensor
        :return: 返回网络输出结果
        """
        return self.net(inputs)
# 过渡块
class DenseTransitionBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, **args):
        """
        密集网络的过渡块
        :param in_channels: 输入额tensor 通道数目
        :param out_channels:  密集块输出的通道数目
        :param args: 其它参数列表，包括激活函数。
        """
        super(DenseTransitionBlock, self).__init__()
        # 检查激活函数设置
        if args.get("activation")==None:
            activation=nn.ReLU
            activation_parm=[False]
        else:
            activation=args.get("activation")
            if args.get("activation_parm") is not None:
                activation_parm=args.get("activation_parm")
            else:
                activation_parm=[]
        # 检查正则化方法
        if args.get("norm_method") is None:
            norm_method=nn.InstanceNorm2d
        else:
            norm_method=args.get("norm_method")
        #　检查编码器还是解码器类型
        if args.get("type")=="encoder":
            assert args.get("pooling_size") is not None,"encoder requires pooling_size argument !"
            pooling_size=args.get("pooling_size")
            self.net=nn.Sequential(
                norm_method(in_channels),
                activation(*activation_parm),
                nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
                nn.AvgPool2d(pooling_size)
            )
        else:
            assert args.get("kernel_size") is not None,"decoder requires kernel_size for ConvTranspose2d function"
            assert args.get("stride") is not None,"decoder requires stride for ConvTranspose2d function"
            assert args.get("padding") is not None,"decoder requires padding for ConvTranspose2d function"
            kernel_size=args.get("kernel_size")
            stride=args.get("stride")
            padding=args.get("padding")
            self.net=nn.Sequential(
                norm_method(in_channels),
                activation(*activation_parm),
                nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,
                                   stride=stride,padding=padding,bias=False)
            )

    def forward(self, x:torch.Tensor):
        """
        网络前向传播函数
        :param x: 输入tensor
        :return:  tensor经过网络之后的结果
        """
        return self.net(x)