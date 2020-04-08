"""
停下休息的时候，别人还在奔跑
write by qianqianjun
2020.03.12
DenseNet 架构的DAE造成了信息损失过大，重建效果不好
参考图像分割领域的 Unet 结构，重现编写
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DenseNet import DenseBlock, DenseTransitionBlock
from setting.parameter import Parameter


##########################   编码器体系结构  ################################
class Encoders(nn.Module):
    def __init__(self,parm:Parameter):
        """
        使用 DenseNet 架构的变形自动编码器 编码器实现
        :param parm:  参数集合
        """
        super(Encoders, self).__init__()
        self.encoder=DenseEncoder(parm, channel=parm.channel, ndf=parm.ndf, ndim=parm.zdim,
                                  texture_gate_channels=parm.texture_gate_channels,
                                  warp_gate_channels=parm.warp_gate_channels)

        self.zImixer=Mixer(in_channel=parm.zdim, out_channel=parm.idim)
        self.zWmixer=Mixer(in_channel=parm.zdim, out_channel=parm.wdim)

    def forward(self,x:torch.Tensor):
        """
        :param x:
        :return:
        """
        self.z,texture_inter_outs,warp_inter_outs=self.encoder(x)
        self.zImg=self.zImixer(self.z)
        self.zWarp=self.zWmixer(self.z)

        # 注意，这里新增加了 texture_inter_outs,warp_inter_outs
        return  self.z,self.zImg,self.zWarp,texture_inter_outs,warp_inter_outs

class DenseEncoder(nn.Module):
    def __init__(self, parm, channel=3, ndf=32, ndim=128,
                 activation=nn.LeakyReLU,activation_parm=None, f_activation=nn.Sigmoid,norm_method=nn.InstanceNorm2d,
                 growth_rate=12, layers_number=None,block_in_channels=None, transition_out_channels=None,
                 texture_gate_channels=None,warp_gate_channels=None):
        super(DenseEncoder, self).__init__()
        # 初始化参数
        if activation_parm is None:
            activation_parm = [0.2, False]
        if layers_number==None:
            layers_number = [6, 12, 24, 16]
        if transition_out_channels==None:
            transition_out_channels = [ndf * 2, ndf * 4, ndf * 8, ndim]
        if block_in_channels==None:
            block_in_channels = [ndf, ndf * 2, ndf * 4, ndf * 8]
        if texture_gate_channels is None:
            texture_gate_channels=[4, 16, 32, 64]
        if warp_gate_channels is None:
            warp_gate_channels=[4,16,32,64]
        self.ndim=ndim

        # 计算过渡块的输入通道数目
        transition_in_channels=[]
        for step,number_layer in enumerate(layers_number):
            transition_in_channels.append(block_in_channels[step] + growth_rate * number_layer)

        # 定义模型结构
        self.net=nn.Sequential()
        # 将输入图片数据变为 [BN,ndf,32,32]
        self.net.add_module(
            "init_conv",
            nn.Conv2d(in_channels=channel,out_channels=ndf,kernel_size=3,stride=2,padding=1)
        )
        # 添加 Densenet 架构的 encoder块和过渡块
        for i in range(1,len(layers_number)):
            self.net.add_module(
                "denseblock{}".format(i),
                nn.Sequential(
                    DenseBlock(in_channels=block_in_channels[i-1],
                               layer_number=layers_number[i-1],
                               growth_rate=growth_rate,
                               use_bottleneck=False,
                               norm_method=norm_method),

                    DenseTransitionBlock(
                        in_channels=transition_in_channels[i-1],
                        out_channels=transition_out_channels[i-1],
                        type="encoder",
                        pooling_size=2,activation=activation,activation_parm=activation_parm
                    )
                )
            )
        self.net.add_module(
            "denseblock{}".format(len(layers_number)),
            nn.Sequential(
                DenseBlock(in_channels=block_in_channels[-1],
                           layer_number=layers_number[-1],
                           growth_rate=growth_rate,
                           use_bottleneck=True),
                DenseTransitionBlock(
                    in_channels=transition_in_channels[-1],
                    out_channels=transition_out_channels[-1],
                    type="encoder",
                    pooling_size=4,activation=activation,activation_parm=activation_parm
                ),
                f_activation()
            )
        )

        # 纹理 跳过连接
        self.textureGate=nn.Sequential()
        for i in range(len(texture_gate_channels)):
            self.textureGate.add_module(
                "textureGate{}".format(i+1),
                nn.Sequential(
                    norm_method(block_in_channels[i]),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=block_in_channels[i], out_channels=texture_gate_channels[i],
                              kernel_size=3, padding=1, stride=1),

                    norm_method(texture_gate_channels[i]),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=texture_gate_channels[i],out_channels=texture_gate_channels[i],
                              kernel_size=3,padding=1,stride=1),

                    norm_method(texture_gate_channels[i]),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=texture_gate_channels[i],out_channels=texture_gate_channels[i],
                              kernel_size=3,stride=1,padding=1)
                )
            )

        # 变形 跳过连接
        self.warpGate=nn.Sequential()
        for i in range(len(warp_gate_channels)):
            self.warpGate.add_module(
                "warpGate{}".format(i+1),
                nn.Sequential(
                    norm_method(block_in_channels[i]),
                    nn.ReLU(),
                    nn.Conv2d(in_channels=block_in_channels[i],out_channels=warp_gate_channels[i],
                              kernel_size=3,padding=1,stride=1),

                    norm_method(warp_gate_channels[i]),
                    nn.ReLU(),
                    nn.Conv2d(warp_gate_channels[i],warp_gate_channels[i],3,1,1),

                    norm_method(warp_gate_channels[i]),
                    nn.ReLU(),
                    nn.Conv2d(warp_gate_channels[i], warp_gate_channels[i], 3, 1, 1)
                )
            )

    def forward(self, x:torch.Tensor):
        init_conv_out=self.net[0](x)
        block1out=self.net[1](init_conv_out)
        block2out = self.net[2](block1out)
        block3out = self.net[3](block2out)
        out = self.net[4](block3out)

        texture_unet_out = []
        texture_unet_out.append(self.textureGate[0](init_conv_out))
        texture_unet_out.append(self.textureGate[1](block1out))
        texture_unet_out.append(self.textureGate[2](block2out))
        texture_unet_out.append(self.textureGate[3](block3out))

        warp_unet_out=[]
        warp_unet_out.append(self.warpGate[0](init_conv_out))
        warp_unet_out.append(self.warpGate[1](block1out))
        warp_unet_out.append(self.warpGate[2](block2out))
        warp_unet_out.append(self.warpGate[3](block3out))

        # 这里改变一下
        # out=out.view(-1,self.ndim)
        return out, texture_unet_out,warp_unet_out

#########################   解码器体系结构   #################################
class Decoders(nn.Module):
    def __init__(self,parm:Parameter):
        """
        密集网络解码器体系结构
        :param parm: 参数set
        """
        super(Decoders, self).__init__()
        self.imageDimension=parm.imgSize
        self.idim=parm.idim
        self.wdim=parm.wdim

        self.decoderT=DenseDecoder(parm, ndim=parm.idim, nc=parm.channel, ngf=parm.ngf,
                                   gate_add_channels=list(reversed(parm.texture_gate_channels)))
        self.decoderW=DenseDecoder(parm,ndim=parm.wdim,nc=2,ngf=parm.ngf,
                                   activation=nn.Tanh,args=[],f_activation=nn.Sigmoid,f_args=[],
                                   gate_add_channels=list(reversed(parm.warp_gate_channels)))

        self.warper=Warper(parm)
        self.integrator=GridSpatialIntegral(parm)

        self.cutter=nn.Hardtanh(-1,1) # 这里有bug

    def forward(self, zI:torch.Tensor, zW:torch.Tensor,
                texture_inter_outs:torch.Tensor,warp_inter_outs:torch.Tensor,basegrid:torch.Tensor):
        """
        前向传播函数
        :param zI:  纹理潜在空间 向量表征
        :param zW:  变形场 潜在空间 向量表征
        :param texture_inter_outs: 编码器的某些层纹理输出，这叫短连接？
        :param basegrid:  基准变形场 向量表征
        :return:
        """
        self.texture=self.decoderT(zI, texture_inter_outs)
        self.diffentialWarping= self.decoderW(zW, warp_inter_outs) * (5.0 / self.imageDimension)
        self.warping=self.integrator(self.diffentialWarping)-1.2
        self.warping=self.cutter(self.warping)
        self.resWarping=self.warping-basegrid
        self.output=self.warper(self.texture,self.warping)
        return self.texture,self.resWarping,self.output,self.warping

class DenseDecoder(nn.Module):
    def __init__(self,parm:Parameter,ndim:int=128,nc:int=1,ngf:int=32,
                 activation=nn.ReLU,args=None,f_activation=nn.Hardtanh,f_args=None,
                 growth_rate=12,layers_number=None,block_in_channels=None,gate_add_channels=None,
                 transition_out_channels=None):
        super(DenseDecoder, self).__init__()

        # 初始化参数
        if args==None:
            args=[False]
        if f_args==None:
            f_args=[0,1]
        if growth_rate==None:
            growth_rate=12
        if layers_number==None:
            layers_number=[16,24,12,6]
        if gate_add_channels is None:
            # 设置设置 Unet 短连接的带宽
            gate_add_channels=[64,32,16,4]
        if block_in_channels==None:
            # 注意，这里增加倍数了，因为使用 Unet 的结构
            block_in_channels=[ngf*8,ngf*4,ngf*2,ngf]
            assert len(gate_add_channels) == len(block_in_channels),"编码器，解码器通道无法配对！"
        if transition_out_channels==None:
            transition_out_channels = [ngf * 4, ngf * 2, ngf, ngf]


        # 计算每一个密集块的输入通道数目
        transition_in_channels=[]
        for step,layer_number in enumerate(layers_number):
            transition_in_channels.append(block_in_channels[step]+gate_add_channels[step] + growth_rate * layer_number)

        self.net=nn.Sequential()
        # 初始化卷积层
        self.net.add_module(
            "init_transposeConv",
            nn.ConvTranspose2d(in_channels=ndim, out_channels=ngf * 8, kernel_size=4,
                                      stride=1, padding=0, bias=False)
        )
        # densenet 结构
        for i in range(1,len(layers_number)+1):
            self.net.add_module(
                "block{}".format(i),
                nn.Sequential(
                    DenseBlock(
                        in_channels=block_in_channels[i-1] + gate_add_channels[i-1],
                        layer_number=layers_number[i-1],
                        growth_rate=growth_rate,
                        use_bottleneck=False
                    ),
                    DenseTransitionBlock(
                        in_channels=transition_in_channels[i-1],
                        out_channels=transition_out_channels[i-1],
                        activation=activation,
                        activation_parm=args,
                        norm_method=nn.InstanceNorm2d,
                        kernel_size=4,padding=1,stride=2
                    )
                )
            )
        self.net.add_module(
            "final_transposeConv",
            nn.Sequential(
                nn.InstanceNorm2d(ngf),
                activation(*args),
                nn.ConvTranspose2d(in_channels=ngf,out_channels=nc,kernel_size=3,
                                   stride=1,padding=1,bias=False),
                f_activation(*f_args)
            )
        )



    def forward(self, x:torch.Tensor, inter_outs:list):
        # 对第一层解码，只是用输入 x
        init_transposeConv_out=self.net[0](x)
        # 第一个 block
        block1_input=torch.cat((inter_outs[-1], init_transposeConv_out), dim=1)
        block1_output=self.net[1](block1_input)

        # 第二个block
        block2_input=torch.cat((inter_outs[-2], block1_output), dim=1)
        block2_output=self.net[2](block2_input)

        # 第三个block
        block3_input=torch.cat((inter_outs[-3], block2_output), dim=1)
        block3_output=self.net[3](block3_input)

        # 第四个block
        block4_input=torch.cat((inter_outs[-4], block3_output), dim=1)
        block4_output=self.net[4](block4_input)

        # 后续层
        out=self.net[5](block4_output)
        return out

#########################  变换层  #########################################
class Mixer(nn.Module):
    def __init__(self,in_channel:int,out_channel:int,norm_method=nn.InstanceNorm2d,activation=nn.ReLU):
        """
        :param in_channel:  输入通道数目
        :param out_channel:  输出通道数目
        """
        super(Mixer, self).__init__()
        self.net=nn.Sequential(
            norm_method(in_channel),
            nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0),
            activation(),
            norm_method(out_channel),
            nn.Conv2d(out_channel,out_channel,1,1,0),
            activation()
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