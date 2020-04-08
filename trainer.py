"""
不想学习就照照镜子，都长这样了，不学习你还想干嘛？
write by qianqianjun
2020.03.13
训练 unet 结构的 DAE
"""
import torch.utils.data as Data

from model.Loss import *
from setting.train_option import *
from tools.DatasetLoader import load_dataset
from tools.DatasetUtil import DataSet
from tools.utils import *

# 设置微调设置
fine_tune=True
# 初始化模型参数，如果有预训练的模型，则进行微调
if fine_tune:
    start_epoch,learning_rate,encoders,decoders=init_model(parm,dataset_name,os.path.abspath("."))
else:
    encoders = Encoders(parm)
    decoders = Decoders(parm)
    nn.Module.apply(encoders,weight_init)
    nn.Module.apply(decoders,weight_init)
    if parm.useCuda:
        encoders = nn.Module.cuda(encoders)
        decoders = nn.Module.cuda(decoders)
    start_epoch=0
    learning_rate=parm.learning_rate
# 初始化训练环境
save_path=init_train_env(parm,dataset_name,os.path.abspath("."))
# 定义优化器
encodersOptimizer=torch.optim.Adam(Encoders.parameters(encoders),
                                   lr=learning_rate,betas=(parm.beta1,0.999))
decodersOptimizer=torch.optim.Adam(Decoders.parameters(decoders),
                                   lr=learning_rate,betas=(parm.beta1,0.999))
# 设置损失函数
ReconstructionLoss=nn.L1Loss() # 重建损失
WarpLoss=TotalVaryLoss(parm)
BiasReduce=BiasReduceLoss(parm)
SmoothL1=TotalVaryLoss(parm)
SmoothL2=SelfSmoothLoss2(parm)

# 加载数据集
train_files,test_files=load_dataset(dataset_name,total_num,train_num)
train_set=DataSet(files=train_files,resizeTo=parm.imgSize)

# 开始训练
Step=0
encoder_lr_scheduler=torch.optim.lr_scheduler.StepLR(
    encodersOptimizer,step_size=learning_rate_weaken_epoch_interval,gamma=learning_rate_weaken_rate)
decoder_lr_scheduler=torch.optim.lr_scheduler.StepLR(
    decodersOptimizer,step_size=learning_rate_weaken_epoch_interval,gamma=learning_rate_weaken_rate)
print("起始学习率：{}".format(learning_rate))
for epoch in range(start_epoch,parm.epochs+ start_epoch):
    # 加载训练图片数据
    train_loader = Data.DataLoader(train_set, batch_size=parm.batchSize, shuffle=True, num_workers=parm.workers)
    # 定义要可视化的变量
    batch_data = None
    I = None
    out = None
    W = None
    baseGrid = None

    # 批训练
    for step, batch_data in enumerate(train_loader, start=0):
        batch_data = batchDataTransform(batch_data, parm.channel)
        baseGrid = getBaseGrid(imgSize=parm.imgSize, Inbatch=True, batchSize=batch_data.size()[0])
        zeroWarp = torch.tensor(
            np.zeros(shape=(1, 2, parm.imgSize, parm.imgSize)), dtype=torch.float32, requires_grad=False)
        if parm.useCuda:
            batch_data = batch_data.cuda()
            baseGrid = baseGrid.cuda()
            zeroWarp = zeroWarp.cuda()
        encodersOptimizer.zero_grad()
        decodersOptimizer.zero_grad()
        # 前向计算
        z, zI, zW, texture_inter_outs ,warp_inter_outs= encoders(batch_data)
        I, W, out, _ = decoders(zI, zW, texture_inter_outs,warp_inter_outs,baseGrid)
        # 计算损失
        loss_reconstruction = ReconstructionLoss(out, batch_data)
        loss_Smooth = WarpLoss(W, weight=1e-6)
        loss_bias_reduce = BiasReduce(W, zeroWarp, weight=1e-2)
        loss_all = loss_reconstruction + loss_bias_reduce + loss_Smooth
        # 反向传播
        loss_all.backward()
        encodersOptimizer.step()
        decodersOptimizer.step()
        if (Step + 1) % log_step_interval == 0:
            loss_val = loss_reconstruction.item() + loss_Smooth.item() + loss_bias_reduce.item()
            print("<epoch:{}/{} , {}> loss-- total:{} recon:{} smooth:{} biasReduce:{}".format(
                epoch + 1, parm.epochs+start_epoch, Step, loss_val, loss_reconstruction.item(), loss_Smooth.item(),
                loss_bias_reduce.item()))
        Step += 1

    # 动态衰减学习率
    encoder_lr_scheduler.step(epoch)
    decoder_lr_scheduler.step(epoch)

    if (epoch + 1) % save_epoch_interval == 0:
        # 可视化训练过程
        gx = (W.data[:, 0, :, :] + baseGrid.data[:, 0, :, :]).unsqueeze(1).clone()
        gy = (W.data[:, 1, :, :] + baseGrid.data[:, 1, :, :]).unsqueeze(1).clone()
        saveIntermediateImage(img_list=batch_data.data.clone(),
                              output_dir=os.path.join(save_path,parm.dirImageOutput),
                              filename="step_{}_img".format(Step), n_sample=n_sample, nrow=n_row, normalize=False)
        saveIntermediateImage(img_list=I.data.clone(),
                              output_dir=os.path.join(save_path,parm.dirImageOutput),
                              filename="step_{}_texture".format(Step), n_sample=n_sample, nrow=n_row, normalize=False)
        saveIntermediateImage(img_list=out.data.clone(),
                              output_dir=os.path.join(save_path,parm.dirImageOutput),
                              filename="step_{}_output".format(Step), n_sample=n_sample, nrow=n_row, normalize=False)
        saveIntermediateImage(img_list=(gx + 1) / 2,
                              output_dir=os.path.join(save_path,parm.dirImageOutput),
                              filename="step_{}_warpx".format(Step), n_sample=n_sample, nrow=n_row, normalize=False)
        saveIntermediateImage(img_list=(gy + 1) / 2,
                              output_dir=os.path.join(save_path,parm.dirImageOutput),
                              filename="step_{}_warpy".format(Step), n_sample=n_sample, nrow=n_row, normalize=False)

# 保存训练好的模型
torch.save(
    nn.Module.state_dict(encoders),
    "{}/encoders.pth".format(os.path.join(save_path,parm.dirCheckpoints)))
torch.save(
    nn.Module.state_dict(decoders),
    "{}/decoders.pth".format(os.path.join(save_path,parm.dirCheckpoints)))
info={}
info["epoch"]=parm.epochs + start_epoch
info["lr"]=encodersOptimizer.state_dict().get("param_groups")[0]["lr"]
with open("{}/info".format(os.path.join(save_path,parm.dirCheckpoints)),"wb") as f:
    pk.dump(info,f)