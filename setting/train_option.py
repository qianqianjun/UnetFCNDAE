# 设置要加载的数据集相关配置
#dataset_name="JAFFE"
dataset_name="AGFW"
# dataset_name="CelebA"
# dataset_name="FGNET"
total_num=1000
train_num=1000
# 训练相关参数
log_step_interval=10
learning_rate_weaken_epoch__interval=80
learning_rate_weaken_rate=0.8

# 保存训练中间结果图像的设置
save_epoch_interval=20
n_sample=25
n_row=5