# 设置要加载的数据集相关配置
# dataset_name="JAFFE"
# dataset_name="AGFW"
#dataset_name="CelebA"
# dataset_name="FGNET"
dataset_name="other"
total_num=160
train_num=160
# 训练相关参数
log_step_interval=50
learning_rate_weaken_epoch_interval=80
learning_rate_weaken_rate=0.8

# 保存训练中间结果图像的设置
save_epoch_interval=30
n_sample=25
n_row=5