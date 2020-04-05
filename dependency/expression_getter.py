import os

import cv2
import torch

from model.UnetDAE import Encoders
from setting.parameter import parameter as parm
from tools.utils import batchDataTransform

images_path="/home/qianqianjun/下载/jaffe"
all_images_path=os.listdir("/home/qianqianjun/下载/jaffe")
def save_estimate_attr(attribute: str, neg_imgs, pos_imgs, encoders):
    neg_tensor = batchDataTransform(torch.tensor(neg_imgs, dtype=torch.float32), channel=3)
    pos_tensor = batchDataTransform(torch.tensor(pos_imgs, dtype=torch.float32), channel=3)
    _, pos_texture_code, pos_warp_code, pos_texture_inters, pos_warp_inters = encoders(pos_tensor)
    _, neg_texture_code, neg_warp_code, neg_texture_inters, neg_warp_inters = encoders(neg_tensor)

    texture_code = torch.mean(pos_texture_code, dim=0) - torch.mean(neg_texture_code, dim=0)
    warp_code = torch.mean(pos_warp_code, dim=0) - torch.mean(neg_warp_code, dim=0)
    texture_inters_attribute = []
    for pos, neg in zip(pos_texture_inters, neg_texture_inters):
        texture_inters_attribute.append(
            torch.mean(pos, dim=0) - torch.mean(neg, dim=0)
        )
    warp_inters_attribute = []
    for pos, neg in zip(pos_warp_inters, neg_warp_inters):
        warp_inters_attribute.append(
            torch.mean(pos, dim=0) - torch.mean(neg, dim=0)
        )
    attr = {}
    attr["texture_code"] = texture_code
    attr["warp_code"] = warp_code
    attr["texture_inter_outs_attribute"] = texture_inters_attribute
    attr["warp_inter_outs_attribute"] = warp_inters_attribute
    import pickle as pk
    with open(attribute, "wb") as f:
        pk.dump(attr, f)
smile_imgs=[]
nature_imgs=[]
angry_imgs=[]
sad_imgs=[]
fear_imgs=[]
disgust_imgs=[]
surprise_imgs=[]
for name in all_images_path:
    if not name.endswith("tiff"):
        continue
    if name.find("HA")!=-1:
        smile_imgs.append(
            cv2.resize(
                cv2.imread(os.path.join(images_path,name)),
                (parm.imgSize,parm.imgSize)))
    if name.find("NE") != -1:
        nature_imgs.append(
            cv2.resize(
                cv2.imread(os.path.join(images_path,name)),
                (parm.imgSize,parm.imgSize)))
    if name.find("AN") != -1:
        angry_imgs.append(
            cv2.resize(
                cv2.imread(os.path.join(images_path,name)),
                (parm.imgSize,parm.imgSize)))
    if name.find("SA") != -1:
        sad_imgs.append(
            cv2.resize(
                cv2.imread(os.path.join(images_path,name)),
                (parm.imgSize,parm.imgSize)))
    if name.find("FE") != -1:
        fear_imgs.append(
            cv2.resize(
                cv2.imread(os.path.join(images_path,name)),
                (parm.imgSize,parm.imgSize)))
    if name.find("DI") != -1:
        disgust_imgs.append(
            cv2.resize(
                cv2.imread(os.path.join(images_path,name)),
                (parm.imgSize,parm.imgSize)))
    if name.find("SU") != -1:
        surprise_imgs.append(
            cv2.resize(
                cv2.imread(os.path.join(images_path,name)),
                (parm.imgSize,parm.imgSize)))
# 加载模型
encoders=Encoders(parm)
model_path="../JAFFE_out/2020-03-27_21:38:52/checkpoints/encoders.pth"
torch.nn.Module.load_state_dict(encoders,torch.load(model_path,map_location=torch.device("cpu")))

# 保存属性
save_estimate_attr("smile",nature_imgs,smile_imgs,encoders)
save_estimate_attr("angry",nature_imgs,angry_imgs,encoders)
save_estimate_attr("sad",nature_imgs,sad_imgs,encoders)
save_estimate_attr("fear",nature_imgs,fear_imgs,encoders)
save_estimate_attr("disgust",nature_imgs,disgust_imgs,encoders)
save_estimate_attr("surprise",nature_imgs,surprise_imgs,encoders)