import os

from tools.DatasetUtil import train_test_split
from tools.utils import ImgSeleter

def load_jaffe()->list:
    all_paths = []
    for name in os.listdir("/home/qianqianjun/下载/jaffe"):
        if not name.endswith(".tiff"):
            continue
        all_paths.append(os.path.join(
            "/home/qianqianjun/下载/jaffe",
            name
        ))
    return all_paths

def load_CelebA()->list:
    attr_path="/home/qianqianjun/CODE/DataSets/CelebA/Anno/list_attr_celeba.txt"
    dataset_image_path="/home/qianqianjun/CODE/DataSets/DaeDatasets"
    selector=ImgSeleter(attr_path,dataset_image_path)
    all_paths=selector.getImagePathsByAttributes(["Male"])
    return all_paths

def load_AGFW()->list:
    all_paths = []
    for curdir, subdir, files in os.walk("/home/qianqianjun/下载/AGFW_cropped/cropped/128/male"):
        if len(files) < 500:
            continue
        for i in range(500):
            all_paths.append(os.path.join(curdir, files[i]))

    return all_paths

def laod_Extend_AGFW():
    all_paths=[]
    for cur,subdir,files in os.walk("/home/qianqianjun/下载/AGFW_cropped/cropped/128/male"):
        if len(files)<100:
            continue
        if cur.find("老人脸") !=-1:
            for file in files:
                all_paths.append(os.path.join(cur,file))
        else:
            for i in range(100):
                all_paths.append(os.path.join(cur,files[i]))
    return all_paths

def load_FGNET()->list:
    path="/home/qianqianjun/下载/FGNET/faceCrop"
    paths=os.listdir(path)
    all_paths=[os.path.join(path,item) for item in paths]
    return all_paths

def load_dataset(dataset_name:str,total_number:int,train_number:int):
    all_paths=[]
    if dataset_name=="JAFFE":
        all_paths=load_jaffe()
    if dataset_name=="AGFW":
        #all_paths=load_AGFW()
        all_paths=laod_Extend_AGFW()
    if dataset_name=="CelebA":
        all_paths=load_CelebA()
    if dataset_name=="FGNET":
        all_paths=load_FGNET()

    assert total_number<len(all_paths),"没有足够的数据，请扩充数据集"
    assert total_number >= train_number ,"训练数据不能多于总数"
    train_files,test_files=train_test_split(all_paths,total_number,train_number)
    return train_files,test_files