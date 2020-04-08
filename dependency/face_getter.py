import os

import cv2
import dlib

path="/home/qianqianjun/桌面/马云"
detector=dlib.get_frontal_face_detector()

files=os.listdir(path)
images=[]
for file in files:
    if file.endswith(".jpg"):
        images.append(os.path.join(path,file))

name=0
save_path="/home/qianqianjun/桌面/马云人脸"
os.makedirs(save_path,exist_ok=True)

for image in images:
    img=cv2.imread(image)
    try:
        coordinates=detector(img)
    except:
        print(name)
        print(image)
        continue
    for coordinate in coordinates:
        if min(coordinate.bottom()-coordinate.top(),coordinate.right()-coordinate.left()) < 64:
            continue
        length=coordinate.right() -coordinate.left()
        #padding=length / 4
        padding=min(coordinate.top(),
                    coordinate.left(),
                    img.shape[1]-coordinate.right(),
                    img.shape[0]-coordinate.bottom())
        padding=int(padding)
        face=img[coordinate.top()-padding:coordinate.bottom()+padding,coordinate.left()-padding:coordinate.right()+padding,:]
        cv2.imwrite(os.path.join(save_path,"{}.jpg".format(name)),face)
        name=name+1