#-*- coding: UTF-8 -*- 
import os
from PIL import Image
import numpy as np
import shutil 

RGB_image_path = './road_data/RGB_Image'
image_path = './road_data/Image'
ORIG_PATH = '/home/RoadMaint-Sample-20180328/'
#由于label me把原始灰度图变成了RGB，所以需要找出原始图片
def find_orignal_img():
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    num=0
    for path in os.listdir(RGB_image_path):
        num+=1
        img_file = os.path.join(RGB_image_path,path)
        im = Image.open(img_file)
        img = np.asarray(im)
        img_shape = img.shape

        for i in range(10):
            o_path = os.path.join(ORIG_PATH,str(i),path)
            if os.path.exists(o_path):
                shutil.copy(o_path,image_path)
                print o_path,path,img_shape,len(img_shape)
    print num
#统计训练样本长宽情况
def com_w_h_rate():
    for path in os.listdir(image_path):
        img_file = os.path.join(image_path,path)
        im = Image.open(img_file)
        img = np.asarray(im)
        img_shape = img.shape
        if len(img_shape) == 3:
            print img_shape,len(img_shape)

if __name__ == '__main__':
    find_orignal_img()
