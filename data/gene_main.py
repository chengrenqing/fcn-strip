import numpy as np
from PIL import Image,ImageDraw
import sys
import os
import shutil
def split_train_val():
    out_path = './road_data/Image/'
    trainval = open('road/VOC2010/ImageSets/Main/trainval.txt','w')
    train = open('road/VOC2010/ImageSets/Main/train.txt','w')
    val = open('road/VOC2010/ImageSets/Main/val.txt','w')

    data_num=0
    for file in os.listdir(out_path):
        s = file.split('.')
        if s[1]=='jpg':
            data_num+=1
    train_num = int(data_num*0.8)
    print data_num,train_num
    index=0
    for file in os.listdir(out_path):
        s = file.split('.')
        if s[1]=='jpg':
            index+=1

            for i in range(12):
                if (index==train_num  or index==data_num) and i==11:
                    n1 = s[0]+'-'+str(i)
                else:
                    n1 = s[0]+'-'+str(i)+'\n'
                if index <= train_num:
                    train.write(n1)
                else: 
                    val.write(n1)
                if index == train_num and i==11:
                    trainval.write(n1+'\n')
                else:
                    trainval.write(n1)
                
    print index
    trainval.close()
    train.close()
    val.close()
if __name__ == '__main__':
    split_train_val()
