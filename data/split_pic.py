import os
from PIL import Image
import numpy as np

def split_pic(dir,h,w,out_dir):
    for file in os.listdir(dir):
        img_file = os.path.join(dir,file)
        im = Image.open(img_file)
        img = np.asarray(im)
        img_shape = img.shape
        print img_shape

        W_cut = int(img_shape[1]/w) + 1
        H_cut = img_shape[0]/h
        print W_cut,H_cut
        index = 0
        for i in range(H_cut):
            for j in range(W_cut):
                x = i*h
                y = j*w
                y2 = y+w
                y2 = min(y2,img_shape[1])
                x2 = x+h
                region = im.crop((y,x,y2,x2))
                o_name = file.split('.')[0] + '-' +str(index)+'.jpg'
                out_name = os.path.join(out_dir,o_name)
                region.save(out_name)
                index+=1

def main():
    im_path = './road_data/Image/'
    out_path = './road/VOC2010/JPEGImages/'
    
    split_pic(im_path,1024,550,out_path)

    print 'finish'
if __name__ == '__main__':
    main()  
