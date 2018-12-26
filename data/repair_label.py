#coding=utf-8
from PIL import Image
import PIL.ImageDraw
import numpy as np
import os
import json
import scipy.io as sio

def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = Image.fromarray(mask)
    # mask.show() 
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def trans_lable(json_file):
    print('Generating dataset from:', json_file)
    with open(json_file) as f:
        data = json.load(f)

        img_file = os.path.join('./road_data/Image/', data['imagePath'])
        img = np.asarray(Image.open(img_file))
        img_shape = img.shape
        shapes = data['shapes']
        class_name_to_id = {'repair':1}

        cls = np.zeros(img_shape[:2], dtype=np.int32)
        for shape in shapes:
            polygons = shape['points']
            label = shape['label']
            print label
            cls_id = class_name_to_id[label]
            # print cls_id
            mask = polygons_to_mask(img_shape[:2], polygons)
            cls[mask] = cls_id
        # print cls.shape

        # cut lable and save mat
        split_w = 550
        split_h = 1024
        mat_dir = './road-context/trainval/'
        #3024x1882 3152x2048
        W_cut = int(img_shape[1]/split_w) + 1
        H_cut = img_shape[0]/split_h
        index = 0

        print W_cut,H_cut
        for i in range(H_cut):
            for j in range(W_cut):
                h_begin = i*split_h
                h_end = h_begin +split_h
                w_begin = j*split_w
                w_end = w_begin+split_w
                # w_end = min(w_end,img_shape[1])
                labelmap = cls[h_begin:h_end,w_begin:w_end]
                out_mat_name = data['imagePath'].split('.')[0] + '-' +str(index)+'.mat'
                # out_mat_name = out_mat_name.encode('utf8')
                out_mat_path = os.path.join(mat_dir,out_mat_name)
                # tt = Image.fromarray(labelmap)
                # tt.show()
                index+=1
                sio.savemat(out_mat_path,{"LabelMap":labelmap})

def main():
    test_path = './road_data/json_file/'
    json_name = 'G30IA-333+728440-333+737720.json'
    for file in os.listdir(test_path):
        if file.split('.')[1] == 'json':
            json_file = os.path.join(test_path,file)
            trans_lable(json_file)
    print 'finish'
if __name__ == '__main__':
    main()  


