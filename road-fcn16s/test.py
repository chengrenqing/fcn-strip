import numpy as np
from PIL import Image
import sys
import caffe
import vis
import os
# the demo image is "2007_000129" from PASCAL VOC
fcn_root = '/home/chengrenqing/fcn.berkeleyvision.org/road-fcn16s/'
model_def = fcn_root + 'deploy.prototxt'
model_weights = fcn_root + 'snapshot/train_iter_4000.caffemodel'

def detect_image(test_image,index):
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    x1 = index*400
    x2 = (index+1)*400
    print index,test_image
    gray = Image.open(test_image)
    gray = gray.crop((x1,0,x2,2048))
    #return gray
    print gray.format, gray.size, gray.mode
    im = gray.convert('RGB')
    print 'im.format,im.size,im.mode:',im.format, im.size, im.mode
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((105.430,105.430,105.430))
    in_ = in_.transpose((2,0,1))
    # init
    caffe.set_device(0)
    caffe.set_mode_gpu()
    # load net
    net = caffe.Net(model_def,model_weights, caffe.TEST)
    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_
    # run net and take argmax for prediction
    net.forward()
    out = net.blobs['score'].data[0].argmax(axis=0)

    print 'score shape:',net.blobs['score'].data[0].shape
    print net.blobs['score'].data[0]
    print net.blobs['score'].data[0].argmax(axis=0)
    # visualize segmentation in PASCAL VOC colors
    voc_palette = vis.make_palette(3)
   # out_im = Image.fromarray(vis.color_seg(out, voc_palette))
   # out_im.save('demo/output.png')
    masked_im = Image.fromarray(vis.vis_seg(im, out, voc_palette))
    return masked_im
    #masked_im.save('demo/visualization.jpg')
def split_check(test_image):
    in_dir = test_image.split('/')
    outname = in_dir[len(in_dir)-1].split('.')[0]+'-res.jpg'
    out_dir = os.path.join('demo/',outname)
    all_pic = {}
    orig = Image.open(test_image)
    print 'image size: ',orig.size
    toImage = Image.new('RGB',orig.size)
    print toImage.size
    for i in range(0,8):
        loc = (i*400,0)
        toImage.paste(detect_image(test_image,i),loc)
    #print out_dir
    toImage.save(out_dir)
if __name__ == '__main__':
    #test_image = '/Users/ColinCheng/Desktop/0/G30IA-331+053000-331+053360.jpg'
    test_image = '/home/RoadMaint-Sample-20180328/1/G30IA-351+756920-351+769240.jpg'
    if len(sys.argv) >= 2:
        test_image = sys.argv[1]
    #print test_image
    split_check(test_image)
    path = 'road_result/'
    num=1
    for file in os.listdir(path):
        file_path = os.path.join(path,file)
        print '----------------------',num
        #split_check(file_path)
        num+=1
    print 'finish'
