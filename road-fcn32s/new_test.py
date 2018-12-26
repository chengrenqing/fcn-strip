import matplotlib.pyplot as plt 
import caffe
import numpy as np
from PIL import Image
import os
from timeit import timeit
import time

fcn_root = '/home/chengrenqing/fcn-strip/road-fcn32s/'
model_def = os.path.join(fcn_root,'deploy.prototxt')
model_weights = os.path.join(fcn_root,'snapshot/train_iter_28000.caffemodel')
def preprocess(gray):
    im = gray.convert('RGB')
    #print im.mode,im.size
    in_ = np.array(im, dtype=np.float32)
    #print in_.shape,in_
    in_ = in_[:,:,::-1] #switch channels RGB -> BGR
    in_ -= np.array((105.430,105.430,105.430)) #substract mean
    in_ = in_.transpose((2,0,1)) #transpose to channel x height x width
    #print in_.shape,in_
    return in_
def inference(test_image):
    gray = Image.open(test_image,'r')
#     plt.imshow(gray,cmap='gray')
#     plt.show()

    height = gray.size[1]
    width = gray.size[0]
    h_split = np.linspace(0,height,2+1, endpoint=True)
    w_split = np.linspace(0,width,5+1,endpoint=True)
    s_height = height/2
    s_width = width/5
    #print gray.size,h_split,w_split,s_height,s_width
    
    data = []
    for i in range(h_split.size-1):
        for j in range(w_split.size-1):
            gray_s = gray.crop((w_split[j],h_split[i],w_split[j+1],h_split[i+1]))
            data.append(preprocess(gray_s))
    data = np.array(data)
    #print type(data),data.shape
    
    print '>>>>begin inference'
    # init
    caffe.set_device(1)
    caffe.set_mode_gpu()
    # load net
    net = caffe.Net(model_def,model_weights, caffe.TEST)
    
    start = time.time()   
    for i in range(data.shape[0]):
        print i,data[i].shape
         # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1,*data[i].shape)
        net.blobs['data'].data[...] = data[i]
        # run net and take argmax for prediction
        net.forward()
        out = net.blobs['score'].data[0].argmax(axis=0)
#         print 'score shape:',net.blobs['score'].data[0].shape
#         print 'score shape:',net.blobs['score'].data.shape
    end = time.time()
    print(end-start)
if __name__ == '__main__':
    cpu_start = time.time()
    test_image = '../data/road_data/Image/G30IA-550+338920-550+338440.jpg'
    inference(test_image)
    cpu_end = time.time() 
    print('cpu:', cpu_end - cpu_start)
