import caffe
import surgery, score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

#weights = '../ilsvrc-nets/vgg16-fcn.caffemodel'
weights = '../ilsvrc-nets/VGG_ILSVRC_16_layers.caffemodel'
vgg_proto = '../ilsvrc-nets/VGG_ILSVRC_16_layers_deploy.prototxt'

# init
caffe.set_device(1)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
#solver.net.copy_from(weights)
#colin new insert
vgg_net = caffe.Net(vgg_proto,weights,caffe.TRAIN)
surgery.transplant(solver.net,vgg_net)
del vgg_net

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('../data/road/VOC2010/ImageSets/Main/val.txt', dtype=str)

for _ in range(50):
    solver.step(30001)
    score.seg_tests(solver, False, val, layer='score')
