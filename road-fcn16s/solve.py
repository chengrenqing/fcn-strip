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

#weights = '../pascalcontext-fcn32s/pascalcontext-fcn32s.caffemodel'
weights = '../road-fcn32s/snapshot/model_530/train_iter_8000.caffemodel'

# init
caffe.set_device(1)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('../data/road/VOC2010/ImageSets/Main/val.txt', dtype=str)

for _ in range(50):
    solver.step(40000)
    score.seg_tests(solver, False, val, layer='score')
