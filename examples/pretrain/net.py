import sys
caffe_root = '../../'
sys.path.insert(0, caffe_root + 'python')
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def conv_relu(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2, pad=0):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride, pad=pad)

def fcn(split):
    n = caffe.NetSpec()
    pydata_params = dict(split=split, mean=(104.00699, 116.66877, 122.67892),
            seed=1337)
    if split == 'train':
        pydata_params['sbdd_dir'] = '../../data/SBD'
        pylayer = 'SBDDSegDataLayer'
    else:
        pydata_params['voc_dir'] = '../data/pascal/VOC2011'
        pylayer = 'VOCSegDataLayer'
    n.data, n.label = L.Python(module='voc_layers', layer=pylayer,
            ntop=2, param_str=str(pydata_params))

    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64)
    n.pool1 = max_pool(n.relu1_2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128)
    n.pool2 = max_pool(n.relu2_2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256)
    n.pool3 = max_pool(n.relu3_3)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
    n.pool4 = max_pool(n.relu4_3, 3, 1, 1)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, 5, 1, 2)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, 5, 1, 2)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, 5, 1, 2)

    # fully conv
    n.conv_dsp, n.relu_dsp = conv_relu(n.relu5_3, 64, ks=1, pad=0)
    n.score_fr = L.Convolution(n.relu_dsp, num_output=21, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.upsample = L.Deconvolution(n.score_fr,
        convolution_param=dict(num_output=21, kernel_size=16, stride=8,
            bias_term=False),
        param=[dict(lr_mult=0)])
    n.crop = crop(n.upsample, n.data)
    n.loss = L.SoftmaxWithLoss(n.crop, n.label,
            loss_param=dict(normalize=False, ignore_label=255))

    return n.to_proto()

def make_net():
    with open('train1.prototxt', 'w') as f:
        f.write(str(fcn('train')))

    #with open('val1.prototxt', 'w') as f:
    #    f.write(str(fcn('seg11valid')))

if __name__ == '__main__':
    make_net()
