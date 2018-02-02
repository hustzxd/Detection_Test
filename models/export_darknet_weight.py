# -*- coding: utf-8 -*-
import os
import sys
import struct

cur_path = os.path.dirname(os.path.realpath(__file__))
caffe_root = os.path.join(cur_path, os.pardir, os.pardir)
pycaffe_path = os.path.join(caffe_root, 'python')
sys.path.append(pycaffe_path)
try:
    import caffe
except ImportError:
    print('cannot import caffe, check pycaffe')
    sys.exit(-1)

import numpy as np

def build_caffe_net(model_file):
    net = caffe.Net(model_file, caffe.TEST)
    net.forward()
    return net

def print_layer_names(net):
    print("Network layers:")
    for name, layer in zip(net._layer_names, net.layers):
        print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))

def splice_weight(weight, start, end):
    if start < 0 or end > weight.shape[0]:
        raise ValueError('out of range: weight shape: {}, start: {}, end: {}'.format(weight.shape[0],
            start, end))
    return weight[start:end]

def main(model_file, darknet_weight_file, caffe_weight_file):
    caffe.set_device(0)
    # caffe.set_model_gpu()
    net = build_caffe_net(model_file)
    
    print_layer_names(net)

    for name, blob in net.blobs.iteritems():
        print('blob name: {}, blob shape: {}'.format(name, blob.data.shape))
    print('=' * 30)
    count = 0
    for name, param in net.params.iteritems():
        print('layer name: {}'.format(name))
        for i in range(len(param)):
            count += np.prod(param[i].data.shape)
            print('{}th param shape: {}'.format(i+1, param[i].data.shape))
        print('*' * 30)

    print('=' * 30)
    print('weight total count: {}'.format(count))
    # transpose flag, the first 4 entries are major, minor, revision and net.seen
    with open(darknet_weight_file, 'rb') as f:
        bytes = f.read()
    major = struct.unpack("i", bytes[:4])[0]
    if major != 0:
        raise ValueError('unknown major version: {}'.format(major))
    minor = struct.unpack("i", bytes[4:8])[0]
    revision = struct.unpack("i", bytes[8:12])[0]
    if minor == 1:
        offset = 4 + 4 + 4 + 4
        netseen = struct.unpack("i", bytes[12:16])[0]  # for int
    elif minor == 2:
        offset = 4 + 4 + 4 + 8
        netseen = struct.unpack("Q", bytes[12:20])[0]  # for size_t (unsigned long)    
    else:
        raise ValueError('unknown minor version: {}'.format(minor))

    print('net info:')
    print('major: {}, minor: {}, revision: {}, netseen: {}'.format(major, minor, revision, netseen))
    
    bytes = bytes[offset:]
    if len(bytes) % 4 != 0:
        raise ValueError('left bytes cannot decoded by float')
    num_float = len(bytes) / 4    
    
    weight = np.array(struct.unpack('f'*num_float, bytes))
    
    assert weight.shape[0] == num_float
    print('weight number: {}'.format(num_float))

    num_weight = num_float

    print('=' * 30)
    print('converting weights...')
    count = 0
    params = net.params.keys()

    nlayers = len(params)
    print('there are {} layers need params'.format(nlayers))

    for i, param in enumerate(params):
        print('index: {:02d}, name: {}'.format(i, param))
        lidx = list(net._layer_names).index(param)
        layer = net.layers[lidx]
        print('layer index: {}'.format(lidx))
        if layer.type == 'Convolution':
            print('convolution layer')
            has_bias = len(net.params[param]) > 1
            ### read bias
            bias_dim = (net.params[param][0].data.shape[0],)  # number of outplanes
            bias_size = np.prod(bias_dim)
            bias = np.reshape(splice_weight(weight, count, count+bias_size), bias_dim)
            print('read bias done')
            if has_bias:
                net.params[param][1].data[...] = bias
                bias = None
            count += bias_size
            ### read bn param if has no bias
            if not has_bias:
                ### read scale ratio
                gamma_dim = bias_dim
                gamma_size = np.prod(gamma_dim)
                gamma = np.reshape(splice_weight(weight, count, count+gamma_size), gamma_size)
                count += gamma_size
                ### read bn mean and variance
                bn_mean_dim = bias_dim
                bn_mean_size = np.prod(bn_mean_dim)
                bn_var_dim = bias_dim
                bn_var_size = np.prod(bn_var_dim)
                bn_mean = np.reshape(splice_weight(weight, count, count+bn_mean_size), bn_mean_dim)
                count += bn_mean_size
                bn_var = np.reshape(splice_weight(weight, count, count+bn_var_size), bn_var_dim)
                count += bn_var_size
                print('read bn gamma/mean/var done')
            else:
                print('no bn layer')
            
            ### read weight
            conv_dim = net.params[param][0].data.shape
            weight_size = np.prod(conv_dim)
            conv_weight = np.reshape(splice_weight(weight, count, count+weight_size), conv_dim)
            count += weight_size
            print('read kernel weight done')
            net.params[param][0].data[...] = conv_weight

        elif layer.type == 'BatchNorm':
            print('bn layer')
            assert bn_mean is not None and bn_var is not None
            net.params[param][0].data[...] = bn_mean  # mean
            net.params[param][1].data[...] = bn_var  # variance
            net.params[param][2].data[...] = 1.0  # scale factor
        elif layer.type == 'Scale':
            print('scale layer')
            assert gamma is not None and bias is not None
            net.params[param][0].data[...] = gamma  # scale
            gamma = None
            net.params[param][1].data[...] = bias   # bias (beta)
            bias = None
        else:
            raise ValueError('unknown layer')

    if count == num_weight:
        print('you are right')
        net.save(caffe_weight_file)
    else:
        print('something error happen, check. weight in caffe: {}, weight read: {}'.format(count, num_weight))

def help():
    print("""Usage python export_darknet_weight.py model_file darknet_weight_file output_caffemodel""")

if __name__ == '__main__':
    if len(sys.argv) != 4:
        help()
        sys.exit(-1)
    else:
        model_file = sys.argv[1]
        darknet_weight_file = sys.argv[2]
        if not os.path.exists(model_file):
            print('cannot find given model file: {}'.format(model_file))
            sys.exit(-1)
        if not os.path.exists(darknet_weight_file):
            print('cannot find given darknet weight file: {}'.format(darknet_weight_file))
            sys.exit(-1)
        main(model_file, darknet_weight_file, sys.argv[3])
