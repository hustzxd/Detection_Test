import os
import sys
import caffe
import matplotlib.image as mpimg
from utils.predict import *
import math


def list_voc_test(imagesetfile):
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    return imagenames


def _write_results_for_eval(_model_def, _model_weights, _devkit_path):
    _res_prefix = './'
    _year = '2007'
    _classes = ('__background__',  # always index 0
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor')
    _res_prefix = os.path.join(
        _res_prefix,
        'results/VOC' + _year,
        'Main')
    imagesetfile = os.path.join(
        _devkit_path,
        'VOC' + _year,
        'ImageSets',
        'Main',
        'test.txt')
    imagepath = os.path.join(
        _devkit_path,
        'VOC' + _year,
        'JPEGImages',
        '{:s}.jpg')
    if not os.path.isdir(_res_prefix):
        os.makedirs(_res_prefix)
    filename = _res_prefix + '/comp4_det_test_{:s}.txt'
    for cls in _classes:
        if cls == '__background__':
            continue
        if not os.path.isfile(filename.format(cls)):
            os.mknod(filename.format(cls))
    imagelist = list_voc_test(imagesetfile)
    list_len = len(imagelist)
    batch_size = 32
    print('batch_size{}'.format(batch_size))
    batch_num = int(list_len / batch_size)
    left_num = list_len - batch_size * batch_num

    caffe.set_device(0)  # if we have multiple GPUs, pick the first one
    caffe.set_mode_gpu()
    net = caffe.Net(_model_def,  # defines the structure of the model
                    _model_weights,  # contains the trained weights
                    caffe.TEST)  # use test mode (e.g., don't perform dropout)
    net.blobs['data'].reshape(batch_size, 3, 416, 416)
    transformer = caffe.io.Transformer({'data': (1, 3, 416, 416)})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    # transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    # transformed_image = transformer.preprocess('data', image)

    for i in range(batch_num + 1):
        print('{}/{}'.format(i, batch_num))
        if i == batch_num:
            batch_size = left_num
            net.blobs['data'].reshape(batch_size, 3, 416, 416)
        transformer_images = np.zeros([batch_size, 3, 416, 416])
        # print(transformer_images.shape)
        new_lens = []
        for j in range(batch_size):
            index = batch_size * i + j
            image = caffe.io.load_image(imagepath.format(imagelist[index]))
            # print('image.shape: {}'.format(image.shape))
            new_len, image = fill_image(image)  # padding the image
            # print('image.shape2: {}'.format(image.shape))
            transformer_images[j,:,:,:] = transformer.preprocess('data', image)
            new_lens.append(new_len)
        # print('transformer_images{}'.format(transformer_images.shape))
        net.blobs['data'].data[...] = transformer_images
        net.forward()
        probs = net.blobs['region_output'].data[0][0]
        rec_results = []
        boxes = []
        for prob in probs:
            batch_id = int(prob[6])
            index = batch_size * i + batch_id
            box = Box(x=prob[0], y=prob[1], w=prob[2], h=prob[3], prob=prob[4], category=int(prob[5]),
                      image=imagelist[index])
            # print('box={}'.format(box))
            boxes.append(box)
        # for j in range(batch_size):
            # index = batch_size * i + j
        for box in boxes:
            lena = mpimg.imread(imagepath.format(box.image))
            height, width = lena.shape[:2]
            # print('height, width: {}  {}'.format(height, width))
            new_len = max(height, width)

            x = box.x
            y = box.y
            h = box.h
            w = box.w
            left = (x - w / 2) * new_len
            left = left - (new_len - width) / 2
            right = (x + w / 2) * new_len
            right = right - (new_len - width) / 2
            top = (y - h / 2) * new_len
            top = top - (new_len - height) / 2
            bot = (y + h / 2) * new_len
            bot = bot - (new_len - height) / 2
            if left < 0:
                left = 0
            if right > width - 1:
                right = width - 1
            if top < 0:
                top = 0
            if bot > height - 1:
                bot = height - 1
            rec_results.append(
                rec_result(image_id=box.image, idx=box.category + 1, conf=box.prob, xmin=left, ymin=top, xmax=right,
                           ymax=bot))
            # print('{} {} {} {}'.format(left, top, right, bot))

        for rec in rec_results:
            # print('rec: {}'.format(rec))
            filename.format(_classes[rec.idx])
            with open(filename.format(_classes[rec.idx]), 'a') as f:
                f.write('{0} {1} {2} {3} {4} {5}\n'.format(rec.image_id, rec.conf, rec.xmin, rec.ymin, rec.xmax, rec.ymax))

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python [prototxt] [caffemodel] [devkit_path_root]')
        exit(0)
    model_def = sys.argv[1]
    model_weights = sys.argv[2]
    devkit_root = sys.argv[3]
    print('{}'.format(devkit_root))
    _write_results_for_eval(model_def, model_weights, devkit_root)
