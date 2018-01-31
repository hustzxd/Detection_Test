import os
import numpy as np
import caffe
import matplotlib.image as mpimg
from utils.predict import *


def list_voc_test(imagesetfile):
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    return imagenames


def predict(pic_name, model_def, model_weights, use_gpu=False):
    if use_gpu:
        caffe.set_device(2)
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    image = caffe.io.load_image(pic_name)
    new_len, image = fill_image(image)  # padding the image

    transformer = caffe.io.Transformer({'data': (1, 3, 416, 416)})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    transformed_image = transformer.preprocess('data', image)
    # print transformed_image.shape

    net = caffe.Net(model_def, model_weights, caffe.TEST)
    net.blobs['data'].reshape(1, 3, 416, 416)
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()

    probs = net.blobs['region_output'].data[0][0]
    #     print probs

    boxes = []
    rec_results = []
    for prob in probs:
        box = Box(prob[0], prob[1], prob[2], prob[3], prob[4], int(prob[5]))
        boxes.append(box)

    lena = mpimg.imread(pic_name)
    height, width = lena.shape[:2]

    for box in boxes:
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
        #         print('{:f} {:f} {:f} {:f}'.format(left, top, right, bot))
        rec_results.append(rec_result(box.category + 1, box.prob, left, top, right, bot))
    return rec_results


def _write_results_for_eval():
    _devkit_path = 'data/Pascal_VOC/VOCdevkit'
    _res_prefix = './'
    _model_def = 'models/darknet_yolov2/deploy-voc.2.1.prototxt'
    _model_weights = 'models/darknet_yolov2/yolo-voc.2.0.caffemodel'
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
    for i, image in enumerate(imagelist):
        if (i % 10) == 0:
            print('{0}/{1}'.format(i, list_len))
        recs = predict(imagepath.format(image), _model_def, _model_weights, use_gpu=True)
        for rec in recs:
            filename.format(_classes[rec.idx])
            with open(filename.format(_classes[rec.idx]), 'a') as f:
                f.write('{0} {1} {2} {3} {4} {5}\n'.format(
                    image, rec.conf, rec.xmin, rec.ymin, rec.xmax, rec.ymax))

    # recs = predict(imagepath, imagelist, _model_def, _model_weights, use_gpu=True)


#     for rec in recs:
#         file.na
_write_results_for_eval()
