import caffe
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import matplotlib.image as mpimg
import sys

class Box(object):
    def __init__(self, x, y, w, h, prob=0.0, category=-1):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.prob = prob
        self.category = category

    def __str__(self):
        return "({0},{1},{2},{3},{4},{5})".format(self.x, self.y, self.w, self.h, self.prob, self.category)


def get_names_from_file(filename):
    result = []
    with open(filename, 'r') as rf:
        lines = rf.readlines()
        for line in lines:
            result.append(line.replace('\n', ''))
    return result


def get_color_from_file(filename):
    colors = []
    with open(filename, 'r') as rf:
        lines = rf.readlines()
        for line in lines:
            words = line.split(r',')
            color = (int(words[0]), int(words[1]), int(words[2]))
            colors.append(color)
    return colors


def draw_image(pic_name, new_len, boxes, namelist_file):
    name_list = get_names_from_file(namelist_file)
    color_list = get_color_from_file('data/Pascal_VOC/ink.color')
    im = Image.open(pic_name)
    draw = ImageDraw.Draw(im)
    lena = mpimg.imread(pic_name)
    height, width = lena.shape[:2]
    print(height, width)
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
        category = name_list[box.category]
        color = color_list[box.category % len(color_list)]
        draw.line((left, top, right, top), fill=color, width=2)
        draw.line((right, top, right, bot), fill=color, width=2)
        draw.line((left, top, left, bot), fill=color, width=2)
        draw.line((left, bot, right, bot), fill=color, width=2)
        font_size = 20
        my_font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-M.ttf", size=font_size)
        draw.text([left + 5, top], category, font=my_font, fill=color)
        print('[{} {} {} {}]'.format(left, top, right, bot))
    #     im.show()
    # plt.imshow(im)
    im.show()
    # plt.savefig(im, 'pre.jpg')


def fill_image(image):
    height = image.shape[0]
    width = image.shape[1]
    new_len = 0
    image_fill = 0
    if height == width:
        image_fill = image
        new_len = height
    if height > width:
        image_fill = np.empty(height * height * 3)
        image_fill.fill(0.5)
        image_fill = image_fill.reshape([height, height, 3])
        diff = height - width
        diff = diff / 2
        image_fill[:, diff:diff + width, :] = image[:, :, :]
        new_len = height
    if height < width:
        image_fill = np.empty(width * width * 3)
        image_fill.fill(0.5)
        image_fill = image_fill.reshape([width, width, 3])
        diff = width - height
        diff = diff / 2
        image_fill[diff:diff + height, :, :] = image[:, :, :]
        new_len = width
    return new_len, image_fill


def main():
    caffe.set_device(0)
    caffe.set_mode_gpu()
    # caffe.set_mode_cpu()
    # pic_name = 'data/Pascal_VOC/VOCdevkit/VOC2007/JPEGImages/000003.jpg'
    image = caffe.io.load_image(pic_name)
    new_len, image = fill_image(image)
    # plt.imshow(image)

    transformer = caffe.io.Transformer({'data': (1, 3, 416, 416)})
    transformer.set_transpose('data', (2, 0, 1))  # move image channels to outermost dimension
    # transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR
    transformed_image = transformer.preprocess('data', image)
    print(transformed_image.shape)

    # model_def = 'models/darknet_yolov2/deploy-voc.2.1.prototxt'
    # model_weights = 'models/darknet_yolov2/yolo-voc.2.0.caffemodel'

    net = caffe.Net(model_def, model_weights, caffe.TEST)
    net.blobs['data'].reshape(1, 3, 416, 416)
    net.blobs['data'].data[...] = transformed_image
    net.forward()

    # feat = net.blobs['region1'].data[0][0]
    # print feat.shape
    # print feat:

    probs = net.blobs['region_output'].data[0][0]
    print(probs)

    boxes = []
    for prob in probs:
        box = Box(prob[0], prob[1], prob[2], prob[3], prob[4], int(prob[5]))
        boxes.append(box)

    draw_image(pic_name, new_len, boxes=boxes, namelist_file='data/Pascal_VOC/label_map.txt')


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python [prototxt] [caffemodel] [image_path]')
        exit(0)
    model_def = sys.argv[1]
    model_weights = sys.argv[2]
    pic_name = sys.argv[3]
    main()
