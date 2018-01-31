import numpy as np
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


class rec_result(object):
    def __init__(self, idx, conf, xmin, ymin, xmax, ymax):
        self.idx = idx
        self.conf = conf
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def __str__(self):
        return '(idx, conf, xmin, ymin, xmax, ymax) = ({0},{1},{2},{3},{4},{5})'.format(
            self.idx, self.conf, self.xmin, self.ymin, self.xmax, self.ymax)


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
