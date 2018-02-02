""" export cfg file to caffe prototxt
"""
import os
import sys
import math


def input_template(batch_size, channels, height, width):
    return """layer {{
  name: "data"
  type: "Input"
  top: "data"
  input_param {{ shape: {{ dim: {} dim: {} dim: {} dim: {} }} }}
}}
""".format(batch_size, channels, height, width)


def conv_template(conv_layer):
    bottom_blob_name = conv_layer.bottom_blob_names[0]
    top_blob_name = conv_layer.top_blob_names[0]
    return """layer {{
  name: "{l.conv_name}"
  type: "Convolution"
  bottom: "{}"
  top: "{}"
  convolution_param {{
    num_output: {l.filters}
    kernel_size: {l.kernel_size}
    pad: {l.pad}
    stride: {l.stride}
    bias_term: {l.has_bias}
  }}
}}
""".format(bottom_blob_name,
           top_blob_name,
           l=conv_layer)


def batchnorm_template(layer):
    blob_name = layer.top_blob_names[0]
    return """layer {{
  name: "{l.bn_name}"
  type: "BatchNorm"
  bottom: "{}"
  top: "{}"
  batch_norm_param {{
    use_global_stats: true
  }}
}}
layer {{
  name: "{l.scale_name}"
  type: "Scale"
  bottom: "{}"
  top: "{}"
  scale_param {{
    bias_term: true
  }}
}}
""".format(blob_name, blob_name, blob_name, blob_name, l=layer)


def nonlinear_template(layer):
    blob_name = layer.top_blob_names[0]
    if layer.activation == 'leaky':
        param = """  relu_param{
    negative_slope: 0.1
  }"""
    else:
        param = ''
    return """layer {{
  name: "{l.activation_name}"
  type: "ReLU"
  bottom: "{}"
  top: "{}"
  {}
}}
""".format(blob_name, blob_name, param, l=layer)


def poolint_template(layer):
    bottom_blob_name = layer.bottom_blob_names[0]
    top_blob_name = layer.top_blob_names[0]
    return """layer {{
  name: "{l.name}"
  type: "Pooling"
  bottom: "{}"
  top: "{}"
  pooling_param {{
    pool: MAX
    kernel_size: {l.kernel_size}
    stride: {l.stride}
  }}
}}
""".format(bottom_blob_name, top_blob_name, l=layer)


def reorg_template(layer):
    bottom_blob_name = layer.bottom_blob_names[0]
    top_blob_name = layer.top_blob_names[0]
    return """layer {{
  name: "{l.name}"
  type: "Reorg"
  bottom: "{}"
  top: "{}"
  reorg_param {{
    stride: {l.stride}
  }}
}}
""".format(bottom_blob_name, top_blob_name, l=layer)


def route_template(route_layer):
    if len(route_layer.bottom_blob_names) == 1:
        return ''
    bottoms = ['  bottom: \"{}\"'.format(x) for x in route_layer.bottom_blob_names]
    bottoms = '\n'.join(bottoms)
    top_blob_name = route_layer.top_blob_names[0]
    return """layer {{
  name: "{l.name}"
  type: "Concat"
{}
  top: "{}"
}}
""".format(bottoms, top_blob_name, l=route_layer)


def region_template(layer):
    bottom_blob_name = layer.bottom_blob_names[0]
    top_blob_name = layer.name + '_act'
    part1 = """layer {{
  name: "{l.region_name}"
  type: "Region"
  bottom: "{}"
  top: "{}"
  region_param {{
    softmax: true
    classes: {l.classes}
    boxes_of_each_grid: {l.num}
  }}
}}
""".format(bottom_blob_name, top_blob_name, l=layer)

    bottom_blob_name = top_blob_name
    top_blob_names = ['  top: \"{}\"'.format(x) for x in layer.top_blob_names]
    top_blob_names = '\n'.join(top_blob_names)

    anchors = ['    anchor_coords {{pw: {} ph: {}}}'.format(x, y) for x, y in zip(layer.ax, layer.ay)]
    anchors = '\n'.join(anchors)

    part2 = """layer {{
  name: "{l.region_output_name}"
  type: "RegionOutput"
  bottom: "{}"
{}
  region_output_param {{
    classes: {l.classes}
    boxes_of_each_grid: {l.num}                                                                                   
{}
  }}
}}
""".format(bottom_blob_name, top_blob_names, anchors, l=layer)
    return part1 + part2


# layer list of the net
layers = []


class Layer(object):
    def __init__(self, name, layer_index):
        self.name = name
        self.layer_index = layer_index
        self.bottom_blob_names = []
        self.top_blob_names = []

    def __repr__(self):
        raise NotImplementedError


class ConvLayer(Layer):
    activation_types = ['relu', 'leaky', 'linear']

    def __init__(self, name, layer_index, **kwargs):
        super(ConvLayer, self).__init__(name, layer_index)
        if 'batch_normalize' in kwargs:
            self.batch_normalize = kwargs['batch_normalize']
        else:
            self.batch_normalize = 0
        if 'filters' not in kwargs:
            raise ValueError('cannot find filter param')
        else:
            self.filters = kwargs['filters']
        if 'size' not in kwargs:
            raise ValueError('cannot find kerner size')
        else:
            self.kernel_size = kwargs['size']
        if 'stride' not in kwargs:
            raise ValueError('cannot find stride param')
        else:
            self.stride = kwargs['stride']
        if 'pad' in kwargs:
            self.pad = kwargs['pad']
        else:
            self.pad = 0
        if self.pad:
            self.pad = self.kernel_size // 2
        if 'activation' in kwargs:
            self.activation = kwargs['activation']
            if self.activation not in self.activation_types:
                raise ValueError('unkown activate type: {}'.format(self.activation))
        else:
            self.activation = 'linear'
        ## conv
        self.conv_name = self.name + '_conv'
        ## batch norm
        if self.batch_normalize:
            self.bn_name = self.name + '_bn'
            self.scale_name = self.name + '_scale'
        ## non linear activation
        self.has_nonlinear = self.activation != 'linear'
        self.activation_name = self.name + '_relu'

        ## bottom blob and top blob
        if layer_index == 1:
            # this is the first conv layer
            self.bottom_blob_names = ['data']
        else:
            prev_layer = layers[-1]
            self.bottom_blob_names = [prev_layer.top_blob_names[0]]
        self.top_blob_names = [self.name]

        ## bias of conv
        self.has_bias = "false " if self.batch_normalize else "true"

        layers.append(self)

    def __repr__(self):
        text = conv_template(self)
        if self.batch_normalize:
            text += batchnorm_template(self)
        if self.has_nonlinear:
            text += nonlinear_template(self)
        return text


class PoolingLayer(Layer):
    def __init__(self, name, layer_index, **kwargs):
        super(PoolingLayer, self).__init__(name, layer_index)
        if 'size' in kwargs:
            self.kernel_size = kwargs['size']
        else:
            raise ValueError('cannot find size in pooling')
        if 'stride' in kwargs:
            self.stride = kwargs['stride']
        else:
            raise ValueError('cannot find stride in pooling')
        prev_layer = layers[-1]
        if len(prev_layer.top_blob_names) != 1:
            raise ValueError('input layer of pooling should have 1 top blob, '
                             'but {} has {}'.format(prev_layer.name,
                                                    len(prev_layer.top_blob_names)))
        self.bottom_blob_names = [prev_layer.top_blob_names[0]]
        self.top_blob_names = [self.name]

        layers.append(self)

    def __repr__(self):
        return poolint_template(self)


class ReorgLayer(Layer):
    def __init__(self, name, layer_index, **kwargs):
        super(ReorgLayer, self).__init__(name, layer_index)
        if 'stride' in kwargs:
            self.stride = kwargs['stride']
        else:
            raise ValueError('cannot find stride in reorg')
        prev_layer = layers[-1]
        prev_top_blob_names = prev_layer.top_blob_names

        if len(prev_top_blob_names) != 1:
            raise ValueError('input layer of reorg should have 1 blob, '
                             'but {} has {}'.format(prev_layer.name,
                                                    len(prev_top_blob_names)))
        self.bottom_blob_names = [prev_top_blob_names[0]]
        self.top_blob_names = [self.name]

        layers.append(self)

    def __repr__(self):
        return reorg_template(self)


class RouteLayer(Layer):
    def __init__(self, name, layer_index, **kwargs):
        super(RouteLayer, self).__init__(name, layer_index)
        if 'layers' in kwargs:
            self.layers = kwargs['layers']
        else:
            raise ValueError('cannot find layers in route')
        if not isinstance(self.layers, list):
            self.layers = [self.layers]

        self.input_layers = [layers[lidx] for lidx in self.layers]

        self.bottom_blob_names = []
        for l in self.input_layers:
            if len(l.top_blob_names) != 1:
                raise ValueError('input layer of route should have only 1 output blob, '
                                 'but {} has {}'.format(l.name, len(l.top_blob_names)))
            self.bottom_blob_names.append(l.top_blob_names[0])

        self.bottom_blob_names = [x.top_blob_names[0] for x in self.input_layers]
        if len(self.bottom_blob_names) == 1:
            # only change path, dont do concat
            self.top_blob_names = [self.bottom_blob_names[0]]
        else:
            self.top_blob_names = [self.name]

        layers.append(self)

    def __repr__(self):
        return route_template(self)


class RegionLayer(Layer):
    def __init__(self, name, layer_index, **kwargs):
        super(RegionLayer, self).__init__(name, layer_index)
        if 'anchors' in kwargs:
            self.anchors = kwargs['anchors']
        else:
            raise ValueError('cannot find anchors in region')
        if 'classes' in kwargs:
            self.classes = kwargs['classes']
        else:
            raise ValueError('cannot find classes in region')
        if 'coords' in kwargs:
            self.coords = kwargs['coords']
            if self.coords != 4:
                raise ValueError('coords should be 4, while get {}'.format(self.coords))
        if 'num' in kwargs:
            self.num = kwargs['num']
        else:
            self.num = 5

        self.region_name = name + '_region'
        self.region_output_name = name + '_output'
        len_anchors = len(self.anchors)
        if len_anchors != 2 * self.num:
            raise ValueError('anchor size dismatch with box number')

        self.ax = [self.anchors[i * 2] for i in range(self.num)]
        self.ay = [self.anchors[i * 2 + 1] for i in range(self.num)]

        prev_layer = layers[-1]
        prev_blobs = prev_layer.top_blob_names

        if len(prev_blobs) != 1:
            raise ValueError('input layer of region should have 1 blob, '
                             'but {} has {}'.format(prev_layer.name, len(prev_blobs)))

        self.bottom_blob_names = [prev_blobs[0]]
        self.top_blob_names = ['pred_box', 'pred_prob']

        layers.append(self)

    def __repr__(self):
        return region_template(self)


def find_param(block):
    # skip []
    items = block[1:]
    d = {}
    for item in items:
        comma_pos = item.index('=')
        key = item[:comma_pos]
        value_str = item[comma_pos + 1:]

        try:
            value = float(value_str)
        except ValueError:
            try:
                value = [float(x) for x in value_str.split(',')]
            except ValueError:
                value = value_str
        # try to convert to int
        if not isinstance(value, str):
            if isinstance(value, list):
                value_int = [int(x) for x in value]
                for x, y in zip(value_int, value):
                    if math.fabs(x - y) > 0.0001:
                        break
                else:
                    value = value_int
            else:
                value_int = int(value)
                if math.fabs(value_int - value) < 0.0001:
                    value = value_int
        d[key] = value

    return d


def parse_block(block, layer_index):
    params = find_param(block)
    if block[0].find('[convolutional]') != -1:
        name = 'conv{}'.format(layer_index)
        return ConvLayer(name, layer_index, **params)
    elif block[0].find('[maxpool]') != -1:
        name = 'pool{}'.format(layer_index)
        return PoolingLayer(name, layer_index, **params)
    elif block[0].find('[reorg]') != -1:
        name = 'reorg{}'.format(layer_index)
        return ReorgLayer(name, layer_index, **params)
    elif block[0].find('[route]') != -1:
        name = 'route{}'.format(layer_index)
        return RouteLayer(name, layer_index, **params)
    elif block[0].find('[region]') != -1:
        name = 'region'
        return RegionLayer(name, layer_index, **params)
    elif block[0].find('[net]') != -1:
        return input_template(1, params['channels'], params['height'], params['width'])


def parse_cfg(cfg_file, output_file):
    with open(cfg_file, 'r') as f:
        lines = f.readlines()
    # remove blank line
    lines = [l.strip().replace(' ', '') for l in lines]
    lines = [l for l in lines if len(l) > 0 and not l.startswith('#')]
    # find block start line number
    split_line_nums = []
    for i, line in enumerate(lines):
        if line.find('[') != -1 and line.find(']') != -1:
            split_line_nums.append(i)
    # net block, for input shape parsing
    net_block = lines[split_line_nums[0]: split_line_nums[1]]
    input_layer = parse_block(net_block, None)

    # skip [net] block
    split_line_nums = split_line_nums[1:]
    num_blocks = len(split_line_nums)

    # add end line number for last block
    split_line_nums.append(len(lines))

    for i, start_line in enumerate(split_line_nums[:-1]):
        layer_index = i + 1
        end_line = split_line_nums[i + 1]
        block = lines[start_line: end_line]

        parse_block(block, layer_index)

    with open(output_file, 'w') as f:
        f.write('name: \"YOLONET\"\n')
        f.write(input_layer)
        for l in layers:
            f.write(str(l))


def help():
    print("""Usage: python export_cfg.py cfg_file output_caffe_prototxt""")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        help()
        sys.exit(-1)
    # cfg_file = '/home/xiaomeifeng/yolo-related/caffe/models/yolo/yolo.cfg'
    cfg_file = sys.argv[1]
    if not os.path.exists(cfg_file):
        raise ValueError('cannot open cfgfile: {}'.format(cfg_file))

    # prototxt_file = './yolo.prototxt'
    prototxt_file = sys.argv[2]
    parse_cfg(cfg_file, prototxt_file)
