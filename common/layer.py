import tensorflow as tf
from collections import defaultdict

_layer_count = defaultdict(lambda: 1)


class Layer:
    mean = 0.0
    stddev = 0.01
    prefix = "layer"

    def __init__(self):
        self.name = self.prefix + "_" + str(_layer_count[self.__class__])
        _layer_count[self.__class__] += 1
        self.network = None

    def set_network(self, network):
        self.network = network

    def connect(self, *args, **kwargs):
        raise NotImplemented

    def __call__(self, *args, **kwargs):
        with tf.variable_scope(self.name):
            return self.connect(*args, **kwargs)


class LSTMLayer(Layer):
    prefix = "lstm"

    def __init__(self, n_hidden, n_classes):
        super().__init__()

        self.n_hidden = n_hidden
        with tf.variable_scope(self.name):
            self.w = tf.get_variable("w", shape=[n_hidden, n_classes],
                                     initializer=tf.contrib.layers.xavier_initializer())
            self.b = tf.Variable(tf.zeros([n_classes]), name="b")
            # self.w = tf.Variable(tf.truncated_normal([n_hidden, n_classes], self.mean, self.stddev),
            #                      name="w")
            # self.b = tf.Variable(tf.truncated_normal([n_classes], self.mean, self.stddev), name="b")

    def connect(self, x):
        # x = tf.unstack(tf.reshape(tf.transpose(data, [0, 3, 1, 2]), [-1, 128 * 12, 12]), 128 * 12, 1)  # 128, 2, 2
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.n_hidden)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
        return tf.matmul(outputs[-1], self.w) + self.b


class BatchNormalizationLayer(Layer):
    prefix = "bn"

    def __init__(self, epsilon=1e-5, momentum=0.9):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum

    def connect(self, x):
        with tf.variable_scope(self.name):
            return tf.contrib.layers.batch_norm(x,
                                                decay=self.momentum,
                                                updates_collections=None,
                                                epsilon=self.epsilon,
                                                scale=True,
                                                is_training=self.network.is_training,
                                                scope=self.name)


class ConvolutionLayer(Layer):
    prefix = "conv"

    def __init__(self, in_ch, out_ch, c_size=3, strides=None, padding="SAME", activation=tf.nn.relu):
        super().__init__()
        if strides is None:
            strides = [1, 1, 1, 1]
        self.in_ch = in_ch
        self.out_ch = out_ch
        if type(c_size) in (list, tuple):
            h_c_size = c_size[0]
            w_c_size = c_size[1]
        else:
            h_c_size = w_c_size = c_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        with tf.variable_scope(self.name):
            self.f = tf.get_variable("f", shape=[h_c_size, w_c_size, self.in_ch, self.out_ch],
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            self.b = tf.Variable(tf.zeros([self.out_ch]), name="b")
            # self.bn_layer = BatchNormalizationLayer()

    def connect(self, data):
        t = tf.nn.conv2d(data, self.f, strides=self.strides, padding=self.padding)
        t = tf.nn.bias_add(t, self.b)
        # t = self.bn_layer.connect(t)
        # with tf.variable_scope(self.name):
        #     t = batch_norm(t, self.out_ch, is_training)
        return self.activation(t) if self.activation else t


class ActivationLayer(Layer):
    prefix = "atv"

    def __init__(self, activation=tf.nn.relu):
        super().__init__()
        self.activation = activation

    def connect(self, x):
        return self.activation(x)


class DeconvolutionLayer(Layer):
    prefix = "deconv"

    def __init__(self, in_shape, out_shape, c_size=3, strides=None, padding="SAME", activation=tf.nn.relu):
        super().__init__()
        if strides is None:
            strides = [1, 1, 1, 1]

        self.out_shape = out_shape
        self.in_shape = in_shape
        if type(c_size) in (list, tuple):
            h_c_size = c_size[0]
            w_c_size = c_size[1]
        else:
            h_c_size = w_c_size = c_size
        self.strides = strides
        self.padding = padding
        self.activation = activation

        with tf.variable_scope(self.name):
            self.f = tf.get_variable("f", shape=[h_c_size, w_c_size, in_shape[-1], out_shape[-1]],
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            self.b = tf.Variable(tf.zeros([out_shape[-1]]), name="b")

            # self.f = tf.Variable(
            #     tf.truncated_normal([h_c_size, w_c_size, in_ch, out_shape[-1]], self.mean, self.stddev),
            #     name="f")
            # self.b = tf.Variable(tf.truncated_normal([out_shape[-1]], self.mean, self.stddev), name="b")

    def connect(self, data):
        t = tf.nn.conv2d_transpose(tf.reshape(data, self.in_shape), self.f, self.out_shape, strides=self.strides,
                                   padding=self.padding)
        t = tf.nn.bias_add(t, self.b)
        return self.activation(t) if self.activation else t


class MaxPoolLayer(Layer):
    prefix = "mpool"

    def __init__(self, k_size=None, strides=None, padding='SAME'):
        super().__init__()

        if strides is None:
            strides = [1, 2, 2, 1]
        if k_size is None:
            k_size = [1, 2, 2, 1]
        self.k_size = k_size
        self.strides = strides
        self.padding = padding

    def connect(self, data):
        return tf.nn.max_pool(data, ksize=self.k_size, strides=self.strides, padding=self.padding, name=self.name)


class RegionalSelectionLayer(Layer):
    prefix = "r_s"

    def __init__(self, region_count, size):
        super().__init__()
        self.region_count = region_count
        self.size = size
        self.s = tf.range(0, size, 1)
        self.region_map = []
        for i in range(2, region_count + 2):
            region = []
            for j in range(size):
                region.append((j // i) % 2)
            self.region_map.append(region)
        self.region_map = tf.constant(self.region_map)

    def connect(self, data, selected_param):  # [20, features], [20]
        return tf.multiply(tf.to_float(tf.reshape(self.region_map[selected_param], [1, -1])),
                           tf.reshape(data, [-1, self.size]))


class AvgPoolLayer(Layer):
    prefix = "apool"

    def __init__(self, k_size=None, strides=None, padding='SAME'):
        super().__init__()

        if strides is None:
            strides = [1, 2, 2, 1]
        if k_size is None:
            k_size = [1, 2, 2, 1]
        self.k_size = k_size
        self.strides = strides
        self.padding = padding

    def connect(self, data):
        return tf.nn.avg_pool(data, ksize=self.k_size, strides=self.strides, padding=self.padding, name=self.name)


class FullConnectedLayer(Layer):
    prefix = "fc"

    def __init__(self, in_ch, out_ch, activation=tf.nn.relu):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.activation = activation
        with tf.variable_scope(self.name):

            if type(in_ch) in (list, tuple):
                self.w = tf.get_variable("w", shape=[sum(self.in_ch), self.out_ch],
                                         initializer=tf.contrib.layers.xavier_initializer())
                # self.w = tf.Variable(tf.truncated_normal([sum(self.in_ch), self.out_ch], self.mean, self.stddev),
                #                      name="w")
            else:
                self.w = tf.get_variable("w", shape=[self.in_ch, self.out_ch],
                                         initializer=tf.contrib.layers.xavier_initializer())
                # self.w = tf.Variable(tf.truncated_normal([self.in_ch, self.out_ch], self.mean, self.stddev),
                #                      name="w")
            self.b = tf.Variable(tf.zeros([self.out_ch]), name="b")

    def connect(self, *args):
        if len(args) > 1:
            # data_min = tf.reduce_mean(data)
            # print(data_min)
            data = tf.reshape(args[0], [-1, self.in_ch[0]])
            i = 1
            in_ch = self.in_ch[0]
            for each in args[1:]:
                data = tf.concat([data, tf.reshape(each, [-1, self.in_ch[i]])], 1)
                in_ch += self.in_ch[i]
                i += 1
            x = tf.reshape(data, [-1, in_ch])
        else:
            data = args[0]  # tf.reshape(args[0], [-1])

            if type(self.in_ch) in (list, tuple):
                x = tf.reshape(data, [-1, self.in_ch[0]])
            else:
                x = tf.reshape(data, [-1, self.in_ch])
        fc = tf.nn.bias_add(tf.matmul(x, self.w), self.b)

        return self.activation(fc)


class DropConnectedLayer(FullConnectedLayer):
    prefix = "d_fc"

    def __init__(self, in_ch, out_ch, rate, activation=tf.nn.relu):
        super().__init__(in_ch, out_ch, activation)
        self.rate = rate
        self.w = tf.nn.dropout(self.w, self.rate)


class DropOutLayer(Layer):
    prefix = "do"

    def __init__(self, rate):
        super().__init__()

        self.rate = rate

    def connect(self, *args):
        data = args[0]
        return tf.nn.dropout(data, self.rate, name=self.name)
