import tensorflow as tf


class ClassificationNetwork:
    def __init__(self, data, learning_rate=0.0001):
        self.is_training = tf.placeholder(tf.bool)
        self.learning_rate = learning_rate
        self.data = data
        self._last_output = self.data
        self.cost_eq = None
        self.pred_val = None
        self.opt = None
        self.acc = None

        self.layer_map = {}
        self.variables = {}
        self._var_idx = 0

    def add_layer(self, layer):
        layer.set_network(self)
        name = layer.name
        if name not in self.layer_map:
            self.layer_map[name] = layer
        else:
            raise Exception
        return layer.name

    def connect(self, layer_name, inputs=None):

        if not inputs:
            out = self.layer_map[layer_name].connect(self._last_output)
        else:
            out = self.layer_map[layer_name].connect(*[input for input in inputs])
        self._last_output = out
        return out

    def add_layer_and_connect(self, layer, input_layer_names=None):
        layer_name = self.add_layer(layer)
        return layer_name, self.connect(layer_name, input_layer_names)
        # return layer_name

    def build(self, true_labels):
        self.cost_eq = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=self._last_output, labels=true_labels))
        self.pred_val = tf.nn.softmax(self._last_output)
        self.opt = tf.train.AdamOptimizer(name="adam", learning_rate=self.learning_rate).minimize(self.cost_eq)
        prediction = tf.equal(tf.argmax(self.pred_val, 1), tf.argmax(true_labels, 1))
        self.acc = tf.reduce_sum(tf.cast(prediction, tf.float32))


class GANNetwork:
    def __init__(self):
        pass
