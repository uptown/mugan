import tensorflow as tf


class ClassificationNetwork:

    def __init__(self, is_training, data):
        self.is_training = is_training
        self.data = data
        self.dropout_rate = tf.placeholder(tf.float32)
        self._last_output = self.data
        self.cost_eq = None
        self.pred_val = None
        self.layer_map = {}

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


class GANNetwork:

    def __init__(self):
        pass