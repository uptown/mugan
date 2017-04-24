from common.layer import *
from common.network import *
import tensorflow as tf

VGG_MEAN = [103.939, 116.779, 123.68]


def build(data, true_labels, keep_rate):
    # rgb_scaled = data * 255.0
    #
    # red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
    # bgr = tf.concat(axis=3, values=[
    #     blue - VGG_MEAN[0],
    #     green - VGG_MEAN[1],
    #     red - VGG_MEAN[2],
    # ])

    network = ClassificationNetwork(data)
    network.add_layer_and_connect(ConvolutionLayer(1, 64, c_size=3))
    network.add_layer_and_connect(ConvolutionLayer(64, 64, c_size=3))
    network.add_layer_and_connect(MaxPoolLayer())

    network.add_layer_and_connect(ConvolutionLayer(64, 128, c_size=3))
    network.add_layer_and_connect(ConvolutionLayer(128, 128, c_size=3))
    network.add_layer_and_connect(MaxPoolLayer())

    network.add_layer_and_connect(ConvolutionLayer(128, 256, c_size=3))
    network.add_layer_and_connect(ConvolutionLayer(256, 256, c_size=3))
    network.add_layer_and_connect(ConvolutionLayer(256, 256, c_size=3))
    network.add_layer_and_connect(ConvolutionLayer(256, 256, c_size=3))
    network.add_layer_and_connect(MaxPoolLayer())

    network.add_layer_and_connect(ConvolutionLayer(256, 512, c_size=3))
    network.add_layer_and_connect(ConvolutionLayer(512, 512, c_size=3))
    network.add_layer_and_connect(ConvolutionLayer(512, 512, c_size=3))
    network.add_layer_and_connect(ConvolutionLayer(512, 512, c_size=3))
    network.add_layer_and_connect(MaxPoolLayer())

    network.add_layer_and_connect(ConvolutionLayer(512, 512, c_size=3))
    network.add_layer_and_connect(ConvolutionLayer(512, 512, c_size=3))
    network.add_layer_and_connect(ConvolutionLayer(512, 512, c_size=3))
    network.add_layer_and_connect(ConvolutionLayer(512, 512, c_size=3))
    network.add_layer_and_connect(MaxPoolLayer())

    network.add_layer_and_connect(DropConnectedLayer(1 * 1 * 512, 4096, keep_rate))

    network.add_layer_and_connect(DropOutLayer(keep_rate))
    network.add_layer_and_connect(DropConnectedLayer(4096, 4096, keep_rate))
    network.add_layer_and_connect(DropOutLayer(keep_rate))
    network.add_layer_and_connect(DropConnectedLayer(4096, 2048, keep_rate))
    network.add_layer_and_connect(DropOutLayer(keep_rate))
    network.add_layer_and_connect(DropConnectedLayer(2048, 512, keep_rate))
    network.add_layer_and_connect(DropOutLayer(keep_rate))
    network.add_layer_and_connect(FullConnectedLayer(512, 10))

    # network.add_layer_and_connect(DropConnectedLayer(4096, 512 * 4 * 4, keep_rate))
    # network.add_layer_and_connect(DeconvolutionLayer([-1, 4, 4, 512], [-1, 8, 8, 256]))
    # network.add_layer_and_connect(DeconvolutionLayer([-1, 8, 8, 256], [-1, 16, 16, 128]))
    # network.add_layer_and_connect(DeconvolutionLayer([-1, 16, 16, 128], [-1, 32, 32, 64]))
    # network.add_layer_and_connect(DeconvolutionLayer([-1, 64, 64, 64], [-1, 128, 128, 32]))
    # network.add_layer_and_connect(DeconvolutionLayer([-1, 128, 128, 32], [-1, 256, 256, 16]))
    # network.add_layer_and_connect(DeconvolutionLayer([-1, 256, 256, 16], [-1, 512, 512, 8]))
    # network.add_layer_and_connect(DeconvolutionLayer([-1, 512, 512, 8], [-1, 512, 512, 4]))
    # network.add_layer_and_connect(DeconvolutionLayer([-1, 512, 512, 4], [-1, 512, 512, 2]))
    # network.add_layer_and_connect(DeconvolutionLayer([-1, 512, 512, 2], [-1, 512, 512, 1]))

    network.build(true_labels)

    def train(session, d, true_out, keep=0.3):
        return session.run(network.opt,
                           feed_dict={data: d, true_labels: true_out, keep_rate: keep})

    def cost(session, d, true_out):
        return session.run(network.cost_eq,
                           feed_dict={data: d, true_labels: true_out, keep_rate: 0})

    def accuracy(session, d, true_out):
        return session.run(network.acc, feed_dict={data: d, true_labels: true_out, keep_rate: 0})

    return train, cost, accuracy
