from common.layer import *
from common.network import *

VGG_MEAN = [103.939, 116.779, 123.68]


def build(data, true_labels, dropout_rate, learning_rate=0.0001):

    rgb_scaled = data * 255.0

    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
    bgr = tf.concat(axis=3, values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ])

    vgg = ClassificationNetwork(bgr)
    vgg.add_layer_and_connect(ConvolutionLayer(3, 64, c_size=3))
    vgg.add_layer_and_connect(ConvolutionLayer(64, 64, c_size=3))
    vgg.add_layer_and_connect(MaxPoolLayer())

    vgg.add_layer_and_connect(ConvolutionLayer(64, 128, c_size=3))
    vgg.add_layer_and_connect(ConvolutionLayer(128, 128, c_size=3))
    vgg.add_layer_and_connect(MaxPoolLayer())

    vgg.add_layer_and_connect(ConvolutionLayer(128, 256, c_size=3))
    vgg.add_layer_and_connect(ConvolutionLayer(256, 256, c_size=3))
    vgg.add_layer_and_connect(ConvolutionLayer(256, 256, c_size=3))
    vgg.add_layer_and_connect(ConvolutionLayer(256, 256, c_size=3))
    vgg.add_layer_and_connect(MaxPoolLayer())

    vgg.add_layer_and_connect(ConvolutionLayer(256, 512, c_size=3))
    vgg.add_layer_and_connect(ConvolutionLayer(512, 512, c_size=3))
    vgg.add_layer_and_connect(ConvolutionLayer(512, 512, c_size=3))
    vgg.add_layer_and_connect(ConvolutionLayer(512, 512, c_size=3))
    vgg.add_layer_and_connect(MaxPoolLayer())

    vgg.add_layer_and_connect(ConvolutionLayer(512, 512, c_size=3))
    vgg.add_layer_and_connect(ConvolutionLayer(512, 512, c_size=3))
    vgg.add_layer_and_connect(ConvolutionLayer(512, 512, c_size=3))
    vgg.add_layer_and_connect(ConvolutionLayer(512, 512, c_size=3))
    vgg.add_layer_and_connect(MaxPoolLayer())

    vgg.add_layer_and_connect(DropConnectedLayer(7 * 7 * 512, 4096, dropout_rate))
    vgg.add_layer_and_connect(DropOutLayer(dropout_rate))
    vgg.add_layer_and_connect(DropConnectedLayer(4096, 4096, dropout_rate))
    vgg.add_layer_and_connect(DropOutLayer(dropout_rate))
    vgg.add_layer_and_connect(FullConnectedLayer(4096, 1000))

    vgg.build(true_labels)


    cost_eq = vgg.cost_eq
    pred_val = vgg.pred_val

    opt = tf.train.AdamOptimizer(name="adam", learning_rate=learning_rate).minimize(cost_eq)
    prediction = tf.equal(tf.argmax(pred_val, 1), tf.argmax(true_labels, 1))
    acc = tf.reduce_sum(tf.cast(prediction, tf.float32))

    def train(session, d, true_out, dropout=0.5):
        return session.run(opt,
                           feed_dict={data: d, true_labels: true_out, dropout_rate: dropout})

    def cost(session, d, true_out):
        return session.run(cost_eq,
                           feed_dict={data: d, true_labels: true_out, dropout_rate: 0})

    def accuracy(session, d, true_out):
        return session.run(acc, feed_dict={data: d, true_labels: true_out, dropout_rate: 0})

    return train, cost, accuracy
