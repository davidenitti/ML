# Copyright 2016 Davide Nitti. All Rights Reserved.
# Licensed under GNU GENERAL PUBLIC LICENSE Version 3
# ==============================================================================

import matplotlib.pyplot as plt

# Import MINST data
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import numpy as np
import math
import tensorflow as tf

# Parameters
global_step = tf.Variable(0, trainable=False, name="global_step")
learning_rate = tf.train.exponential_decay(0.003,
                                           global_step,
                                           100,
                                           0.995,
                                           staircase=True)
training_epochs = 150
batch_size = 150
display_step = 2

# Network Parameters
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])



def autoenc(_X, h_shape, _weights, _biases):
    """autoencoder that returns the reconstructed data and the encoded data"""
    layer = _X
    for h in xrange(0, len(h_shape) - 1):
        _weights[h] = tf.Variable(
                            tf.random_normal([h_shape[h], h_shape[h + 1]], stddev=1. / math.sqrt(float(h_shape[h]))))
        _biases[h] = tf.Variable(tf.zeros([h_shape[h + 1]]))
        layer = tf.nn.tanh(tf.add(tf.matmul(layer, _weights[h]), _biases[h]))
        if h == len(h_shape) / 2 - 1:
            code_layer = layer
    return layer, code_layer


# Store layers weight & bias
weights = {}
biases = {}

# Construct model
dim = 20
pred, code = autoenc(x, [n_input, 200, 110, 70, 30, dim, 30, 70, 110, 200, n_input], weights, biases)
pred = pred * 0.5 + 0.5 # pred is between 0 and 1
loss = tf.reduce_mean((pred - x) ** 2)
for ww in weights:
    loss += tf.nn.l2_loss(weights[ww]) * 0.000001
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.7).minimize(loss,
                                                                                          global_step=global_step)

# Initializing the variables
init = tf.initialize_all_variables()

plt.rcParams['figure.figsize'] = (6.0, 5.0);
plt.rcParams['image.interpolation'] = 'nearest'

plt.ion()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            plt.clf()
            numhidden1vis = 10
            rec = sess.run(pred, feed_dict={x: mnist.train.images[0:numhidden1vis]})
            rec = np.maximum(np.minimum(1, rec), 0)

            code_train = sess.run(code, feed_dict={x: mnist.train.images})
            # sampling a code assuming the codes are distributed as a Gaussian and computing the reconstruction
            sampled = sess.run(pred, feed_dict={
                code: np.random.multivariate_normal(np.mean(code_train, 0), np.cov(code_train.T), 20)})
            #sampled = np.maximum(np.minimum(1, sampled), 0)

            plt.subplot(4, numhidden1vis + 1,  1)
            plt.text(0, -9., 'Training digits', fontsize=14)
            plt.subplot(4, numhidden1vis + 1, 1+ numhidden1vis + 1)
            plt.text(0, -9., 'Reconstructed digits', fontsize=14)
            plt.subplot(4, numhidden1vis + 1, 1 + numhidden1vis + 1 + numhidden1vis + 1)
            plt.text(0, -9., 'Generated digits', fontsize=14)
            plt.subplot(4, numhidden1vis + 1, 1 + numhidden1vis + 1 + numhidden1vis + 1 + numhidden1vis + 1)
            plt.text(0, -9., 'Morphing', fontsize=14)
            for ind in range(0, numhidden1vis):
                plt.subplot(4, numhidden1vis + 1, ind + 1)

                plt.imshow(mnist.train.images[ind].reshape(28, 28), vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
                plt.axis('off')
                plt.subplot(4, numhidden1vis + 1, ind + 1 + numhidden1vis + 1)
                plt.imshow(rec[ind].reshape(28, 28), vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
                plt.axis('off')
                plt.subplot(4, numhidden1vis + 1, ind + 1 + numhidden1vis + 1 + numhidden1vis + 1)
                plt.imshow(sampled[ind].reshape(28, 28), vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
                plt.axis('off')
                alpha = 1.0 * ind / (numhidden1vis - 1)
                # morphing: generating a reconstruction from a code "between" the codes of 2 given numbers
                # the new code is a weighted average of 2 codes, with weights alpha and (1-alpha)
                morph = sess.run(pred, feed_dict={code: (1 - alpha) * code_train[0:1] + (alpha) * code_train[1:2]})
                plt.subplot(4, numhidden1vis + 1, ind + 1 + numhidden1vis + 1 + numhidden1vis + 1 + numhidden1vis + 1)
                plt.imshow(morph.reshape(28, 28), vmin=0, vmax=1, cmap=plt.get_cmap('gray'))
                plt.axis('off')
            plt.draw()
            plt.pause(0.01)
            print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost), "learn rate {:.9f}".format(
                learning_rate.eval())
            # print mean.eval(),vv.eval()
    print "Optimization Finished!"