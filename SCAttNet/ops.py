""" ops.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages


def xavier_initializer(uniform=True, seed=None, dtype=tf.float32):
    return tf.contrib.layers.xavier_initializer(
        uniform=uniform, seed=seed, dtype=dtype)


def conv2d(incoming, num_filters, filter_size, stride=1, pad='SAME',
           activation=tf.identity,
           weight_init=xavier_initializer(),
           bias_init=tf.constant_initializer(0.0),
           reuse=False, name="conv2d"):

    x = incoming

    input_shape = incoming.get_shape().as_list()
    filter_shape = [filter_size, filter_size, input_shape[-1], num_filters]
    strides = [1, stride, stride, 1]
    bias_shape = [num_filters]

    with tf.variable_scope(name, reuse=reuse) as scope:
        weight = tf.get_variable(name + "_weight", filter_shape,
                                 initializer=weight_init)
        bias = tf.get_variable(name + "_bias", bias_shape,
                               initializer=bias_init)

        conved = tf.nn.conv2d(x, weight, strides, pad)
        conved = tf.nn.bias_add(conved, bias)
        output = activation(conved)

    return output


def maxpool2d(incoming, pool_size, stride=2, pad='SAME', name="maxpool2d"):
    x = incoming
    input_shape = incoming.get_shape().as_list()
    filter_shape = [1, pool_size, pool_size, 1]
    strides = [1, stride, stride, 1]

    with tf.name_scope(name) as scope:
        pooled = tf.nn.max_pool(x, filter_shape, strides, pad)

    return pooled


def maxpool2d_with_argmax(incoming, pool_size=2, stride=2,
                          name='maxpool_with_argmax'):
    x = incoming
    filter_shape = [1, pool_size, pool_size, 1]
    strides = [1, stride, stride, 1]

    with tf.name_scope(name):
        _, mask = tf.nn.max_pool_with_argmax(
            x, ksize=filter_shape, strides=strides, padding='SAME')
        mask = tf.stop_gradient(mask)

        pooled = tf.nn.max_pool(
            x, ksize=filter_shape, strides=strides, padding='SAME')

    return pooled, mask


def upsample(incoming, size, name='upsample'):
    x = incoming
    with tf.name_scope(name) as scope:
        resized = tf.image.resize_nearest_neighbor(x, size=size)
    return resized


# https://github.com/Pepslee/tensorflow-contrib/blob/master/unpooling.py
def maxunpool2d(incoming, mask, stride=2, name='unpool'):
    x = incoming

    input_shape = incoming.get_shape().as_list()
    strides = [1, stride, stride, 1]
    output_shape = (input_shape[0],
                    input_shape[1] * strides[1],
                    input_shape[2] * strides[2],
                    input_shape[3])

    flat_output_shape = [output_shape[0], np.prod(output_shape[1:])]
    with tf.name_scope(name):
        flat_input_size = tf.size(x)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=mask.dtype),
                                 shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(mask) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        mask_ = tf.reshape(mask, [flat_input_size, 1])
        mask_ = tf.concat([b, mask_], 1)

        x_ = tf.reshape(x, [flat_input_size])
        ret = tf.scatter_nd(mask_, x_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
        return ret


# https://github.com/tflearn/tflearn/blob/master/tflearn/layers/normalization.py
def batch_norm(incoming, phase_train,
               epsilon=1e-4, alpha=0.1, decay=0.9,
               beta_init=tf.constant_initializer(0.0),
               gamma_init=tf.random_normal_initializer(mean=1.0, stddev=0.002),
               reuse=False, name='batch_norm'):

    x = incoming

    input_shape = incoming.get_shape().as_list()

    depth = input_shape[-1]
    with tf.variable_scope(name, reuse=reuse) as scope:
        beta = tf.get_variable(name + '_beta', shape=depth,
                               initializer=beta_init, trainable=True)
        gamma = tf.get_variable(name + '_gamma', shape=depth,
                                initializer=gamma_init, trainable=True)

        axes = list(range(len(input_shape) - 1))
        batch_mean, batch_variance = tf.nn.moments(incoming, axes) # channel
        moving_mean = tf.get_variable(
            name + '_moving_mean', shape=depth,
            initializer=tf.zeros_initializer(),
            trainable=False)
        moving_variance = tf.get_variable(
            name + '_moving_variance', shape=depth,
            initializer=tf.constant_initializer(1.0),
            trainable=False)

        def update():
            update_moving_mean = moving_averages.assign_moving_average(
                moving_mean, batch_mean, decay, zero_debias=False)
            update_moving_variance = moving_averages.assign_moving_average(
                moving_variance, batch_variance,  decay, zero_debias=False)

            with tf.control_dependencies(
                [update_moving_mean, update_moving_variance]):
                return tf.identity(batch_mean), tf.identity(batch_variance)

        mean, variance = tf.cond(phase_train,
                                 update,
                                 lambda: (moving_mean, moving_variance))

        output = tf.nn.batch_normalization(
            x, mean, variance, beta, gamma, epsilon)

    return output


def relu(incoming, summary=False, name='relu'):
    x = incoming
    with tf.name_scope(name) as scope:
        output = tf.nn.relu(x)
    return output
