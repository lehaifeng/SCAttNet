# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 09:23:34 2018

@author: qkj
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

from ops import *


def channel_spatial_block(input_feature, name, ratio=8):
    """Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    with tf.variable_scope(name):
        scale,attention_feature = channel_attention(input_feature, 'ch_at', ratio)
        concat,attention_feature = spatial_attention(attention_feature, 'sp_at')
    return attention_feature


def channel_attention(input_feature, name, ratio=8):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name):
        channel = input_feature.get_shape()[-1]
        avg_pool = tf.reduce_mean(input_feature, axis=[1, 2], keep_dims=True)

        assert avg_pool.get_shape()[1:] == (1, 1, channel)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_0',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel // ratio)
        avg_pool = tf.layers.dense(inputs=avg_pool,
                                   units=channel,
                                   kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer,
                                   name='mlp_1',
                                   reuse=None)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)

        max_pool = tf.reduce_max(input_feature, axis=[1, 2], keep_dims=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel // ratio,
                                   activation=tf.nn.relu,
                                   name='mlp_0',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel // ratio)
        max_pool = tf.layers.dense(inputs=max_pool,
                                   units=channel,
                                   name='mlp_1',
                                   reuse=True)
        assert max_pool.get_shape()[1:] == (1, 1, channel)
        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')


    return scale,input_feature * scale


def spatial_attention(input_feature, name):
    kernel_size = 7
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    with tf.variable_scope(name):
        avg_pool = tf.reduce_mean(input_feature, axis=[3], keep_dims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(input_feature, axis=[3], keep_dims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = tf.concat([avg_pool, max_pool], 3)
        assert concat.get_shape()[-1] == 2

        concat = tf.layers.conv2d(concat,
                                  filters=1,
                                  kernel_size=[kernel_size, kernel_size],
                                  strides=[1, 1],
                                  padding="same",
                                  activation=None,
                                  kernel_initializer=kernel_initializer,
                                  use_bias=False,
                                  name='conv')
        assert concat.get_shape()[-1] == 1
        concat = tf.sigmoid(concat, 'sigmoid')


    return concat,input_feature * concat


def n_enc_block(inputs, phase_train, n, k, name):
    h = inputs
    with tf.variable_scope(name):
        for i in range(n):
            h = conv2d(h, k, 3, stride=1, name='conv_{}'.format(i + 1))
            h = batch_norm(h, phase_train, name='bn_{}'.format(i + 1))
            h = relu(h, name='relu_{}'.format(i + 1))
        h, mask = maxpool2d_with_argmax(h, name='maxpool_{}'.format(i + 1))
       
    return h, mask


def encoder(inputs, phase_train, name='encoder'):
    with tf.variable_scope(name):
        h, mask_1 = n_enc_block(inputs, phase_train, n=2, k=64, name='block_1')
        h, mask_2 = n_enc_block(h, phase_train, n=2, k=128, name='block_2')
        h, mask_3 = n_enc_block(h, phase_train, n=3, k=256, name='block_3')
        h, mask_4 = n_enc_block(h, phase_train, n=3, k=512, name='block_4')
        h, mask_5 = n_enc_block(h, phase_train, n=3, k=512, name='block_5')
    return h, [mask_5, mask_4, mask_3, mask_2, mask_1]


def n_dec_block(inputs, mask, adj_k, phase_train, n, k, name):
    in_shape = inputs.get_shape().as_list()
    with tf.variable_scope(name):
        h = maxunpool2d(inputs, mask, name='unpool')
        for i in range(n):
            if i == (n - 1) and adj_k:
                h = conv2d(h, k / 2, 3, stride=1, name='conv_{}'.format(i + 1))
            else:
                h = conv2d(h, k, 3, stride=1, name='conv_{}'.format(i + 1))
            h = batch_norm(h, phase_train, name='bn_{}'.format(i + 1))
            h = relu(h, name='relu_{}'.format(i + 1))
        
    return h

def dec_last_conv(inputs, phase_train, k, name):
    with tf.variable_scope(name):
        h = conv2d(inputs, k, 1, name='conv')
    return h


def decoder(inputs, mask, phase_train, name='decoder'):
    with tf.variable_scope(name):

        h = n_dec_block(inputs, mask[0], False, phase_train, n=3, k=512, name='block_5')

        h = n_dec_block(h, mask[1], True, phase_train, n=3, k=512, name='block_4')

        h = n_dec_block(h, mask[2], True, phase_train, n=3, k=256, name='block_3')

        h = n_dec_block(h, mask[3], True, phase_train, n=2, k=128, name='block_2')

        h = n_dec_block(h, mask[4], True, phase_train, n=2, k=64, name='block_1')
        
        h=channel_spatial_block(h,'csb')
       
        h = dec_last_conv(h, phase_train, k=6, name='last_conv')
    logits = h
    return logits

def inference(inputs, phase_train):
    with tf.variable_scope('segnet'):
        h, mask = encoder(inputs, phase_train, name='encoder')
        logits = decoder(h, mask, phase_train, name='decoder')
    return logits

