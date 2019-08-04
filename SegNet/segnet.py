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
        h = dec_last_conv(h, phase_train, k=6, name='last_conv')
    logits = h
    return logits

def inference(inputs, phase_train):
    with tf.variable_scope('segnet'):
        h, mask = encoder(inputs, phase_train, name='encoder')
        logits = decoder(h, mask, phase_train, name='decoder')
    return logits