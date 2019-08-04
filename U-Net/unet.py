# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 17:24:49 2018

@author: lhf
"""

from __future__ import print_function
import numpy as np
import os
import tensorflow as tf

_DEPTH = 2
_CHANNELS = [32, 64, 128,256,512]

def unet_model(input, is_training=True):
  convs = []
  # Down sampling
  for c in _CHANNELS:
    for _ in range(_DEPTH):
      input = tf.layers.conv2d(inputs=input,
                              filters=c,
                              kernel_size=(3, 3),
                              kernel_initializer=tf.variance_scaling_initializer(),
                              padding='SAME',
                              activation=None)
      input = tf.layers.batch_normalization(inputs=input,
                                         training=is_training,
                                         center=True,
                                         scale=True,
                                         fused=True)
      input = tf.nn.relu(input)

    convs.append(input)
    if not c == _CHANNELS[-1]:
      input = tf.layers.max_pooling2d(inputs=input,
                                      pool_size=(2, 2),
                                      strides=(2, 2))
    
        

  convs = reversed(convs[:-1])

  for index, c in enumerate(reversed(_CHANNELS[:-1])):
    input = tf.image.resize_images(input, tf.shape(input)[1:3] * 2)
    input = tf.layers.conv2d(inputs=input,
                             filters=c,
                             kernel_size=(3, 3),
                             kernel_initializer=tf.variance_scaling_initializer(),
                             padding='SAME',
                             activation=None)
    
        
    input = tf.concat([input, convs.next()], axis=3)
    for _ in range(_DEPTH):
      input = tf.layers.conv2d(inputs=input,
                               filters=c,
                               kernel_size=(3, 3),
                               kernel_initializer=tf.variance_scaling_initializer(),
                               padding='SAME',
                               activation=None)
      input = tf.layers.batch_normalization(inputs=input,
                                            training=is_training,
                                            center=True,
                                            scale=True,
                                            fused=True)
      input = tf.nn.relu(input)
      
        

  logits = tf.layers.conv2d(inputs=input,
                            filters=6,
                            kernel_size=(1, 1),
                            kernel_initializer=tf.variance_scaling_initializer(),
                            padding='SAME',activation=None)

  return logits
  
