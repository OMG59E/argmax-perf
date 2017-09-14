#!/usr/bin/python

from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import time
import tensorflow as tf
import numpy as np

bsize = 64
hsize = 30000

with tf.device('/gpu:0'):
    data = tf.get_variable('data', [bsize, hsize], dtype=tf.float32)
    #data = tf.ones([bsize, hsize], tf.float32)
    #out = tf.arg_max(data, 1)
    out = tf.argmax2d(data)

config = tf.ConfigProto(allow_soft_placement=False,
                        log_device_placement=True)
with tf.Session(config=config) as sess:
    for i in range(10):
        t0 = time.time() * 1e6
        sess.run(out, feed_dict = {data: np.ones([bsize, hsize], np.float32)})
        #sess.run(out)
        t1 = time.time() * 1e6
        print("%d us" % (t1 - t0))

